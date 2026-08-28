#!/usr/bin/env python3
"""
FULL 7-LAYER GRASP PIPELINE -- real finger movement, real force feedback,
NO attach/glue mechanism anywhere. The object is held purely by friction
from the fingers actually closing on it.
"""
import time
import json
import base64
import os
import subprocess
import numpy as np
import requests
import cv2
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from sensor_msgs.msg import JointState, Image
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from cv_bridge import CvBridge
import ikpy.chain
import sys

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"
URDF_PATH = "/tmp/real_robot_exact.urdf"
EXPERIENCE_LOG = "/home/mahimaa/ur_gz_ws/experience_log.json"

ARM_JOINTS = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
              'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
FINGER_GROUPS = ["R_Index", "R_Middle", "R_Ring", "R_Pinky", "R_Thumb"]
FINGER_JOINT_NAMES = [f"{g}_Pitch" for g in FINGER_GROUPS]
FINGER_SECONDARY_JOINTS = {
    "R_Index": ["R_Index_Flexor", "R_Index_DIP"],
    "R_Middle": ["R_Middle_Flexor", "R_Middle_DIP"],
    "R_Ring": ["R_Ring_Flexor", "R_Ring_DIP"],
    "R_Pinky": ["R_Pinky_Flexor", "R_Pinky_DIP"],
    "R_Thumb": ["R_Thumb_Flexor", "R_Thumb_DIP"],
}

EFFORT_CONTACT_THRESHOLD = 0.12
EFFORT_DANGER_THRESHOLD = 0.60
MAX_CLOSE_STEPS = 30
CLOSE_STEP_SIZE = 0.05
STEP_DURATION = 0.25
# *_Pitch upper limit is 1.309 rad; *_Flexor/*_DIP (commanded at 0.8x this value) are
# limited to 1.047 rad. 1.25 keeps both under their limit with margin at fragility=0.
MAX_PITCH_CEILING = 1.25

# find_station.py swept 70 candidate (X, Y) combinations and computed the REAL
# shoulder-pan mismatch for each via solve_ik (not a guess) -- confirmed that no
# station gets close to a "natural" (near-zero mismatch) reach for apple_06's low
# grasp height; the best of all 70 was still 35.6 degrees off. This is the best one
# found: (1.40, -0.90). The twisted arm shape is an inherent characteristic of
# reaching this low with the hand pointed straight down, not a station-placement bug.
DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y = 1.40, -0.90
DELIVERY_ROBOT_YAW = 1.5708
# Place target in the delivery station's local (base_footprint-relative) frame --
# same frame/convention as apple grasp targets, deliberately on the opposite side
# of the robot from the apple row (negative x_local) so it can't overlap an apple.
# x_local magnitude (0.6) chosen so the crate's world position clears the mobile
# base's own rotated footprint (0.6x0.8m) with margin -- see crate model comments.
# UNVERIFIED against the real solver/sim -- test this before running pick_all_apples.py.
CRATE_LOCAL_XY = (-0.6, 0.0)
CRATE_LOCAL_Z = 0.08


def local_to_world(x_local, y_local, robot_x, robot_y, robot_yaw):
    cos_yaw = np.cos(robot_yaw)
    sin_yaw = np.sin(robot_yaw)
    return (x_local * cos_yaw - y_local * sin_yaw + robot_x,
            x_local * sin_yaw + y_local * cos_yaw + robot_y)


def world_to_local(world_x, world_y, robot_x, robot_y, robot_yaw):
    """Inverse of local_to_world: express a world (x, y) in the robot-frame
    convention solve_ik expects, undoing the station's teleported position/yaw."""
    dx = world_x - robot_x
    dy = world_y - robot_y
    cos_yaw = np.cos(-robot_yaw)
    sin_yaw = np.sin(-robot_yaw)
    return (dx * cos_yaw - dy * sin_yaw, dx * sin_yaw + dy * cos_yaw)


# World-frame position the crate must be spawned at in apple_world.world for the
# place target above to actually land inside it. See apple_gripper_sim/models/crate.
CRATE_WORLD_XY = local_to_world(CRATE_LOCAL_XY[0], CRATE_LOCAL_XY[1],
                                 DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)


def build_chain():
    chain = ikpy.chain.Chain.from_urdf_file(
        URDF_PATH,
        base_elements=[
            'base_footprint', 'base_footprint_joint', 'mobile_base_link',
            'base_joint', 'base_link',
            'shoulder_pan_joint', 'shoulder_link',
            'shoulder_lift_joint', 'upper_arm_link', 'elbow_joint',
            'forearm_link', 'wrist_1_joint', 'wrist_1_link',
            'wrist_2_joint', 'wrist_2_link', 'wrist_3_joint',
            'wrist_3_link', 'wrist_3-flange', 'flange',
            'flange-tool0', 'tool0', 'tool0_to_dexhand', 'dexhand_base_link'
        ]
    )
    mask = [False] * len(chain.links)
    for i, link in enumerate(chain.links):
        if link.name in ARM_JOINTS:
            mask[i] = True
    chain.active_links_mask = mask
    return chain


# 1cm was too tight -- borderline-converging targets could pass or fail depending on
# tiny floating-point differences between runs/environments. 2cm has enough margin for
# this task (finger closing envelope is far larger) while still being solidly accurate.
IK_ERROR_TOLERANCE = 0.025
IK_FALLBACK_ERROR_CEILING = 0.05
# Was +1 (untested guess). User confirmed the hand was touching the ground beside
# the apple, not the apple itself, with the wrist position otherwise verified
# correct -- consistent with fingers facing the wrong horizontal direction.
# Flipping to test the opposite sign.
HAND_FACING_SIGN = -1


def solve_ik(chain, target_xyz, init=None):
    # Computed early so guesses can be seeded AT the target's actual direction --
    # confirmed by testing that fixed-angle presets alone (0.3/-0.3 rad) let the solver
    # converge to a ~132-138 degree "mirror" configuration regardless of where the
    # target actually was (reproduced with expected_pan both at 0 and 12.8 degrees,
    # landing on nearly the same wrong shoulder angle both times) -- the fixed presets
    # just weren't close enough to the correct basin for this arm/orientation combo.
    expected_pan = np.arctan2(target_xyz[1], target_xyz[0])

    guesses = []
    if init is not None:
        guesses.append(init)
    presets = [
        {'shoulder_lift_joint': -1.5708, 'wrist_1_joint': -1.5708, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -1.0, 'elbow_joint': 1.2, 'wrist_1_joint': -1.7, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -0.8, 'elbow_joint': 1.6, 'wrist_1_joint': -2.3, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -2.0, 'elbow_joint': 1.5, 'wrist_1_joint': 0.0, 'wrist_2_joint': -1.5708},
        {'shoulder_pan_joint': 0.3, 'shoulder_lift_joint': -1.2, 'elbow_joint': 1.8, 'wrist_2_joint': -1.5708},
        {'shoulder_pan_joint': -0.3, 'shoulder_lift_joint': -1.4, 'elbow_joint': 1.4, 'wrist_1_joint': -1.5, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -0.6, 'elbow_joint': 1.0, 'wrist_1_joint': -1.9, 'wrist_2_joint': -1.5708},
        # Low-reach configurations: arm swung further down/out, biased for targets near
        # the ground where the straight-down orientation constraint is harder to satisfy.
        {'shoulder_lift_joint': -1.8, 'elbow_joint': 2.0, 'wrist_1_joint': -1.77, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -0.4, 'elbow_joint': 0.8, 'wrist_1_joint': -1.98, 'wrist_2_joint': -1.5708},
        # Same rest pose used by reset_everything() -- a known-sane, non-extreme
        # posture -- but with shoulder_pan rotated to actually face the target.
        {'shoulder_pan_joint': expected_pan, 'shoulder_lift_joint': -1.2, 'elbow_joint': 1.5,
         'wrist_1_joint': -1.9, 'wrist_2_joint': 0.0},
        {'shoulder_pan_joint': expected_pan, 'shoulder_lift_joint': -0.9, 'elbow_joint': 1.3,
         'wrist_1_joint': -1.97, 'wrist_2_joint': -1.5708},
        {'shoulder_pan_joint': expected_pan, 'shoulder_lift_joint': -0.5, 'elbow_joint': 0.9,
         'wrist_1_joint': -2.0, 'wrist_2_joint': -1.5708},
        {},
    ]
    for preset in presets:
        g = [0.0] * len(chain.links)
        for i, link in enumerate(chain.links):
            if link.name in preset:
                g[i] = preset[link.name]
        guesses.append(g)

    valid_solutions = []
    best_solution, best_error = None, float('inf')
    for g in guesses:
        solution = chain.inverse_kinematics(
            target_xyz, initial_position=g,
            target_orientation=[0, 0, -1], orientation_mode='Z'
        )
        achieved = chain.forward_kinematics(solution)[:3, 3]
        error = np.linalg.norm(np.array(target_xyz) - achieved)
        if error < best_error:
            best_error, best_solution = error, solution
        if error < IK_ERROR_TOLERANCE:
            valid_solutions.append(solution)

    if not valid_solutions:
        # Nothing hit the normal tolerance -- fall back to the closest solution found,
        # as long as it's not wildly off, rather than failing outright on a near-miss.
        if best_solution is not None and best_error < IK_FALLBACK_ERROR_CEILING:
            valid_solutions = [best_solution]
        else:
            print(f"[solve_ik] UNREACHABLE target={target_xyz}: best error across "
                  f"{len(guesses)} guesses was {best_error:.4f}m "
                  f"(tolerance={IK_ERROR_TOLERANCE}m, fallback ceiling={IK_FALLBACK_ERROR_CEILING}m)")
            return None

    def pan_of(sol):
        for link, a in zip(chain.links, sol):
            if link.name == 'shoulder_pan_joint':
                return float(a)
        return 0.0

    def angle_diff(a, b):
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return abs(d)

    def total_motion(sol):
        return sum(abs(a) for link, a in zip(chain.links, sol) if link.name in ARM_JOINTS)

    # Only a Z-axis constraint above ("point down") leaves the hand's rotation around
    # that now-vertical axis free -- it can end up facing any horizontal direction,
    # including away from the target (a hard full-orientation constraint was tried and
    # made many positions unreachable, so this is a soft preference instead).
    # HAND_FACING_SIGN flips which way "toward" means if this guess is backwards.
    horiz = np.array([target_xyz[0], target_xyz[1], 0.0])
    horiz_norm = np.linalg.norm(horiz)
    desired_facing = (horiz / horiz_norm) if horiz_norm > 1e-6 else np.array([1.0, 0.0, 0.0])
    desired_facing = desired_facing * HAND_FACING_SIGN

    def facing_alignment(sol):
        hand_x_axis = chain.forward_kinematics(sol)[:3, 0]
        return float(np.dot(hand_x_axis, desired_facing))

    # PRIMARILY avoid "mirror" arm configurations (same wrist position, shoulder rotated
    # to roughly the opposite side, elbow flipped) -- a real bug that slipped through the
    # previous all-or-nothing 60-degree cutoff (it fell back to accepting ANY solution,
    # including mirrored ones, when nothing passed). Sorting by pan mismatch first, with
    # facing/motion only as tiebreaks among comparably-good arm configurations, prevents
    # that regardless of whether anything happens to clear a fixed threshold.
    best = min(valid_solutions, key=lambda sol: (
        angle_diff(pan_of(sol), expected_pan), -facing_alignment(sol), total_motion(sol)))
    joints = {link.name: float(a) for link, a in zip(chain.links, best) if link.name in ARM_JOINTS}
    return [joints[j] for j in ARM_JOINTS], best


class FullLayerGraspNode(Node):
    def __init__(self, robot_x=DELIVERY_ROBOT_X, robot_y=DELIVERY_ROBOT_Y, robot_yaw=DELIVERY_ROBOT_YAW):
        super().__init__('full_layer_grasp_node')
        self.robot_x = robot_x
        self.robot_y = robot_y
        self.robot_yaw = robot_yaw
        self.target_name = None
        self.chain = build_chain()
        self.bridge = CvBridge()

        self.pose_sub = None
        self.joint_sub = self.create_subscription(
            JointState, '/joint_states', self._joint_cb, 10)
        self.camera_sub = self.create_subscription(
            Image, '/gripper_camera', self._camera_cb, 10)

        self.arm_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.hand_pub = self.create_publisher(
            JointTrajectory, '/dexhand_controller/joint_trajectory', 10)

        self.target_pose = None
        self.latest_joint_state = {}
        self.latest_frame = None

    def set_target(self, target_name):
        if self.pose_sub is not None:
            self.destroy_subscription(self.pose_sub)
        self.target_name = target_name
        self.target_pose = None
        self.pose_sub = self.create_subscription(
            Pose, f'/model/{target_name}/pose', self._pose_cb, 10)

    def world_to_local(self, world_x, world_y):
        return world_to_local(world_x, world_y, self.robot_x, self.robot_y, self.robot_yaw)

    def _pose_cb(self, msg):
        self.target_pose = msg

    def _joint_cb(self, msg):
        for name, pos, eff in zip(msg.name, msg.position, msg.effort):
            self.latest_joint_state[name] = (pos, eff)

    def _camera_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception:
            pass

    def wait_for(self, check_fn, timeout=5.0):
        start = time.time()
        while rclpy.ok() and not check_fn() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return check_fn()

    def teleport(self, x, y, yaw):
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        result = subprocess.run(
            'ign service -s /world/apple_world/set_pose '
            '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
            '--timeout 2000 '
            f"--req 'name: \"ur\" position: {{x: {x} y: {y} z: 0.0}} "
            f"orientation: {{x: 0 y: 0 z: {qz} w: {qw}}}'",
            shell=True, capture_output=True, text=True
        )
        self.get_logger().info(
            f"Base teleport to ({x:.3f}, {y:.3f}, yaw={yaw:.3f}): "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}")
        time.sleep(1.5)

    def reset_everything(self):
        self.get_logger().info("=== RESET: base position ===")
        self.teleport(self.robot_x, self.robot_y, self.robot_yaw)

        self.get_logger().info("=== RESET: arm to natural rest pose ===")
        home = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]
        self.send_arm_trajectory(home, 4.0)
        time.sleep(4.5)

        self.get_logger().info("=== RESET: hand fully open ===")
        self.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
        time.sleep(1.5)

    def layer1_vlm_analysis(self):
        self.get_logger().info("[Layer 1] Capturing gripper camera frame for VLM...")
        if not self.wait_for(lambda: self.latest_frame is not None, timeout=5.0):
            self.get_logger().warn("[Layer 1] No camera frame -- using fallback defaults.")
            return {"object_name": self.target_name, "fragility_score": 5,
                    "estimated_material": "unknown", "recommended_grip_force": "medium",
                    "confidence": 0.0, "notes": "no camera data available"}

        success, buffer = cv2.imencode(".jpg", self.latest_frame)
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        prompt = """You are a robotic grasping assistant. Look at the image and respond
with ONLY a valid JSON object (no markdown) with these exact keys:
{
  "object_name": "best guess at what the object is",
  "fragility_score": 0-10 integer, 0=indestructible, 10=extremely fragile,
  "estimated_material": "e.g. fruit skin, plastic, ceramic",
  "recommended_grip_force": "low, medium, or high",
  "confidence": 0.0-1.0 float,
  "notes": "one short sentence"
}"""
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL_NAME, "prompt": prompt, "images": [image_b64],
                "stream": False, "format": "json"
            }, timeout=240)  # CPU-only inference is slow, especially cold start
            response.raise_for_status()
            raw = response.json().get("response", "").strip()
            result = json.loads(raw)
            self.get_logger().info(f"[Layer 1] VLM analysis: {result}")
            return result
        except Exception as e:
            self.get_logger().warn(f"[Layer 1] VLM call failed ({e}) -- using fallback.")
            return {"object_name": self.target_name, "fragility_score": 5,
                    "estimated_material": "unknown", "recommended_grip_force": "medium",
                    "confidence": 0.0, "notes": f"VLM unavailable: {e}"}

    def layer2_imagination(self, vlm_result):
        fragility = vlm_result.get("fragility_score", 5)
        object_mass = 0.18
        mu = 0.8
        g = 9.81
        min_force_n = (object_mass * g) / (2 * mu)
        safety_margin = 1.0 + (10 - fragility) / 10.0
        target_effort = min_force_n * safety_margin
        # *_Pitch joints are hard-limited to 1.309 rad and *_Flexor/*_DIP to 1.047 rad
        # (see dexhandv2_right.urdf). command_fingers() sends secondary joints at 0.8x
        # this value, so the ceiling must leave both limits with margin.
        max_pitch = MAX_PITCH_CEILING - (fragility / 10.0) * 0.5
        self.get_logger().info(
            f"[Layer 2] Physics plan: min_force={min_force_n:.2f}N, "
            f"target_effort_est={target_effort:.2f}, max_pitch={max_pitch:.2f}")
        return {"max_pitch": max_pitch, "target_effort_est": target_effort}

    def command_fingers(self, positions_by_group, step_duration):
        all_names = list(FINGER_JOINT_NAMES)
        all_positions = [positions_by_group[g] for g in FINGER_GROUPS]
        for g in FINGER_GROUPS:
            for j_name in FINGER_SECONDARY_JOINTS[g]:
                all_names.append(j_name)
                all_positions.append(positions_by_group[g] * 0.8)
        msg = JointTrajectory()
        msg.joint_names = all_names
        point = JointTrajectoryPoint()
        point.positions = all_positions
        point.time_from_start.nanosec = int(step_duration * 1e9)
        msg.points = [point]
        self.hand_pub.publish(msg)

    def layer3_4_close_with_feedback(self, plan, vlm_result):
        fragility = vlm_result.get("fragility_score", 5)
        force_word = vlm_result.get("recommended_grip_force", "medium")
        speed_scale = {"low": 1.6, "medium": 1.0, "high": 0.6}.get(force_word, 1.0)
        step_duration = STEP_DURATION * speed_scale
        contact_threshold = EFFORT_CONTACT_THRESHOLD * (1.0 + (10 - fragility) / 20.0)

        self.get_logger().info(
            f"[Layer 3/4] Closing: max_pitch={plan['max_pitch']:.2f}, "
            f"step_duration={step_duration:.2f}s, contact_threshold={contact_threshold:.3f}Nm")

        contacted = {g: False for g in FINGER_GROUPS}
        current = {g: 0.0 for g in FINGER_GROUPS}
        max_effort_seen = {g: 0.0 for g in FINGER_GROUPS}

        for step in range(MAX_CLOSE_STEPS):
            if all(contacted.values()):
                self.get_logger().info(f"[Layer 3/4] All fingers contacted after {step} steps.")
                break
            for g in FINGER_GROUPS:
                if not contacted[g]:
                    current[g] = min(current[g] + CLOSE_STEP_SIZE, plan['max_pitch'])
            self.command_fingers(current, step_duration)
            time.sleep(step_duration)
            rclpy.spin_once(self, timeout_sec=0.1)

            for g in FINGER_GROUPS:
                name = f"{g}_Pitch"
                pos = self.latest_joint_state.get(name, (0, 0))[0]
                effort = abs(self.latest_joint_state.get(name, (0, 0))[1])
                max_effort_seen[g] = max(max_effort_seen[g], effort)
                if effort > contact_threshold and not contacted[g]:
                    contacted[g] = True
                    self.get_logger().info(f"  [Layer 3] {g}: contact (effort={effort:.3f}Nm)")
                elif step == MAX_CLOSE_STEPS - 1 and not contacted[g]:
                    self.get_logger().info(
                        f"  [Layer 3] {g}: NO contact. commanded={current[g]:.3f}rad "
                        f"actual_pos={pos:.3f}rad peak_effort={max_effort_seen[g]:.4f}Nm "
                        f"(threshold={contact_threshold:.3f}Nm)")

        n_contacted = sum(contacted.values())
        return n_contacted >= 3

    def layer5_smooth_lift(self, grasp_joints):
        self.get_logger().info("[Layer 5] Smooth joint-space lift...")
        lift_joints = list(grasp_joints)
        lift_joints[1] -= 0.35
        self.send_arm_trajectory(lift_joints, 5.0)
        return lift_joints

    def layer6_safety_monitor(self, duration=6.0):
        self.get_logger().info("[Layer 6] Monitoring for dangerous force spikes...")
        start = time.time()
        aborted = False
        while time.time() - start < duration:
            rclpy.spin_once(self, timeout_sec=0.2)
            for g in FINGER_GROUPS:
                name = f"{g}_Pitch"
                effort = abs(self.latest_joint_state.get(name, (0, 0))[1])
                if effort > EFFORT_DANGER_THRESHOLD:
                    self.get_logger().warn(
                        f"[Layer 6] DANGER: {g} effort={effort:.3f}Nm -- opening hand and aborting hold.")
                    aborted = True
                    break
            if aborted:
                break
        if aborted:
            self.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 0.5)
            time.sleep(0.5)
        else:
            self.get_logger().info("[Layer 6] No safety violations detected.")
        return not aborted

    def layer_place(self):
        """Carry the held apple to the crate at the delivery station and release it."""
        self.get_logger().info("[Place] Teleporting to delivery station...")
        prev_x, prev_y, prev_yaw = self.robot_x, self.robot_y, self.robot_yaw
        self.teleport(DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)
        self.robot_x, self.robot_y, self.robot_yaw = DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW

        x_local, y_local = CRATE_LOCAL_XY
        approach_result = solve_ik(self.chain, [x_local, y_local, CRATE_LOCAL_Z + 0.15])
        target_result = solve_ik(self.chain, [x_local, y_local, CRATE_LOCAL_Z])
        if approach_result is None or target_result is None:
            self.get_logger().error(
                "[Place] Crate position unreachable -- apple still held, will drop on next reset.")
            self.robot_x, self.robot_y, self.robot_yaw = prev_x, prev_y, prev_yaw
            return False

        approach_joints, _ = approach_result
        place_joints, _ = target_result

        self.get_logger().info("[Place] Moving above crate...")
        self.send_arm_trajectory(approach_joints, 3.5)
        time.sleep(4.0)

        self.get_logger().info("[Place] Lowering into crate...")
        self.send_arm_trajectory(place_joints, 2.5)
        time.sleep(3.0)

        self.get_logger().info("[Place] Releasing...")
        self.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
        time.sleep(1.5)

        self.get_logger().info("[Place] Retreating...")
        retreat_joints = list(place_joints)
        retreat_joints[1] -= 0.3
        self.send_arm_trajectory(retreat_joints, 3.0)
        time.sleep(3.5)

        self.robot_x, self.robot_y, self.robot_yaw = prev_x, prev_y, prev_yaw
        return True

    def layer7_log_experience(self, record):
        history = []
        if os.path.exists(EXPERIENCE_LOG):
            try:
                with open(EXPERIENCE_LOG) as f:
                    history = json.load(f)
            except Exception:
                history = []
        history.append(record)
        with open(EXPERIENCE_LOG, "w") as f:
            json.dump(history, f, indent=2)
        self.get_logger().info(f"[Layer 7] Logged experience to {EXPERIENCE_LOG}")

    def send_arm_trajectory(self, joint_positions, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = ARM_JOINTS
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.arm_pub.publish(msg)

    def run_for_target(self, target_name):
        """Run the full pick-and-place sequence for one apple at this node's
        currently-configured station (self.robot_x/y/yaw). Returns a result dict."""
        self.set_target(target_name)
        self.reset_everything()

        pose = None
        if self.wait_for(lambda: self.target_pose is not None, timeout=5.0):
            pose = self.target_pose
        if pose is None:
            self.get_logger().error(f"No live pose for {target_name} -- aborting.")
            return {"target": target_name, "success": False, "reason": "no_live_pose"}

        x, y = self.world_to_local(pose.position.x, pose.position.y)
        z_center = pose.position.z

        self.get_logger().info(
            f"[Coordinate fix] World pos ({pose.position.x:.3f}, {pose.position.y:.3f}) "
            f"-> Robot-frame pos ({x:.3f}, {y:.3f})")

        # Confirmed via the mesh itself (Index_Tip_1's visual origin z = -0.163947 in
        # dexhandv2_right.urdf): fingers are ~0.164m long from the wrist
        # (dexhand_base_link) to fingertip. The previous grasp height (apple top + 1cm)
        # put the wrist only ~9cm above the ground -- an open, straight finger from
        # there reaches ~7cm BELOW the floor, physically impossible, so fingers were
        # resting on the ground before closing even started (confirmed: user reported
        # fingers touching the ground, and R_Middle spiking to 33Nm during lift,
        # consistent with a finger wedged against something immovable). Positioning
        # the wrist so an open, straight fingertip lands at the apple's own center
        # height gives the closing/curling motion room to actually wrap the object
        # instead of bottoming out on the floor first.
        FINGER_LENGTH = 0.164
        grasp_z = z_center + FINGER_LENGTH

        approach_target = [x, y, grasp_z + 0.15]
        grasp_target = [x, y, grasp_z]

        approach_result = solve_ik(self.chain, approach_target)
        if approach_result is None:
            self.get_logger().error(f"UNREACHABLE: {approach_target}")
            return {"target": target_name, "success": False, "reason": "unreachable_approach"}
        approach, _ = approach_result

        grasp_result = solve_ik(self.chain, grasp_target)
        if grasp_result is None:
            self.get_logger().error(f"UNREACHABLE: {grasp_target}")
            return {"target": target_name, "success": False, "reason": "unreachable_grasp"}
        grasp, grasp_full_sol = grasp_result

        expected_pan = np.degrees(np.arctan2(grasp_target[1], grasp_target[0]))
        actual_pan = np.degrees(grasp[0])
        self.get_logger().info(
            f"[Arm-config check] grasp joints (deg)="
            f"{[f'{np.degrees(a):.1f}' for a in grasp]} "
            f"expected_shoulder_pan={expected_pan:.1f} actual_shoulder_pan={actual_pan:.1f} "
            f"(large mismatch here means a 'mirror' arm configuration, not a hand-facing issue)")

        achieved_robot_frame = self.chain.forward_kinematics(grasp_full_sol)[:3, 3]
        world_check_x, world_check_y = local_to_world(
            achieved_robot_frame[0], achieved_robot_frame[1],
            self.robot_x, self.robot_y, self.robot_yaw)
        self.get_logger().info(
            f"[SELF-CHECK] Hand will move to WORLD ({world_check_x:.3f}, {world_check_y:.3f}) "
            f"-- compare to apple's world pos above.")

        self.get_logger().info("Opening hand, moving to approach position...")
        self.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
        self.send_arm_trajectory(approach, 3.5)
        time.sleep(4.0)

        self.get_logger().info("Lowering to grasp position...")
        self.send_arm_trajectory(grasp, 2.5)
        time.sleep(3.0)
        rclpy.spin_once(self, timeout_sec=0.5)
        if self.target_pose is not None:
            dx = self.target_pose.position.x - pose.position.x
            dy = self.target_pose.position.y - pose.position.y
            dz = self.target_pose.position.z - pose.position.z
            drift = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
            self.get_logger().info(
                f"[Pre-close check] Apple pose after lowering: "
                f"({self.target_pose.position.x:.3f}, {self.target_pose.position.y:.3f}, "
                f"{self.target_pose.position.z:.3f}), drift from original={drift:.3f}m")

        vlm_result = self.layer1_vlm_analysis()
        plan = self.layer2_imagination(vlm_result)
        grip_ok = self.layer3_4_close_with_feedback(plan, vlm_result)

        self.layer5_smooth_lift(grasp)
        safety_ok = self.layer6_safety_monitor(duration=6.0)

        pose_before_world = (pose.position.x, pose.position.y, pose.position.z)
        rclpy.spin_once(self, timeout_sec=0.5)
        pose_after_lift_world = None
        if self.target_pose is not None:
            pose_after_lift_world = (self.target_pose.position.x, self.target_pose.position.y,
                                      self.target_pose.position.z)

        lifted_ok = (grip_ok and safety_ok and pose_after_lift_world is not None
                     and pose_after_lift_world[2] > pose_before_world[2] + 0.03)

        placed_ok = False
        pose_final_world = pose_after_lift_world
        if lifted_ok:
            placed_ok = self.layer_place()
            for _ in range(10):
                rclpy.spin_once(self, timeout_sec=0.2)
            if self.target_pose is not None:
                pose_final_world = (self.target_pose.position.x, self.target_pose.position.y,
                                     self.target_pose.position.z)
                dist_to_crate = float(np.hypot(pose_final_world[0] - CRATE_WORLD_XY[0],
                                                pose_final_world[1] - CRATE_WORLD_XY[1]))
                if dist_to_crate > 0.3:
                    self.get_logger().warn(
                        f"[Place] Apple settled {dist_to_crate:.2f}m from crate center -- missed.")
                    placed_ok = False

        success = lifted_ok and placed_ok

        self.layer7_log_experience({
            "timestamp": time.time(),
            "object": target_name,
            "station": {"robot_x": self.robot_x, "robot_y": self.robot_y, "robot_yaw": self.robot_yaw},
            "vlm_analysis": vlm_result,
            "grip_plan": plan,
            "grip_contacted_fingers_ok": grip_ok,
            "safety_ok": safety_ok,
            "lifted_ok": lifted_ok,
            "placed_ok": placed_ok,
            "pose_before_world": pose_before_world,
            "pose_after_lift_world": pose_after_lift_world,
            "pose_final_world": pose_final_world,
            "success": success,
        })

        self.get_logger().info(f"=== RESULT for {target_name}: {'SUCCESS' if success else 'FAILED'} ===")
        return {"target": target_name, "success": success, "grip_ok": grip_ok,
                "safety_ok": safety_ok, "lifted_ok": lifted_ok, "placed_ok": placed_ok}


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 full_layer_grasp.py <object_name>")
        sys.exit(1)
    rclpy.init()
    node = FullLayerGraspNode()
    node.run_for_target(sys.argv[1])
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
