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
EXPERIENCE_LOG = "/home/tt501/ur_gz_ws/experience_log.json"

ROBOT_X, ROBOT_Y = 1.375, -0.55
ROBOT_YAW = 1.5708

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


def solve_ik(chain, target_xyz, init=None):
    guesses = []
    if init is not None:
        guesses.append(init)
    presets = [
        {'shoulder_lift_joint': -1.5708, 'wrist_1_joint': -1.5708, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -1.0, 'elbow_joint': 1.2, 'wrist_1_joint': -1.7, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -0.8, 'elbow_joint': 1.6, 'wrist_1_joint': -2.3, 'wrist_2_joint': -1.5708},
        {'shoulder_lift_joint': -2.0, 'elbow_joint': 1.5, 'wrist_1_joint': 0.0, 'wrist_2_joint': -1.5708},
        {'shoulder_pan_joint': 0.3, 'shoulder_lift_joint': -1.2, 'elbow_joint': 1.8, 'wrist_2_joint': -1.5708},
        {},
    ]
    for preset in presets:
        g = [0.0] * len(chain.links)
        for i, link in enumerate(chain.links):
            if link.name in preset:
                g[i] = preset[link.name]
        guesses.append(g)

    valid_solutions = []
    for g in guesses:
        solution = chain.inverse_kinematics(
            target_xyz, initial_position=g,
            target_orientation=[0, 0, -1], orientation_mode='Z'
        )
        achieved = chain.forward_kinematics(solution)[:3, 3]
        error = np.linalg.norm(np.array(target_xyz) - achieved)
        if error < 0.01:
            valid_solutions.append(solution)

    if not valid_solutions:
        return None

    expected_pan = np.arctan2(target_xyz[1], target_xyz[0])

    def pan_of(sol):
        for link, a in zip(chain.links, sol):
            if link.name == 'shoulder_pan_joint':
                return float(a)
        return 0.0

    def angle_diff(a, b):
        d = (a - b + np.pi) % (2 * np.pi) - np.pi
        return abs(d)

    facing_correct = [s for s in valid_solutions if angle_diff(pan_of(s), expected_pan) < 1.05]
    candidates = facing_correct if facing_correct else valid_solutions

    def total_motion(sol):
        return sum(abs(a) for link, a in zip(chain.links, sol) if link.name in ARM_JOINTS)

    best = min(candidates, key=total_motion)
    joints = {link.name: float(a) for link, a in zip(chain.links, best) if link.name in ARM_JOINTS}
    return [joints[j] for j in ARM_JOINTS], best


class FullLayerGraspNode(Node):
    def __init__(self, target_name):
        super().__init__('full_layer_grasp_node')
        self.target_name = target_name
        self.chain = build_chain()
        self.bridge = CvBridge()

        self.pose_sub = self.create_subscription(
            Pose, f'/model/{target_name}/pose', self._pose_cb, 10)
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

    def reset_everything(self):
        self.get_logger().info("=== RESET: base position ===")
        reset_result = subprocess.run(
            'ign service -s /world/apple_world/set_pose '
            '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
            '--timeout 2000 '
            "--req 'name: \"ur\" position: {x: 1.375 y: -0.55 z: 0.0} "
            "orientation: {x: 0 y: 0 z: 0.7071 w: 0.7071}'",
            shell=True, capture_output=True, text=True
        )
        self.get_logger().info(
            f"Base reset: stdout={reset_result.stdout.strip()!r} stderr={reset_result.stderr.strip()!r}")
        time.sleep(1.5)

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
        max_pitch = 1.4 - (fragility / 10.0) * 0.5
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
                effort = abs(self.latest_joint_state.get(name, (0, 0))[1])
                if effort > contact_threshold and not contacted[g]:
                    contacted[g] = True
                    self.get_logger().info(f"  [Layer 3] {g}: contact (effort={effort:.3f}Nm)")

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
                    self.get_logger().warn(f"[Layer 6] DANGER: {g} effort={effort:.3f}Nm!")
                    aborted = True
        if not aborted:
            self.get_logger().info("[Layer 6] No safety violations detected.")
        return not aborted

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

    def run(self):
        self.reset_everything()

        pose = None
        if self.wait_for(lambda: self.target_pose is not None, timeout=5.0):
            pose = self.target_pose
        if pose is None:
            self.get_logger().error(f"No live pose for {self.target_name} -- aborting.")
            return

        world_x = pose.position.x - ROBOT_X
        world_y = pose.position.y - ROBOT_Y
        cos_yaw = np.cos(-ROBOT_YAW)
        sin_yaw = np.sin(-ROBOT_YAW)
        x = world_x * cos_yaw - world_y * sin_yaw
        y = world_x * sin_yaw + world_y * cos_yaw
        z_center = pose.position.z

        self.get_logger().info(
            f"[Coordinate fix] World pos ({pose.position.x:.3f}, {pose.position.y:.3f}) "
            f"-> Robot-frame pos ({x:.3f}, {y:.3f})")

        z_top = z_center + 0.02

        approach_target = [x, y, z_top + 0.15]
        grasp_target = [x, y, z_top - 0.02]

        approach_result = solve_ik(self.chain, approach_target)
        if approach_result is None:
            self.get_logger().error(f"UNREACHABLE: {approach_target}")
            return
        approach, _ = approach_result

        grasp_result = solve_ik(self.chain, grasp_target)
        if grasp_result is None:
            self.get_logger().error(f"UNREACHABLE: {grasp_target}")
            return
        grasp, grasp_full_sol = grasp_result

        achieved_robot_frame = self.chain.forward_kinematics(grasp_full_sol)[:3, 3]
        ax, ay = achieved_robot_frame[0], achieved_robot_frame[1]
        cos_yaw2 = np.cos(ROBOT_YAW)
        sin_yaw2 = np.sin(ROBOT_YAW)
        world_check_x = ax * cos_yaw2 - ay * sin_yaw2 + ROBOT_X
        world_check_y = ax * sin_yaw2 + ay * cos_yaw2 + ROBOT_Y
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

        vlm_result = self.layer1_vlm_analysis()
        plan = self.layer2_imagination(vlm_result)
        grip_ok = self.layer3_4_close_with_feedback(plan, vlm_result)

        self.layer5_smooth_lift(grasp)
        safety_ok = self.layer6_safety_monitor(duration=6.0)

        pose_before_world = (pose.position.x, pose.position.y, pose.position.z)
        rclpy.spin_once(self, timeout_sec=0.5)
        pose_after_world = None
        if self.target_pose is not None:
            pose_after_world = (self.target_pose.position.x, self.target_pose.position.y,
                                 self.target_pose.position.z)

        success = (grip_ok and safety_ok and pose_after_world is not None
                   and pose_after_world[2] > pose_before_world[2] + 0.03)

        self.layer7_log_experience({
            "timestamp": time.time(),
            "object": self.target_name,
            "vlm_analysis": vlm_result,
            "grip_plan": plan,
            "grip_contacted_fingers_ok": grip_ok,
            "safety_ok": safety_ok,
            "pose_before_world": pose_before_world,
            "pose_after_world": pose_after_world,
            "success": success,
        })

        self.get_logger().info(f"=== RESULT for {self.target_name}: {'SUCCESS' if success else 'FAILED'} ===")


def main():
    if len(sys.argv) < 2:
        print("Usage: python3 full_layer_grasp.py <object_name>")
        sys.exit(1)
    rclpy.init()
    node = FullLayerGraspNode(sys.argv[1])
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
