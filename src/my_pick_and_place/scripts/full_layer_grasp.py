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
import xml.etree.ElementTree as ET
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
import tf2_ros

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


def compute_grasp_reward(contacted, max_effort_seen, steps_taken, lift_gain):
    """Score one closing attempt, for the learned-closing training loop. Built from
    signals the pipeline already computes (contact per finger, peak effort per
    finger, how many closing steps it took, and how much the apple actually rose
    during the lift) -- no new instrumentation needed, just a scoring function on
    top of what run_for_target was already measuring.

    Reward shape, in plain terms: more real contact is good; 3+ fingers (the
    existing success bar) is a clear bonus; the apple genuinely rising during lift
    is the strongest signal of an actual grasp; any finger crossing the danger
    threshold (crushing) is penalized hard; a big spread between contacted fingers'
    peak efforts (one finger doing all the work) is penalized; closing quickly is
    mildly rewarded so the policy doesn't learn to stall.
    """
    n_contacted = sum(contacted.values())
    reward = 2.0 * n_contacted
    if n_contacted >= 3:
        reward += 10.0

    if lift_gain is not None:
        if lift_gain > 0.03:
            reward += 15.0
        else:
            reward -= 5.0

    for g in FINGER_GROUPS:
        if max_effort_seen.get(g, 0.0) > EFFORT_DANGER_THRESHOLD:
            reward -= 8.0

    contacted_efforts = [max_effort_seen[g] for g in FINGER_GROUPS if contacted.get(g)]
    if len(contacted_efforts) >= 2:
        reward -= 2.0 * float(np.std(contacted_efforts))

    reward -= 0.03 * steps_taken
    return reward

# -0.70 was close enough that reaching down onto the apple row required the forearm
# to sweep in low and shallow, straight through the table -- shoulder_lift_joint was
# hitting its 150Nm effort limit and getting physically stuck near 0deg no matter
# what it was commanded to (confirmed NOT a self-collision or the camera pole, both
# ruled out with real data). -0.90 solves with shoulder_lift=-18deg -- mild, and in
# the same (negative) direction the joint already moves freely in during the rest
# pose reset -- giving the arm room to approach from a steeper angle that clears the
# table. Every station tested beyond -0.90 was UNREACHABLE, so this is the farthest,
# safest working distance.
DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y = 1.25, -0.90
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


def is_pose_within_table_bounds(x, y, z):
    """A hand-bumped apple can occasionally get a bad physics impulse and end up
    flung far from the table (confirmed directly: one seen at world (15.9, -74.0),
    tens of meters away) rather than just nudged a few cm. apple_table spans roughly
    x=[-0.2, 2.4], y=[-0.25, 0.25]; this bound is deliberately generous around that
    so it only catches genuinely broken positions, not normal drift. Shared between
    run_for_target's abort check and reset_target_apple_position's capture guard, so
    a corrupted position never gets accidentally locked in as a reset target."""
    return -1.0 <= x <= 4.0 and -2.0 <= y <= 2.0 and 0.0 <= z <= 1.0


# World-frame position the crate must be spawned at in apple_world.world for the
# place target above to actually land inside it. See apple_gripper_sim/models/crate.
CRATE_WORLD_XY = local_to_world(CRATE_LOCAL_XY[0], CRATE_LOCAL_XY[1],
                                 DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)


ARM_ONLY_URDF_PATH = "/tmp/real_robot_exact_arm_only.urdf"
# dexhand_base_link must end up with NO children in the pruned URDF, so it's the
# chain's unambiguous leaf/end-effector -- everything mounted on it (5 fingers, the
# gripper camera) has to go, not just the fingers.
_HAND_ROOT_LINK = 'dexhand_base_link'


def _write_arm_only_urdf(source_path, dest_path):
    """ikpy.Chain treats whatever the URDF's last reachable link is as the chain's
    end effector for inverse_kinematics()'s internal optimization -- it does NOT stop
    at 'dexhand_base_link' just because build_chain()'s base_elements lists it last.
    Since the real URDF branches at dexhand_base_link (5 fingers, plus the gripper
    camera), ikpy silently keeps walking into whichever branch appears first in the
    file (the thumb) and solves IK to place the THUMB TIP there instead of the wrist
    -- confirmed by dumping chain.links and seeing R_Thumb_Yaw/Roll/... appear after
    tool0_to_dexhand. That mismatched frame (offset ~13cm, and rotated onto the
    thumb's own skewed mounting axis) is what forced the solver into contorted,
    wrong-looking arm shapes: it was satisfying "point the thumb down" while
    self-checking a "thumb at target" position, not the wrist. Dropping every joint
    mounted on dexhand_base_link here makes it the chain's real, unambiguous last
    link, so IK actually solves for wrist placement. Finger motion is commanded
    separately via command_fingers()/the dexhand_controller topic, and the gripper
    camera doesn't need to be part of the arm-planning chain at all, so neither loses
    anything by being left out of this IK-only copy of the URDF.
    """
    tree = ET.parse(source_path)
    root = tree.getroot()

    # Map every link to its direct children, from ALL joints (unfiltered), so the walk
    # below can follow a finger chain all the way to its tip -- fingers are several
    # joints deep (Yaw -> Roll -> Pitch -> Flexor -> DIP), not just one hop from
    # dexhand_base_link, so a filter that only looked at each joint's own immediate
    # parent (an earlier version of this function) missed everything past the first
    # joint in each chain, leaving deeper joints referencing links that had already
    # been deleted -- an invalid, dangling URDF. Collecting the full downstream set
    # first and then dropping every joint whose CHILD is in it removes each chain
    # completely, at any depth, in one pass.
    all_children_of = {}
    for joint in root.findall('joint'):
        parent = joint.find('parent').get('link')
        child = joint.find('child').get('link')
        all_children_of.setdefault(parent, []).append(child)

    to_remove = set()
    frontier = list(all_children_of.get(_HAND_ROOT_LINK, []))
    while frontier:
        link_name = frontier.pop()
        if link_name in to_remove:
            continue
        to_remove.add(link_name)
        frontier.extend(all_children_of.get(link_name, []))

    for joint in root.findall('joint'):
        if joint.find('child').get('link') in to_remove:
            root.remove(joint)
    for link in root.findall('link'):
        if link.get('name') in to_remove:
            root.remove(link)

    tree.write(dest_path)


def build_chain():
    _write_arm_only_urdf(URDF_PATH, ARM_ONLY_URDF_PATH)
    chain = ikpy.chain.Chain.from_urdf_file(
        ARM_ONLY_URDF_PATH,
        base_elements=[
            'base_footprint', 'base_footprint_joint', 'mobile_base_link',
            'base_joint', 'base_link',
            # ur_description deliberately inserts a fixed 180deg Z rotation here --
            # documented in the URDF itself: "frames of the robot/controller have X+
            # pointing backwards... introduce the necessary rotation over Z (of pi
            # rad)" -- to align the ROS base_link convention with the UR controller's
            # own. This base_elements list used to jump straight from 'base_link' to
            # 'shoulder_pan_joint', skipping this joint entirely; chain.links then
            # showed no entry for it at all, meaning ikpy built the arm WITHOUT this
            # rotation while the real robot (via robot_state_publisher, straight from
            # the same URDF) has it. Every angle solve_ik ever computed for the
            # shoulder was therefore in a frame rotated 180deg from the real one --
            # confirmed by an exhaustive set of sweeps (station position, robot yaw,
            # target height, combined distance+height) all showing the same ~165-172
            # degree pan offset no matter what was varied, with the "wrong-looking"
            # solution's elbow/lift values otherwise being the genuinely natural ones.
            'base_link-base_link_inertia', 'base_link_inertia',
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

# build_chain() now loads a pruned URDF (see _write_arm_only_urdf) where
# dexhand_base_link/'tool0_to_dexhand' is a true leaf with nothing mounted on it --
# confirmed by dumping chain.links, which now ends at index 11: 'tool0_to_dexhand'.
# That makes ikpy's own default forward_kinematics() (the plain last-link result,
# with no full_kinematics search) already the correct wrist pose, and -- critically
# -- the exact same frame chain.inverse_kinematics() targets internally, so the
# solver and this check are now guaranteed consistent. An earlier version of this
# function used full_kinematics=True to hunt down the wrist by name as a workaround
# for the old branching (unpruned) chain; keeping that after pruning turned out to
# be worse than just deleting it -- it produced a flat, distance-independent ~0.13m
# error on every single station in a reachability sweep (a hallmark of reading a
# misaligned/wrong frame, not a real reach limit), which went away entirely once
# reverted to plain forward_kinematics().
HAND_JOINT_NAME = 'tool0_to_dexhand'


def hand_fk(chain, solution):
    """Forward kinematics of the actual wrist/palm mount point -- see HAND_JOINT_NAME
    comment above for why this is now just a thin, explicit wrapper around ikpy's own
    default forward_kinematics() rather than a full_kinematics()-based name search."""
    return chain.forward_kinematics(solution)


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

    # Locking shoulder_pan_joint to expected_pan (an earlier version of this function)
    # was built to fight a "mirror configuration" symptom that turned out to have two
    # real, separate causes elsewhere: a stale 1.4m offset baked into the URDF's base
    # frame, and IK solving for the DexHand's thumb tip instead of the actual wrist
    # (both now fixed -- see the comments on _write_arm_only_urdf and HAND_JOINT_NAME).
    # With both of those fixed, forcing pan to expected_pan turned out to make MANY
    # genuinely reachable targets UNREACHABLE outright: a direct, unlocked
    # inverse_kinematics() call for a close, low, straight-down-pointing target
    # converged cleanly (~1cm error) at shoulder_pan=169deg, nowhere near the "expected"
    # 0deg -- because reaching a close target while keeping the hand vertical often
    # genuinely requires an unintuitive shoulder angle for this arm's geometry, not
    # because anything is wrong. expected_pan is still useful as a SEED (below, in the
    # presets), just not as a hard constraint the optimizer can't move away from.

    # target_orientation is fed into the optimizer as ONE term in its combined cost
    # (position + orientation), so a guess can converge with near-zero position error
    # while still trading away orientation -- e.g. shoulder_lift=-157deg, elbow=0deg,
    # a straight-armed sweep-back branch that lands on the right xyz but never actually
    # points the hand down at the table (confirmed visually: the hand ends up roughly
    # horizontal, not descending onto the apple). Filtering only on position error, as
    # before, let that branch through untouched. Now checking the achieved Z-axis
    # against straight-down explicitly, and requiring both, is what actually catches it.
    desired_z_axis = np.array([0.0, 0.0, -1.0])
    ORIENTATION_DOT_MIN = 0.9  # ~26 degrees off straight-down, still clearly "pointing down"
    ORIENTATION_DOT_MIN_LOOSE = 0.7  # ~46 degrees -- fallback tier if nothing is that clean

    results = []
    best_solution, best_error = None, float('inf')
    for g in guesses:
        solution = chain.inverse_kinematics(
            target_xyz, initial_position=g,
            target_orientation=[0, 0, -1], orientation_mode='Z'
        )
        fk = hand_fk(chain, solution)
        achieved_pos = fk[:3, 3]
        achieved_z = fk[:3, 2]
        error = np.linalg.norm(np.array(target_xyz) - achieved_pos)
        orientation_dot = float(np.dot(achieved_z, desired_z_axis))
        results.append((solution, error, orientation_dot))
        if error < best_error:
            best_error, best_solution = error, solution

    valid_solutions = [sol for sol, err, dot in results
                        if err < IK_ERROR_TOLERANCE and dot > ORIENTATION_DOT_MIN]
    if not valid_solutions:
        valid_solutions = [sol for sol, err, dot in results
                            if err < IK_FALLBACK_ERROR_CEILING and dot > ORIENTATION_DOT_MIN_LOOSE]

    # ikpy has no concept of collisions -- joint limits here are +-2*pi (URDF), wide
    # enough that a mathematically valid solution can still swing an arm joint far past
    # center, physically sweeping it through the robot's own body to get there.
    # Confirmed with real data: commanded shoulder_lift=-193deg was NEVER reached --
    # /joint_states showed it stuck at -213deg with 71Nm of torque (every other joint
    # was <1Nm). A +-180deg cap on all arm joints still let a -162deg, straight-elbow
    # (0deg) solution through -- same collision risk, just under the wire -- and a
    # verified-good alternative for the same target used shoulder_lift=-18deg, so
    # shoulder_lift specifically (the joint that swings the heavy upper arm back into
    # the mobile base/ground) gets a much tighter cap than the rest.
    SHOULDER_LIFT_SWING_LIMIT = 2.35  # ~135 degrees

    def within_normal_swing(sol):
        for link, a in zip(chain.links, sol):
            if link.name not in ARM_JOINTS:
                continue
            limit = SHOULDER_LIFT_SWING_LIMIT if link.name == 'shoulder_lift_joint' else np.pi
            if abs(a) > limit + 1e-6:
                return False
        return True

    normal_swing_solutions = [sol for sol in valid_solutions if within_normal_swing(sol)]
    if normal_swing_solutions:
        valid_solutions = normal_swing_solutions

    if not valid_solutions:
        # Nothing hit position+orientation together -- fall back to the closest
        # solution by position alone, as long as it's not wildly off, rather than
        # failing outright on a near-miss (this may still be poorly oriented, but an
        # unreachable-looking failure is more visible/debuggable than silently
        # returning a bad-orientation solution from the tier above).
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

    # Sorting by pan mismatch / facing_alignment first (an earlier version of this
    # function) kept selecting extreme, straight-elbow configurations -- e.g.
    # shoulder_lift=-162deg with elbow=0deg, confirmed by screenshot AND real
    # /joint_states data to physically collide with the robot's own body -- because
    # those criteria say nothing about HOW extreme a solution's joint angles are, and
    # facing_alignment in particular is a hand-tuned heuristic (HAND_FACING_SIGN) from
    # earlier, buggier debugging that has no verified relationship to collision safety.
    # total_motion (total joint movement from zero) is a direct, physically meaningful
    # proxy for "does this look like a normal reach or a contorted one" -- minimizing
    # it first favors a moderately-bent elbow and a shoulder that isn't swung to an
    # extreme, exactly the property the earlier criteria failed to capture.
    best = min(valid_solutions, key=lambda sol: (
        total_motion(sol), angle_diff(pan_of(sol), expected_pan), -facing_alignment(sol)))
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
        self.captured_apple_home_poses = {}

        # Real ground-truth position checks, done BY THE SCRIPT ITSELF rather than a
        # human manually timing a second tf2_echo terminal against the log -- that
        # approach kept producing mistimed readings (catching the approach position,
        # or pre-motion stale data, instead of the true settled grasp position) no
        # matter how carefully the timing instructions were followed. A direct lookup
        # right when the code itself knows the arm has settled has no such ambiguity.
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

    def real_wrist_position(self, timeout_sec=1.0):
        """Look up dexhand_base_link's REAL position relative to base_footprint,
        straight from TF -- the same ground truth `tf2_echo` reports, but queried at
        the exact right moment instead of guessed by a human watching two terminals."""
        try:
            t = self.tf_buffer.lookup_transform(
                'base_footprint', 'dexhand_base_link', rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=timeout_sec))
            p = t.transform.translation
            return (p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"[real_wrist_position] TF lookup failed: {e}")
            return None

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
        for name, pos, vel, eff in zip(msg.name, msg.position, msg.velocity, msg.effort):
            self.latest_joint_state[name] = (pos, vel, eff)

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

    def wait_for_settled(self, joint_names, vel_threshold=0.02, timeout=15.0, min_wait=0.0):
        """Block until every named joint's real velocity is near zero, instead of
        guessing a fixed sleep duration. Real TF data showed the wrist still visibly
        moving (and not monotonically -- swinging back and forth) more than 6 seconds
        after the "lowering" trajectory command was sent, well past any fixed sleep
        we'd tried. No PID gains are defined anywhere for these joints, so Gazebo's
        ros2_control plugin is running on its own internal default -- likely
        under-tuned for the extra mass the DexHand adds at the wrist compared to the
        stock end effector it was probably tuned for. Polling real velocity is safe
        (pure Python, no risk to the working robot config) and correct regardless of
        how long the real settling time actually turns out to be.

        min_wait exists because a bare velocity check has a real failure mode: called
        right after publishing a new trajectory, the controller hasn't necessarily
        started moving yet, so /joint_states can still be reporting the PREVIOUS
        (already-settled) position's near-zero velocity -- causing an immediate,
        false "already settled" return before the arm has moved at all. Confirmed
        directly: [Pre-close check] once fired just 0.045s after the lowering
        command, meaning the fingers closed at the old approach height, never having
        descended. min_wait should be set to at least the commanded trajectory
        duration, so the check can't exit before the arm has had a chance to move.
        """
        start = time.time()
        while rclpy.ok() and (time.time() - start) < min_wait:
            rclpy.spin_once(self, timeout_sec=0.1)
        last_max_vel = None
        while rclpy.ok() and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
            velocities = [abs(self.latest_joint_state.get(name, (0, 0, 0))[1]) for name in joint_names]
            if velocities:
                last_max_vel = max(velocities)
                if last_max_vel < vel_threshold:
                    return True
        if last_max_vel is not None:
            self.get_logger().warn(
                f"[wait_for_settled] timed out after {timeout:.1f}s, "
                f"max joint velocity was {last_max_vel:.4f} rad/s "
                f"(threshold={vel_threshold:.4f})")
        return False

    def teleport_model(self, model_name, x, y, z, yaw=0.0, settle_sec=1.5):
        """Set any model's pose directly via Gazebo's set_pose service -- the same
        mechanism teleport() already used for the robot ("ur"), generalized so it can
        also reset an apple back to its real starting position between training
        attempts (nothing did that before; each attempt used to just continue from
        wherever the apple drifted to after the previous one, which isn't a fair
        comparison between different closing-policy candidates)."""
        qz = float(np.sin(yaw / 2.0))
        qw = float(np.cos(yaw / 2.0))
        result = subprocess.run(
            'ign service -s /world/apple_world/set_pose '
            '--reqtype ignition.msgs.Pose --reptype ignition.msgs.Boolean '
            '--timeout 2000 '
            f"--req 'name: \"{model_name}\" position: {{x: {x} y: {y} z: {z}}} "
            f"orientation: {{x: 0 y: 0 z: {qz} w: {qw}}}'",
            shell=True, capture_output=True, text=True
        )
        self.get_logger().info(
            f"Teleport {model_name} to ({x:.3f}, {y:.3f}, {z:.3f}, yaw={yaw:.3f}): "
            f"stdout={result.stdout.strip()!r} stderr={result.stderr.strip()!r}")
        if settle_sec:
            time.sleep(settle_sec)

    def teleport(self, x, y, yaw):
        self.teleport_model('ur', x, y, 0.0, yaw)

    def reset_target_apple_position(self, target_name):
        """Capture target_name's real live position the first time it's seen, then on
        every later call reset it back there via teleport_model() -- so each grasp
        attempt during training starts from the SAME apple position, regardless of
        whether a previous attempt bumped it. Silently does nothing the first time
        (nothing to reset back to yet) and if the target has no live pose at all."""
        if self.target_pose is None:
            return
        if target_name not in self.captured_apple_home_poses:
            p = self.target_pose
            if not is_pose_within_table_bounds(p.position.x, p.position.y, p.position.z):
                self.get_logger().warn(
                    f"[Apple reset] {target_name}'s live position ({p.position.x:.2f}, "
                    f"{p.position.y:.2f}, {p.position.z:.2f}) looks corrupted -- NOT "
                    f"capturing it as home. Restart Gazebo to respawn apples cleanly.")
                return
            self.captured_apple_home_poses[target_name] = (
                p.position.x, p.position.y, p.position.z)
            self.get_logger().info(
                f"[Apple reset] Captured {target_name}'s home position: "
                f"({p.position.x:.3f}, {p.position.y:.3f}, {p.position.z:.3f})")
            return
        hx, hy, hz = self.captured_apple_home_poses[target_name]
        self.teleport_model(target_name, hx, hy, hz, settle_sec=1.0)

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

    def layer3_4_close_with_feedback(self, plan, vlm_result, grasp_target, grasp,
                                      policy_params=None):
        """policy_params, if given, is a dict {g: {'speed': float, 'threshold_offset':
        float}} letting a learned policy override each finger's closing speed and
        contact threshold individually. Left as None, behavior is IDENTICAL to the
        original fixed schedule (same speed_scale/contact_threshold math, same loop) --
        deliberate, so the working pipeline never changes unless a policy is explicitly
        supplied, and the classical version stays available as a safety-net fallback.
        grasp_target/grasp are the [x, y, z] IK target and the joint solution currently
        aiming the wrist there -- needed here (not just in run_for_target) so a mid-close
        push can be corrected for in real time; the (possibly updated) grasp is returned
        so the caller lifts from wherever the wrist actually ended up, not the stale
        pre-closing position."""
        fragility = vlm_result.get("fragility_score", 5)
        force_word = vlm_result.get("recommended_grip_force", "medium")
        speed_scale = {"low": 1.6, "medium": 1.0, "high": 0.6}.get(force_word, 1.0)
        step_duration = STEP_DURATION * speed_scale
        base_contact_threshold = EFFORT_CONTACT_THRESHOLD * (1.0 + (10 - fragility) / 20.0)

        finger_step_size = {}
        contact_threshold = {}
        for g in FINGER_GROUPS:
            params = (policy_params or {}).get(g, {})
            finger_step_size[g] = CLOSE_STEP_SIZE * params.get('speed', 1.0)
            contact_threshold[g] = base_contact_threshold + params.get('threshold_offset', 0.0)

        self.get_logger().info(
            f"[Layer 3/4] Closing: max_pitch={plan['max_pitch']:.2f}, "
            f"step_duration={step_duration:.2f}s, "
            f"contact_threshold={base_contact_threshold:.3f}Nm"
            + (" (learned per-finger params active)" if policy_params else ""))
        # Diagnostic for the actual_pos=0.000rad-on-every-finger anomaly: if this ever
        # reads 0, the trajectory message has no subscriber and is being silently
        # dropped -- a real, checkable cause rather than a guess. If it reads >=1 here
        # but fingers still don't move, the problem is downstream of the publish.
        self.get_logger().info(
            f"[Layer 3/4] hand_pub subscriber count: {self.hand_pub.get_subscription_count()}")

        contacted = {g: False for g in FINGER_GROUPS}
        current = {g: 0.0 for g in FINGER_GROUPS}
        max_effort_seen = {g: 0.0 for g in FINGER_GROUPS}
        steps_taken = MAX_CLOSE_STEPS
        apple_pose_at_recenter = self.target_pose
        recentered_during_close = False

        for step in range(MAX_CLOSE_STEPS):
            if all(contacted.values()):
                self.get_logger().info(f"[Layer 3/4] All fingers contacted after {step} steps.")
                steps_taken = step
                break
            for g in FINGER_GROUPS:
                if not contacted[g]:
                    current[g] = min(current[g] + finger_step_size[g], plan['max_pitch'])
            self.command_fingers(current, step_duration)
            time.sleep(step_duration)
            rclpy.spin_once(self, timeout_sec=0.1)

            for g in FINGER_GROUPS:
                name = f"{g}_Pitch"
                pos = self.latest_joint_state.get(name, (0, 0, 0))[0]
                effort = abs(self.latest_joint_state.get(name, (0, 0, 0))[2])
                max_effort_seen[g] = max(max_effort_seen[g], effort)
                if effort > contact_threshold[g] and not contacted[g]:
                    contacted[g] = True
                    self.get_logger().info(f"  [Layer 3] {g}: contact (effort={effort:.3f}Nm)")
                elif step == MAX_CLOSE_STEPS - 1 and not contacted[g]:
                    self.get_logger().info(
                        f"  [Layer 3] {g}: NO contact. commanded={current[g]:.3f}rad "
                        f"actual_pos={pos:.3f}rad peak_effort={max_effort_seen[g]:.4f}Nm "
                        f"(threshold={contact_threshold[g]:.3f}Nm)")

            # Reactive reposition, one-shot per attempt like the pre-close re-center:
            # a finger closing on the apple can shove it sideways before enough other
            # fingers make contact, same physical effect as the lowering-phase bump
            # already handled above -- just happening mid-close instead of before it.
            # Pause, re-aim the wrist at the apple's real current position, and
            # continue closing from there instead of grabbing at empty air where it
            # used to be.
            if (not recentered_during_close and apple_pose_at_recenter is not None
                    and self.target_pose is not None):
                dx = self.target_pose.position.x - apple_pose_at_recenter.position.x
                dy = self.target_pose.position.y - apple_pose_at_recenter.position.y
                dz = self.target_pose.position.z - apple_pose_at_recenter.position.z
                drift = (dx ** 2 + dy ** 2 + dz ** 2) ** 0.5
                if drift > 0.02:
                    self.get_logger().info(
                        f"[Mid-close re-center] Apple drifted {drift:.3f}m during closing "
                        f"(step {step}) -- pausing to re-aim at its real current position.")
                    new_local_x, new_local_y = self.world_to_local(
                        self.target_pose.position.x, self.target_pose.position.y)
                    recenter_target = [new_local_x, new_local_y, grasp_target[2]]
                    recenter_result = solve_ik(self.chain, recenter_target)
                    if recenter_result is not None:
                        grasp, _ = recenter_result
                        self.send_arm_trajectory(grasp, 1.0)
                        # Confirmed directly: under Gazebo's GUI on this VM, the
                        # simulation runs well below real-time (joints still moving
                        # fast -- e.g. 1.2 rad/s -- after the old 6s wall-clock
                        # timeout), so a short timeout gives up before the arm has
                        # actually finished moving and reads a stale, wrong position.
                        # wait_for_settled measures real (wall-clock) time, so a
                        # generous timeout here just costs extra real seconds under
                        # normal (real-time) headless operation -- it doesn't change
                        # behavior when the sim is keeping up.
                        self.wait_for_settled(ARM_JOINTS, vel_threshold=0.05, timeout=30.0,
                                              min_wait=2.0)
                        # Twice now, fingers have frozen (stopped tracking new commanded
                        # positions, despite near-zero effort) right after a re-center's
                        # arm-trajectory-plus-long-wait sequence in the same attempt --
                        # both times the freeze point matched roughly where the fingers
                        # were when the re-center fired. Checking the subscriber count
                        # again here (already logged once before closing started) tells
                        # us directly whether the hand controller's connection is
                        # actually dropping, instead of guessing.
                        self.get_logger().info(
                            f"[Mid-close re-center] hand_pub subscriber count after "
                            f"re-center: {self.hand_pub.get_subscription_count()}")
                        real_wrist = self.real_wrist_position()
                        if real_wrist is not None:
                            self.get_logger().info(
                                f"[Mid-close re-center] re-solved for "
                                f"({recenter_target[0]:.3f}, {recenter_target[1]:.3f}, "
                                f"{recenter_target[2]:.3f}), real wrist now "
                                f"({real_wrist[0]:.3f}, {real_wrist[1]:.3f}, "
                                f"{real_wrist[2]:.3f})")
                    else:
                        self.get_logger().warn(
                            f"[Mid-close re-center] target {recenter_target} UNREACHABLE -- "
                            f"proceeding with the stale target.")
                    recentered_during_close = True

        n_contacted = sum(contacted.values())
        return {
            "success": n_contacted >= 3,
            "contacted": contacted,
            "max_effort_seen": max_effort_seen,
            "steps_taken": steps_taken,
            "grasp": grasp,
        }

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
                effort = abs(self.latest_joint_state.get(name, (0, 0, 0))[2])
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

    def run_for_target(self, target_name, closing_policy_params=None):
        """Run the full pick-and-place sequence for one apple at this node's
        currently-configured station (self.robot_x/y/yaw). Returns a result dict.
        closing_policy_params, if given, is passed straight through to
        layer3_4_close_with_feedback (see its docstring) -- left as None, this
        function's behavior is unchanged from before the learned-closing work."""
        self.set_target(target_name)
        self.wait_for(lambda: self.target_pose is not None, timeout=5.0)
        # Move the robot to its rest pose FIRST, then reset the apple -- not the other
        # way around. reset_everything() only lifts the arm ~20deg (shoulder_lift) from
        # wherever the previous attempt's grasp left it, so right after an attempt the
        # hand can still be sitting close to the apple's home spot; teleporting the
        # apple back there before the robot is clear risks spawning it overlapping the
        # hand's collision geometry, which Gazebo resolves with a violent separating
        # impulse -- confirmed as the cause of apples being flung far off the table
        # right after a reset. First call for a given target just captures its live
        # position as "home" (the robot hasn't touched the apple yet at that point, so
        # doing this after reset_everything() doesn't change what gets captured); every
        # later call resets it back there -- otherwise each attempt just continues from
        # wherever a previous one left it, which isn't a fair comparison between
        # different closing-policy candidates during training.
        self.reset_everything()
        self.reset_target_apple_position(target_name)

        pose = None
        if self.wait_for(lambda: self.target_pose is not None, timeout=5.0):
            pose = self.target_pose
        if pose is None:
            self.get_logger().error(f"No live pose for {target_name} -- aborting.")
            return {"target": target_name, "success": False, "reason": "no_live_pose"}

        # Running IK against a corrupted apple position wastes a full attempt on a
        # target that was never real -- catch it here and abort cleanly instead.
        if not is_pose_within_table_bounds(pose.position.x, pose.position.y, pose.position.z):
            self.get_logger().error(
                f"[Sanity check] {target_name} world pose ({pose.position.x:.2f}, "
                f"{pose.position.y:.2f}, {pose.position.z:.2f}) is way outside the "
                f"table area -- looks like leftover physics corruption from an earlier "
                f"run, not a real target. Restart Gazebo to respawn apples cleanly.")
            return {"target": target_name, "success": False, "reason": "apple_position_corrupted"}

        x, y = self.world_to_local(pose.position.x, pose.position.y)
        z_center = pose.position.z

        self.get_logger().info(
            f"[Coordinate fix] World pos ({pose.position.x:.3f}, {pose.position.y:.3f}) "
            f"-> Robot-frame pos ({x:.3f}, {y:.3f})")

        # Measured directly via TF (finger_geometry_check.py, not a guess): with the
        # hand open, the fingertips' average position relative to dexhand_base_link
        # (the wrist frame IK aims) is NOT at the wrist's own origin -- it sits offset
        # by about (0.065, -0.006, 0.102)m in the wrist's own local frame. Aiming the
        # WRIST directly at the apple (the old approach, which only ever added a
        # single fixed height for FINGER_LENGTH) left the FINGERS off-center by this
        # same amount -- confirmed visually: the apple sat near one edge of the open
        # hand instead of centered between the fingers, and got pushed away instead
        # of enveloped. Correcting for this needs the wrist's actual achieved
        # orientation (a fixed offset in the wrist's own frame maps to a different
        # world/local-frame offset depending on which way the wrist ends up facing),
        # so this solves IK twice: once for a rough target to get that orientation,
        # then again for the corrected target so the FINGERS -- not the bare wrist --
        # land at the apple's real measured position.
        FINGERTIP_CENTROID_HAND_FRAME = np.array([0.0651, -0.0058, 0.1024])

        rough_target = [x, y, z_center + 0.164]
        rough_result = solve_ik(self.chain, rough_target)
        if rough_result is None:
            self.get_logger().error(f"UNREACHABLE: {rough_target}")
            return {"target": target_name, "success": False, "reason": "unreachable_grasp"}
        _, rough_full_sol = rough_result
        wrist_rotation = hand_fk(self.chain, rough_full_sol)[:3, :3]
        offset_local = wrist_rotation @ FINGERTIP_CENTROID_HAND_FRAME
        grasp_target = [x - offset_local[0], y - offset_local[1], z_center - offset_local[2]]
        self.get_logger().info(
            f"[Finger-center correction] fingertip offset in local frame = "
            f"({offset_local[0]:.3f}, {offset_local[1]:.3f}, {offset_local[2]:.3f}) -- "
            f"corrected wrist target ({grasp_target[0]:.3f}, {grasp_target[1]:.3f}, "
            f"{grasp_target[2]:.3f}) so fingers land at ({x:.3f}, {y:.3f}, {z_center:.3f})")

        # 0.15m hover was untested reach margin, not a real requirement -- the grasp
        # point itself solves with ~0 error at this station (see find_station.py), but
        # adding 0.15m of extra height pushed the wrist just past real reach (0.059m
        # error vs 0.05m fallback ceiling). 0.08m still clears the apple (radius
        # ~0.04m) by a comfortable margin before descending, well inside reach.
        approach_target = [grasp_target[0], grasp_target[1], grasp_target[2] + 0.08]

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

        achieved_robot_frame = hand_fk(self.chain, grasp_full_sol)[:3, 3]
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
        grasp_traj_duration = 2.5
        # [Real joint check] pinpointed a genuine, repeatable steady-state error: only
        # shoulder_lift_joint lands noticeably off from where it's commanded (e.g.
        # commanded=13.0 real=-1.5deg, a 14.5deg gap) while every other joint matches
        # within 0.3deg -- exactly the joint that has to fight gravity to hold the
        # arm+DexHand up, and no PID gains are defined anywhere to correct for it.
        # Rather than hardcode a fixed compensation angle (which would only be valid
        # for this exact pose, not a real fix), close the loop: measure the real
        # error after settling and re-solve for a corrected target that accounts for
        # whatever the real discrepancy turns out to be, repeating until it actually
        # converges or we run out of attempts. This adapts to the real, measured
        # error regardless of its size or cause, instead of guessing a number.
        MAX_CORRECTION_ITERS = 3
        POSITION_TOLERANCE = 0.02
        current_target = list(grasp_target)
        real_wrist = None
        for correction_iter in range(MAX_CORRECTION_ITERS):
            self.send_arm_trajectory(grasp, grasp_traj_duration)
            # Confirmed with real TF data (base_footprint -> dexhand_base_link), not
            # a guess: the wrist keeps visibly moving for well over 6 seconds after
            # this trajectory command is sent. No fixed sleep duration is reliable
            # against a real, variable settling time like that, so wait for the
            # arm's own actual velocity to confirm it has genuinely stopped instead.
            # min_wait=grasp_traj_duration (not an independent guess) prevents the
            # check from exiting on stale, pre-motion /joint_states data before the
            # controller has even started executing this trajectory -- confirmed to
            # happen directly: [Pre-close check] once fired just 0.045s after send.
            # timeout raised from 20s -- confirmed directly that Gazebo's GUI on this
            # VM runs well below real-time (joints still moving at 0.5-0.6 rad/s after
            # the old 20s wall-clock timeout), so 20s wasn't enough real time for a
            # real (just slow) trajectory to actually finish under the GUI's load.
            settled = self.wait_for_settled(ARM_JOINTS, vel_threshold=0.05, timeout=60.0,
                                             min_wait=grasp_traj_duration)
            if not settled:
                self.get_logger().warn(
                    "[Lowering] Arm did not settle to near-zero velocity within 20s -- "
                    "proceeding anyway, but the wrist may still be moving.")

            real_wrist = self.real_wrist_position()
            if real_wrist is None:
                break
            rwx, rwy, rwz = real_wrist
            werr = ((rwx - grasp_target[0]) ** 2 + (rwy - grasp_target[1]) ** 2
                    + (rwz - grasp_target[2]) ** 2) ** 0.5
            self.get_logger().info(
                f"[Real wrist check #{correction_iter}] intended=({grasp_target[0]:.3f}, "
                f"{grasp_target[1]:.3f}, {grasp_target[2]:.3f}) real=({rwx:.3f}, {rwy:.3f}, "
                f"{rwz:.3f}) error={werr:.3f}m")
            # shoulder_lift_joint landed at EXACTLY the same real angle (-1.5deg)
            # across three attempts commanding it to 13.0, 27.4, and 40.5deg -- never
            # moving even 0.1deg despite a 27.5deg spread of different commands, while
            # every other joint tracked its own command closely each time. That rules
            # out ordinary gravity sag (which would still respond somewhat to
            # different commands) in favor of either a genuine physical obstruction
            # (the joint straining hard against something, most likely a collision
            # with the mobile base directly below the shoulder) or the command simply
            # never reaching this joint. Effort distinguishes the two: near-zero means
            # the command isn't landing; high means it's genuinely stuck fighting
            # something.
            real_vs_commanded = []
            for jname, commanded_val in zip(ARM_JOINTS, grasp):
                state = self.latest_joint_state.get(jname, (None, None, None))
                real_val, real_effort = state[0], state[2]
                if real_val is not None:
                    real_vs_commanded.append(
                        f"{jname}: commanded={np.degrees(commanded_val):.1f} "
                        f"real={np.degrees(real_val):.1f}deg effort={real_effort:.2f}Nm")
            self.get_logger().info("[Real joint check] " + " | ".join(real_vs_commanded))

            if werr < POSITION_TOLERANCE:
                break
            if correction_iter == MAX_CORRECTION_ITERS - 1:
                break

            error_vec = [grasp_target[i] - real_wrist[i] for i in range(3)]
            current_target = [current_target[i] + error_vec[i] for i in range(3)]
            corrected_result = solve_ik(self.chain, current_target)
            if corrected_result is None:
                self.get_logger().warn(
                    f"[Correction] corrected target {current_target} UNREACHABLE -- "
                    f"keeping previous commanded joints.")
                break
            grasp, grasp_full_sol = corrected_result
            self.get_logger().info(
                f"[Correction] re-solving for corrected target ({current_target[0]:.3f}, "
                f"{current_target[1]:.3f}, {current_target[2]:.3f})")

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

            # The hand (even open) can brush the apple during the final part of the
            # descent and nudge it sideways along the table before closing even
            # starts -- confirmed directly: a 0.073m drift was measured here while
            # the wrist itself landed within 0.013m of ITS intended target, meaning
            # the apple moved, not the arm. Closing on the original target after
            # that just grabs empty air where the apple used to be. Re-aiming at
            # wherever the apple actually ended up makes the sequence robust to
            # this instead of assuming it stayed put.
            if drift > 0.02:
                self.get_logger().info(
                    f"[Re-center] Apple drifted {drift:.3f}m during lowering -- "
                    f"re-aiming at its real current position.")
                new_local_x, new_local_y = self.world_to_local(
                    self.target_pose.position.x, self.target_pose.position.y)
                recenter_target = [new_local_x, new_local_y, grasp_target[2]]
                recenter_result = solve_ik(self.chain, recenter_target)
                if recenter_result is not None:
                    grasp, grasp_full_sol = recenter_result
                    self.send_arm_trajectory(grasp, 1.5)
                    # timeout raised from 10s -- same GUI slow-motion issue as the
                    # other wait_for_settled calls: a short wall-clock timeout gives
                    # up before a real (just slow, under GUI load) trajectory finishes.
                    self.wait_for_settled(ARM_JOINTS, vel_threshold=0.05, timeout=30.0,
                                          min_wait=2.0)
                    real_wrist = self.real_wrist_position()
                    if real_wrist is not None:
                        self.get_logger().info(
                            f"[Re-center] re-solved for ({recenter_target[0]:.3f}, "
                            f"{recenter_target[1]:.3f}, {recenter_target[2]:.3f}), real "
                            f"wrist now ({real_wrist[0]:.3f}, {real_wrist[1]:.3f}, "
                            f"{real_wrist[2]:.3f})")
                else:
                    self.get_logger().warn(
                        f"[Re-center] target {recenter_target} UNREACHABLE -- "
                        f"proceeding with the stale target.")

        vlm_result = self.layer1_vlm_analysis()
        plan = self.layer2_imagination(vlm_result)
        grip_result = self.layer3_4_close_with_feedback(
            plan, vlm_result, grasp_target, grasp, closing_policy_params)
        grip_ok = grip_result["success"]
        grasp = grip_result.get("grasp", grasp)

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

        lift_gain = None
        if pose_after_lift_world is not None:
            lift_gain = pose_after_lift_world[2] - pose_before_world[2]
        reward = compute_grasp_reward(
            grip_result["contacted"], grip_result["max_effort_seen"],
            grip_result["steps_taken"], lift_gain)

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
            "reward": reward,
            "closing_policy_params": closing_policy_params,
        })

        self.get_logger().info(
            f"=== RESULT for {target_name}: {'SUCCESS' if success else 'FAILED'} "
            f"(reward={reward:.2f}) ===")
        return {"target": target_name, "success": success, "grip_ok": grip_ok,
                "safety_ok": safety_ok, "lifted_ok": lifted_ok, "placed_ok": placed_ok,
                "reward": reward}


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
