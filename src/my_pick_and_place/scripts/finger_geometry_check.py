#!/usr/bin/env python3
"""
One-off diagnostic: where do the fingertips actually sit relative to
dexhand_base_link (the wrist frame IK aims), both with the hand OPEN and
fully CLOSED?

Motivation: after fixing the open-hand centering offset (this script's own
earlier measurement), a real grasp attempt showed all 5 fingers reaching
their full commanded closure (1.000rad) with only baseline noise effort --
never registering real contact. So the fingers ARE moving and closing
correctly, but their curling path seems to miss the apple's surface
entirely. The open-hand centroid this script already measured doesn't tell
us where the fingers end up at full closure -- this measures that directly
via TF (same ground-truth method real_wrist_position() already uses),
instead of guessing why the curl misses.

Usage: with the simulation running (hand controller active) and the arm
positioned somewhere reasonable (doesn't need to be at a grasp target -- this
only cares about the hand's own geometry, relative to its own wrist frame),
run:
  python3 finger_geometry_check.py
"""
import time

import rclpy
from rclpy.node import Node
import tf2_ros
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from full_layer_grasp import FINGER_GROUPS, FINGER_JOINT_NAMES, FINGER_SECONDARY_JOINTS

FINGERTIP_LINKS = ["Index_Tip_1", "Midle_Tip_1", "Ring_Tip_1", "Pinky_Tip_1", "Thumb_Tip_1"]


class FingerGeometryCheck(Node):
    def __init__(self):
        super().__init__('finger_geometry_check')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)
        self.hand_pub = self.create_publisher(JointTrajectory, '/dexhand_controller/joint_trajectory', 10)

    def lookup(self, link):
        try:
            t = self.tf_buffer.lookup_transform(
                'dexhand_base_link', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=3.0))
            p = t.transform.translation
            return (p.x, p.y, p.z)
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed for {link}: {e}")
            return None

    def command_hand(self, pitch, duration_sec):
        all_names = list(FINGER_JOINT_NAMES)
        all_positions = [pitch for _ in FINGER_GROUPS]
        for g in FINGER_GROUPS:
            for j_name in FINGER_SECONDARY_JOINTS[g]:
                all_names.append(j_name)
                all_positions.append(pitch * 0.8)
        msg = JointTrajectory()
        msg.joint_names = all_names
        point = JointTrajectoryPoint()
        point.positions = all_positions
        point.time_from_start.sec = int(duration_sec)
        msg.points = [point]
        self.hand_pub.publish(msg)


def measure(node, label):
    positions = {}
    for link in FINGERTIP_LINKS:
        pos = node.lookup(link)
        positions[link] = pos
        if pos is not None:
            print(f"{link}: relative to dexhand_base_link ({label}) = "
                  f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        else:
            print(f"{link}: NO TF DATA")
    valid = [p for p in positions.values() if p is not None]
    if valid:
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        cz = sum(p[2] for p in valid) / len(valid)
        print(f"Fingertip centroid ({label}): ({cx:.4f}, {cy:.4f}, {cz:.4f})")
        return (cx, cy, cz)
    print(f"No fingertip TF data at all for {label}.")
    return None


def main():
    rclpy.init()
    node = FingerGeometryCheck()
    print("Waiting for TF...")
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.2)

    print("\n=== OPEN ===")
    node.command_hand(0.0, 1.0)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(1.5)
    open_centroid = measure(node, "open")

    print("\n=== CLOSED (pitch=1.0) ===")
    node.command_hand(1.0, 2.0)
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    time.sleep(2.5)
    closed_centroid = measure(node, "closed")

    if open_centroid and closed_centroid:
        dx = closed_centroid[0] - open_centroid[0]
        dy = closed_centroid[1] - open_centroid[1]
        dz = closed_centroid[2] - open_centroid[2]
        print(f"\nCentroid moved by ({dx:.4f}, {dy:.4f}, {dz:.4f}) from open to closed.")
        print("The grasp code aims the OPEN centroid at the apple's position. If the "
              "CLOSED centroid ends up meaningfully far from that same point (compare "
              "the deltas above to the apple's ~0.04m radius), the fingers are curling "
              "past or short of where the apple actually is, which would explain full "
              "closure with no real contact.")

    node.command_hand(0.0, 1.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
