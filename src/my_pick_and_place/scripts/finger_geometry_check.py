#!/usr/bin/env python3
"""
One-off diagnostic: where do the fingertips actually sit relative to
dexhand_base_link (the wrist frame IK aims), with the hand open?

Motivation: a screenshot showed the apple sitting near the edge of the open
hand instead of centered between the fingers -- closing on it that way pushes
it out sideways instead of enveloping it. The grasp code only ever aims the
wrist using the apple's height (z_center + FINGER_LENGTH); it never checks
whether the fingers' own X/Y span is actually centered under the wrist. If
it isn't, every attempt aims slightly off-center by a fixed, measurable
amount -- this script measures that amount directly via TF (same ground-truth
method real_wrist_position() already uses), instead of guessing a correction.

Usage: with the simulation running and the hand in (or near) its open/rest
pose, run:
  python3 finger_geometry_check.py
"""
import rclpy
from rclpy.node import Node
import tf2_ros

FINGERTIP_LINKS = ["Index_Tip_1", "Midle_Tip_1", "Ring_Tip_1", "Pinky_Tip_1", "Thumb_Tip_1"]


class FingerGeometryCheck(Node):
    def __init__(self):
        super().__init__('finger_geometry_check')
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

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


def main():
    rclpy.init()
    node = FingerGeometryCheck()
    print("Waiting for TF...")
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.2)

    positions = {}
    for link in FINGERTIP_LINKS:
        pos = node.lookup(link)
        positions[link] = pos
        if pos is not None:
            print(f"{link}: relative to dexhand_base_link = "
                  f"({pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f})")
        else:
            print(f"{link}: NO TF DATA")

    valid = [p for p in positions.values() if p is not None]
    if valid:
        cx = sum(p[0] for p in valid) / len(valid)
        cy = sum(p[1] for p in valid) / len(valid)
        cz = sum(p[2] for p in valid) / len(valid)
        print(f"\nFingertip centroid relative to dexhand_base_link (open pose): "
              f"({cx:.4f}, {cy:.4f}, {cz:.4f})")
        print("If cx/cy aren't close to 0, that's a real, measured offset between "
              "the wrist (what IK aims) and the fingers' actual span center -- "
              "the grasp target should be shifted by this amount so the apple "
              "ends up centered between the fingers instead of near one edge.")
    else:
        print("\nNo fingertip TF data at all -- check the robot is spawned and "
              "robot_state_publisher is running.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
