#!/usr/bin/env python3
"""
Isolate shoulder_lift_joint's 150Nm saturation: is it contact, or the joint itself?

What we know from the station sweep (arm_table_clearance_check.py): shoulder_lift
saturates at exactly its 150Nm limit at EVERY station distance tested (-0.70 to
-0.90), while every arm link measured at least 0.135m clear of the table. Its real
angle also never gets meaningfully positive no matter what's commanded -- across
five stations it landed at -13.1, -0.5, +5.9, +8.3, -1.4 deg against commands of
-8.8 to +20.6 deg.

Two candidate explanations, and they need separating:
  (a) CONTACT -- the hand/fingers are pressing on the apple or table, and the
      reaction force is what loads the shoulder.
  (b) THE ARM ITSELF -- upper_arm_link swings down into mobile_base_link's box
      (which spans z=0.035..0.285, with the shoulder at z=0.448 and a 0.425m upper
      arm, so positive shoulder_lift angles bring the elbow right down onto it), or
      the joint controller simply can't hold the load.

This sweeps shoulder_lift alone with the arm folded so the hand stays HIGH and far
from the table, apple, and everything else. If saturation still appears at the same
angles with nothing to touch, it's (b) and no amount of station repositioning will
fix it. If effort stays normal throughout, it's (a).

Usage: python3 shoulder_lift_sweep.py
"""
import time

import numpy as np
import rclpy

from full_layer_grasp import FullLayerGraspNode, ARM_JOINTS, FINGER_GROUPS

# Elbow/wrist folded so the hand rides high and close in, well clear of the table
# (top z=0.40) and of the mobile base box (top z=0.285) throughout the sweep. Only
# shoulder_lift changes, so anything that shows up is attributable to that joint.
ELBOW = np.radians(-90.0)
WRIST_1 = np.radians(-90.0)
WRIST_2 = np.radians(-90.0)
WRIST_3 = 0.0
PAN = 0.0

SWEEP_DEG = [-90, -75, -60, -45, -30, -20, -10, 0, 10, 20, 30]

SHOULDER_LIFT_LIMIT = 150.0
SATURATION_MARGIN = 0.98


def settle(node, seconds, interval=0.5):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(int(interval / 0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
        max_vel = 0.0
        for jname in ARM_JOINTS:
            _, vel, _ = node.latest_joint_state.get(jname, (None, None, None))
            if vel is not None:
                max_vel = max(max_vel, abs(vel))
        if max_vel < 0.05 and (time.time() - start) > 3.0:
            return True
    return False


def main():
    rclpy.init()
    node = FullLayerGraspNode()

    print("Waiting for the arm controller to connect...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.arm_pub.get_subscription_count() > 0:
            break
    if node.arm_pub.get_subscription_count() == 0:
        print("WARNING: no subscriber on the arm trajectory topic -- is the simulation "
              "running with controllers active?")
        node.destroy_node()
        rclpy.shutdown()
        return

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)

    print("\nSweeping shoulder_lift with the arm folded (hand high, nothing to touch).")
    print("Positive angles swing the upper arm DOWN toward the mobile base box.\n")
    print(f"{'commanded':>10} {'real':>8} {'gap':>7} {'effort':>9} {'elbow_eff':>10}  note")

    results = []
    for deg in SWEEP_DEG:
        joints = [PAN, np.radians(deg), ELBOW, WRIST_1, WRIST_2, WRIST_3]
        node.send_arm_trajectory(joints, 3.0)
        settle(node, seconds=12.0)

        pos, _, eff = node.latest_joint_state.get('shoulder_lift_joint', (None, None, None))
        _, _, elbow_eff = node.latest_joint_state.get('elbow_joint', (None, None, None))
        if pos is None:
            print(f"{deg:10.1f} {'no data':>8}")
            continue

        real_deg = np.degrees(pos)
        gap = abs(real_deg - deg)
        saturated = abs(eff) >= SHOULDER_LIFT_LIMIT * SATURATION_MARGIN
        note = ""
        if saturated:
            note = "SATURATED"
        if gap > 5.0:
            note += (" / " if note else "") + "did NOT reach"
        print(f"{deg:10.1f} {real_deg:8.1f} {gap:7.1f} {eff:9.2f} {elbow_eff:10.2f}  {note}")
        results.append((deg, real_deg, gap, abs(eff), saturated))

    print()
    saturated_angles = [d for d, _, _, _, s in results if s]
    stuck_angles = [d for d, _, g, _, _ in results if g > 5.0]

    if not saturated_angles:
        print("shoulder_lift never saturated anywhere in this sweep with the hand held "
              "clear. That points to CONTACT (the hand/fingers pressing on the apple or "
              "table) as the real load during grasp attempts, not the joint or the "
              "mobile base -- so where the hand is placed matters, and repositioning is "
              "a valid fix.")
    else:
        print(f"shoulder_lift SATURATED at commanded angles: {saturated_angles}")
        print(f"Failed to reach commanded angle at: {stuck_angles}")
        print("With the hand held high and nothing to touch, this cannot be the table or "
              "the apple. It is the arm itself -- most likely upper_arm_link hitting "
              "mobile_base_link's box (top at z=0.285, directly below the shoulder at "
              "z=0.448) as positive angles swing the arm down onto it. No station "
              "repositioning will fix that; the base geometry or the arm's mounting "
              "height has to change.")

    # Park somewhere harmless rather than leaving the arm straining.
    node.send_arm_trajectory([0.0, -1.2, 1.5, -1.9, 0.0, 0.0], 4.0)
    settle(node, seconds=8.0)

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
