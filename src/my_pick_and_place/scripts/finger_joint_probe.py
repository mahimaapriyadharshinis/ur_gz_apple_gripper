#!/usr/bin/env python3
"""
Which finger joints actually respond to commands, and which exist at all?

After removing the broken mimic tags from the DexHand URDF (all five DIP joints
were slaved to master joints that do not exist), the fingers stopped moving
ENTIRELY -- fingertip TF positions became identical at open, pitch=1.0 and
pitch=1.25, where before they at least rotated ~2cm at the knuckle. So the change
either did not fully take effect, or it broke the joints in a different way.

Fingertip positions can't distinguish those cases. This reads the joints
themselves: whether each finger joint appears in /joint_states at all (if it
doesn't, the simulator never created it), and whether its real position changes
when commanded.

Usage: python3 finger_joint_probe.py
"""
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, FINGER_GROUPS, FINGER_JOINT_NAMES, FINGER_SECONDARY_JOINTS,
    MAX_PITCH_CEILING,
)

ALL_FINGER_JOINTS = list(FINGER_JOINT_NAMES)
for _g in FINGER_GROUPS:
    ALL_FINGER_JOINTS.extend(FINGER_SECONDARY_JOINTS[_g])


def snapshot(node):
    return {j: node.latest_joint_state.get(j, (None, None, None))[0]
            for j in ALL_FINGER_JOINTS}


def spin(node, seconds):
    end = time.time() + seconds
    while time.time() < end:
        rclpy.spin_once(node, timeout_sec=0.1)


def main():
    rclpy.init()
    node = FullLayerGraspNode()

    print("Waiting for the hand controller to connect...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.hand_pub.get_subscription_count() > 0:
            break
    print(f"hand_pub subscriber count: {node.hand_pub.get_subscription_count()}")

    spin(node, 2.0)

    # Which finger joints does the simulator actually publish? A joint missing here was
    # never created in the sim, which is a different failure from one that exists but
    # refuses to move.
    present = [j for j in ALL_FINGER_JOINTS if node.latest_joint_state.get(j) is not None]
    missing = [j for j in ALL_FINGER_JOINTS if j not in present]
    print(f"\nfinger joints present in /joint_states: {len(present)}/{len(ALL_FINGER_JOINTS)}")
    if missing:
        print(f"  MISSING (never created in the sim): {', '.join(missing)}")

    all_states = sorted(node.latest_joint_state.keys())
    print(f"\nall joints the sim reports ({len(all_states)}):")
    print("  " + ", ".join(all_states))

    print("\nOpening hand...")
    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
    spin(node, 3.0)
    before = snapshot(node)

    print(f"Closing hand to pitch={MAX_PITCH_CEILING}...")
    node.command_fingers({g: MAX_PITCH_CEILING for g in FINGER_GROUPS}, 2.0)
    spin(node, 4.0)
    after = snapshot(node)

    print(f"\n{'joint':22s} {'commanded':>10s} {'before':>9s} {'after':>9s} "
          f"{'moved':>8s}  status")
    for g in FINGER_GROUPS:
        for j in [f"{g}_Pitch"] + FINGER_SECONDARY_JOINTS[g]:
            commanded = (MAX_PITCH_CEILING if j.endswith("_Pitch")
                         else MAX_PITCH_CEILING * 0.8)
            b, a = before.get(j), after.get(j)
            if b is None or a is None:
                print(f"{j:22s} {commanded:10.3f} {'n/a':>9s} {'n/a':>9s} "
                      f"{'n/a':>8s}  NOT IN /joint_states")
                continue
            moved = a - b
            if abs(moved) < 0.01:
                status = "DID NOT MOVE"
            elif abs(a - commanded) < 0.05:
                status = "reached command"
            else:
                status = "moved, short of command"
            print(f"{j:22s} {commanded:10.3f} {b:9.3f} {a:9.3f} {moved:8.3f}  {status}")

    stuck = [j for g in FINGER_GROUPS for j in [f"{g}_Pitch"] + FINGER_SECONDARY_JOINTS[g]
             if before.get(j) is not None and after.get(j) is not None
             and abs(after[j] - before[j]) < 0.01]
    print()
    if not stuck:
        print("Every finger joint moved -- the hand is being driven correctly.")
    elif len(stuck) == len(ALL_FINGER_JOINTS):
        print("NO finger joint moved at all. The commands are not reaching the joints "
              "(controller/interface problem), rather than the fingers being blocked -- "
              "check /tmp/sim_launch.log for controller or interface errors.")
    else:
        print(f"These joints did not move: {', '.join(stuck)}")
        print("Joints that move prove the command path works, so the stuck ones are "
              "either blocked, at a limit, or not actually commandable.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
