#!/usr/bin/env python3
"""
Put the apple where the fingers actually close, and see if it grips.

Why: measured with the hand working (finger_geometry_check.py), thumb-to-index
span only closes to 9.11cm, while every apple is 7.36-8.64cm across. So the hand
can NEVER pinch an apple between fingertips -- the tips stay wider than the fruit,
and the thumb is already at its 1.047 rad joint limit. The height sweep confirmed
the consequence: at heights where the fingertips sat correctly between the table
and the apple's top, the apple was clearly reached (it moved 11-13cm) yet peak
finger force never exceeded 0.05Nm against a 0.12Nm threshold -- the fingers brush
it and it rolls away before any force can build.

That rules out a pinch and leaves a power grasp: the apple held in the pocket the
fingers curl into, pressed toward the palm. Every attempt so far has aimed either
the wrist or the OPEN fingertips at the apple, but the fingers converge to a point
0.063m forward and 0.083m below the wrist -- the apple has never been put there.

This aims that closed-fingertip centroid at the apple, using the wrist's real
achieved orientation to rotate the offset (the same two-pass approach
full_layer_grasp.py uses), and tries small variations around it since the pocket's
exact centre is itself a measured estimate.

Usage: python3 pocket_grasp_test.py [target_name]
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, APPLE_HOME_Z,
    DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, hand_fk, ARM_JOINTS, FINGER_GROUPS,
    EFFORT_CONTACT_THRESHOLD, MAX_PITCH_CEILING,
)

# Fingertip centroid at full closure, in the hand's own frame, measured via TF with
# the fingertip joints working. This is the centre of the pocket the fingers curl
# into -- where an object has to be to end up gripped rather than brushed.
CLOSED_CENTROID_HAND_FRAME = np.array([0.0634, -0.0056, 0.0834])

# The pocket centre is a measured estimate, and the apple has a 4cm radius, so nudge
# the aim around it rather than betting everything on one point. Offsets are applied
# along the hand's own approach axis (+ve = deeper into the palm).
DEPTH_OFFSETS = [0.00, -0.02, -0.04, 0.02]

REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]


def settle(node, seconds, joints=ARM_JOINTS, thresh=0.05):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
        vels = [abs(node.latest_joint_state.get(j, (0, 0, 0))[1] or 0.0) for j in joints]
        if vels and max(vels) < thresh and (time.time() - start) > 3.0:
            return True
    return False


def close_and_measure(node):
    contacted = {g: False for g in FINGER_GROUPS}
    peak = {g: 0.0 for g in FINGER_GROUPS}
    current = {g: 0.0 for g in FINGER_GROUPS}
    for _ in range(26):
        for g in FINGER_GROUPS:
            if not contacted[g]:
                current[g] = min(current[g] + 0.05, MAX_PITCH_CEILING)
        node.command_fingers(current, 0.25)
        for _ in range(3):
            rclpy.spin_once(node, timeout_sec=0.1)
        for g in FINGER_GROUPS:
            _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
            eff = abs(eff or 0.0)
            peak[g] = max(peak[g], eff)
            if eff > EFFORT_CONTACT_THRESHOLD:
                contacted[g] = True
        if all(contacted.values()):
            break
    return contacted, peak


def apple_xyz(node):
    if node.target_pose is None:
        return None
    p = node.target_pose.position
    return np.array([p.x, p.y, p.z])


def attempt(node, target_name, depth_offset):
    print(f"\n{'=' * 72}\nPOCKET AIM, depth offset {depth_offset:+.3f}m\n{'=' * 72}")

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    node.robot_x, node.robot_y, node.robot_yaw = wx, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW
    node.reset_everything()
    node.reset_target_apple_position(target_name)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    before = apple_xyz(node)

    x, y = world_to_local(wx, wy, node.robot_x, node.robot_y, node.robot_yaw)
    apple_local = np.array([x, y, APPLE_HOME_Z])

    # Pass 1: a rough solve just to learn which way the wrist ends up facing, since the
    # pocket offset is expressed in the hand's own frame and has to be rotated into the
    # robot frame before it means anything.
    rough = solve_ik(node.chain, list(apple_local + np.array([0, 0, 0.164])))
    if rough is None:
        print("  rough solve UNREACHABLE")
        return {"offset": depth_offset, "ok": False}
    wrist_rot = hand_fk(node.chain, rough[1])[:3, :3]

    pocket = CLOSED_CENTROID_HAND_FRAME.copy()
    pocket[2] += depth_offset
    offset_local = wrist_rot @ pocket
    wrist_target = apple_local - offset_local

    print(f"  apple at local ({apple_local[0]:.3f}, {apple_local[1]:.3f}, {apple_local[2]:.3f})")
    print(f"  pocket offset rotated -> ({offset_local[0]:.3f}, {offset_local[1]:.3f}, "
          f"{offset_local[2]:.3f})")
    print(f"  wrist target ({wrist_target[0]:.3f}, {wrist_target[1]:.3f}, "
          f"{wrist_target[2]:.3f})")

    approach = solve_ik(node.chain, list(wrist_target + np.array([0, 0, 0.09])))
    grasp = solve_ik(node.chain, list(wrist_target))
    if approach is None or grasp is None:
        print("  UNREACHABLE")
        return {"offset": depth_offset, "ok": False}

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
    node.send_arm_trajectory(approach[0], 3.5)
    settle(node, 12.0)
    node.send_arm_trajectory(grasp[0], 3.0)
    settle(node, 20.0)

    real = node.real_wrist_position()
    if real:
        err = float(np.linalg.norm(np.array(real) - wrist_target))
        print(f"  wrist real ({real[0]:.3f}, {real[1]:.3f}, {real[2]:.3f}) err={err:.3f}m")

    contacted, peak = close_and_measure(node)
    n = sum(contacted.values())
    print(f"  fingers contacted: {n}/5")
    print("  peak efforts: " + ", ".join("%s=%.3f" % (g, peak[g]) for g in FINGER_GROUPS))

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    after = apple_xyz(node)
    moved = float(np.linalg.norm(after - before)) if before is not None and after is not None else None
    lifted = float(after[2] - before[2]) if before is not None and after is not None else None
    if moved is not None:
        print(f"  apple moved {moved:.3f}m (height change {lifted:+.3f}m)")

    return {"offset": depth_offset, "ok": True, "contacts": n, "peak": peak,
            "moved": moved, "lifted": lifted}


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    if target_name not in APPLE_HOME_WORLD_XY:
        print(f"Unknown target {target_name}")
        return

    print(f"Target {target_name}. Aiming the CLOSED fingertip centroid "
          f"{CLOSED_CENTROID_HAND_FRAME} at the apple, not the wrist or the open hand.")

    rclpy.init()
    node = FullLayerGraspNode()
    node.set_target(target_name)
    node.wait_for(lambda: node.target_pose is not None, timeout=5.0)
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.arm_pub.get_subscription_count() > 0:
            break

    results = [attempt(node, target_name, d) for d in DEPTH_OFFSETS]

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'depth':>8} {'contacts':>9} {'max_effort':>11} {'apple_moved':>12} {'lifted':>9}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['offset']:+8.3f} {'UNREACHABLE':>9}")
            continue
        mx = max(r["peak"].values())
        moved = f"{r['moved']:.3f}m" if r["moved"] is not None else "n/a"
        lifted = f"{r['lifted']:+.3f}m" if r["lifted"] is not None else "n/a"
        print(f"{r['offset']:+8.3f} {r['contacts']:>7}/5 {mx:11.3f} {moved:>12} {lifted:>9}")

    best = max((r for r in results if r.get("ok")),
               key=lambda r: (r["contacts"], max(r["peak"].values())), default=None)
    if best and best["contacts"] >= 3:
        print(f"\nGRIP: {best['contacts']}/5 fingers at depth offset "
              f"{best['offset']:+.3f}m. This is a working grasp configuration.")
    elif best and best["contacts"] > 0:
        print(f"\nPartial: best was {best['contacts']}/5 at depth offset "
              f"{best['offset']:+.3f}m. Worth sweeping finer around that depth.")
    else:
        print("\nStill no contact anywhere in the pocket. Since the hand demonstrably "
              "reaches the apple (it moves), and cannot pinch it (closed span 9.11cm vs "
              "an 8.00cm apple), the remaining options are making the apple bigger than "
              "the closed span, or holding it against the palm/table rather than in "
              "free space.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
