#!/usr/bin/env python3
"""
Find the grasp height that actually works, by testing several and measuring.

Everything measured so far says the usable height window is narrow and we have
never tried inside it:

  table top    = 0.400   apple centre = 0.440   apple top = 0.480
  fingertips hang ~0.117m below the wrist (finger_geometry_check.py, four long
  fingers; the thumb is much shorter at 0.077 and drags the 5-finger average down
  to 0.102, which is why using that average put the wrist too low)

  wrist 0.604 (old height)       -> fingertips ~0.487 -> just ABOVE the apple:
                                    fingers close in thin air, zero contact.
                                    Confirmed: all 5 fingers reached full closure
                                    with only baseline-noise effort.
  wrist 0.542 (0.102 correction) -> fingertips ~0.425 -> only 2cm above the TABLE,
                                    with fingers spread wide open around the apple.
                                    Add the measured 3-4deg shoulder droop
                                    (shoulder_lift_sweep.py) and they reach the
                                    table. Confirmed: shoulder_lift pinned at its
                                    150Nm limit at every station distance tested.

So the fingertips need to land between ~0.42 (clear of the table) and ~0.48 (below
the apple's top) -- a wrist height around 0.55-0.58, never tested. This sweeps that
range, and at each height actually closes the fingers and reports whether real
contact happened, rather than inferring it.

Usage: python3 grasp_height_sweep.py [target_name] [z1,z2,z3...]
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, APPLE_HOME_Z,
    DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, ARM_JOINTS, FINGER_GROUPS,
    EFFORT_CONTACT_THRESHOLD, MAX_PITCH_CEILING,
)

# Measured: the arm bottoms out around wrist z=0.585 -- commanding anything lower
# just saturates shoulder_lift/wrist_1/wrist_2 and still lands at 0.584-0.590. So
# only heights at or above that floor are actually testable.
DEFAULT_HEIGHTS = [0.590, 0.600, 0.610, 0.620]

# Thumb-to-index fingertip distance, measured via TF (finger_geometry_check.py).
# The hand only opens to 8.73cm and closes to 7.87cm, so an 8.00cm apple leaves just
# 3.6mm of clearance per side going in, and barely any grip travel once around it.
HAND_SPAN_OPEN = 0.0873
HAND_SPAN_CLOSED = 0.0787

# Each apple's real collision radius, straight from its own model.sdf -- they are NOT
# all the same, and the differences matter a lot against a hand this tight.
APPLE_RADIUS = {
    "apple_01": 0.04000, "apple_02": 0.03680, "apple_03": 0.04320, "apple_04": 0.03960,
    "apple_05": 0.04000, "apple_06": 0.04000, "apple_07": 0.03973, "apple_08": 0.03987,
    "apple_09": 0.03800, "apple_10": 0.04000,
}
FINGERTIP_LINKS = ["Index_Tip_1", "Midle_Tip_1", "Ring_Tip_1", "Pinky_Tip_1", "Thumb_Tip_1"]
TABLE_TOP_Z = 0.40
REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]

JOINT_EFFORT_LIMITS = {
    'shoulder_pan_joint': 150.0, 'shoulder_lift_joint': 150.0, 'elbow_joint': 150.0,
    'wrist_1_joint': 28.0, 'wrist_2_joint': 28.0, 'wrist_3_joint': 28.0,
}


def settle(node, seconds, interval=0.5):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(int(interval / 0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
        max_vel = max((abs(node.latest_joint_state.get(j, (0, 0, 0))[1] or 0.0))
                      for j in ARM_JOINTS)
        if max_vel < 0.05 and (time.time() - start) > 3.0:
            return True
    return False


def fingertip_heights(node):
    heights = {}
    for link in FINGERTIP_LINKS:
        try:
            t = node.tf_buffer.lookup_transform(
                'base_footprint', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            heights[link] = t.transform.translation.z
        except Exception:
            heights[link] = None
    return heights


def close_and_measure_contact(node):
    """Close the fingers in steps, exactly like the real grasp does, and report which
    ones register real contact force."""
    contacted = {g: False for g in FINGER_GROUPS}
    peak = {g: 0.0 for g in FINGER_GROUPS}
    current = {g: 0.0 for g in FINGER_GROUPS}

    for _ in range(24):
        for g in FINGER_GROUPS:
            if not contacted[g]:
                current[g] = min(current[g] + 0.05, MAX_PITCH_CEILING)
        node.command_fingers(current, 0.25)
        time.sleep(0.25)
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


def apple_pos(node):
    if node.target_pose is None:
        return None
    p = node.target_pose.position
    return (p.x, p.y, p.z)


def test_height(node, target_name, wrist_z):
    print(f"\n{'=' * 72}\nWRIST HEIGHT {wrist_z:.3f}\n{'=' * 72}")

    # Stand the robot alongside whichever apple is being tested, rather than always at
    # apple_06's x. The station's x is what puts the apple within a natural reach; with
    # it fixed, any other apple sits far off to the side (apple_02 would be 1.34m away
    # diagonally) and the test would measure that awkward reach instead of the grasp.
    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    node.robot_x, node.robot_y, node.robot_yaw = wx, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW

    node.reset_everything()
    node.reset_target_apple_position(target_name)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    before = apple_pos(node)

    x, y = world_to_local(wx, wy, node.robot_x, node.robot_y, node.robot_yaw)

    approach = solve_ik(node.chain, [x, y, wrist_z + 0.08])
    grasp = solve_ik(node.chain, [x, y, wrist_z])
    if approach is None or grasp is None:
        print("  UNREACHABLE at this height")
        return {"z": wrist_z, "ok": False}

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
    node.send_arm_trajectory(approach[0], 3.5)
    settle(node, 10.0)
    node.send_arm_trajectory(grasp[0], 3.0)
    settle(node, 20.0)

    saturated = []
    for jname in ARM_JOINTS:
        _, _, eff = node.latest_joint_state.get(jname, (None, None, None))
        limit = JOINT_EFFORT_LIMITS.get(jname)
        if eff is not None and limit and abs(eff) >= limit * 0.98:
            saturated.append(jname.replace("_joint", ""))

    wrist = node.real_wrist_position()
    tips = fingertip_heights(node)
    tip_vals = [v for v in tips.values() if v is not None]
    lowest_tip = min(tip_vals) if tip_vals else None

    print(f"  wrist: commanded z={wrist_z:.3f} real={wrist[2]:.3f}"
          if wrist else "  wrist: TF lookup failed")
    if lowest_tip is not None:
        print(f"  lowest fingertip z={lowest_tip:.3f} "
              f"({lowest_tip - TABLE_TOP_Z:+.3f}m vs table top)")
    print(f"  saturated joints: {', '.join(saturated) if saturated else 'none'}")

    contacted, peak = close_and_measure_contact(node)
    n = sum(contacted.values())
    print(f"  fingers contacted: {n}/5  "
          f"peak efforts: {', '.join('%s=%.3f' % (g, peak[g]) for g in FINGER_GROUPS)}")

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    after = apple_pos(node)
    moved = None
    if before and after:
        moved = float(np.linalg.norm(np.array(after) - np.array(before)))
        print(f"  apple moved {moved:.3f}m during this attempt")

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
    time.sleep(1.0)

    return {"z": wrist_z, "ok": True, "saturated": saturated, "contacted": n,
            "lowest_tip": lowest_tip, "moved": moved}


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    if target_name not in APPLE_HOME_WORLD_XY:
        print(f"Unknown target {target_name}")
        return
    heights = ([float(v) for v in sys.argv[2].split(",")]
               if len(sys.argv) > 2 else DEFAULT_HEIGHTS)

    print(f"Target {target_name}; testing wrist heights {heights}")
    radius = APPLE_RADIUS.get(target_name)
    if radius:
        diameter = 2 * radius
        margin = (HAND_SPAN_OPEN - diameter) / 2.0
        squeeze = diameter - HAND_SPAN_CLOSED
        print(f"(table top {TABLE_TOP_Z}, apple centre {APPLE_HOME_Z}, "
              f"apple top {APPLE_HOME_Z + radius:.3f})")
        print(f"{target_name} diameter {diameter * 100:.2f}cm vs hand span "
              f"{HAND_SPAN_OPEN * 100:.2f}cm open / {HAND_SPAN_CLOSED * 100:.2f}cm closed")
        print(f"  -> {margin * 100:+.2f}cm clearance per side going in, "
              f"{squeeze * 100:+.2f}cm of squeeze once around it")
        if margin <= 0:
            print("  -> WARNING: apple is WIDER than the hand opens; it cannot be "
                  "enclosed at all at this size.")
        elif squeeze <= 0:
            print("  -> WARNING: hand cannot close smaller than this apple, so it can "
                  "never actually grip it, only touch it.")

    rclpy.init()
    node = FullLayerGraspNode()
    node.set_target(target_name)
    node.wait_for(lambda: node.target_pose is not None, timeout=5.0)

    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.arm_pub.get_subscription_count() > 0:
            break

    results = [test_height(node, target_name, z) for z in heights]

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'wrist_z':>8} {'lowest_tip':>11} {'vs_table':>9} {'saturated':>22} "
          f"{'contacts':>9} {'apple_moved':>12}")
    for r in results:
        if not r.get("ok"):
            print(f"{r['z']:8.3f} {'UNREACHABLE':>11}")
            continue
        tip = f"{r['lowest_tip']:.3f}" if r["lowest_tip"] is not None else "n/a"
        vs = (f"{r['lowest_tip'] - TABLE_TOP_Z:+.3f}"
              if r["lowest_tip"] is not None else "n/a")
        sat = ", ".join(r["saturated"]) or "none"
        moved = f"{r['moved']:.3f}m" if r["moved"] is not None else "n/a"
        print(f"{r['z']:8.3f} {tip:>11} {vs:>9} {sat:>22} "
              f"{r['contacted']:>7}/5 {moved:>12}")

    good = [r for r in results
            if r.get("ok") and not r["saturated"] and r["contacted"] >= 3]
    if good:
        best = max(good, key=lambda r: r["contacted"])
        print(f"\nWorking heights (no saturated joint, 3+ fingers in contact): "
              f"{', '.join('%.3f' % r['z'] for r in good)}")
        print(f"Best: wrist z={best['z']:.3f} with {best['contacted']}/5 fingers")
    else:
        partial = [r for r in results if r.get("ok") and r["contacted"] > 0]
        if partial:
            print("\nNo height got 3+ fingers, but some made contact: " +
                  ", ".join("z=%.3f (%d/5)" % (r["z"], r["contacted"]) for r in partial))
            print("Worth sweeping between/around those before concluding anything.")
        else:
            print("\nNo height produced any finger contact. If the lowest fingertip is "
                  "sitting in the right band (between the table top and the apple's "
                  "top) and still nothing touches, the problem is the fingers' spread "
                  "being wider than the apple, not the height.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
