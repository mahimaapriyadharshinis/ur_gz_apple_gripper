"""
Find the grasp height that actually works, by testing several and measuring.

Re-run this after the fingertip-joint fix: every height conclusion drawn before it
was measured against a hand whose fingertip joints were frozen, so they are all
suspect.

Geometry as now measured (finger_geometry_check.py, with the hand actually
closing):

  table top 0.400,  apple centre 0.440,  apple top 0.480
  hand OPEN:   fingertips hang 0.164m below the wrist, thumb-index span 15.38cm
  hand CLOSED: fingertip centroid rises to 0.083m below the wrist, span 9.11cm

So with the hand open the fingertips only clear the table above wrist z=0.564, and
as the fingers close they rise and converge -- meaning the apple wants to end up in
the volume the fingers sweep through, not where the open fingertips start.

At each height this actually closes the fingers and reports real contact, fingertip
clearance above the table, joint saturation, and how far the apple moved, rather
than inferring any of it.

Usage: python3 grasp_height_sweep.py [target_name] [z1,z2,z3...]
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, apple_home_z, APPLE_RADIUS,
    DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, ARM_JOINTS, FINGER_GROUPS,
    EFFORT_CONTACT_THRESHOLD, MAX_PITCH_CEILING,
)

# The earlier "arm bottoms out at 0.585" floor was an artefact of the broken hand:
# with the fingertip joints frozen, the fingers stuck out rigidly 16.4cm below the
# wrist, hit the table, and physically blocked the arm's descent (which is also what
# pinned shoulder_lift at 150Nm). Now that the fingers actually curl, that limit no
# longer applies, so this sweeps a wider band again.
#
# With the hand OPEN the fingertips hang 0.164m below the wrist, so they clear the
# table (0.400) only above wrist z=0.564. As they close, the fingertip centroid rises
# to 0.083m below the wrist, so the closing fingers converge around the apple rather
# than below it.
DEFAULT_HEIGHTS = [0.570, 0.585, 0.600, 0.615, 0.630]

# Thumb-to-index fingertip distance, measured via TF (finger_geometry_check.py)
# AFTER the fingertip-joint fix. The hand opens to 15.38cm and closes to 9.11cm.
# The earlier 8.73/7.87cm figures were measured on the broken hand and on stale TF
# data, and badly understated what the hand can do.
HAND_SPAN_OPEN = 0.1538
HAND_SPAN_CLOSED = 0.0911

# Radii now live in full_layer_grasp.py (APPLE_RADIUS) so there is one source of
# truth; they were duplicated here and went stale after the apples were resized.
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
        centre = apple_home_z(target_name)
        print(f"(table top {TABLE_TOP_Z}, apple centre {centre:.3f}, "
              f"apple top {centre + radius:.3f})")
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
