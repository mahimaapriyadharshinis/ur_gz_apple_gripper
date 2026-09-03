#!/usr/bin/env python3
"""
Station sweep: which robot distance can actually hold the real grasp pose?

Confirmed directly by an earlier version of this script: commanding the wrist to
the real, correct grasp height (from the measured fingertip reach) pegs
shoulder_lift_joint at exactly its 150Nm limit -- but NOT because of a collision.
Every arm link measured at least 0.135m clear of the table, the arm was completely
stationary (max_vel=0.000), and it reached every other commanded joint angle
exactly. That is gravity: at this station the arm has to extend nearly horizontally
to reach the apple, which is the worst possible lever arm for the shoulder, so it
saturates and droops ~3deg short.

That also means this project's earlier "arm-vs-table collision" diagnosis of the
same 150Nm symptom -- fixed by moving the station FARTHER away (-0.70 -> -0.90) --
was very likely backwards: farther means more extension, which makes the gravity
torque worse, not better.

This sweeps candidate station distances with real physics, reporting peak joint
effort and real table clearance for each, so the station is chosen on measured
evidence rather than a guess.

Usage:
  python3 arm_table_clearance_check.py [target_name] [y1,y2,y3...]
e.g.
  python3 arm_table_clearance_check.py apple_06 -0.70,-0.75,-0.80,-0.85,-0.90
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, APPLE_HOME_Z,
    DELIVERY_ROBOT_X, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, ARM_JOINTS, FINGER_GROUPS,
)

# Real measured fingertip reach (finger_geometry_check.py) -- the height that
# actually needs to be reached for the fingers to touch the apple.
GRASP_Z_OFFSET = 0.102

ARM_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link",
             "wrist_1_link", "wrist_2_link", "wrist_3_link", "dexhand_base_link"]

# apple_table in apple_world.world: model pose (1.125, 0.0, 0.0) + collision box's
# local pose (0, 0, 0.20), size 2.60 x 0.50 x 0.40.
TABLE_WORLD_CENTER_XY = (1.125, 0.0)
TABLE_SIZE_XY = (2.60, 0.50)
TABLE_TOP_Z = 0.40

# Each UR5e joint's own declared max effort -- a joint sitting exactly at this value
# is saturated (can't produce any more torque), which is the signature we're hunting.
JOINT_EFFORT_LIMITS = {
    'shoulder_pan_joint': 150.0, 'shoulder_lift_joint': 150.0, 'elbow_joint': 150.0,
    'wrist_1_joint': 28.0, 'wrist_2_joint': 28.0, 'wrist_3_joint': 28.0,
}
SATURATION_MARGIN = 0.98  # within 2% of the limit counts as saturated

REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]


def table_local_bbox(robot_y):
    cx, cy = TABLE_WORLD_CENTER_XY
    hx, hy = TABLE_SIZE_XY[0] / 2.0, TABLE_SIZE_XY[1] / 2.0
    corners_world = [(cx + sx * hx, cy + sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
    corners_local = [world_to_local(wx, wy, DELIVERY_ROBOT_X, robot_y, DELIVERY_ROBOT_YAW)
                      for wx, wy in corners_world]
    xs = [c[0] for c in corners_local]
    ys = [c[1] for c in corners_local]
    return min(xs), max(xs), min(ys), max(ys)


def sample_until_settled(node, seconds=25.0, interval=1.0):
    """Watch joint efforts/velocities until the arm stops moving (or time runs out).
    Returns (peak_effort_by_joint, settled)."""
    peak = {j: 0.0 for j in ARM_JOINTS}
    start = time.time()
    settled = False
    while time.time() - start < seconds:
        for _ in range(int(interval / 0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
        max_vel = 0.0
        for jname in ARM_JOINTS:
            pos, vel, eff = node.latest_joint_state.get(jname, (None, None, None))
            if vel is not None:
                max_vel = max(max_vel, abs(vel))
            if eff is not None:
                peak[jname] = max(peak[jname], abs(eff))
        if max_vel < 0.05 and (time.time() - start) > 6.0:
            settled = True
            break
    return peak, settled


def link_clearances(node, robot_y):
    """Min vertical clearance above the table top, over links actually over the table."""
    min_x, max_x, min_y, max_y = table_local_bbox(robot_y)
    worst = None
    worst_link = None
    overlaps = []
    for link in ARM_LINKS:
        try:
            t = node.tf_buffer.lookup_transform(
                'base_footprint', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=1.0))
            p = t.transform.translation
        except Exception:
            continue
        if min_x <= p.x <= max_x and min_y <= p.y <= max_y:
            clearance = p.z - TABLE_TOP_Z
            if worst is None or clearance < worst:
                worst, worst_link = clearance, link
            if clearance <= 0:
                overlaps.append(link)
    return worst, worst_link, overlaps


def test_station(node, target_name, robot_y):
    print(f"\n{'=' * 70}\nSTATION Y={robot_y:.2f}\n{'=' * 70}")

    node.teleport(DELIVERY_ROBOT_X, robot_y, DELIVERY_ROBOT_YAW)
    node.robot_x, node.robot_y, node.robot_yaw = DELIVERY_ROBOT_X, robot_y, DELIVERY_ROBOT_YAW

    # Park in the rest pose first so each station is tested from the same starting
    # configuration, not from whatever strained pose the previous station left behind.
    node.send_arm_trajectory(REST_POSE, 4.0)
    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.0)
    sample_until_settled(node, seconds=12.0)

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    x, y = world_to_local(wx, wy, DELIVERY_ROBOT_X, robot_y, DELIVERY_ROBOT_YAW)
    grasp_target = [x, y, APPLE_HOME_Z + GRASP_Z_OFFSET]

    result = solve_ik(node.chain, grasp_target)
    if result is None:
        print(f"  UNREACHABLE at this station: {grasp_target}")
        return {"robot_y": robot_y, "reachable": False}
    joints, _ = result
    print(f"  grasp target (local) = ({grasp_target[0]:.3f}, {grasp_target[1]:.3f}, "
          f"{grasp_target[2]:.3f})")
    print(f"  IK solution (deg) = {[f'{np.degrees(a):.1f}' for a in joints]}")

    node.send_arm_trajectory(joints, 4.0)
    peak, settled = sample_until_settled(node, seconds=25.0)

    saturated = []
    for jname in ARM_JOINTS:
        limit = JOINT_EFFORT_LIMITS.get(jname)
        if limit and peak[jname] >= limit * SATURATION_MARGIN:
            saturated.append(jname)

    print(f"  settled={settled}")
    for jname, commanded in zip(ARM_JOINTS, joints):
        pos, _, eff = node.latest_joint_state.get(jname, (None, None, None))
        if pos is None:
            continue
        gap = abs(np.degrees(pos - commanded))
        mark = " SATURATED" if jname in saturated else ""
        print(f"    {jname:20s} commanded={np.degrees(commanded):7.1f} "
              f"real={np.degrees(pos):7.1f} gap={gap:5.1f}deg "
              f"peak_effort={peak[jname]:6.1f}Nm{mark}")

    clearance, worst_link, overlaps = link_clearances(node, robot_y)
    if clearance is None:
        print("  no link over the table footprint")
    else:
        print(f"  closest link over table: {worst_link} at {clearance:+.3f}m vs table top")
    if overlaps:
        print(f"  LINKS OVERLAPPING TABLE: {', '.join(overlaps)}")

    return {
        "robot_y": robot_y, "reachable": True, "saturated": saturated,
        "peak": peak, "clearance": clearance, "overlaps": overlaps,
        "settled": settled,
    }


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    if target_name not in APPLE_HOME_WORLD_XY:
        print(f"Unknown target {target_name}")
        return
    if len(sys.argv) > 2:
        candidates = [float(v) for v in sys.argv[2].split(",")]
    else:
        candidates = [-0.70, -0.75, -0.80, -0.85, -0.90]

    print(f"Target: {target_name}  (real grasp height = apple center + {GRASP_Z_OFFSET}m)")
    print(f"Testing stations Y = {candidates}")

    rclpy.init()
    node = FullLayerGraspNode()

    # ROS2 discovery has to finish before the first command, or it goes into the void
    # (confirmed directly: the arm silently stayed in its spawn pose).
    print("Waiting for the arm controller to connect...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.arm_pub.get_subscription_count() > 0:
            break
    if node.arm_pub.get_subscription_count() == 0:
        print("WARNING: no subscriber on the arm trajectory topic -- is the simulation "
              "running with controllers active?")

    results = []
    for robot_y in candidates:
        results.append(test_station(node, target_name, robot_y))

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    print(f"{'Y':>7} {'reach':>6} {'saturated joints':>28} {'clearance':>10}")
    for r in results:
        if not r["reachable"]:
            print(f"{r['robot_y']:7.2f} {'NO':>6} {'--':>28} {'--':>10}")
            continue
        sat = ", ".join(j.replace("_joint", "") for j in r["saturated"]) or "none"
        clr = f"{r['clearance']:+.3f}m" if r["clearance"] is not None else "n/a"
        print(f"{r['robot_y']:7.2f} {'yes':>6} {sat:>28} {clr:>10}")

    good = [r for r in results
            if r["reachable"] and not r["saturated"] and not r["overlaps"]]
    if good:
        best = max(good, key=lambda r: r["clearance"] if r["clearance"] is not None else 0)
        usable = ", ".join("{:.2f}".format(r["robot_y"]) for r in good)
        print(f"\nUsable stations (no saturated joint, no table overlap): {usable}")
        print(f"Best clearance among those: Y={best['robot_y']:.2f}")
    else:
        print("\nNo station in this sweep holds the real grasp pose without saturating "
              "a joint or overlapping the table. Try closer distances, or the arm may "
              "genuinely lack the torque for this reach with the DexHand attached.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
