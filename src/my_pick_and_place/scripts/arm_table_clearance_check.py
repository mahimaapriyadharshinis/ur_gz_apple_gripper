#!/usr/bin/env python3
"""
One-off diagnostic: which arm link is actually hitting the table?

Motivation: commanding the wrist to the real, correct grasp height (0.542,
computed from the real measured fingertip reach) makes shoulder_lift_joint peg
at its 150Nm effort limit -- a real physical collision, confirmed directly.
But find_station.py (pure kinematics) shows this exact station reaches that
height with an easy, modest joint angle (-8.8deg) -- so it's NOT a
reachability problem, and searching for a different station is pointless
without knowing what's actually being hit. ikpy has no concept of collision
geometry at all, so this checks the REAL arm link positions via TF (same
ground-truth method real_wrist_position() already uses) against the table's
known physical boundaries, instead of guessing or eyeballing a screenshot.

Usage: with the simulation running (robot spawned, controllers active), run:
  python3 arm_table_clearance_check.py [target_name]
Commands the arm toward the same problem height used for target_name (apple_06
by default) and reports each link's real position vs. the table's real extent.
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, APPLE_HOME_Z,
    DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, ARM_JOINTS,
)

# Real measured fingertip reach (finger_geometry_check.py) -- the height that
# actually needs to be reached for the fingers to touch the apple, and the one
# confirmed to cause the collision.
GRASP_Z_OFFSET = 0.102

ARM_LINKS = ["shoulder_link", "upper_arm_link", "forearm_link",
             "wrist_1_link", "wrist_2_link", "wrist_3_link", "dexhand_base_link"]

# apple_table in apple_world.world: model pose (1.125, 0.0, 0.0) + collision box's
# local pose (0, 0, 0.20), size 2.60 x 0.50 x 0.40 -- world corners/top computed
# directly from those declared values, not measured/guessed.
TABLE_WORLD_CENTER_XY = (1.125, 0.0)
TABLE_SIZE_XY = (2.60, 0.50)
TABLE_TOP_Z = 0.40


def table_local_bbox():
    cx, cy = TABLE_WORLD_CENTER_XY
    hx, hy = TABLE_SIZE_XY[0] / 2.0, TABLE_SIZE_XY[1] / 2.0
    corners_world = [(cx + sx * hx, cy + sy * hy) for sx in (-1, 1) for sy in (-1, 1)]
    corners_local = [world_to_local(wx, wy, DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)
                      for wx, wy in corners_world]
    xs = [c[0] for c in corners_local]
    ys = [c[1] for c in corners_local]
    return min(xs), max(xs), min(ys), max(ys)


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    if target_name not in APPLE_HOME_WORLD_XY:
        print(f"Unknown target {target_name}")
        return

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    x, y = world_to_local(wx, wy, DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)
    grasp_target = [x, y, APPLE_HOME_Z + GRASP_Z_OFFSET]
    print(f"Target: {target_name}, real grasp height target = {grasp_target}")

    rclpy.init()
    node = FullLayerGraspNode()

    result = solve_ik(node.chain, grasp_target)
    if result is None:
        print(f"UNREACHABLE: {grasp_target}")
        node.destroy_node()
        rclpy.shutdown()
        return
    joints, _ = result

    # Wait for the trajectory publisher to actually connect to the controller before
    # publishing. Confirmed directly: publishing immediately after node creation sent
    # the command into the void (arm stayed in its spawn pose, all efforts normal,
    # nothing moved) because ROS2 discovery hadn't finished yet. The normal grasp flow
    # never hits this because reset_everything()'s teleport + sleeps give discovery
    # plenty of time before the first arm command.
    print("Waiting for the arm controller to connect...")
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.arm_pub.get_subscription_count() > 0:
            break
    print(f"arm_pub subscriber count: {node.arm_pub.get_subscription_count()}")
    if node.arm_pub.get_subscription_count() == 0:
        print("WARNING: no subscriber on the arm trajectory topic -- the command will "
              "go nowhere. Is the simulation actually running with controllers active?")

    min_x, max_x, min_y, max_y = table_local_bbox()
    print(f"\nTable footprint in robot-local frame: x=[{min_x:.3f}, {max_x:.3f}] "
          f"y=[{min_y:.3f}, {max_y:.3f}], top_z={TABLE_TOP_Z:.3f}")

    print("\nCommanding arm toward the grasp height (may peg at max effort -- expected)...")
    node.send_arm_trajectory(joints, 4.0)

    # Sample continuously rather than measuring once after "settling". Two reasons,
    # both confirmed directly: (1) a single measurement caught the arm mid-flight
    # (still moving at 0.44 rad/s) and showed normal efforts, which reads as "no
    # collision" purely because it was taken too early; (2) the collision can be
    # transient -- a link can strike the table during the motion even if the final
    # pose looks clear. Sampling catches whichever moment things actually go wrong.
    SAMPLE_SECONDS = 60.0
    SAMPLE_INTERVAL = 2.0
    start = time.time()
    worst_efforts = {j: 0.0 for j in ARM_JOINTS}
    overlaps_seen = set()

    while time.time() - start < SAMPLE_SECONDS:
        for _ in range(int(SAMPLE_INTERVAL / 0.1)):
            rclpy.spin_once(node, timeout_sec=0.1)
        elapsed = time.time() - start

        positions = {}
        for link in ARM_LINKS:
            try:
                t = node.tf_buffer.lookup_transform(
                    'base_footprint', link, rclpy.time.Time(),
                    timeout=rclpy.duration.Duration(seconds=0.5))
                p = t.transform.translation
                positions[link] = (p.x, p.y, p.z)
            except Exception:
                positions[link] = None

        flagged = []
        for link, pos in positions.items():
            if pos is None:
                continue
            px, py, pz = pos
            if min_x <= px <= max_x and min_y <= py <= max_y and pz <= TABLE_TOP_Z:
                flagged.append(f"{link} at ({px:.3f}, {py:.3f}, {pz:.3f})")
                overlaps_seen.add(link)

        max_vel = 0.0
        effort_str = []
        for jname in ARM_JOINTS:
            state = node.latest_joint_state.get(jname, (None, None, None))
            if state[1] is not None:
                max_vel = max(max_vel, abs(state[1]))
            if state[2] is not None:
                worst_efforts[jname] = max(worst_efforts[jname], abs(state[2]))
                if abs(state[2]) > 100.0:
                    effort_str.append(f"{jname}={state[2]:.1f}Nm")

        line = f"t={elapsed:5.1f}s  max_vel={max_vel:.3f}"
        if effort_str:
            line += "  HIGH EFFORT: " + ", ".join(effort_str)
        if flagged:
            line += "  OVERLAPS TABLE: " + "; ".join(flagged)
        print(line)

        if max_vel < 0.05 and elapsed > 6.0:
            print("Arm settled.")
            break

    print(f"\nFinal link positions (robot-local frame):")
    for link in ARM_LINKS:
        try:
            t = node.tf_buffer.lookup_transform(
                'base_footprint', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=2.0))
            p = t.transform.translation
            inside_xy = min_x <= p.x <= max_x and min_y <= p.y <= max_y
            below_top = p.z <= TABLE_TOP_Z
            clearance = p.z - TABLE_TOP_Z
            flag = " <-- OVERLAPS TABLE VOLUME" if (inside_xy and below_top) else ""
            over = "over table" if inside_xy else "clear of table footprint"
            print(f"{link:18s}: ({p.x:.3f}, {p.y:.3f}, {p.z:.3f})  "
                  f"{over}, {clearance:+.3f}m vs table top{flag}")
        except Exception as e:
            print(f"{link:18s}: TF lookup failed: {e}")

    print()
    for jname, commanded in zip(ARM_JOINTS, joints):
        state = node.latest_joint_state.get(jname, (None, None, None))
        if state[2] is not None:
            gap = abs(np.degrees(state[0] - commanded))
            note = "  <-- did NOT reach commanded" if gap > 5.0 else ""
            print(f"  {jname}: commanded={np.degrees(commanded):.1f}deg "
                  f"real={np.degrees(state[0]):.1f}deg effort={state[2]:.2f}Nm "
                  f"(peak seen {worst_efforts[jname]:.1f}Nm){note}")

    if overlaps_seen:
        print(f"\nLinks that overlapped the table volume at any point: "
              f"{', '.join(sorted(overlaps_seen))}")
    else:
        print("\nNo arm link's ORIGIN entered the table volume at any point. Note this "
              "checks link origins, not full mesh extents -- a link whose origin stays "
              "clear can still have geometry dipping into the table, so a pegged effort "
              "with no overlap flagged here means the contact is from link geometry "
              "rather than the origin point.")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
