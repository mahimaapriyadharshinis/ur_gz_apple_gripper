#!/usr/bin/env python3
"""
Multi-station orchestration for the 7-layer grasp pipeline.

A UR5e-class arm's real, IK-verified reach is ~0.8m from a single mobile-base
position (see PROJECT_CONTEXT.md, section 6). The 10 apples span 2.25m, so no
single station can reach all of them. This script:

  1. Builds the real IK chain and tests reachability of every apple from a set
     of candidate mobile-base "stations" (using the actual solve_ik solver,
     not a distance estimate).
  2. Greedily picks the smallest set of stations that covers every reachable
     apple.
  3. Drives FullLayerGraspNode through the full pick-and-place sequence for
     each apple, teleporting the mobile base between stations as needed.
  4. Prints a summary on top of what Layer 7 already logs to experience_log.json.

Run `python3 pick_all_apples.py --plan-only` first to sanity-check the station
plan without touching ROS/Gazebo.
"""
import sys

import rclpy

from full_layer_grasp import (
    build_chain, solve_ik, world_to_local,
    FullLayerGraspNode, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW, EXPERIENCE_LOG,
)

APPLE_NAMES = [f"apple_{i:02d}" for i in range(1, 11)]
# Static layout from apple_gripper_sim/worlds/apple_world.world -- apples sit 0.25m
# apart along x at y=0.00. Used only to PLAN which station reaches which apple; the
# actual grasp always targets the live /model/<name>/pose, never this snapshot.
APPLE_WORLD_XY = {name: (0.25 * i, 0.00) for i, name in enumerate(APPLE_NAMES)}
APPLE_WORLD_Z = 0.05

STATION_Y = DELIVERY_ROBOT_Y
STATION_YAW = DELIVERY_ROBOT_YAW
CANDIDATE_STATION_X = [round(-0.2 + 0.2 * i, 2) for i in range(15)]  # -0.2 .. 2.6


FINGER_LENGTH = 0.164  # must match full_layer_grasp.py's FINGER_LENGTH


def apple_reachable(chain, station_x, apple_world_xy):
    x_local, y_local = world_to_local(apple_world_xy[0], apple_world_xy[1],
                                       station_x, STATION_Y, STATION_YAW)
    grasp_z = APPLE_WORLD_Z + FINGER_LENGTH
    approach = solve_ik(chain, [x_local, y_local, grasp_z + 0.15])
    grasp = solve_ik(chain, [x_local, y_local, grasp_z])
    return approach is not None and grasp is not None


def plan_stations(chain):
    reachability = {}
    for sx in CANDIDATE_STATION_X:
        reachability[sx] = {name for name in APPLE_NAMES
                             if apple_reachable(chain, sx, APPLE_WORLD_XY[name])}
        print(f"  station x={sx:+.2f} reaches {sorted(reachability[sx])}")

    uncovered = set(APPLE_NAMES)
    stations = []
    while uncovered:
        best_x, best_new = None, set()
        for sx, reached in reachability.items():
            new = reached & uncovered
            if len(new) > len(best_new):
                best_x, best_new = sx, new
        if not best_new:
            print(f"  WARNING: no candidate station reaches {sorted(uncovered)}.")
            break
        stations.append((best_x, sorted(best_new)))
        uncovered -= best_new

    return stations


def main():
    plan_only = "--plan-only" in sys.argv

    print("=== Building IK chain and testing reachability from candidate stations ===")
    chain = build_chain()
    stations = plan_stations(chain)

    print("\n=== STATION PLAN ===")
    for sx, apples in stations:
        print(f"  station x={sx:+.2f}, y={STATION_Y}, yaw={STATION_YAW} -> {apples}")
    covered = {a for _, apples in stations for a in apples}
    missing = [a for a in APPLE_NAMES if a not in covered]
    if missing:
        print(f"  UNREACHABLE from any candidate station: {missing}")

    if plan_only:
        print("\n--plan-only: not touching ROS/Gazebo.")
        return

    rclpy.init()
    node = FullLayerGraspNode()
    results = []
    for sx, apples in stations:
        node.robot_x, node.robot_y, node.robot_yaw = sx, STATION_Y, STATION_YAW
        for apple in apples:
            print(f"\n=== Picking {apple} from station x={sx:+.2f} ===")
            results.append(node.run_for_target(apple))

    node.destroy_node()
    rclpy.shutdown()

    print("\n=== SUMMARY ===")
    for r in results:
        status = "OK" if r.get("success") else "FAILED"
        print(f"  {r['target']}: {status} ({r})")
    for a in missing:
        print(f"  {a}: SKIPPED (unreachable from any station)")

    n_success = sum(1 for r in results if r.get("success"))
    print(f"\n{n_success}/{len(APPLE_NAMES)} apples picked and placed successfully "
          f"({len(missing)} unreachable). Full detail in {EXPERIENCE_LOG}.")


if __name__ == '__main__':
    main()
