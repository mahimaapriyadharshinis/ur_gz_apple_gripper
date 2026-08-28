#!/usr/bin/env python3
"""
One-shot diagnostic: sweep candidate station distances/positions for reaching
apple_06's grasp point, and report the actual shoulder_pan mismatch for each --
instead of guessing one station at a time by hand.

Usage: python3 find_station.py
"""
import numpy as np

from full_layer_grasp import build_chain, solve_ik, world_to_local, ARM_JOINTS

APPLE_WORLD_XY = (1.25, 0.0)
APPLE_WORLD_Z = 0.44  # apple_table top (0.40) + apple radius (0.04)
FINGER_LENGTH = 0.164  # must match full_layer_grasp.py's FINGER_LENGTH


def pan_mismatch_deg(chain, robot_x, robot_y, robot_yaw):
    x, y = world_to_local(APPLE_WORLD_XY[0], APPLE_WORLD_XY[1], robot_x, robot_y, robot_yaw)
    grasp_target = [x, y, APPLE_WORLD_Z + FINGER_LENGTH]

    expected_pan = np.degrees(np.arctan2(y, x))
    result = solve_ik(chain, grasp_target)
    if result is None:
        return x, y, expected_pan, None, None, None
    joints, _ = result
    actual_pan = np.degrees(joints[0])
    shoulder_lift = np.degrees(joints[1])
    mismatch = abs(((actual_pan - expected_pan + 180) % 360) - 180)
    return x, y, expected_pan, actual_pan, shoulder_lift, mismatch


def main():
    print("Building IK chain...")
    chain = build_chain()

    print(f"\nApple_06 at world {APPLE_WORLD_XY}, grasp target local frame testing:\n")
    print(f"{'ROBOT_X':>8} {'ROBOT_Y':>8} {'local_x':>8} {'local_y':>8} "
          f"{'exp_pan':>8} {'act_pan':>8} {'lift':>8} {'mismatch':>9}")

    results = []
    # Sweep both perpendicular distance (Y, extended further out than before -- closer
    # distances kept forcing a "face backward, redirect via the wrist" solution instead
    # of a natural forward-facing reach) and lateral position (X).
    for robot_y in [-0.45, -0.5, -0.55, -0.6, -0.65, -0.7, -0.75, -0.8, -0.85, -0.9,
                     -0.95, -1.0, -1.05, -1.1, -1.15, -1.2]:
        for robot_x in [1.0, 1.1, 1.2, 1.25, 1.3, 1.4, 1.5]:
            x, y, exp_pan, act_pan, lift, mismatch = pan_mismatch_deg(chain, robot_x, robot_y, 1.5708)
            if mismatch is None:
                print(f"{robot_x:8.2f} {robot_y:8.2f} {x:8.3f} {y:8.3f} "
                      f"{exp_pan:8.1f} {'UNREACHABLE':>8} {'--':>8} {'--':>9}")
            else:
                print(f"{robot_x:8.2f} {robot_y:8.2f} {x:8.3f} {y:8.3f} "
                      f"{exp_pan:8.1f} {act_pan:8.1f} {lift:8.1f} {mismatch:9.1f}")
                results.append((mismatch, robot_x, robot_y, lift))

    if results:
        # A station is only genuinely "natural" if BOTH pan and shoulder_lift are
        # modest -- low pan mismatch alone isn't enough, since a backward-pan solution
        # can still show a small mismatch if expected_pan itself happens to be near
        # 180 degrees for that particular (x, y).
        natural = [r for r in results if abs(r[3]) < 60]
        print(f"\n=== BEST 5 NATURAL (mismatch low AND |shoulder_lift| < 60deg) ===")
        for mismatch, rx, ry, lift in sorted(natural, key=lambda r: r[0])[:5]:
            print(f"  ROBOT_X={rx:.2f} ROBOT_Y={ry:.2f}  mismatch={mismatch:.1f}deg  shoulder_lift={lift:.1f}deg")
        if not natural:
            print("  None found -- no station in this sweep reaches apple_06 without "
                  "either a large pan mismatch or an extreme shoulder_lift.")
    else:
        print("\nNothing reachable at all in this sweep -- something more fundamental is wrong.")


if __name__ == '__main__':
    main()
