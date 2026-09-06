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

Aiming that pocket at the apple is not enough on its own, though: with the hand
open the fingertips hang 0.164m below the wrist and ground out on the table, so the
arm cannot descend past ~0.57 and the pocket never gets below the apple's top. So
this also PRE-SHAPES the fingers before descending -- curling them partway retracts
the fingertips enough to reach, while the hand still spans far wider than the apple.

It aims the closed-fingertip centroid at the apple using the wrist's real achieved
orientation to rotate the offset (the same two-pass approach full_layer_grasp.py
uses), and sweeps the pre-shape angle.

Usage: python3 pocket_grasp_test.py [target_name]
"""
import sys
import time

import numpy as np
import rclpy

from full_layer_grasp import (
    FullLayerGraspNode, APPLE_HOME_WORLD_XY, apple_home_z, APPLE_RADIUS,
    DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW,
    world_to_local, solve_ik, hand_fk, ARM_JOINTS, FINGER_GROUPS,
    PALM_DOWN_ROTATION,
    EFFORT_CONTACT_THRESHOLD, MAX_PITCH_CEILING,
    THUMB_GRASP_YAW, THUMB_GRASP_ROLL,
)

# Fingertip centroid at full closure, in the hand's own frame, measured via TF with
# the fingertip joints working. This is the centre of the pocket the fingers curl
# into -- where an object has to be to end up gripped rather than brushed.
CLOSED_CENTROID_HAND_FRAME = np.array([0.0634, -0.0056, 0.0834])

# Pre-shape: how far the fingers are curled BEFORE descending.
#
# With the hand fully open the fingertips hang 0.164m below the wrist, so they strike
# the table (0.400) at any wrist height below 0.564 -- measured directly, the arm
# bottoms out at ~0.57 no matter what is commanded. But the grip pocket sits 0.083m
# below the wrist, so putting it at the apple's centre (0.440) needs the wrist at
# 0.523. Those two cannot both hold with an open hand, which is why the fingers end up
# closing over the apple's TOP and it squirts away.
#
# Curling the fingers partway first retracts the fingertips enough to descend, while
# the hand still spans far more than the apple: 15.38cm open, 9.17cm at pitch 1.0, so
# roughly 11.5cm at pitch 0.7 against an 8.00cm apple. This sweeps that pre-shape.
# Each case is (palm_down, preshape). palm_down pins the hand's full orientation so
# the palm faces the table and the fingers curl up under the apple; without it only
# the wrist axis is constrained and the palm's roll is random, which is why identical
# commands gave 5/5 contacts one run and 1/5 the next.
CASES = [
    (False, 0.0),   # baseline: what we have been running all along
    (True, 0.0),
    (True, 0.3),
    (True, 0.5),
]

REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]

# How far to raise the wrist after closing. Contact proves the fingers reached the
# apple; only lifting proves the grip actually holds it.
LIFT_HEIGHT = 0.15

# How close the apple must still be to the wrist afterwards to count as held.
# Roughly the hand's own size -- further than this and it is not in the hand.
HOLD_DISTANCE = 0.20


def settle(node, seconds, joints=ARM_JOINTS, thresh=0.05):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
        vels = [abs(node.latest_joint_state.get(j, (0, 0, 0))[1] or 0.0) for j in joints]
        if vels and max(vels) < thresh and (time.time() - start) > 3.0:
            return True
    return False


# Closing in +0.05 rad steps and only checking force every ~0.3s let the position
# controller keep driving hard into the apple between samples: R_Middle was measured
# peaking at 7-11Nm against a 0.12Nm contact threshold -- roughly 110N on a 0.18kg
# apple, which simply launches it. The first finger to arrive threw the apple clear
# before the other four got near it, which is why every attempt stops at 1/5.
# Smaller steps checked far more often stop a finger when it first feels the apple
# instead of long after.
CLOSE_STEP = 0.015
CHECKS_PER_STEP = 2


def close_and_measure(node, start_pitch=0.0, apple_local=None, radius=None):
    contacted = {g: False for g in FINGER_GROUPS}
    peak = {g: 0.0 for g in FINGER_GROUPS}
    current = {g: start_pitch for g in FINGER_GROUPS}
    steps = int((MAX_PITCH_CEILING - start_pitch) / CLOSE_STEP) + 2
    for _ in range(steps):
        for g in FINGER_GROUPS:
            if not contacted[g]:
                current[g] = min(current[g] + CLOSE_STEP, MAX_PITCH_CEILING)
        node.command_fingers(current, 0.08, thumb_yaw=THUMB_GRASP_YAW,
                             thumb_roll=THUMB_GRASP_ROLL)
        for _ in range(CHECKS_PER_STEP):
            rclpy.spin_once(node, timeout_sec=0.05)
            for g in FINGER_GROUPS:
                _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
                eff = abs(eff or 0.0)
                peak[g] = max(peak[g], eff)
                if eff > EFFORT_CONTACT_THRESHOLD and not contacted[g]:
                    # Force alone cannot tell the apple from the table -- a run with
                    # fingers jammed in the tabletop reported 4/5 "contacts" at
                    # saturated 100Nm while the apple never moved. Only count it if
                    # this fingertip is actually at the apple.
                    if apple_local is not None and radius is not None:
                        if not node.fingertips_near_apple(apple_local, radius).get(g):
                            continue
                    contacted[g] = True
                    # Hold this finger exactly where it is the moment it feels the
                    # apple, so it stops pushing instead of driving on to its target.
                    pos = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))[0]
                    if pos is not None:
                        current[g] = pos
        if all(contacted.values()):
            break
    return contacted, peak


def apple_xyz(node):
    if node.target_pose is None:
        return None
    p = node.target_pose.position
    return np.array([p.x, p.y, p.z])


def attempt(node, target_name, palm_down, preshape):
    label = "palm-DOWN" if palm_down else "free-roll (baseline)"
    print(f"\n{'=' * 72}\n{label}, pre-shape {preshape:.2f}\n{'=' * 72}")
    rot = PALM_DOWN_ROTATION if palm_down else None

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    node.robot_x, node.robot_y, node.robot_yaw = wx, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW
    node.reset_everything()
    node.reset_target_apple_position(target_name)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    before = apple_xyz(node)

    x, y = world_to_local(wx, wy, node.robot_x, node.robot_y, node.robot_yaw)
    apple_local = np.array([x, y, apple_home_z(target_name)])

    # Pass 1: a rough solve just to learn which way the wrist ends up facing, since the
    # pocket offset is expressed in the hand's own frame and has to be rotated into the
    # robot frame before it means anything.
    rough = solve_ik(node.chain, list(apple_local + np.array([0, 0, 0.164])),
                     target_rotation=rot)
    if rough is None:
        print("  rough solve UNREACHABLE")
        return {"preshape": preshape, "palm_down": palm_down, "ok": False}
    wrist_rot = hand_fk(node.chain, rough[1])[:3, :3]

    offset_local = wrist_rot @ CLOSED_CENTROID_HAND_FRAME
    wrist_target = apple_local - offset_local

    print(f"  apple at local ({apple_local[0]:.3f}, {apple_local[1]:.3f}, {apple_local[2]:.3f})")
    print(f"  pocket offset rotated -> ({offset_local[0]:.3f}, {offset_local[1]:.3f}, "
          f"{offset_local[2]:.3f})")
    print(f"  wrist target ({wrist_target[0]:.3f}, {wrist_target[1]:.3f}, "
          f"{wrist_target[2]:.3f})")

    approach = solve_ik(node.chain, list(wrist_target + np.array([0, 0, 0.09])),
                        target_rotation=rot)
    grasp = solve_ik(node.chain, list(wrist_target), target_rotation=rot)
    if approach is None or grasp is None:
        print("  UNREACHABLE")
        return {"preshape": preshape, "palm_down": palm_down, "ok": False}

    # Pre-shape BEFORE descending, so the fingertips are retracted on the way down and
    # the wrist can actually reach the pocket height instead of the fingers grounding
    # out on the table first.
    node.command_fingers({g: preshape for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    settle(node, 6.0, joints=[f"{g}_Pitch" for g in FINGER_GROUPS], thresh=0.02)
    node.send_arm_trajectory(approach[0], 3.5)
    settle(node, 12.0)
    node.send_arm_trajectory(grasp[0], 3.0)
    settle(node, 20.0)

    real = node.real_wrist_position()
    if real:
        err = float(np.linalg.norm(np.array(real) - wrist_target))
        print(f"  wrist real ({real[0]:.3f}, {real[1]:.3f}, {real[2]:.3f}) err={err:.3f}m")

    # Verify the PALM actually ended up where it was asked to. The hand's +X axis is
    # the palm normal (fingers along +Z, thumb opposing along +X), so palm-down means
    # that axis pointing at the table. Reporting it makes a silently-ignored
    # orientation request obvious in the log rather than only in the viewer.
    palm = node.real_palm_normal()
    if palm is not None:
        down = float(np.dot(palm, np.array([0.0, 0.0, -1.0])))
        tilt = float(np.degrees(np.arccos(max(-1.0, min(1.0, down)))))
        print(f"  palm normal ({palm[0]:+.2f}, {palm[1]:+.2f}, {palm[2]:+.2f}) -- "
              f"{tilt:.0f}deg from straight down "
              f"({'PARALLEL to ground' if tilt < 25 else 'NOT parallel'})")

    contacted, peak = close_and_measure(
        node, start_pitch=preshape, apple_local=apple_local,
        radius=APPLE_RADIUS.get(target_name, 0.0555))
    n = sum(contacted.values())
    print(f"  fingers contacted: {n}/5")
    print("  peak efforts: " + ", ".join("%s=%.3f" % (g, peak[g]) for g in FINGER_GROUPS))

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    after_close = apple_xyz(node)
    moved = (float(np.linalg.norm(after_close - before))
             if before is not None and after_close is not None else None)
    if moved is not None:
        print(f"  apple moved {moved:.3f}m while closing")

    # Actually LIFT. Contact alone proves the fingers reached the apple; only raising
    # it proves the grip holds. Keeping the fingers commanded where they stopped, so
    # the hold is maintained through the lift rather than relaxing.
    lifted = None
    lift_target = list(wrist_target + np.array([0, 0, LIFT_HEIGHT]))
    lift = solve_ik(node.chain, lift_target, target_rotation=rot)
    if lift is None:
        print("  lift target UNREACHABLE -- cannot test the hold")
    else:
        print(f"  lifting {LIFT_HEIGHT:.2f}m...")
        node.send_arm_trajectory(lift[0], 3.0)
        settle(node, 15.0)
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
        after_lift = apple_xyz(node)
        wrist_after = node.real_wrist_position()
        if before is not None and after_lift is not None:
            lifted = float(after_lift[2] - before[2])
            # Height alone is NOT enough: an apple flung across the room can land
            # higher than it started and score as a success. Measured directly -- one
            # attempt threw the apple 79m, ended +0.400m up, and was reported HELD.
            # A real hold means the apple is still in the hand.
            near = None
            if wrist_after is not None:
                near = float(np.linalg.norm(after_lift - np.array(wrist_after)))
            held = (lifted > LIFT_HEIGHT * 0.5 and near is not None
                    and near < HOLD_DISTANCE)
            near_s = f"{near:.3f}m from wrist" if near is not None else "wrist unknown"
            print(f"  apple height change after lift: {lifted:+.3f}m, {near_s} "
                  f"({'HELD' if held else 'not held'})")
            if not held:
                lifted = None

    return {"preshape": preshape, "palm_down": palm_down, "ok": True,
            "contacts": n, "peak": peak,
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

    results = [attempt(node, target_name, pd, ps) for pd, ps in CASES]

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'case':>22} {'contacts':>9} {'max_effort':>11} {'apple_moved':>12} {'lifted':>9}")
    for r in results:
        if not r.get("ok"):
            nm = ('palm-down' if r['palm_down'] else 'free-roll') + f" ps={r['preshape']:.1f}"
            print(f"{nm:>22} {'UNREACHABLE':>9}")
            continue
        mx = max(r["peak"].values())
        moved = f"{r['moved']:.3f}m" if r["moved"] is not None else "n/a"
        lifted = f"{r['lifted']:+.3f}m" if r["lifted"] is not None else "n/a"
        nm = ('palm-down' if r['palm_down'] else 'free-roll') + f" ps={r['preshape']:.1f}"
        print(f"{nm:>22} {r['contacts']:>7}/5 {mx:11.3f} {moved:>12} {lifted:>9}")

    held = [r for r in results
            if r.get("ok") and r.get("lifted") is not None
            and r["lifted"] > LIFT_HEIGHT * 0.5]
    if held:
        b = max(held, key=lambda r: r["contacts"])
        print(f"\nLIFTED: pre-shape {b['preshape']:.2f} held the apple through a "
              f"{LIFT_HEIGHT:.2f}m lift with {b['contacts']}/5 fingers. That is a "
              f"complete grasp.")

    best = max((r for r in results if r.get("ok")),
               key=lambda r: (r["contacts"], max(r["peak"].values())), default=None)
    if best and best["contacts"] >= 3:
        print(f"\nGRIP: {best['contacts']}/5 fingers at depth offset "
              f"{best['preshape']:.2f}. This is a working grasp configuration.")
    elif best and best["contacts"] > 0:
        print(f"\nPartial: best was {best['contacts']}/5 at depth offset "
              f"{best['preshape']:.2f}. Worth sweeping finer around it.")
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
