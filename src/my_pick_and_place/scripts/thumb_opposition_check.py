#!/usr/bin/env python3
"""
Where should the thumb sit so it opposes the fingers instead of blocking them?

The thumb has two joints that position it relative to the palm -- R_Thumb_Yaw
(+/-0.524 rad) and R_Thumb_Roll (+/-0.349 rad), both mounted straight onto
dexhand_base_link. Neither has ever been commanded: command_fingers only sent
Pitch/Flexor/DIP, so the thumb has sat in its default splayed pose for the whole
project.

That default is the problem. Measured via TF, the thumb tip sits at hand-frame
X=+0.119m while the four fingers sit at X=+0.007m. Hand-X is the palm normal, so
with the palm facing down (which is what we now command) the thumb points straight
at the table, 12cm below the fingers. It grounds out before the fingers reach
anything -- measured directly as 15.3Nm on the thumb against 0.38-0.40Nm on every
other finger, with the hand closing on empty space.

This sweeps yaw and roll and measures, for each combination, where the thumb tip
actually ends up relative to the fingers: how far it still protrudes along the palm
normal (less is better -- it stops grounding out) and how well it opposes the
fingers across the grasp pocket (more is better -- that is what lets it pinch).

Usage: python3 thumb_opposition_check.py
"""
import time

import numpy as np
import rclpy

from full_layer_grasp import FullLayerGraspNode, FINGER_GROUPS, FINGERTIP_LINK

YAWS = [-0.50, -0.25, 0.0, 0.25, 0.50]
ROLLS = [-0.30, 0.0, 0.30]

FINGER_TIPS = ["Index_Tip_1", "Midle_Tip_1", "Ring_Tip_1", "Pinky_Tip_1"]


def tip(node, link):
    try:
        t = node.tf_buffer.lookup_transform(
            'dexhand_base_link', link, rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0))
        p = t.transform.translation
        return np.array([p.x, p.y, p.z])
    except Exception:
        return None


def settle_thumb(node, yaw, roll, timeout=12.0, min_wait=2.0):
    """Wait until the thumb joints actually REACH the commanded angles.

    Waiting for "no longer changing" is not enough on its own: called right after
    publishing, the joints have not started moving yet, so unchanged reads as settled
    and the measurement is taken before anything happened. That produced byte-identical
    rows for different commands -- each row showing the PREVIOUS setting's result.
    Requiring the commanded value to be reached (and a minimum wait first) removes the
    ambiguity.
    """
    start = time.time()
    while time.time() - start < timeout:
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
        if time.time() - start < min_wait:
            continue
        y = node.latest_joint_state.get('R_Thumb_Yaw', (None, None, None))[0]
        r = node.latest_joint_state.get('R_Thumb_Roll', (None, None, None))[0]
        if y is None or r is None:
            continue
        if abs(y - yaw) < 0.02 and abs(r - roll) < 0.02:
            return True, y, r
    y = node.latest_joint_state.get('R_Thumb_Yaw', (None, None, None))[0]
    r = node.latest_joint_state.get('R_Thumb_Roll', (None, None, None))[0]
    return False, y, r


def main():
    rclpy.init()
    node = FullLayerGraspNode()
    for _ in range(100):
        rclpy.spin_once(node, timeout_sec=0.1)
        if node.hand_pub.get_subscription_count() > 0:
            break

    print("Sweeping thumb yaw/roll with the hand open.\n")
    print(f"{'yaw':>6} {'roll':>6} {'real_yaw':>9} {'thumb_X':>9} "
          f"{'gap_to_fingers':>15} {'opposition':>11}")

    rows = []
    for yaw in YAWS:
        for roll in ROLLS:
            node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                                 thumb_yaw=yaw, thumb_roll=roll)
            reached, real_yaw, real_roll = settle_thumb(node, yaw, roll)
            if not reached:
                ry = f"{real_yaw:.3f}" if real_yaw is not None else "?"
                rr = f"{real_roll:.3f}" if real_roll is not None else "?"
                print(f"{yaw:6.2f} {roll:6.2f}   did NOT reach command "
                      f"(real yaw={ry} roll={rr}) -- reading skipped")
                continue

            th = tip(node, FINGERTIP_LINK["R_Thumb"])
            fingers = [tip(node, l) for l in FINGER_TIPS]
            fingers = [f for f in fingers if f is not None]
            if th is None or not fingers:
                print(f"{yaw:6.2f} {roll:6.2f}   TF lookup failed")
                continue
            fc = np.mean(fingers, axis=0)

            # How far the thumb still sticks out along the palm normal (hand +X).
            # This is what grounds out on the table when the palm faces down.
            protrude = float(th[0])
            gap = float(np.linalg.norm(th - fc))
            # Opposition: the thumb should sit across the pocket from the fingers, i.e.
            # displaced from them mainly along the palm normal rather than beside them.
            d = th - fc
            n = float(np.linalg.norm(d))
            opposition = float(abs(d[0]) / n) if n > 1e-6 else 0.0

            print(f"{yaw:6.2f} {roll:6.2f} {real_yaw:9.3f} {th[0]:9.4f} "
                  f"{gap:15.4f} {opposition:11.3f}")
            rows.append((yaw, roll, protrude, gap, opposition))

    if rows:
        # Want the thumb pulled IN (small protrusion, so it clears the table) while
        # still far enough from the fingers to have an apple between them.
        usable = [r for r in rows if r[3] > 0.06]
        best = min(usable or rows, key=lambda r: r[2])
        # Look the default up rather than indexing a fixed row -- rows can now be
        # skipped when the thumb fails to reach a commanded angle, so a fixed index
        # would either crash or silently report the wrong row.
        default = next((r for r in rows if abs(r[0]) < 1e-6 and abs(r[1]) < 1e-6), None)
        was = f"{default[2]:.4f}m" if default else "not measured"
        print(f"\nLeast protruding while still {best[3]:.3f}m from the fingers: "
              f"yaw={best[0]:.2f} roll={best[1]:.2f} (protrudes {best[2]:.4f}m; "
              f"at the never-commanded default yaw=0 roll=0 it was {was})")
        print("Lower protrusion means the thumb stops hitting the table first; the gap "
              "must stay wider than an apple radius or there is nowhere to hold it.")

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=0.0, thumb_roll=0.0)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
