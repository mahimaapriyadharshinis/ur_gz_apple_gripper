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
    FINGERTIP_LINK, TABLE_TOP_Z,
)

# Fingertip centroid at full closure, in the hand's own frame, measured via TF with
# the fingertip joints working. This is the centre of the pocket the fingers curl
# into -- where an object has to be to end up gripped rather than brushed.
_CLOSED_CENTROID_MEASURED = np.array([0.0634, -0.0056, 0.0834])

# Aiming the measured closed-fingertip centroid at the apple puts the wrist only
# 0.105m from the apple's CENTRE. With a 0.0555m radius the apple's surface then
# reaches to within 0.05m of the wrist -- and the DexHand's palm body is roughly
# 0.08-0.10m across, so the palm is driven straight into the apple. In hand
# coordinates the apple would occupy from z=0.028 outward while the palm sits at
# z=0: they overlap.
#
# That matches the measurement exactly: an attempt with the wrist at err=0.000m and
# the palm 0deg from parallel -- perfect on both counts -- still knocked the apple
# 0.140m during the descent, before any finger moved. Position and orientation were
# never the problem; the target point itself was inside the hand.
#
# Pushing the aim point further out along the hand's forward axis puts the apple in
# the open span between fingers and thumb instead of against the palm.
PALM_CLEARANCE = 0.045
CLOSED_CENTROID_HAND_FRAME = _CLOSED_CENTROID_MEASURED + np.array([0.0, 0.0, PALM_CLEARANCE])

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
# Each case pins the hand's full orientation rather than only the wrist axis: leaving
# the palm's roll free is why identical commands once gave 5/5 contacts on one run and
# 1/5 on the next. What that orientation should BE is now the variable -- see palm_tilt
# in CASES below, and tilted_palm_rotation() for why flat turned out to be wrong.
#
# thumb_yaw was fixed at -0.50 because that pulls the thumb closest in and gives it
# the most table clearance (protrudes 0.0946m vs 0.1229m at +0.50). But it also
# gives the LEAST room between thumb and fingers -- 0.135m against 0.179m at +0.50 --
# and the apple is 0.111m across. Optimising for clearance quietly optimised against
# having anywhere to put the apple, so this sweeps the trade-off with real grasps
# rather than assuming which end matters more.
# Fixed now, and settled by measurement rather than argument.
#
# Palm tilt 30deg: with the palm FLAT the four fingers met the apple 43, 44, 52 and 55mm
# ABOVE its equator -- the pinky a millimetre from the top pole -- while the thumb sat
# 21mm below, 70mm apart, unable to oppose. At 30deg they meet it at +5, +3, +4 and +4mm
# with the thumb at -6mm: 10mm apart, all five essentially on the apple's widest circle.
# That is the grasp geometry the project has been missing. 45deg and 60deg were worse
# (fingers 9-14mm INSIDE the apple, 0/5 and 1/5 contacts).
PALM_TILT = np.radians(30.0)

# How far out along the fingers, in the hand's own frame. Settled earlier: 0.060 and
# 0.085 drove the palm into the apple; 0.120 gave the evenest fingertip spread.
AIM_DEPTH = 0.120

# Tilt 45deg with 0.090 back-off is the best geometry found: it is the only setting
# that has put a finger BELOW the apple's equator (ring at -12mm) with the pinky on it
# at +2mm. 30deg gets more fingers but all of them 13-26mm above the equator, where
# squeezing pushes the apple down and out; 60deg misses entirely.
#
# What 45deg does not fix is that index and middle never touch the apple at all. They
# start 14-16mm further from it than the pinky and peak at 0.023Nm -- below the 0.046Nm
# free-air noise floor -- so they close past it into empty space. The apple is sitting
# off to the ring/pinky side of the hand rather than in the middle of the four fingers,
# which is the gap it keeps escaping through.
#
# The cause is in the aim itself. The across-the-hand target has always been the mean of
# all FIVE fingertips, and the thumb sits well off to one side, so that mean is pulled
# away from the four fingers' own centre line. This sweeps that offset.
PALM_TILT = np.radians(45.0)
PALM_OFFSET = 0.0900

CASES = [
    # (palm_tilt, preshape, lateral, palm_offset)
    (PALM_TILT, 0.3,  0.000, PALM_OFFSET),   # control: the five-fingertip mean
    (PALM_TILT, 0.3, -0.015, PALM_OFFSET),
    (PALM_TILT, 0.3, +0.015, PALM_OFFSET),
    (PALM_TILT, 0.3, +0.030, PALM_OFFSET),
]

REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]

# How far to raise the wrist after closing. Contact proves the fingers reached the
# apple; only lifting proves the grip actually holds it.
LIFT_HEIGHT = 0.15

# How far back along the hand's forward axis to start, so the fingers move in
# beside the apple rather than being lowered through it.
APPROACH_BACKOFF = 0.16

# How close the apple must still be to the wrist afterwards to count as held.
# Roughly the hand's own size -- further than this and it is not in the hand.
HOLD_DISTANCE = 0.20

# Fraction of the measured wrist error to correct per iteration. Full
# correction overshoots and oscillates; damping it converges.
CORRECTION_GAIN = 0.6
# Attempt 1 of the thumb-last run converged 0.233 -> 0.150 -> 0.113 -> 0.075 -> 0.058
# and then simply ran out of iterations, starting the grasp with the arm still 0.031m
# out and still moving -- it knocked the apple 0.219m during the descent. It was
# improving the whole way, so give it room to finish. Attempts that start close still
# converge in one pass and cost nothing.
CORRECTION_ITERS = 9


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
# Slowed further still. Even at 0.015 rad steps the force jumped straight from 0.044Nm
# (nothing) to 36-40Nm (a hard slam) with nothing in between -- measured with R_Pinky
# at 36.840 and 40.308 while every other finger sat at ~0.99Nm. There is no gentle
# middle, so the finger is already driving hard into the apple by the time contact is
# noticed, and it shoves it away. Smaller steps, checked more often, with a longer
# command window so the joint tracks rather than lunges.
CLOSE_STEP = 0.003
CHECKS_PER_STEP = 5
STEP_COMMAND_TIME = 0.30

# The four fingers idle at 0.043-0.046Nm, so anything meaningfully above that is real
# contact. The old 0.12Nm threshold sat far enough above the noise that a finger had
# already begun pushing before it tripped.
# Set at runtime by calibrate_contact_threshold(). The old fixed 0.075 was chosen
# against the IDLE noise floor (0.032-0.046Nm), but the fingers are MOVING when they
# close, and a moving finger appears to read far more than that on its own. The evidence:
# attempts reporting 5/5 contacts had index, middle and ring peaking at 0.198/0.204/0.206
# and at 0.133/0.129/0.122 -- near-identical across three fingers, which is a common
# cause, not three separate collisions -- while a finger genuinely on the apple reads its
# 1.500Nm cap. If closing noise alone crosses 0.075 then every finger "contacts"
# immediately, freezes short of the fruit, and the squeeze drives it into empty air,
# which is exactly the 0.01-0.04Nm holding force every run ends with.
CONTACT_THRESHOLD = 0.075

# Filled in per finger by calibrate_contact_threshold(); falls back to the flat value.
CONTACT_THRESHOLD_BY_FINGER = {}

# After every finger has touched, squeeze a little further to actually GRIP.
#
# Freezing each finger the instant it feels contact produced touch without grip:
# measured R_Index=0.127, R_Middle=0.114, R_Ring=0.108 -- barely above the 0.075
# detection threshold, i.e. resting on the apple rather than holding it. The apple
# then simply stayed behind when the arm lifted. A real hold needs the fingers to
# keep closing past first contact until they are pressing.
SQUEEZE_EXTRA = 0.12
SQUEEZE_STEP = 0.006
SQUEEZE_FORCE_CAP = 3.0

# How far past first contact to drive the thumb before the fingers close, and the force
# at which to stop. First contact alone measured 0.103-0.284Nm, which the fingers then
# overwhelmed at 1.5-5.9Nm.
THUMB_PRELOAD_EXTRA = 0.10
THUMB_PRELOAD_FORCE = 1.2
THUMB_PRELOAD_STEP = 0.002

# How far above free-air closing noise a reading must be to count as real contact.
CONTACT_MARGIN = 2.0


# The gripper camera is not just a sensor: ur5e_dexhand.xacro gives gripper_camera_link
# a 0.04 x 0.04 x 0.02m COLLISION box, mounted 0.06m off the hand's centre line at
# z=0.08. In hand coordinates it therefore occupies z=0.070 to 0.090 -- and the
# fingertips converge at z=0.0834, dead centre of that band. The camera sits at exactly
# the depth the fingers close through, on one side of the hand. The apple itself clears
# it by 23-44mm at every aim tested, so it is not hitting the fruit; whether a FINGER
# passes through it has never been measured.
CAMERA_LINK = "gripper_camera_link"
CAMERA_HALF_EXTENTS = np.array([0.02, 0.02, 0.01])


def camera_clearances(node):
    """Each fingertip's distance to the camera's collision box surface, in metres.

    Negative means the fingertip is inside the box, i.e. the camera is physically
    blocking that finger from closing.
    """
    out = {}
    for g, link in FINGERTIP_LINK.items():
        try:
            t = node.tf_buffer.lookup_transform(
                CAMERA_LINK, link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p_ = t.transform.translation
            local = np.array([p_.x, p_.y, p_.z])
            outside = np.maximum(np.abs(local) - CAMERA_HALF_EXTENTS, 0.0)
            out[g] = float(np.linalg.norm(outside))
        except Exception:
            out[g] = None
    return out


def calibrate_contact_threshold(node):
    """Close the hand in free air and measure what a MOVING finger reads with nothing
    to touch. Returns a threshold safely above that, or None if it cannot measure.

    Everything downstream depends on telling "this finger is on the apple" from "this
    finger is moving", and that line has never actually been measured -- only assumed
    from the idle noise floor, which is a different quantity.
    """
    print("Calibrating: closing the hand in free air to measure moving-finger noise...")
    node.send_arm_trajectory(REST_POSE, 4.0)
    settle(node, 8.0)
    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    peak = {g: 0.0 for g in FINGER_GROUPS}
    pos = 0.0
    while pos < MAX_PITCH_CEILING:
        pos = min(pos + CLOSE_STEP, MAX_PITCH_CEILING)
        node.command_fingers({g: pos for g in FINGER_GROUPS}, STEP_COMMAND_TIME,
                             thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
        for _ in range(CHECKS_PER_STEP):
            rclpy.spin_once(node, timeout_sec=0.08)
            for g in FINGER_GROUPS:
                _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
                peak[g] = max(peak[g], abs(eff or 0.0))

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    print("  closing in free air, peak effort per finger: "
          + ", ".join("%s=%.3f" % (g.replace("R_", ""), peak[g]) for g in FINGER_GROUPS))
    if max(peak.values(), default=0.0) <= 0.0:
        print("  could not measure -- keeping the existing threshold")
        return None
    # Per finger, not one number for all: measured, the four fingers read 0.029Nm moving
    # in free air while the thumb reads 0.053Nm. Taking the worst of the five and
    # applying it everywhere makes the fingers 2x less sensitive than they need to be,
    # for no reason other than the thumb being noisier.
    out = {g: max(peak[g] * CONTACT_MARGIN, peak[g] + 0.05) for g in FINGER_GROUPS}
    print("  contact is only credible above: "
          + ", ".join("%s=%.3f" % (g.replace("R_", ""), out[g]) for g in FINGER_GROUPS))
    return out


def tilted_palm_rotation(tilt_rad):
    """Palm orientation, tilted from flat-down by tilt_rad about the robot's Y axis.

    Measured with the palm flat (tilt 0), the four fingers meet the apple 51, 52, 70 and
    82 degrees up from its equator -- the pinky within 1mm of the exact top pole, and in
    one attempt the ring and pinky at +66mm and +61mm on a 55.5mm apple, i.e. above the
    fruit entirely, touching nothing. Meanwhile the thumb sits 22 degrees BELOW the
    equator. Fingers and thumb are 65-76mm apart in height on a 111mm apple, so they
    never oppose each other, and the apple escapes through the gap.

    That is not a tuning failure. A finger descending onto a sphere can only reach its
    top cap, and with the palm flat the hand's own body sits 8mm above the apple's crown,
    so there is no path around the equator. Tilting the palm lets the fingers come at the
    apple from the side, where they can reach past its widest point.
    """
    if tilt_rad is None:
        return None
    c, s_ = float(np.cos(tilt_rad)), float(np.sin(tilt_rad))
    r_y = np.array([[c, 0.0, s_], [0.0, 1.0, 0.0], [-s_, 0.0, c]])
    return r_y @ PALM_DOWN_ROTATION


def fingertip_height_vs_apple(node, group, apple_local):
    """Fingertip height minus the apple's centre height, in metres.

    Positive means the finger met the apple above its equator. A sphere resting on a
    table can only be reached on its upper half from above, and contacts confined to
    the upper half push it down and out instead of trapping it -- so this distinguishes
    "not pressing hard enough" from "pressing in a place that cannot lift".
    """
    if apple_local is None:
        return None
    try:
        t = node.tf_buffer.lookup_transform(
            'base_footprint', FINGERTIP_LINK[group], rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5))
        return float(t.transform.translation.z) - float(apple_local[2])
    except Exception:
        return None


def fingertip_gaps(node, apple_local, radius):
    """Each fingertip's distance to the apple's SURFACE, in the robot frame.

    Negative means the fingertip is already inside the apple's radius. This says
    whether the apple is centred between the fingers or sitting off to one side --
    which effort readings alone cannot distinguish from a weak grip.
    """
    if apple_local is None or radius is None:
        return {}
    out = {}
    for g, link in FINGERTIP_LINK.items():
        try:
            t = node.tf_buffer.lookup_transform(
                'base_footprint', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p = t.transform.translation
            d = float(np.linalg.norm(
                np.array([p.x, p.y, p.z]) - np.asarray(apple_local)))
            out[g] = d - radius
        except Exception:
            out[g] = None
    return out


def simulation_alive(node, timeout=10.0):
    """Is the sim publishing a robot state we can actually measure against?"""
    start = time.time()
    while time.time() - start < timeout:
        rclpy.spin_once(node, timeout_sec=0.2)
        if node.real_wrist_position() is not None:
            return True
    return False


def thumb_tip_clearance(node):
    """Height of the thumb tip above the table top, in metres (None if TF fails)."""
    try:
        t = node.tf_buffer.lookup_transform(
            'base_footprint', FINGERTIP_LINK["R_Thumb"], rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=1.0))
        return float(t.transform.translation.z) - TABLE_TOP_Z
    except Exception:
        return None


def close_and_measure(node, start_pitch=0.0, apple_local=None, radius=None,
                      thumb_yaw=THUMB_GRASP_YAW):
    """Close the four fingers first, then bring the thumb in last.

    The thumb tip sits at hand-frame x=0.119 while the apple spans x=0.008 to 0.119 --
    so with the thumb held part-closed through the approach it is already occupying
    the space the apple needs, and it has nowhere to go when closing starts. Measured
    repeatedly as the thumb taking 12-21Nm while every other finger read 0.04Nm.

    Wrapping the fingers first and only then closing the thumb is how a hand actually
    grasps: the fingers cage the object, and the thumb closes last to secure it.
    """
    contacted = {g: False for g in FINGER_GROUPS}
    peak = {g: 0.0 for g in FINGER_GROUPS}
    # Height of each fingertip, relative to the apple's own centre, at the moment it
    # first makes contact. This is the question force readings cannot answer: a sphere
    # resting on a table can only be reached on its TOP half from above, and squeezing
    # the top half of a sphere drives it down and out rather than into the hand. If
    # every contact is above the equator the grasp cannot lift, however hard it presses.
    contact_height = {g: None for g in FINGER_GROUPS}
    current = {g: start_pitch for g in FINGER_GROUPS}
    # Thumb held wide open while the fingers work.
    current["R_Thumb"] = 0.0

    def sample():
        for g in FINGER_GROUPS:
            _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
            eff = abs(eff or 0.0)
            peak[g] = max(peak[g], eff)
            limit = CONTACT_THRESHOLD_BY_FINGER.get(g, CONTACT_THRESHOLD)
            if eff > limit and not contacted[g]:
                if apple_local is not None and radius is not None:
                    if not node.fingertips_near_apple(apple_local, radius).get(g):
                        continue
                contacted[g] = True
                contact_height[g] = fingertip_height_vs_apple(node, g, apple_local)
                pos = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))[0]
                if pos is not None:
                    current[g] = pos

    def drive(groups, limit):
        steps = int((limit - start_pitch) / CLOSE_STEP) + 2
        for _ in range(steps):
            moved = False
            for g in groups:
                if not contacted[g] and current[g] < limit:
                    current[g] = min(current[g] + CLOSE_STEP, limit)
                    moved = True
            node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                 thumb_roll=THUMB_GRASP_ROLL)
            for _ in range(CHECKS_PER_STEP):
                rclpy.spin_once(node, timeout_sec=0.08)
                sample()
            if not moved or all(contacted[g] for g in groups):
                break

    fingers = [g for g in FINGER_GROUPS if g != "R_Thumb"]

    # The thumb comes in FIRST, but only after the approach is over -- it stays wide
    # open while the arm moves in, then closes just far enough to touch.
    #
    # Closing it LAST was measured to be too late. The four fingers are side by side,
    # so they all push the apple the same way: with nothing opposing them the apple is
    # shoved out of the hand before the thumb ever arrives. Attempt 4 of the previous
    # run showed it exactly -- fingers pressing at 1.5, 1.5 and 3.1Nm, the apple sliding
    # 0.021m, and every finger finishing at 0.04Nm, the idle noise floor, holding air.
    # The only attempt that lifted the apple at all was the one where the thumb happened
    # to be jammed into it at 11.5Nm, which is a wedge, not a grasp.
    #
    # Closing the thumb to first contact before the fingers move gives them something to
    # close against. drive() stops a finger the moment it feels the apple, so this is a
    # light backstop, not a squeeze.
    print("  thumb closing to first contact, to give the fingers a backstop...")
    drive(["R_Thumb"], MAX_PITCH_CEILING)
    # First contact alone is far too light to oppose anything: measured at 0.284Nm and
    # 0.103Nm while the fingers then pressed at 1.5-5.9Nm, so the "backstop" simply gave
    # way. Close the thumb a further fixed amount, capped by force, so it is genuinely
    # bearing on the apple before the fingers arrive.
    if contacted["R_Thumb"]:
        pushed = 0.0
        while pushed < THUMB_PRELOAD_EXTRA:
            _, _, eff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
            if abs(eff or 0.0) >= THUMB_PRELOAD_FORCE:
                break
            # Checking the force only once per step let it overshoot enormously: the cap
            # is 1.2Nm and the thumb was measured reaching 16.79Nm, because contact force
            # jumps from ~0.1Nm to tens of Nm between two samples with nothing in
            # between. Take a much smaller step and re-read within it.
            pushed += THUMB_PRELOAD_STEP
            current["R_Thumb"] = min(current["R_Thumb"] + THUMB_PRELOAD_STEP,
                                     MAX_PITCH_CEILING)
            node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                 thumb_roll=THUMB_GRASP_ROLL)
            stop = False
            for _ in range(CHECKS_PER_STEP):
                rclpy.spin_once(node, timeout_sec=0.08)
                _, _, e = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
                if abs(e or 0.0) >= THUMB_PRELOAD_FORCE:
                    stop = True
                    break
            if stop:
                break
        _, _, teff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
        peak["R_Thumb"] = max(peak["R_Thumb"], abs(teff or 0.0))
        print(f"  thumb preloaded to {abs(teff or 0.0):.2f}Nm -- now closing the four "
              f"fingers against it")
    else:
        print("  thumb found nothing -- the fingers will have no backstop")
    drive(fingers, MAX_PITCH_CEILING)
    print("  fingers wrapped: %d/4" % sum(1 for g in fingers if contacted[g]))

    # Squeeze phase: close past first contact so the fingers actually hold.
    squeezed = 0.0
    while squeezed < SQUEEZE_EXTRA:
        squeezed += SQUEEZE_STEP
        for g in FINGER_GROUPS:
            if not contacted[g]:
                continue
            _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
            if abs(eff or 0.0) < SQUEEZE_FORCE_CAP:
                current[g] = min(current[g] + SQUEEZE_STEP, MAX_PITCH_CEILING)
        node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                             thumb_roll=THUMB_GRASP_ROLL)
        for _ in range(CHECKS_PER_STEP):
            rclpy.spin_once(node, timeout_sec=0.08)
            for g in FINGER_GROUPS:
                _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
                peak[g] = max(peak[g], abs(eff or 0.0))

    holding = {g: abs(node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))[2] or 0.0)
               for g in FINGER_GROUPS}
    n_sq = sum(1 for g in FINGER_GROUPS if contacted[g])
    print(f"  after squeeze ({n_sq}/5 fingers had contact to squeeze), holding force: "
          + ", ".join("%s=%.2f" % (g, holding[g]) for g in FINGER_GROUPS))

    known = {g: h for g, h in contact_height.items() if h is not None}
    if known:
        print("  where each finger met the apple (+ is above its equator, - is below):")
        print("    " + ", ".join(f"{g.replace('R_', '')}={h * 1000:+.0f}mm"
                                 for g, h in known.items()))
        below = [g for g, h in known.items() if h < 0.0]
        print(f"    {len(below)}/{len(known)} contacts are BELOW the equator"
              + ("" if below else
                 "  <-- every contact is on the TOP half: squeezing drives the apple "
                 "down and out, and nothing holds it up during the lift"))
    return contacted, peak


def apple_xyz(node):
    if node.target_pose is None:
        return None
    p = node.target_pose.position
    return np.array([p.x, p.y, p.z])


def attempt(node, target_name, palm_tilt, preshape, lateral, palm_offset):
    label = (f"tilt {np.degrees(palm_tilt):.0f}deg, "
             f"lateral {lateral * 1000:+.0f}mm")
    print(f"\n{'=' * 72}\n{label}, pre-shape {preshape:.2f}\n{'=' * 72}")
    rot = tilted_palm_rotation(palm_tilt)

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    node.robot_x, node.robot_y, node.robot_yaw = wx, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW
    node.reset_everything()
    if not node.reset_target_apple_position(target_name):
        # Two attempts in one run began against a still-flying apple: one ended with the
        # arm unable to reach it at all (fingertips 495mm away, IK error 0.338m) and the
        # other threw it 3.4m off the table. Neither measured anything about the grasp.
        print("  SKIPPED: the apple would not stop moving, so this attempt would "
              "measure nothing.")
        return {"ok": False, "palm_offset": palm_offset, "lateral": lateral, "palm_tilt": palm_tilt,
                "unsettled": True}
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)
    before = apple_xyz(node)

    # Aim at where the apple ACTUALLY is, not its nominal home. It can roll after the
    # reset, and aiming at the home position then puts the hand where the apple used to
    # be -- measured directly: runs with the palm correctly parallel and the wrist
    # within 29mm of target still got 0/5 contacts because the apple had moved 1-2m.
    live = apple_xyz(node)
    if live is not None:
        drift_from_home = float(np.linalg.norm(live[:2] - np.array([wx, wy])))
        if drift_from_home > 0.005:
            print(f"  apple is {drift_from_home:.3f}m from its home position -- "
                  f"aiming at where it actually is")
        x, y = world_to_local(live[0], live[1], node.robot_x, node.robot_y, node.robot_yaw)
        apple_local = np.array([x, y, live[2]])
    else:
        x, y = world_to_local(wx, wy, node.robot_x, node.robot_y, node.robot_yaw)
        apple_local = np.array([x, y, apple_home_z(target_name)])

    # Pass 1: a rough solve just to learn which way the wrist ends up facing, since the
    # pocket offset is expressed in the hand's own frame and has to be rotated into the
    # robot frame before it means anything.
    rough = solve_ik(node.chain, list(apple_local + np.array([0, 0, 0.164])),
                     target_rotation=rot)
    if rough is None:
        print("  rough solve UNREACHABLE")
        return {"preshape": preshape, "palm_tilt": palm_tilt,
            "thumb_yaw": thumb_yaw, "palm_offset": palm_offset, "lateral": lateral, "ok": False}
    wrist_rot = hand_fk(node.chain, rough[1])[:3, :3]

    # Where in the hand the apple should sit. The first two components stay as
    # measured; AIM_DEPTH replaces the third -- how far out along the fingers -- so the
    # sweep can compare holding the apple against the palm (~0.06) against holding it
    # at the fingertips (0.128, what we had been doing).
    # palm_offset is the distance from the wrist to the apple ALONG THE PALM NORMAL --
    # i.e. how far the hand backs off before closing. At 30deg tilt the fingertips were
    # measured starting at +5, +2, -3 and -6mm from the apple surface: a mean of -0.5mm,
    # so the hand is already inside the fruit before the close begins and there is
    # nowhere left to travel. Backing the hand off along this axis gives the fingers
    # room to sweep in and actually wrap.
    # The lateral term decides where the apple sits ACROSS the hand. It has always been
    # _CLOSED_CENTROID_MEASURED[1], the mean of all FIVE fingertips -- but the thumb sits
    # off to one side, so including it drags the aim off the four fingers' own centre.
    # Measured consequence: index and middle start 14-16mm further from the apple than
    # the pinky and peak at 0.023Nm, which is BELOW the 0.046Nm free-air noise. They
    # never touch it. They sweep past beside it, which is the gap the apple escapes
    # through.
    aim_point = np.array([palm_offset,
                          _CLOSED_CENTROID_MEASURED[1] + lateral,
                          AIM_DEPTH])
    offset_local = wrist_rot @ aim_point
    wrist_target = apple_local - offset_local

    print(f"  apple at local ({apple_local[0]:.3f}, {apple_local[1]:.3f}, {apple_local[2]:.3f})")
    print(f"  pocket offset rotated -> ({offset_local[0]:.3f}, {offset_local[1]:.3f}, "
          f"{offset_local[2]:.3f})")
    print(f"  wrist target ({wrist_target[0]:.3f}, {wrist_target[1]:.3f}, "
          f"{wrist_target[2]:.3f})")

    # Approach from BESIDE the apple at grasp height, not from above it.
    #
    # Measured directly: essentially all the apple's movement happens during the
    # descent, before a single finger moves -- 0.034m, 1.260m, 0.770m, 0.672m during
    # the descent versus 0.000-0.113m during the closing itself. With the palm facing
    # down the fingers stick out horizontally 0.164m while the wrist sits 0.083m behind
    # the apple, so they hang directly over it and lowering the hand rakes them
    # straight through it.
    #
    # Backing off along the hand's own forward axis and then moving in horizontally
    # lets the fingers arrive alongside the apple instead of on top of it.
    back_off = wrist_rot @ np.array([0.0, 0.0, -APPROACH_BACKOFF])
    approach_pos = wrist_target + back_off + np.array([0.0, 0.0, 0.02])
    approach = solve_ik(node.chain, list(approach_pos), target_rotation=rot)
    grasp = solve_ik(node.chain, list(wrist_target), target_rotation=rot)
    if approach is None or grasp is None:
        print("  UNREACHABLE")
        return {"preshape": preshape, "palm_tilt": palm_tilt,
            "thumb_yaw": thumb_yaw, "palm_offset": palm_offset, "lateral": lateral, "ok": False}

    # Pre-shape BEFORE descending, so the fingertips are retracted on the way down and
    # the wrist can actually reach the pocket height instead of the fingers grounding
    # out on the table first.
    # Thumb stays fully open during the approach so it is not occupying the space the
    # apple has to enter.
    pre = {g: preshape for g in FINGER_GROUPS}
    pre["R_Thumb"] = 0.0
    node.command_fingers(pre, 1.5, thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    settle(node, 6.0, joints=[f"{g}_Pitch" for g in FINGER_GROUPS], thresh=0.02)
    print(f"  approaching from ({approach_pos[0]:.3f}, {approach_pos[1]:.3f}, "
          f"{approach_pos[2]:.3f}) -- {APPROACH_BACKOFF:.2f}m back, then moving in "
          f"sideways rather than descending onto the apple")
    node.send_arm_trajectory(approach[0], 3.5)
    settle(node, 12.0)
    node.send_arm_trajectory(grasp[0], 3.0)
    settle(node, 20.0)

    # Close the loop on the wrist, the same way full_layer_grasp.py does. A single
    # trajectory lands 0.043-0.395m off depending on the run, and the apple's radius is
    # only 0.055m -- so a one-shot move puts the hand roughly an apple-width away and
    # the fingers close beside it. Measuring the real error and re-solving for a
    # target offset by it converges to ~0.015m in the main pipeline.
    corrected = list(wrist_target)
    for correction_i in range(CORRECTION_ITERS):
        real_now = node.real_wrist_position()
        if real_now is None:
            break
        err_now = float(np.linalg.norm(np.array(real_now) - wrist_target))
        if err_now < 0.02:
            break
        # Apply only part of the measured error. Feeding the FULL error back made the
        # loop overshoot and bounce rather than settle -- measured sequences like
        # 0.062 -> 0.034 -> 0.036 and 0.052 -> 0.052 -> 0.049 that never converge.
        # That matters enormously here: grasps succeed at ~0.005m error (5/5 fingers)
        # and fail completely at 0.024-0.049m (0/5), so the last 2cm decides everything.
        error_vec = (wrist_target - np.array(real_now)) * CORRECTION_GAIN
        corrected = list(np.array(corrected) + error_vec)
        again = solve_ik(node.chain, corrected, target_rotation=rot)
        if again is None:
            print(f"  correction {correction_i + 1}: corrected target unreachable, "
                  f"keeping {err_now:.3f}m error")
            break
        print(f"  correction {correction_i + 1}: err {err_now:.3f}m -> re-solving")
        node.send_arm_trajectory(again[0], 2.0)
        settle(node, 15.0)

    real = node.real_wrist_position()
    if real is None:
        # No wrist position means the TF tree is broken -- the usual message is
        # "base_footprint and dexhand_base_link are not part of the same tree", which
        # means joint data has stopped arriving and the simulation is dead or paused.
        # Closing the hand at that point measures nothing; a whole run was once spent
        # grasping blind because this fell through instead of stopping.
        print("  ABORT: cannot read the wrist position -- the TF tree is broken, which "
              "means the simulation has stopped publishing joint states.")
        print("  Restart the simulation (Terminal 1) before running this again.")
        return {"ok": False, "palm_offset": palm_offset, "lateral": lateral, "palm_tilt": palm_tilt,
                "dead_sim": True}
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

    # Direct check that the thumb is not stabbing the table. Arithmetic said it was:
    # thumb protrudes 0.1190m below the palm, wrist at 0.521, palm facing down puts the
    # tip at 0.402 against a 0.400 table. That matches the measured effort exactly --
    # 18.4/69.8/16.5Nm on the thumb while every other finger sat at its 1.50 cap, which
    # is a joint pressed into something solid, not a motor pushing an apple. Measure it
    # rather than trusting the arithmetic.
    thumb_clear = thumb_tip_clearance(node)
    if thumb_clear is not None:
        print(f"  thumb tip is {thumb_clear * 1000:+.0f}mm above the table"
              + ("" if thumb_clear > 0.010
                 else "  <-- GROUNDED OUT: it is pushing the table, not the apple"))

    # Did the thumb actually HOLD the yaw it was told to? In one attempt the commanded
    # -0.50 behaved geometrically like 0.00 (implied protrusion 0.107 against 0.080 and
    # 0.087 on the attempts that worked), which would mean the joint is being pushed off
    # its command -- its effort cap is 1.5Nm like every other finger joint, so it can be
    # back-driven by contact.
    real_yaw = node.latest_joint_state.get('R_Thumb_Yaw', (None, None, None))[0]
    if real_yaw is not None:
        drift = abs(real_yaw - thumb_yaw)
        print(f"  thumb yaw commanded {thumb_yaw:+.2f}, actually at {real_yaw:+.2f}"
              + ("" if drift < 0.05 else f"  <-- OFF BY {drift:.2f} rad, being pushed back"))

    # Which fingers are actually within reach of the apple BEFORE closing starts?
    # The run that prompted this had 5/5 "contacts" and yet finished with index, middle
    # and ring all reading 0.04-0.05Nm -- the idle noise floor, i.e. touching nothing.
    # Only one finger per attempt was ever loaded. That is the signature of the apple
    # sitting off to one side of the hand rather than in the middle of the closing arc,
    # so measure each fingertip's own gap instead of trusting the centroid.
    gaps = fingertip_gaps(node, apple_local,
                          APPLE_RADIUS.get(target_name, 0.0555))
    if gaps:
        print("  fingertip gap to the apple surface before closing:")
        print("    " + ", ".join(
            f"{g.replace('R_', '')}={v * 1000:+.0f}mm" if v is not None else f"{g}=?"
            for g, v in gaps.items()))
        four = [v for g, v in gaps.items() if g != "R_Thumb" and v is not None]
        if len(four) == 4:
            print(f"    spread across the four fingers: {(max(four) - min(four)) * 1000:.0f}mm"
                  + ("  <-- even, the apple is centred"
                     if (max(four) - min(four)) < 0.008
                     else "  <-- uneven, the apple is off to one side"))
        reach = [g for g, v in gaps.items() if v is not None and v < 0.05]
    cam = {g: v for g, v in camera_clearances(node).items() if v is not None}
    if cam:
        print("  fingertip clearance to the gripper camera's collision box:")
        print("    " + ", ".join(f"{g.replace('R_', '')}={v * 1000:+.0f}mm"
                                 for g, v in cam.items()))
        blocked = [g for g, v in cam.items() if v < 0.005]
        if blocked:
            print("    BLOCKED BY THE CAMERA: "
                  + ", ".join(g.replace("R_", "") for g in blocked)
                  + " -- the camera body is in the way of the grasp")
        print(f"    {len(reach)}/5 fingers start within 50mm of the apple"
              + ("" if len(reach) >= 4
                 else "  <-- the apple is not centred in the hand's closing arc"))

    # Split the measurement: how far did the apple move during the DESCENT, before a
    # single finger moved? The fingers read 0.044Nm (baseline noise) while the apple
    # moved up to 0.8m, so something that is not a finger joint is hitting it -- most
    # likely the hand's own body on the way down. Attributing that to "closing" hid it.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    after_descent = apple_xyz(node)
    if before is not None and after_descent is not None:
        moved_descent = float(np.linalg.norm(after_descent - before))
        print(f"  apple moved {moved_descent:.3f}m DURING THE DESCENT "
              f"(before any finger moved)")
        if moved_descent > 0.02:
            print("    -> the apple is being knocked by the arm/hand body, not the "
                  "fingers; closing never gets a chance")

    contacted, peak = close_and_measure(
        node, start_pitch=preshape, apple_local=apple_local,
        radius=APPLE_RADIUS.get(target_name, 0.0555), thumb_yaw=THUMB_GRASP_YAW)
    n = sum(contacted.values())
    print(f"  fingers contacted: {n}/5")
    print("  peak efforts: " + ", ".join("%s=%.3f" % (g, peak[g]) for g in FINGER_GROUPS))

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    after_close = apple_xyz(node)
    moved = (float(np.linalg.norm(after_close - before))
             if before is not None and after_close is not None else None)
    if moved is not None:
        during_closing = (float(np.linalg.norm(after_close - after_descent))
                          if after_descent is not None else None)
        extra = (f" (of which {during_closing:.3f}m during closing itself)"
                 if during_closing is not None else "")
        print(f"  apple moved {moved:.3f}m total{extra}")

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
            # apple_xyz() is in WORLD coordinates but real_wrist_position() is in the
            # robot's own base_footprint frame -- comparing them directly was measuring
            # nothing. It reported 0.447m for an apple that was really 0.108m from the
            # wrist, so "not held" could not be trusted. Convert the apple into the
            # robot frame first.
            ax, ay = world_to_local(after_lift[0], after_lift[1],
                                    node.robot_x, node.robot_y, node.robot_yaw)
            after_lift_local = np.array([ax, ay, after_lift[2]])
            # Height alone is NOT enough: an apple flung across the room can land
            # higher than it started and score as a success. Measured directly -- one
            # attempt threw the apple 79m, ended +0.400m up, and was reported HELD.
            # A real hold means the apple is still in the hand.
            near = None
            if wrist_after is not None:
                near = float(np.linalg.norm(after_lift_local - np.array(wrist_after)))
            held = (lifted > LIFT_HEIGHT * 0.5 and near is not None
                    and near < HOLD_DISTANCE)
            near_s = (f"{near:.3f}m from wrist (robot frame)" if near is not None
                      else "wrist unknown")
            print(f"  apple height change after lift: {lifted:+.3f}m, {near_s} "
                  f"({'HELD' if held else 'not held'})")
            if not held:
                lifted = None

    return {"preshape": preshape, "palm_tilt": palm_tilt,
            "thumb_yaw": thumb_yaw, "palm_offset": palm_offset, "lateral": lateral,
            "ok": True, "contacts": n, "peak": peak,
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

    global CONTACT_THRESHOLD_BY_FINGER
    measured = calibrate_contact_threshold(node)
    if measured is not None:
        CONTACT_THRESHOLD_BY_FINGER = measured

    if not simulation_alive(node):
        print()
        print("The simulation is not publishing a usable robot state: the TF tree "
              "from base_footprint to the hand is missing or broken.")
        print("Nothing measured here would mean anything, so this is stopping now.")
        print("Restart Terminal 1 (./start_everything.sh gui), wait for the arm to "
              "appear AND the controller messages to stop, then try again.")
        node.destroy_node()
        rclpy.shutdown()
        return

    results = []
    for pt, ps, lat, po in CASES:
        r = attempt(node, target_name, pt, ps, lat, po)
        results.append(r)
        if r.get("dead_sim"):
            print()
            print("Stopping the remaining attempts -- the simulation died mid-run.")
            break

    print(f"\n{'=' * 72}\nSUMMARY\n{'=' * 72}")
    print(f"{'case':>22} {'contacts':>9} {'max_effort':>11} {'apple_moved':>12} {'lifted':>9}")
    for r in results:
        if not r.get("ok"):
            nm = f"lateral={r['lateral'] * 1000:+.0f}mm"
            why = ("SKIPPED" if r.get("unsettled")
                   else "DEAD SIM" if r.get("dead_sim") else "UNREACHABLE")
            print(f"{nm:>22} {why:>9}")
            continue
        mx = max(r["peak"].values())
        moved = f"{r['moved']:.3f}m" if r["moved"] is not None else "n/a"
        lifted = f"{r['lifted']:+.3f}m" if r["lifted"] is not None else "n/a"
        nm = f"lateral={r['lateral'] * 1000:+.0f}mm"
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
