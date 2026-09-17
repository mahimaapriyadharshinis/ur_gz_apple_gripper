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
    FINGERTIP_LINK, TABLE_TOP_Z, FINGER_SECONDARY_JOINTS,
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

# Settled by measurement.
#
# LATERAL +15mm centres the apple across the four fingers: their distances to it come
# out within 2mm of each other, against 11mm at the old aim, 24mm at -15mm and 11mm at
# +30mm. That matters because at the old aim index and middle peaked at 0.023Nm --
# BELOW the free-air noise floor, i.e. never touching the apple at all -- and at +15mm
# they peak at 1.500Nm, the joint cap. They went from missing the fruit to gripping it,
# and dropped below its equator (index -1mm, middle -6mm) for the first time.
#
# Both centred cases lifted the apple: +0.043m at +15mm and +0.048m at +30mm.
#
# Tilt 45deg / back-off 0.090 unchanged -- the only pairing that puts fingers on and
# below the apple's widest circle.
PALM_TILT = np.radians(45.0)
PALM_OFFSET = 0.0900
TESTED_PALM_OFFSET = PALM_OFFSET
LATERAL = 0.015

# THE FIRST COMPLETE GRASP IN THIS PROJECT, and the configuration that produced it.
#
#   palm tilt 45deg, back-off 0.090, lateral +15mm, pre-shape 0.40, squeeze 0.12
#   -> 5/5 fingers, contacts at index -16mm, middle -19mm, ring -9mm, pinky -5mm and
#      thumb +14mm, so FOUR of five below the apple's equator with the thumb over the
#      top. Apple moved 0.044m in total and then rose 0.085m through the lift and
#      stayed 0.123m from the wrist: HELD.
#
# Every false-success guard added over the past week had to pass for that verdict --
# height above half the commanded lift, apple still within a hand-width of the wrist,
# both compared in the same coordinate frame, contacts confirmed by fingertip TF
# position rather than effort alone, and the apple settled before the attempt began.
#
# The squeeze theory that motivated the last sweep was wrong, and wrong in the opposite
# direction: squeezing moved the apple only 0.001-0.010m at every setting, and MORE
# squeeze gave monotonically more contacts, a deeper wrap and the lift. 0.00 got 0/5,
# 0.02 got 1/5, 0.05 got 3/5, 0.12 got 5/5 and the pick.
#
# One success out of four attempts is not yet a reliable grasp. These four cases are
# identical so the next run measures how often it actually works.
SQUEEZE = 0.12
# Pre-shape 0.25 and grasp lowered 12mm (GRASP_DROP), settled 15 Sep. The old 0.4 / 8mm
# picked apple_10 (12cm) on 14 Sep but 0 of 24 on 15 Sep: the half-closed fingers pushed
# the big apple 13mm sideways while closing and index/middle lost it. A sweep of height x
# opening gave 0.25 / 12mm: apple_10 4 of 4 (thumb 116-117deg, apple slid 1mm while
# closing, +14.9-15.0cm); apple_01 3 of 3 and apple_06 3 of 3 unchanged.
PRESHAPE = 0.25
GRASP_DROP = 0.012

# Thumb roll for the current attempt. R_Thumb_Roll (+/-0.349 rad) has been held at 0.0
# for this entire project; set per attempt from CASES.
THUMB_ROLL = THUMB_GRASP_ROLL

# One attempt's key numbers, filled in as it runs and printed as a table at the end, so a
# run can be read at a glance instead of reconstructed from hundreds of log lines.
REC = {}

# The finger command in force when closing finished, so the thumb can be eased off before
# lifting without disturbing the fingers.
LAST_FINGER_CMD = {}

CASES = [
    # (tilt, preshape, lateral, method, lowered, thumb cap, empty control, ease thumb before
    #  lift, wait out the grasp move)
    #
    # Waiting out the grasp move, A/B tested (91aaa7a): 4 of 4 picked. With the wait
    # (attempts 1 and 3) the arm landed 0mm from the grasp pose, needed no corrections,
    # disturbed the apple 8mm, lifted it +15.2 and +15.1cm, and wrist_1 peaked at 12Nm.
    # Without it (2 and 4) the first move was measured 40-50mm off mid-move, one
    # correction knocked the apple 10-13mm, and wrist_1 hit its 28Nm limit and
    # shoulder_lift 150Nm at the very start of the lift -- so those limit readings were
    # the arm still driving through an unfinished correction, not the apple's load.
    # Attempt 1, which had failed in every previous run, picked.
    #
    # Now on for every attempt, to confirm it holds over four more.
    (PALM_TILT, PRESHAPE, LATERAL, "pick", GRASP_DROP, True, False, False, True),
    (PALM_TILT, PRESHAPE, LATERAL, "pick", GRASP_DROP, True, False, False, True),
    (PALM_TILT, PRESHAPE, LATERAL, "pick", GRASP_DROP, True, False, False, True),
    (PALM_TILT, PRESHAPE, LATERAL, "pick", GRASP_DROP, True, False, False, True),
]

REST_POSE = [0.0, -1.2, 1.5, -1.9, 0.0, 0.0]

# How far to raise the wrist after closing. Contact proves the fingers reached the
# apple; only lifting proves the grip actually holds it.
LIFT_HEIGHT = 0.15

# How long the lift is commanded to take. 3.0s is what every result so far used.
# apple_10 fails in the first 2-5cm of the lift with all five fingers still gripping
# after the squeeze (1 of 4 and 2 of 5 on 16-17 Sep), so the start of the lift -- when
# the apple's weight transfers from the table to the fingers -- is the suspect.
LIFT_SECONDS = 3.0

# Re-grip during the lift. The fingers hold a POSITION, not a force: once the apple
# settles a millimetre the contact is lost and nothing re-tightens, which is what the
# traces show (1.50Nm on four fingers, then 0.00Nm a quarter of a second later). With
# REGRIP on, a finger whose load has dropped is commanded slightly further closed,
# while the apple is still with the hand.
REGRIP = False
REGRIP_STEP = 0.010
REGRIP_MAX = 0.060
REGRIP_MIN_GAP_S = 0.05      # simulated seconds between re-grips

# How far back along the hand's forward axis to start, so the fingers move in
# beside the apple rather than being lowered through it.
APPROACH_BACKOFF = 0.16
# Simulated seconds the approach move runs before the grasp move replaces it (see where
# it is used). 0.6s sits inside the range, about 0.4-1.0s, under which apple_10 picked.
APPROACH_SIM_S = 0.6

# Where the wrist correction runs: this far back from the grasp pose along the hand's
# forward axis. It was 40mm, on the assumption that was "clear of the apple". It was not.
# The first move to this pose lands 58-78mm off target (78, 68, 58, 61mm in one run;
# 65-118mm in earlier ones) -- solve_ik accepts answers up to 25mm off and the arm sags on
# top -- so 40mm of clearance was smaller than the error being corrected. 120mm is larger
# than the worst first-move error ever logged, so nothing done here can reach the apple.
PREGRASP_STANDOFF = 0.120

# The final approach covers PREGRASP_STANDOFF in steps this many, i.e. 10mm each, with the
# real wrist position checked after every one.
APPROACH_STEPS = 12

# The hand is not allowed to start towards the apple until its MEASURED position at the
# pre-grasp pose is this close. In the run that prompted this, every attempt drove in
# while still 58-78mm out of place, and every knock happened during that drive.
APPROACH_GATE = 0.012

# During the final approach, the most the real wrist may stray from the straight line
# before the approach is abandoned rather than continued into the apple.
# Raised from 15mm. Two attempts converged at the pre-grasp pose to 3-4mm and were then
# stopped at approach step 3 or 4 for straying 16-19mm -- just over the old limit -- so
# the hand never reached the apple at all. A limit this arm cannot meet does not keep
# the apple safe; it just guarantees no grasp. The knock check still stops the approach
# the moment the apple actually moves.
TRACK_LIMIT = 0.025

# Largest single-joint change allowed for one 10mm approach step or the lift. A 10mm step
# needs a few hundredths of a radian, so anything past this is IK switching arm
# configuration, which swings the hand through space.
FINE_MOVE_MAX_JUMP = 0.25

# Largest single-joint change allowed for a correction made at the pre-grasp pose. This
# was wrongly held to the fine limit (0.35 rad), and a 58-78mm correction legitimately
# needs ~0.4 rad -- so every correction in the run was refused as a "configuration flip",
# the hand was never corrected, and it drove in 6-8cm off target. A real flip is well over
# a radian on some joint; this only catches those.
COARSE_MOVE_MAX_JUMP = 1.0

# Fine positioning uses the arm's Jacobian, not repeated IK solves (see servo_to).
SERVO_MAX_LINEAR = 0.010   # most the wrist is asked to move in one servo step
SERVO_MAX_JOINT = 0.06     # most any joint may turn in one servo step, rad
SERVO_DAMPING = 0.02       # damped least squares, keeps steps sane near singularities
SERVO_MOVE_TIME = 0.8      # seconds per servo step
SERVO_TOL = 0.004          # how close the pre-grasp correction aims to get
SERVO_MAX_ITERS = 25       # pre-grasp correction budget; a 75mm error needs ~8 steps
APPROACH_TOL = 0.004       # per-waypoint tolerance on the final approach
APPROACH_SERVO_ITERS = 6   # servo steps allowed per 10mm approach waypoint
SERVO_ROT_TOL_DEG = 3.0    # palm orientation must also be this close

# The approach that produced the pick, with two limits adjusted from the side-by-side run.
#
# Tolerance 20mm -> 12mm. One attempt landed 14mm off, which the 20mm limit accepted with
# no correction at all -- the hand sat 10mm to the side, the four fingers started 8, 13,
# 19 and 25mm from the apple, and only 2/5 touched it. The attempt that made one
# correction, to 11mm, got 5/5.
#
# Corrections 9 -> 3. Across thirteen earlier attempts, those making 0-1 correction moves
# next to the apple disturbed it 2-16mm (and include the one pick); those making 6-9
# disturbed it 41-340mm and all failed. Three allows the one or two a 12mm limit needs.
PICK_WRIST_TOLERANCE = 0.012
PICK_CORRECTION_ITERS = 3

# Apple displacement between two readings that counts as the arm having hit it. A
# settled apple drifts under 2mm.
KNOCK_THRESHOLD = 0.010

# If every fingertip is further than this from the apple's surface when it moves, the hand
# cannot have moved it.
HAND_CLEAR_OF_APPLE = 0.030

# Below this angle between thumb and fingers (about the apple's centre) the grasp has an
# open side, so squeezing drives the apple out rather than trapping it.
OPPOSITION_MIN_DEG = 120.0

# How close the apple must still be to the wrist afterwards to count as held.
# Roughly the hand's own size -- further than this and it is not in the hand.
HOLD_DISTANCE = 0.20

# Fraction of the measured wrist error to correct per iteration. Full
# correction overshoots and oscillates; damping it converges.
# How close the wrist has to get before the grasp starts. This was 0.020m, which is
# four times larger than the difference between the one attempt that picked the apple up
# and four identical attempts that failed. Those four landed the wrist 4-6mm lower, which
# flipped index, middle and ring from just OUTSIDE the apple (+3 to +6mm) to just INSIDE
# it (-3 to 0mm) -- and a finger that starts inside the fruit cannot wrap around it, only
# shove it. The apple moved 0.110-0.211m during closing in those attempts against 0.029m
# in the success. A 20mm tolerance on a grasp decided by 6mm was never good enough.
WRIST_TOLERANCE = 0.008

CORRECTION_GAIN = 0.6
# Attempt 1 of the thumb-last run converged 0.233 -> 0.150 -> 0.113 -> 0.075 -> 0.058
# and then simply ran out of iterations, starting the grasp with the arm still 0.031m
# out and still moving -- it knocked the apple 0.219m during the descent. It was
# improving the whole way, so give it room to finish. Attempts that start close still
# converge in one pass and cost nothing.
CORRECTION_ITERS = 9


def current_sim_time(node):
    for _ in range(3):
        rclpy.spin_once(node, timeout_sec=0.05)
    return getattr(node, "joint_stamp", None)


def wait_until_sim(node, sim_start, sim_seconds, wall_cap=300.0):
    """Spin until sim_seconds of simulated time have passed since sim_start. Returns the
    simulated seconds that had passed, or None without a simulation clock."""
    if sim_start is None:
        return None
    t0 = time.time()
    waited = 0.0
    while time.time() - t0 < wall_cap:
        rclpy.spin_once(node, timeout_sec=0.1)
        now = getattr(node, "joint_stamp", None)
        if now is not None:
            waited = now - sim_start
            if waited >= sim_seconds:
                break
    return waited


def settle(node, seconds, joints=ARM_JOINTS, thresh=0.05, min_wait=3.0):
    start = time.time()
    while time.time() - start < seconds:
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.1)
        vels = [abs(node.latest_joint_state.get(j, (0, 0, 0))[1] or 0.0) for j in joints]
        if vels and max(vels) < thresh and (time.time() - start) > min_wait:
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
SQUEEZE_EXTRA = 0.12  # see CASES: more squeeze measured strictly better, not worse
SQUEEZE_STEP = 0.006
SQUEEZE_FORCE_CAP = 3.0

# How far past first contact to drive the thumb before the fingers close, and the force
# at which to stop. First contact alone measured 0.103-0.284Nm, which the fingers then
# overwhelmed at 1.5-5.9Nm.
THUMB_PRELOAD_EXTRA = 0.10
THUMB_PRELOAD_FORCE = 1.2
THUMB_PRELOAD_STEP = 0.002

# Enforce the preload cap by backing the thumb off, when CAP_THUMB is on for the attempt.
# The preload loop stops advancing once it reads 1.2Nm, but the thumb is position-
# controlled and its command is left where the spike was read, so the push it keeps up
# afterwards was measured at 5.5-13.3Nm -- up to eleven times the cap.
THUMB_BACKOFF_STEP = 0.004
THUMB_BACKOFF_MAX = 0.10
CAP_THUMB = False

# How the thumb preload decides it is pressing (attempt's thumb_mode). "spike": stop on
# the first effort sample over the cap, a few messages after each step -- the tested
# behaviour. apple_10's 4 picks on 14 Sep backed the thumb off 0.012-0.076 rad after it
# (thumb 112-114deg around the apple from the fingers); all its 15 Sep failures backed off
# 0-0.004 rad (58-109deg): how deep the thumb got depended on when a spike was sampled.
# "steady": each step waits THUMB_STEP_SIM_S of simulated time and the thumb counts as
# pressing only when the MEDIAN effort over that wait passes the cap, so the depth no
# longer depends on sampling luck.
THUMB_MODE = "spike"
THUMB_STEP_SIM_S = 0.10

# How long each closing step (finger, thumb preload, back-off, squeeze) waits before the
# next command. None: CHECKS_PER_STEP incoming messages of ANY subscription, the tested
# behaviour -- but the node also receives camera images, whose rate follows PC load, so
# the fingers close at a different speed in simulated time from session to session
# (closing + squeeze measured 0.42-1.77s simulated across identical apple_10 attempts on
# 15 Sep). A number: wait that many seconds of SIMULATED time per step.
CLOSE_STEP_SIM_S = None

# A finger counts as still gripping only above this. The idle floor measured 0.01-0.09Nm
# across all three joints; genuinely loaded fingers read 0.87-1.50Nm. The contact
# threshold (~0.07Nm) sits inside that idle band, so it overstated gripping.
GRIP_HOLD_NM = 0.30

# Grip check before lifting (see close_and_measure). Every apple_10 failure on 15 Sep
# started its lift with at most 1 finger and a thumb at 0-1.5Nm pressing; its picks had
# 3-4 fingers and a thumb at 3-8Nm. Light apples picked with 1 finger + thumb 1.5Nm, so
# the check asks for 2 fingers and a thumb at 1.5Nm, and only ever ADDS squeeze -- which
# the squeeze sweep measured as strictly better, moving the apple 1-10mm at most.
# Re-centring the hand on the apple's measured position before closing (attempt's
# recentre option): only when the apple has been pushed more than RECENTRE_MIN_M, as a
# RECENTRE_MOVE_S-second move in simulated time.
RECENTRE_MIN_M = 0.005
RECENTRE_MOVE_S = 2.0

GRIP_MIN_FINGERS = 2
GRIP_MIN_THUMB_NM = 1.5
GRIP_SETTLE_SIM_S = 0.3     # simulated seconds for the fingers to follow each command
GRIP_EXTRA_STEP = 0.02      # rad added per check to digits that are not pressing
# Extra squeeze switched OFF (0.0): the check still measures and reports the grip, but no
# longer adds squeeze. Tested on apple_10 it never produced a firm grip (0/4 fingers
# pressing after 0.10 rad extra, twice) and the extra closing pushed the apple 28-30mm and
# swung the thumb to 58-60 degrees from the fingers (95-105 without it). apple_02 did not
# need it (4/4 fingers and 9.4Nm thumb straight after the normal squeeze).
GRIP_EXTRA_MAX = 0.0        # most extra squeeze the check may add

# During the lift, how often to sample, for how long, and how far the apple must fall
# behind the hand before it counts as slipping.
LIFT_SAMPLE_S = 0.2
# Watch the WHOLE lift. Six seconds was not enough: the lift is commanded as 150mm in 3s,
# yet in all four attempts the hand had risen only 12-19mm after 6s, and the apple had
# kept pace with it throughout. Whatever loses the apple happens after that. Watch until
# the hand has stopped rising, up to this long.
#
# 60s and 8s, not 25s and 2s. In the thumb-ease run the hand rose in steps with pauses of
# about 2s between them (attempt 3: held at 18mm from 6.4-7.4s and at 20mm from 8.4-9.4s,
# then kept rising), so a 2s pause ended the watch on an arm that had not stopped: attempt
# 2 was cut off at 15mm and attempt 3 at 65mm, and both were scored as stalls. Attempt 4
# was still rising at 5mm/s when the 25s limit ended its watch at 102mm.
LIFT_WATCH_S = 60.0
LIFT_STILL_S = 8.0          # hand counted as stopped after rising <2mm over this long
#
# Watch the lift in SIMULATED time when the clock is available. Gazebo's speed is not
# constant: lifting an empty hand ran at 3.7s simulated per 42s (0.09x real time), but
# lifting a hard-squeezed apple ran at 1.0s per 60s (0.017x) and 0.6s per 31s (0.02x),
# so the 60s wall-clock watch ended while the arm had had a third of its 3.0s move, and
# attempts with the apple held firmly in the hand (4/4 fingers loaded, 0-1mm behind the
# hand) were scored ARM STALLED. The lift is judged only once the arm has had the whole
# commanded duration plus a margin; the still-rule also counts in simulated time.
LIFT_SIM_DONE_S = 3.3       # simulated seconds: the 3.0s lift plus a margin
LIFT_SIM_STILL_S = 0.5      # simulated seconds without rising, after that, ends the watch
LIFT_WALL_CAP_S = 900.0     # give up if the simulation will not get there at all
LIFT_FROZEN_WALL_S = 90.0   # the simulation clock not advancing at all for this long
# 15mm, not 5mm. As the lift starts the apple settles a few millimetres down into the
# cradle of the fingers and then rises with the hand at that offset: the one picked apple
# in the last run sat a steady 6mm behind the hand from 1.4s to 25s and rose 11.3cm. At 5mm
# that settling was reported as a slip.
SLIP_MM = 15.0
# The apple counts as having stayed in the hand for the whole lift if it ends no more than
# this far behind it.
FOLLOW_MM = 15.0

# Thumb load to ease down to before lifting, when RELAX is on for the attempt.
#
# With the apple held, wrist_1 reached its 28Nm limit in every grasp; lifting the empty hand
# from the same pose it needed 7Nm. The apple's weight can add at most 2.6Nm there (0.544kg
# on a 0.48m lever at the very most), so at least 18Nm is coming from something else about
# holding it. The largest force in the grasp is the thumb: 6.5-10Nm at lift start, peaking
# at 22-23Nm. In a contact simulation, squeezing an object that hard can produce contact
# forces that load the wrist. 3Nm is what the thumb settled to while the one full lift held
# the apple (3.15Nm from 5s to 25s), so the grip is not expected to need more.
THUMB_LIFT_NM = 3.0
THUMB_RELAX_STEP = 0.004
THUMB_RELAX_MAX = 0.20
# The UR5e's declared joint effort limits; a joint at >=98% of its limit is saturated.
ARM_EFFORT_LIMIT = {'shoulder_pan_joint': 150.0, 'shoulder_lift_joint': 150.0,
                    'elbow_joint': 150.0, 'wrist_1_joint': 28.0, 'wrist_2_joint': 28.0,
                    'wrist_3_joint': 28.0}

# How far above free-air closing noise a reading must be to count as real contact.
CONTACT_MARGIN = 2.0

# A reading at or above this during a FREE-AIR close is the hand touching itself, not
# noise: the joint effort cap is 1.5Nm and both the pinky and thumb reached exactly that
# with nothing in the hand.
SELF_COLLISION_NM = 0.8


# The gripper camera is not just a sensor: ur5e_dexhand.xacro gives gripper_camera_link
# a 0.04 x 0.04 x 0.02m COLLISION box, mounted 0.06m off the hand's centre line at
# z=0.08. In hand coordinates it therefore occupies z=0.070 to 0.090 -- and the
# fingertips converge at z=0.0834, dead centre of that band. The camera sits at exactly
# the depth the fingers close through, on one side of the hand. The apple itself clears
# it by 23-44mm at every aim tested, so it is not hitting the fruit; whether a FINGER
# passes through it has never been measured.
CAMERA_LINK = "gripper_camera_link"
CAMERA_HALF_EXTENTS = np.array([0.02, 0.02, 0.01])


SPREAD_JOINTS = ["R_Index_Yaw", "R_Middle_Yaw", "R_Ring_Yaw", "R_Pinky_Yaw"]


def print_finger_spread(node, when):
    """Log the four finger spread (Yaw) joints. They are in dexhand_controller but were
    never commanded (allow_partial_joints_goal), so they hold whatever angle they had
    when the controller started -- which can differ between Gazebo sessions."""
    vals = {j: node.latest_joint_state.get(j, (None, None, None))[0] for j in SPREAD_JOINTS}
    REC["spread_joints"] = vals
    cmd = getattr(node, "finger_yaw", None)
    print(f"  finger spread joints {when} (commanded: "
          f"{'never' if cmd is None else f'{cmd:+.3f}'}): "
          + ", ".join(f"{j[2:-4]}={v:+.4f}" if v is not None else f"{j[2:-4]}=?"
                      for j, v in vals.items()))
    return vals


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


# Free-air calibration readings seen across normal runs: 0.012-0.046Nm on the fingers and
# 0.034-0.056Nm on the thumb. Twice a run began with readings far above that (apple_07:
# fingers 0.149-0.316; apple_08: ring 0.620), which set touch thresholds up to 1.24Nm --
# close to the 1.5Nm a finger can push -- i.e. the hand was closing on something. A
# reading above CALIBRATION_SANE_NM is redone once; if it is still high, the thresholds
# fall back to the middle of those produced by normal calibrations.
CALIBRATION_SANE_NM = 0.10
DEFAULT_CONTACT_THRESHOLDS = {"R_Index": 0.08, "R_Middle": 0.08, "R_Ring": 0.08,
                              "R_Pinky": 0.08, "R_Thumb": 0.10}


def calibrate_contact_threshold(node, _retry=True):
    """Close the hand in free air and measure what a MOVING finger reads with nothing
    to touch. Returns a threshold safely above that, or None if it cannot measure.

    Everything downstream depends on telling "this finger is on the apple" from "this
    finger is moving", and that line has never actually been measured -- only assumed
    from the idle noise floor, which is a different quantity.
    """
    print("Calibrating: closing the hand in free air to measure moving-finger noise...")
    t0 = current_sim_time(node)
    node.send_arm_trajectory(REST_POSE, 4.0)
    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    # Wait for the move and the opening to finish in simulated time. The previous
    # process can end with an apple still gripped after its lift, and a wall-clock wait
    # of a few seconds is a fraction of a second of simulation, so the hand could start
    # "calibrating" with that apple still between its fingers.
    wait_until_sim(node, t0, 4.5)
    settle(node, 8.0)
    for _ in range(30):
        rclpy.spin_once(node, timeout_sec=0.1)

    # Keep every sample, not a running maximum. A hand closing on nothing still ends up
    # touching ITSELF -- measured directly, the pinky and thumb both hit their 1.500Nm
    # joint cap during a free-air close, which is a collision with the palm or each
    # other, not noise. Taking the maximum let that one spike set the threshold to
    # 3.000Nm, and the next grasp then discarded a pinky that reached 1.500Nm on the
    # apple, because 1.500 < 3.000. The calibration threw away a working finger.
    samples = {g: [] for g in FINGER_GROUPS}
    pos = 0.0
    while pos < MAX_PITCH_CEILING:
        pos = min(pos + CLOSE_STEP, MAX_PITCH_CEILING)
        node.command_fingers({g: pos for g in FINGER_GROUPS}, STEP_COMMAND_TIME,
                             thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
        for _ in range(CHECKS_PER_STEP):
            rclpy.spin_once(node, timeout_sec=0.08)
            for g in FINGER_GROUPS:
                _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
                samples[g].append(abs(eff or 0.0))

    # 90th percentile, not the peak: self-collision happens only in the last part of the
    # closure, so it is a small minority of samples and a percentile ignores it while
    # still sitting above ordinary movement noise.
    peak = {}
    for g in FINGER_GROUPS:
        vals = sorted(v for v in samples[g] if v < SELF_COLLISION_NM)
        if not vals:
            print(f"  {g.replace('R_', '')} was against something for the WHOLE close "
                  f"-- cannot calibrate it")
            return None
        peak[g] = vals[int(len(vals) * 0.90)] if len(vals) > 1 else vals[0]

    node.command_fingers({g: 0.0 for g in FINGER_GROUPS}, 1.5,
                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_GRASP_ROLL)
    for _ in range(20):
        rclpy.spin_once(node, timeout_sec=0.1)

    print("  closing in free air, peak effort per finger: "
          + ", ".join("%s=%.3f" % (g.replace("R_", ""), peak[g]) for g in FINGER_GROUPS))
    if max(peak.values(), default=0.0) <= 0.0:
        print("  could not measure -- keeping the existing threshold")
        return None
    abnormal = [g for g in FINGER_GROUPS if peak[g] > CALIBRATION_SANE_NM]
    if abnormal:
        names = ", ".join(g.replace("R_", "") for g in abnormal)
        if _retry:
            print(f"  ABNORMAL: {names} read above {CALIBRATION_SANE_NM:.2f}Nm in free air -- "
                  f"the hand is touching something; opening and calibrating again")
            return calibrate_contact_threshold(node, _retry=False)
        print(f"  ABNORMAL again: {names} -- using the normal thresholds instead: "
              + ", ".join("%s=%.3f" % (g.replace("R_", ""), DEFAULT_CONTACT_THRESHOLDS[g])
                          for g in FINGER_GROUPS))
        return dict(DEFAULT_CONTACT_THRESHOLDS)
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


def full_chain_vector(chain, arm):
    """ARM_JOINTS-ordered angles expanded into ikpy's full per-link vector."""
    v = [0.0] * len(chain.links)
    for i, link in enumerate(chain.links):
        if link.name in ARM_JOINTS:
            v[i] = float(arm[ARM_JOINTS.index(link.name)])
    return v


def real_wrist_rotation(node):
    """dexhand_base_link's real orientation in base_footprint, from TF, as a 3x3 matrix."""
    try:
        t = node.tf_buffer.lookup_transform(
            'base_footprint', 'dexhand_base_link', rclpy.time.Time(),
            timeout=rclpy.duration.Duration(seconds=0.5))
    except Exception:
        return None
    q = t.transform.rotation
    x, y, z, w = q.x, q.y, q.z, q.w
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def arm_jacobian(chain, arm, eps=1e-4):
    """6x6 Jacobian of the wrist (position, then rotation) by finite differences."""
    base = hand_fk(chain, full_chain_vector(chain, arm))
    p0, r0 = base[:3, 3], base[:3, :3]
    jac = np.zeros((6, len(ARM_JOINTS)))
    for j in range(len(ARM_JOINTS)):
        a = list(arm)
        a[j] += eps
        t = hand_fk(chain, full_chain_vector(chain, a))
        jac[:3, j] = (t[:3, 3] - p0) / eps
        m = t[:3, :3] @ r0.T
        jac[3:, j] = np.array([m[2, 1] - m[1, 2], m[0, 2] - m[2, 0],
                               m[1, 0] - m[0, 1]]) / (2 * eps)
    return jac


def servo_to(node, target_pos, target_rot, tol, max_iters, label, on_step=None,
             verbose=True):
    """Move the real wrist to target_pos in small Jacobian steps, measuring after each.

    Replaces "measure the error, ask solve_ik again" for small moves. solve_ik may return
    an answer up to 25mm off (50mm on fallback), and every new solve has a DIFFERENT error,
    so feeding the measured error back into it chased the solver's own noise: logged, the
    "sag compensation" carried was 25mm/37mm/13mm and the very next 10mm step then missed
    by 20mm/33mm/23mm; and in another attempt the accumulated target walked 117mm off the
    safe pose while the joint changes grew from 0.28 to 0.60 rad, until a fingertip was
    6mm from the apple.

    Here each step is computed from where the wrist MEASURABLY is, capped at
    SERVO_MAX_LINEAR of travel and SERVO_MAX_JOINT per joint, so there is nothing to
    accumulate and no step can lunge or flip the arm's configuration.

    Returns (status, final_error_m, steps_taken); status is "reached", "max_iters",
    "knock" (on_step returned False) or "no_tf".
    """
    for i in range(1, max_iters + 1):
        real = node.real_wrist_position()
        q = arm_now(node)
        if real is None or q is None:
            return "no_tf", None, i - 1
        dp = np.asarray(target_pos, dtype=float) - np.array(real)
        err = float(np.linalg.norm(dp))
        rot_now = real_wrist_rotation(node)
        if rot_now is None:
            rot_now = hand_fk(node.chain, full_chain_vector(node.chain, q))[:3, :3]
        if target_rot is not None:
            tr = np.asarray(target_rot)
            e_rot = 0.5 * sum(np.cross(rot_now[:, c], tr[:, c]) for c in range(3))
            ang = float(np.degrees(np.arccos(np.clip(
                (np.trace(rot_now.T @ tr) - 1.0) / 2.0, -1.0, 1.0))))
        else:
            e_rot = np.zeros(3)
            ang = 0.0
        # Position alone is not "reached": a palm at the right spot but tilted 10deg the
        # wrong way closes the fingers on the wrong side of the apple.
        if err < tol and ang < SERVO_ROT_TOL_DEG:
            return "reached", err, i - 1
        if err > SERVO_MAX_LINEAR:
            dp = dp * (SERVO_MAX_LINEAR / err)
        jac = arm_jacobian(node.chain, q)
        task = np.concatenate([dp, e_rot])
        dq = jac.T @ np.linalg.solve(jac @ jac.T + (SERVO_DAMPING ** 2) * np.eye(6), task)
        biggest = float(np.max(np.abs(dq)))
        if biggest > SERVO_MAX_JOINT:
            dq = dq * (SERVO_MAX_JOINT / biggest)
        if verbose:
            print(f"  {label} {i}: {err * 1000:.0f}mm off, palm {ang:.1f}deg off, "
                  f"largest joint change {float(np.max(np.abs(dq))):.3f} rad")
        node.send_arm_trajectory(list(np.array(q) + dq), SERVO_MOVE_TIME)
        settle(node, 6.0, min_wait=SERVO_MOVE_TIME + 0.3)
        if on_step is not None and not on_step(f"{label} {i}"):
            return "knock", err, i
    real = node.real_wrist_position()
    if real is None:
        return "no_tf", None, max_iters
    err = float(np.linalg.norm(np.asarray(target_pos, dtype=float) - np.array(real)))
    return ("reached" if err < tol else "max_iters"), err, max_iters


def arm_now(node):
    """The arm's current joint angles, in ARM_JOINTS order, or None if not all known."""
    vals = []
    for j in ARM_JOINTS:
        pos = node.latest_joint_state.get(j, (None, None, None))[0]
        if pos is None:
            return None
        vals.append(float(pos))
    return vals


def joint_jump(target, current):
    """Largest single-joint change between two ARM_JOINTS-ordered angle lists."""
    if target is None or current is None:
        return 0.0
    return float(max(abs(a - b) for a, b in zip(target, current)))


def live_apple_local(node, fallback):
    """The apple's CURRENT position in the robot frame, or fallback if unavailable.

    The gap and contact-height checks used the position the apple was placed at. When
    the apple has already been knocked, that is a lie: one attempt reported "even, the
    apple is centred" with every fingertip 4-11mm from the surface, while the apple had
    in fact been pushed 0.167m away. The contact gate had the same flaw -- a real touch
    on an apple that had shifted a few centimetres could be rejected as "not near".
    """
    now = apple_xyz(node)
    if now is None:
        return fallback
    x, y = world_to_local(now[0], now[1], node.robot_x, node.robot_y, node.robot_yaw)
    return np.array([x, y, now[2]])


def thumb_opposition(node, centre):
    """Angle between the thumb and the four fingers, measured about the apple's centre.

    180deg means the thumb is directly across the apple from the fingers, so squeezing
    traps it. A small angle means thumb and fingers are on the same side and the apple
    has an open side to be squeezed out through -- the gap it keeps escaping by.
    Returns (angle_deg, thumb_to_nearest_finger_m) or None.
    """
    if centre is None:
        return None
    tips = {}
    for g, link in FINGERTIP_LINK.items():
        try:
            t = node.tf_buffer.lookup_transform(
                'base_footprint', link, rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.5))
            p_ = t.transform.translation
            tips[g] = np.array([p_.x, p_.y, p_.z])
        except Exception:
            return None
    fingers = [tips[g] for g in FINGER_GROUPS if g != "R_Thumb"]
    v_thumb = tips["R_Thumb"] - centre
    v_fingers = np.mean(fingers, axis=0) - centre
    denom = np.linalg.norm(v_thumb) * np.linalg.norm(v_fingers)
    if denom < 1e-9:
        return None
    cos_a = float(np.clip(np.dot(v_thumb, v_fingers) / denom, -1.0, 1.0))
    angle = float(np.degrees(np.arccos(cos_a)))
    nearest = float(min(np.linalg.norm(tips["R_Thumb"] - f) for f in fingers))
    return angle, nearest


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
                      thumb_yaw=THUMB_GRASP_YAW, squeeze_extra=SQUEEZE_EXTRA,
                      after_contact=None, during_squeeze=None, observe_rad=0.03):
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
                where = (live_apple_local(node, apple_local)
                         if apple_local is not None else None)
                if where is not None and radius is not None:
                    if not node.fingertips_near_apple(where, radius).get(g):
                        continue
                contacted[g] = True
                contact_height[g] = fingertip_height_vs_apple(node, g, where)
                pos = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))[0]
                if pos is not None:
                    current[g] = pos

    step_count = [0]

    def step_wait(on_sample=None, stop=None):
        """Wait out one closing step; returns True if stop() became true."""
        step_count[0] += 1
        if CLOSE_STEP_SIM_S is None:
            for _ in range(CHECKS_PER_STEP):
                rclpy.spin_once(node, timeout_sec=0.08)
                if on_sample:
                    on_sample()
                if stop and stop():
                    return True
            return False
        start = getattr(node, "joint_stamp", None)
        wall = time.time()
        spins = 0
        while time.time() - wall < 30.0:
            rclpy.spin_once(node, timeout_sec=0.05)
            spins += 1
            if on_sample:
                on_sample()
            if stop and stop():
                return True
            now = getattr(node, "joint_stamp", None)
            if start is None or now is None:
                if spins >= CHECKS_PER_STEP:
                    return False
            elif now - start >= CLOSE_STEP_SIM_S:
                return False
        return False

    # Read-only touch record over the WHOLE closing (only when a during_squeeze hook is
    # given): every finger's position, command and strongest-joint load after each step.
    closing_samples = []

    def sample_closing(phase):
        if during_squeeze is None:
            return
        loads = finger_loads(node)
        row = {"phase": phase, "fingers": {}}
        for g in FINGER_GROUPS:
            pos = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))[0]
            row["fingers"][g] = {"pos": pos, "cmd": current[g], "load": loads[g],
                                 "contact_flag": bool(contacted[g])}
        closing_samples.append(row)

    def drive(groups, limit):
        steps = int((limit - start_pitch) / CLOSE_STEP) + 2
        for _ in range(steps):
            moved = False
            for g in groups:
                if not contacted[g] and current[g] < limit:
                    current[g] = min(current[g] + CLOSE_STEP, limit)
                    moved = True
            node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                 thumb_roll=THUMB_ROLL)
            step_wait(on_sample=sample)
            sample_closing("close")
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
    close_sim_start = current_sim_time(node)
    close_wall_start = time.time()
    drive(["R_Thumb"], MAX_PITCH_CEILING)
    # First contact alone is far too light to oppose anything: measured at 0.284Nm and
    # 0.103Nm while the fingers then pressed at 1.5-5.9Nm, so the "backstop" simply gave
    # way. Close the thumb a further fixed amount, capped by force, so it is genuinely
    # bearing on the apple before the fingers arrive.
    if contacted["R_Thumb"] and THUMB_MODE == "steady":
        def steady_thumb_effort():
            """Median thumb effort over THUMB_STEP_SIM_S of simulated time."""
            vals = []
            t_start = current_sim_time(node)
            wall_start = time.time()
            while time.time() - wall_start < 60.0:
                rclpy.spin_once(node, timeout_sec=0.05)
                _, _, e = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
                vals.append(abs(e or 0.0))
                now = getattr(node, "joint_stamp", None)
                if t_start is None or now is None or now - t_start >= THUMB_STEP_SIM_S:
                    if len(vals) >= 3:
                        break
            return float(np.median(vals)) if vals else 0.0

        contact_pos = current["R_Thumb"]
        pushed = 0.0
        raw = steady_thumb_effort()
        while raw < THUMB_PRELOAD_FORCE and pushed < THUMB_PRELOAD_EXTRA:
            pushed += THUMB_PRELOAD_STEP
            current["R_Thumb"] = min(current["R_Thumb"] + THUMB_PRELOAD_STEP,
                                     MAX_PITCH_CEILING)
            node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                 thumb_roll=THUMB_ROLL)
            raw = steady_thumb_effort()
        print(f"  thumb (steady mode) pushed {pushed:.3f} rad past first contact, "
              f"steady push {raw:.2f}Nm")
        if CAP_THUMB:
            backed = 0.0
            # Let the command catch up before judging the push it keeps up.
            raw = steady_thumb_effort()
            while raw > THUMB_PRELOAD_FORCE and backed < THUMB_BACKOFF_MAX:
                backed += THUMB_BACKOFF_STEP
                current["R_Thumb"] = max(current["R_Thumb"] - THUMB_BACKOFF_STEP, 0.0)
                node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                     thumb_roll=THUMB_ROLL)
                raw = steady_thumb_effort()
            print(f"  thumb backed off {backed:.3f} rad to hold its preload cap")
        REC["thumb_depth"] = current["R_Thumb"] - contact_pos
        peak["R_Thumb"] = max(peak["R_Thumb"], raw)
        REC["thumb_push"] = raw
        print(f"  thumb preloaded to {raw:.2f}Nm (steady), command "
              f"{REC['thumb_depth']:+.3f} rad past first contact -- now closing the four "
              f"fingers against it")
    elif contacted["R_Thumb"]:
        contact_pos = current["R_Thumb"]
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
                                 thumb_roll=THUMB_ROLL)
            if step_wait(stop=lambda: abs(node.latest_joint_state.get(
                    "R_Thumb_Pitch", (0, 0, 0))[2] or 0.0) >= THUMB_PRELOAD_FORCE):
                break
        step_wait()
        step_wait()
        _, _, teff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
        raw = abs(teff or 0.0)
        if CAP_THUMB and raw > THUMB_PRELOAD_FORCE:
            # Back the command off until the thumb's sustained push is within its cap.
            # Across seven attempts the four fingers stayed loaded after the squeeze
            # when the thumb pushed 5.5-6.2Nm (and the apple rose 4.3-5.9cm), but lost
            # their load in all four attempts where it pushed 8.8-13.3Nm (1.8-3.7cm).
            backed = 0.0
            while raw > THUMB_PRELOAD_FORCE and backed < THUMB_BACKOFF_MAX:
                backed += THUMB_BACKOFF_STEP
                current["R_Thumb"] = max(current["R_Thumb"] - THUMB_BACKOFF_STEP, 0.0)
                node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                                     thumb_roll=THUMB_ROLL)
                step_wait()
                _, _, teff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
                raw = abs(teff or 0.0)
            print(f"  thumb backed off {backed:.3f} rad to hold its preload cap")
        REC["thumb_depth"] = current["R_Thumb"] - contact_pos
        peak["R_Thumb"] = max(peak["R_Thumb"], raw)
        REC["thumb_push"] = raw
        print(f"  thumb preloaded to {raw:.2f}Nm, command {REC['thumb_depth']:+.3f} rad "
              f"past first contact -- now closing the four fingers against it")
    else:
        print("  thumb found nothing -- the fingers will have no backstop")
    drive(fingers, MAX_PITCH_CEILING)
    print("  fingers wrapped: %d/4" % sum(1 for g in fingers if contacted[g]))

    # Optional hook between first contact and the squeeze (fragility_grasp.py). It may
    # press the contacted fingers a little further with probe() to feel the object, and
    # return {"squeeze_extra": ...} to choose how hard to squeeze. Any probing counts
    # towards the squeeze, so the total closing past first contact is still exactly
    # squeeze_extra. With after_contact=None nothing here runs and the grasp is the
    # tested one.
    probe_total = [0.0]

    def probe(amount, settle_sim_s=0.3):
        """Close every contacted finger a further `amount` rad, wait settle_sim_s of
        simulated time, and report how far each actually moved and how its load changed."""
        groups = [g for g in FINGER_GROUPS if contacted[g]]
        before = {}
        for g in groups:
            pos, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))
            before[g] = (pos, abs(eff or 0.0))
        for g in groups:
            current[g] = min(current[g] + amount, MAX_PITCH_CEILING)
        node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                             thumb_roll=THUMB_ROLL)
        wait_until_sim(node, current_sim_time(node), settle_sim_s, wall_cap=60.0)
        out = {}
        for g in groups:
            pos, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))
            p0, e0 = before[g]
            out[g] = {"commanded_rad": amount,
                      "moved_rad": None if (pos is None or p0 is None) else pos - p0,
                      "effort_before_nm": e0, "effort_after_nm": abs(eff or 0.0)}
        probe_total[0] += amount
        return out

    if after_contact is not None:
        override = after_contact(node, {"contacted": dict(contacted), "probe": probe,
                                        "squeeze_extra": squeeze_extra}) or {}
        if "squeeze_extra" in override:
            squeeze_extra = float(override["squeeze_extra"])
            print(f"  squeeze chosen after contact: {squeeze_extra:.3f} rad past first "
                  f"contact ({probe_total[0]:.3f} rad of it already used by the probe)")

    # Squeeze phase: close past first contact so the fingers actually hold.
    # Measure what the squeeze itself costs. Fingers reach 1.500Nm -- the joint cap,
    # genuine hard contact -- and then finish at 0.00-0.02Nm holding nothing, while the
    # apple moves 4.7-5.1cm in that same phase. The squeeze is the prime suspect for
    # pushing the apple out of the hand it just closed around.
    before_squeeze = apple_xyz(node)
    squeezed = probe_total[0]

    # Passive touch (fragility_grasp.py): read, never command. Each sample is taken after a
    # normal squeeze step, so the fingers move exactly as in the tested grasp.
    touch_samples = []
    touch_decided = [during_squeeze is None]

    def sample_touch(squeezed_so_far):
        row = {"squeezed_rad": squeezed_so_far, "fingers": {}}
        for g in FINGER_GROUPS:
            if contacted[g]:
                pos, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))
                row["fingers"][g] = {"pos": pos, "effort": abs(eff or 0.0),
                                     "cmd": current[g]}
        touch_samples.append(row)

    if during_squeeze is not None:
        sample_touch(squeezed)

    while squeezed < squeeze_extra:
        squeezed += SQUEEZE_STEP
        for g in FINGER_GROUPS:
            if not contacted[g]:
                continue
            _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
            if abs(eff or 0.0) < SQUEEZE_FORCE_CAP:
                current[g] = min(current[g] + SQUEEZE_STEP, MAX_PITCH_CEILING)
        node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                             thumb_roll=THUMB_ROLL)

        def track_peak():
            for g in FINGER_GROUPS:
                _, _, eff = node.latest_joint_state.get(f"{g}_Pitch", (0, 0, 0))
                peak[g] = max(peak[g], abs(eff or 0.0))
        step_wait(on_sample=track_peak)

        if during_squeeze is not None:
            sample_touch(squeezed)
            sample_closing("squeeze")
            if not touch_decided[0] and squeezed >= observe_rad - 1e-9:
                touch_decided[0] = True
                override = during_squeeze(node, {"samples": list(touch_samples),
                                                 "closing": list(closing_samples),
                                                 "squeeze_extra": squeeze_extra,
                                                 "squeezed_rad": squeezed}) or {}
                if "squeeze_extra" in override:
                    # Never undo squeeze already applied.
                    squeeze_extra = max(float(override["squeeze_extra"]), squeezed)
                    print(f"  squeeze set by touch+vision after {squeezed:.3f} rad of it: "
                          f"{squeeze_extra:.3f} rad past first contact")

    # How the closing ran in time, and how far each finger is commanded PAST where it
    # actually stands. The same code picked apple_10 4/4 at 0.05-0.07x real time and 0/2
    # at 0.11-0.12x, with the hand arriving the same way, so the squeeze itself differs
    # with simulation speed. A position-controlled finger only presses as hard as its
    # command is ahead of it: this is that lead, measured rather than assumed.
    close_sim_end = current_sim_time(node)
    if close_sim_start is not None and close_sim_end is not None:
        sim_used = close_sim_end - close_sim_start
        wall_used = time.time() - close_wall_start
        REC["close_sim_s"] = sim_used
        per_step = sim_used / max(step_count[0], 1)
        REC["close_step_sim_s"] = per_step
        print(f"  closing + squeeze took {sim_used:.2f}s of simulated time in {wall_used:.0f}s "
              f"({sim_used / max(wall_used, 1e-6):.3f}x real time), {step_count[0]} steps, "
              f"{per_step * 1000:.1f}ms simulated per step "
              f"({'paced by messages' if CLOSE_STEP_SIM_S is None else 'paced in sim time'})")
    lead = {}
    for g in FINGER_GROUPS:
        actual = node.latest_joint_state.get(f"{g}_Pitch", (None, None, None))[0]
        if actual is not None:
            lead[g] = current[g] - actual
    if lead:
        REC["squeeze_lead"] = lead
        print("  command ahead of actual finger position (what makes it press): "
              + ", ".join(f"{g.replace('R_', '')}={v:+.3f}rad" for g, v in lead.items()))

    after_squeeze = apple_xyz(node)
    if (any(contacted.values()) and before_squeeze is not None
            and after_squeeze is not None):
        moved = float(np.linalg.norm(np.array(after_squeeze) - np.array(before_squeeze)))
        print(f"  the squeeze itself moved the apple {moved:.3f}m"
              + ("" if moved < 0.010
                 else "  <-- the squeeze is pushing the apple out of the hand"))

    # Each finger has three joints -- Pitch at the knuckle, then Flexor and DIP -- and when
    # a finger wraps an apple the outer two carry most of the load. Reading Pitch alone
    # reported the four fingers of the one SUCCESSFUL pick as holding 0.02Nm each, the
    # idle floor, while that grasp lifted the apple 8.5cm -- so "the fingers let go"
    # conclusions drawn from that number were wrong. Take the strongest of the three.
    def finger_load(g):
        joints = [f"{g}_Pitch"] + list(FINGER_SECONDARY_JOINTS.get(g, []))
        return max(abs(node.latest_joint_state.get(j, (0, 0, 0))[2] or 0.0)
                   for j in joints)

    holding = {g: finger_load(g) for g in FINGER_GROUPS}

    # Grip check before lifting. The squeeze commands a fixed amount and the lift used to
    # follow regardless of what the hand was actually doing. apple_10 (0.70kg) then failed
    # 10 of 10 attempts on 15 Sep with the fingers commanded the full squeeze past their
    # real position but reading 0.01-0.02Nm (index, middle, pinky) and only one finger plus
    # the thumb pressing at lift start; its earlier picks had 3-4 fingers at 1.5Nm and the
    # thumb at 3-8Nm. Wait for the fingers in simulated time, measure, and squeeze further
    # only the contacted digits that are not pressing, until the grip is firm.
    fingers_only = [g for g in FINGER_GROUPS if g != "R_Thumb"]

    def firm(h):
        pressing = sum(1 for g in fingers_only if h[g] > GRIP_HOLD_NM)
        return pressing >= GRIP_MIN_FINGERS and h["R_Thumb"] >= GRIP_MIN_THUMB_NM

    t_catch = current_sim_time(node)
    wait_until_sim(node, t_catch, GRIP_SETTLE_SIM_S)
    holding = {g: finger_load(g) for g in FINGER_GROUPS}
    extra_used = 0.0
    tries = 0
    while not firm(holding) and extra_used < GRIP_EXTRA_MAX - 1e-9:
        tries += 1
        extra_used += GRIP_EXTRA_STEP
        slack = [g for g in FINGER_GROUPS if contacted[g] and (
            holding[g] < (GRIP_MIN_THUMB_NM if g == "R_Thumb" else GRIP_HOLD_NM))]
        for g in slack:
            current[g] = min(current[g] + GRIP_EXTRA_STEP, MAX_PITCH_CEILING)
        node.command_fingers(current, STEP_COMMAND_TIME, thumb_yaw=THUMB_GRASP_YAW,
                             thumb_roll=THUMB_ROLL)
        t_step = current_sim_time(node)
        wait_until_sim(node, t_step, GRIP_SETTLE_SIM_S)
        holding = {g: finger_load(g) for g in FINGER_GROUPS}
        for g in FINGER_GROUPS:
            peak[g] = max(peak[g], holding[g])
        print(f"  grip check {tries}: squeezed {', '.join(s.replace('R_', '') for s in slack) or 'nothing'} "
              f"a further {GRIP_EXTRA_STEP:.3f} rad -> "
              + ", ".join("%s=%.2f" % (g.replace('R_', ''), holding[g]) for g in FINGER_GROUPS))
    REC["grip_extra"] = extra_used
    REC["grip_firm"] = firm(holding)
    pressing_now = sum(1 for g in fingers_only if holding[g] > GRIP_HOLD_NM)
    print(f"  grip check: {pressing_now}/4 fingers pressing, thumb {holding['R_Thumb']:.2f}Nm -- "
          + ("FIRM" if REC["grip_firm"] else
             f"still NOT firm after {extra_used:.3f} rad extra squeeze"))

    REC["holding"] = sum(1 for g in FINGER_GROUPS if holding[g] > GRIP_HOLD_NM)
    n_sq = sum(1 for g in FINGER_GROUPS if contacted[g])
    print(f"  after squeeze ({n_sq}/5 fingers had contact to squeeze), holding force "
          f"(strongest of each finger's 3 joints): "
          + ", ".join("%s=%.2f" % (g, holding[g]) for g in FINGER_GROUPS))
    # Simulation clock at this moment, so fingertip-sensor logs (tactile_check.py --csv,
    # which records the same /joint_states stamp) can be cut into phases of the grasp.
    stamp = getattr(node, "joint_stamp", None)
    if stamp is not None:
        REC["sim_after_squeeze"] = stamp
        print(f"  [phase] squeeze finished at sim t={stamp:.3f}s")

    known = {g: h for g, h in contact_height.items() if h is not None}
    if known:
        print("  where each finger met the apple (+ is above its equator, - is below):")
        print("    " + ", ".join(f"{g.replace('R_', '')}={h * 1000:+.0f}mm"
                                 for g, h in known.items()))
        below = [g for g, h in known.items() if h < 0.0]
        REC["below"] = f"{len(below)}/{len(known)}"
        print(f"    {len(below)}/{len(known)} contacts are BELOW the equator"
              + ("" if below else
                 "  <-- every contact is on the TOP half: squeezing drives the apple "
                 "down and out, and nothing holds it up during the lift"))
    LAST_FINGER_CMD.clear()
    LAST_FINGER_CMD.update(current)
    return contacted, peak


def finger_loads(node):
    """Strongest of each finger's three joint efforts (Pitch, Flexor, DIP), in Nm."""
    out = {}
    for g in FINGER_GROUPS:
        joints = [f"{g}_Pitch"] + list(FINGER_SECONDARY_JOINTS.get(g, []))
        out[g] = max(abs(node.latest_joint_state.get(j, (0, 0, 0))[2] or 0.0)
                     for j in joints)
    return out


def apple_xyz(node):
    if node.target_pose is None:
        return None
    p = node.target_pose.position
    return np.array([p.x, p.y, p.z])


def attempt(node, target_name, palm_tilt, preshape, lateral, method, drop=0.0,
            cap_thumb=False, empty=False, relax=False, finish_grasp_move=False,
            before_close=None, recentre=False, thumb_mode="spike", after_contact=None,
            during_squeeze=None, observe_rad=0.03):
    """One full grasp attempt: reset, approach, close, lift, measure.

    before_close, if given, is called as before_close(node) once the hand is in position
    and just before the fingers close. It may return a dict of overrides; currently
    "squeeze_extra" (rad past first contact) is honoured. This is how the main pipeline
    lets the vision layer look at the apple from the grasp pose and set the grip.
    Left as None, the attempt is exactly the tested one.
    """
    global THUMB_ROLL
    THUMB_ROLL = THUMB_GRASP_ROLL
    palm_offset = PALM_OFFSET
    REC.clear()
    REC["preshape"] = preshape
    REC["method"] = method
    REC["drop"] = drop
    global CAP_THUMB
    CAP_THUMB = cap_thumb
    REC["cap"] = "on" if cap_thumb else "off"
    REC["empty"] = empty
    REC["recentre"] = "on" if recentre else "off"
    global THUMB_MODE
    THUMB_MODE = thumb_mode
    REC["thumb"] = thumb_mode
    REC["relax"] = "on" if relax else "off"
    REC["finish"] = "yes" if finish_grasp_move else "no"
    label = (f"{'EMPTY-HAND CONTROL' if empty else 'GRASP'}   "
             f"grasp move waited out: {'YES' if finish_grasp_move else 'no'}   "
             f"thumb cap {'ON' if cap_thumb else 'off'}   grasp lowered {drop * 1000:.0f}mm   "
             f"pre-shape {preshape:.1f}")
    print(f"\n{'=' * 72}\n{label}\n{'=' * 72}")
    rot = tilted_palm_rotation(palm_tilt)

    wx, wy = APPLE_HOME_WORLD_XY[target_name]
    node.robot_x, node.robot_y, node.robot_yaw = wx, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW
    node.reset_everything()
    if not node.reset_target_apple_position(target_name):
        # Two attempts in one run began against a still-flying apple: one ended with the
        # arm unable to reach it at all (fingertips 495mm away, IK error 0.338m) and the
        # other threw it 3.4m off the table. Neither measured anything about the grasp.
        REC["settled"] = "no"
        if getattr(node, "last_reset_stale", False):
            print("  SIM FROZE: the apple's position stopped updating during the reset -- "
                  "restart Gazebo; this attempt measured nothing.")
            return {"ok": False, "palm_offset": palm_offset, "lateral": lateral,
                    "palm_tilt": palm_tilt, "sim_frozen": True}
        print("  SKIPPED: the apple would not stop moving, so this attempt would "
              "measure nothing.")
        return {"ok": False, "palm_offset": palm_offset, "lateral": lateral, "palm_tilt": palm_tilt,
                "unsettled": True}
    REC["settled"] = "yes"
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
            "thumb_yaw": THUMB_GRASP_YAW, "palm_offset": palm_offset, "lateral": lateral, "ok": False}
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
    # Lower the grasp on purpose. How far below the apple's middle the fingers meet it
    # decides the lift, attempt for attempt:
    #   fingers meeting it on average 12mm below the middle -> lifted 8.5cm (the pick)
    #                                 10mm below             -> 5.9cm
    #                                 at the middle          -> 4.3cm and 2.0cm
    #                                 9mm above              -> -0.2cm
    # and the two best attempts are the two where the arm happened to settle LOWER than
    # its 0.604 target (0.597 and 0.587). Rather than depend on where it settles, aim low.
    wrist_target = wrist_target + np.array([0.0, 0.0, -drop])

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
            "thumb_yaw": THUMB_GRASP_YAW, "palm_offset": palm_offset, "lateral": lateral, "ok": False}

    # Pre-shape BEFORE descending, so the fingertips are retracted on the way down and
    # the wrist can actually reach the pocket height instead of the fingers grounding
    # out on the table first.
    # Thumb stays fully open during the approach so it is not occupying the space the
    # apple has to enter.
    pre = {g: preshape for g in FINGER_GROUPS}
    pre["R_Thumb"] = 0.0
    node.command_fingers(pre, 1.5, thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_ROLL)
    settle(node, 6.0, joints=[f"{g}_Pitch" for g in FINGER_GROUPS], thresh=0.02)
    print(f"  approaching from ({approach_pos[0]:.3f}, {approach_pos[1]:.3f}, "
          f"{approach_pos[2]:.3f}) -- {APPROACH_BACKOFF:.2f}m back, then moving in "
          f"sideways rather than descending onto the apple")
    # Track the apple across every arm move so a knock is attributed to the move that
    # caused it, instead of being inferred afterwards from a single before/after number.
    watch = {"last": apple_xyz(node), "knocks": []}

    def check_knock(label):
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        now = apple_xyz(node)
        d = 0.0
        if watch["last"] is not None and now is not None:
            d = float(np.linalg.norm(now - watch["last"]))
            if d > KNOCK_THRESHOLD:
                watch["knocks"].append((label, d))
                # Was any part of the hand close enough to have done it? If every
                # fingertip was well clear, the apple moved on its own (rolled), and
                # calling that a knock sends the investigation after the wrong thing.
                gaps_now = fingertip_gaps(node, live_apple_local(node, apple_local),
                                          APPLE_RADIUS.get(target_name, 0.0555))
                near = min((v for v in gaps_now.values() if v is not None), default=None)
                if near is not None and near > HAND_CLEAR_OF_APPLE:
                    print(f"  !! apple moved {d:.3f}m during: {label} -- but the nearest "
                          f"fingertip was {near * 1000:.0f}mm away, so the hand did not "
                          f"touch it: the apple ROLLED on its own")
                else:
                    near_s = f"{near * 1000:.0f}mm" if near is not None else "unknown"
                    print(f"  !! apple moved {d:.3f}m during: {label} -- nearest "
                          f"fingertip {near_s} away")
        watch["last"] = now
        return d

    def not_positioned(reason):
        print(f"  NOT APPROACHING THE APPLE: {reason}")
        now = apple_xyz(node)
        if before is not None and now is not None:
            REC["apple_moved"] = float(np.linalg.norm(now - before))
        return {"preshape": preshape, "palm_tilt": palm_tilt,
                "thumb_yaw": THUMB_GRASP_YAW, "palm_offset": palm_offset,
                "lateral": lateral, "ok": False, "not_positioned": True,
                "reason": reason}

    approach_cmd_sim = current_sim_time(node)
    node.send_arm_trajectory(approach[0], 3.5)
    if approach_cmd_sim is not None:
        # A fixed slice of SIMULATED time, not a 12s wall-clock wait. The grasp move is
        # sent before the approach move completes, so how far the arm has got toward the
        # back-off pose decides the path the hand takes into the apple -- and a wall-clock
        # wait made that depend on how fast Gazebo happened to run. At about 0.4-1.0s of
        # simulated time (0.03-0.09x real time) apple_10 was disturbed 15-17mm and picked
        # 5 of 5; at about 1.3s (0.10-0.11x) it was disturbed 19-21mm and picked 0 of 2;
        # waiting out the whole 3.5s move disturbed apple_06 72mm and picked 0 of 4.
        waited = wait_until_sim(node, approach_cmd_sim, APPROACH_SIM_S)
        print(f"  approach move given {waited:.2f}s of simulated time before the grasp move")
    else:
        settle(node, 12.0)
    check_knock("the move to the approach pose")

    if method == "pick":
        # EXACTLY the approach that produced the only successful pick (commit 9fabc26):
        # one IK move straight to the grasp pose, then up to 9 corrections there, stopping
        # once within 20mm. It lost the apple often afterwards -- but a large share of those
        # losses were the apple rolling away on its own, a perfect sphere with nothing to
        # stop a roll, which the flat base has since fixed. So it gets a fair re-test here,
        # side by side with the servo approach, instead of being judged on runs where the
        # apple would not stay still. Knocks are logged but do not abort.
        grasp_cmd_sim = current_sim_time(node)
        node.send_arm_trajectory(grasp[0], 3.0)
        settle(node, 20.0)
        if finish_grasp_move:
            # Let the 3.0s grasp move finish in simulated time before measuring, so the
            # corrections are not computed against an arm still travelling. Only this
            # move: waiting out the approach move as well (7e769f3) let the arm reach the
            # true back-off pose, and the swing in from there knocked the apple 72mm.
            waited = wait_until_sim(node, grasp_cmd_sim, 3.5)
            settle(node, 10.0)
            if waited is not None:
                print(f"  waited for the grasp move to finish: {waited:.1f}s of simulated time "
                      f"since it was commanded")
                if waited < 3.0:
                    # The simulation clock stopped. On apple_03 it reported 0.0s after the
                    # 300s wait, the arm had not moved (164mm off, three corrections all
                    # 164mm), and carrying on drove the hand through the apple and scored
                    # the attempt NOT HELD. Stop here instead: nothing about the grasp can
                    # be measured on a frozen simulation.
                    print("  SIM FROZE: the simulation clock did not advance during the grasp "
                          "move -- abandoning this attempt")
                    return {"ok": False, "sim_frozen": True, "palm_offset": palm_offset,
                            "lateral": lateral, "palm_tilt": palm_tilt}
        check_knock("the move to the grasp pose")
        real0 = node.real_wrist_position()
        if real0 is not None:
            REC["first_err"] = float(np.linalg.norm(np.array(real0) - wrist_target))
            print(f"  first move landed {REC['first_err'] * 1000:.0f}mm from the grasp pose")
        corrected = list(wrist_target)
        for correction_i in range(PICK_CORRECTION_ITERS):
            real_now = node.real_wrist_position()
            if real_now is None:
                break
            err_now = float(np.linalg.norm(np.array(real_now) - wrist_target))
            if err_now < PICK_WRIST_TOLERANCE:
                break
            corrected = list(np.array(corrected)
                             + (wrist_target - np.array(real_now)) * CORRECTION_GAIN)
            again = solve_ik(node.chain, corrected, target_rotation=rot)
            if again is None:
                print(f"  correction {correction_i + 1}: corrected target unreachable, "
                      f"keeping {err_now:.3f}m error")
                break
            print(f"  correction {correction_i + 1}: err {err_now:.3f}m -> re-solving")
            node.send_arm_trajectory(again[0], 2.0)
            settle(node, 15.0)
            check_knock(f"correction {correction_i + 1}")
        real_end = node.real_wrist_position()
        if real_end is not None:
            REC["pre_err"] = float(np.linalg.norm(np.array(real_end) - wrist_target))
        REC["steps"] = "direct"
        if recentre:
            # Re-centre the hand on where the apple actually is. The fingertips brush the
            # apple on the way in and push it sideways -- 1mm for apple_01, 8-12mm for
            # apple_06, 15-22mm for apple_10 -- and the grasp then closes around where it
            # was. apple_10 picked with the thumb 112-115deg across from the fingers and
            # failed 14 of 14 on 15 Sep at 87-109deg, with only one finger pressing.
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.05)
            live = np.array(live_apple_local(node, apple_local), dtype=float)
            shift = np.array([live[0] - apple_local[0], live[1] - apple_local[1], 0.0])
            REC["recentre_shift"] = float(np.linalg.norm(shift))
            print(f"  re-centre: the apple is {np.linalg.norm(shift) * 1000:.0f}mm "
                  f"(x {shift[0] * 1000:+.0f}, y {shift[1] * 1000:+.0f}) from where the hand aimed")
            if np.linalg.norm(shift) > RECENTRE_MIN_M:
                new_target = np.array(wrist_target) + shift
                here = arm_now(node)
                sol = solve_ik(node.chain, list(new_target), target_rotation=rot, current=here)
                jump = joint_jump(sol[0], here) if sol is not None else None
                if sol is None or jump > FINE_MOVE_MAX_JUMP:
                    print("  re-centre: skipped -- "
                          + ("unreachable" if sol is None else f"needs a {jump:.2f} rad joint swing"))
                else:
                    t_rc = current_sim_time(node)
                    node.send_arm_trajectory(sol[0], RECENTRE_MOVE_S)
                    wait_until_sim(node, t_rc, RECENTRE_MOVE_S + 0.5)
                    settle(node, 10.0)
                    check_knock("the re-centre move")
                    wrist_target = new_target
                    apple_local = live
                    real_rc = node.real_wrist_position()
                    if real_rc is not None:
                        print(f"  re-centre: hand moved {np.linalg.norm(shift) * 1000:.0f}mm; wrist now "
                              f"{np.linalg.norm(np.array(real_rc) - new_target) * 1000:.0f}mm from "
                              f"the re-centred target")
        now = apple_xyz(node)
        if before is not None and now is not None:
            REC["apple_moved"] = float(np.linalg.norm(now - before))
    else:
        # One coarse IK move to the pre-grasp pose -- the only large move near the apple.
        pregrasp_pos = wrist_target + wrist_rot @ np.array([0.0, 0.0, -PREGRASP_STANDOFF])
        here = arm_now(node)
        pregrasp = solve_ik(node.chain, list(pregrasp_pos), target_rotation=rot, current=here)
        if pregrasp is None:
            return not_positioned("the pre-grasp pose is unreachable")
        jump = joint_jump(pregrasp[0], here)
        if jump > COARSE_MOVE_MAX_JUMP:
            return not_positioned(f"the move to pre-grasp needs a {jump:.2f} rad joint swing "
                                  f"(a configuration flip)")
        node.send_arm_trajectory(pregrasp[0], 2.5)
        settle(node, 15.0)
        check_knock("the move to the pre-grasp pose")
        real0 = node.real_wrist_position()
        if real0 is not None:
            REC["first_err"] = float(np.linalg.norm(np.array(real0) - pregrasp_pos))
            print(f"  first move landed {REC['first_err'] * 1000:.0f}mm from the pre-grasp pose")

        def still_ok(label):
            return check_knock(label) <= KNOCK_THRESHOLD

        # Fine correction, clear of the apple, by Jacobian servo.
        status, pre_err, steps = servo_to(node, pregrasp_pos, rot, SERVO_TOL, SERVO_MAX_ITERS,
                                          "correction", on_step=still_ok)
        REC["pre_err"] = pre_err
        if status == "no_tf" or pre_err is None:
            return not_positioned("cannot read the wrist position")
        if status == "knock":
            return not_positioned("the apple moved while the hand was being corrected")
        if pre_err > APPROACH_GATE:
            return not_positioned(
                f"after {steps} correction steps the hand is still {pre_err * 1000:.0f}mm from "
                f"its pre-grasp pose (limit {APPROACH_GATE * 1000:.0f}mm)")
        print(f"  hand verified {pre_err * 1000:.0f}mm from its pre-grasp pose after {steps} "
              f"correction steps, {PREGRASP_STANDOFF * 1000:.0f}mm clear of the apple")

        # Final approach: 10mm waypoints on a straight line, each reached by servo and checked.
        print(f"  moving in {PREGRASP_STANDOFF * 1000:.0f}mm in {APPROACH_STEPS} checked "
              f"10mm steps")
        worst_track = 0.0
        REC["steps"] = f"0/{APPROACH_STEPS}"
        for k in range(1, APPROACH_STEPS + 1):
            waypoint = pregrasp_pos + (wrist_target - pregrasp_pos) * (k / APPROACH_STEPS)
            status, err_k, _ = servo_to(node, waypoint, rot, APPROACH_TOL, APPROACH_SERVO_ITERS,
                                        f"approach step {k}/{APPROACH_STEPS}",
                                        on_step=still_ok, verbose=False)
            if status == "no_tf" or err_k is None:
                return not_positioned("lost the wrist position during the approach")
            if status == "knock":
                return not_positioned(
                    f"the apple moved during approach step {k}/{APPROACH_STEPS}, so the hand "
                    f"stopped instead of pushing it further")
            worst_track = max(worst_track, err_k)
            REC["steps"] = f"{k}/{APPROACH_STEPS}"
            if err_k > TRACK_LIMIT:
                return not_positioned(
                    f"at approach step {k}/{APPROACH_STEPS} the hand was {err_k * 1000:.0f}mm off "
                    f"the approach line (limit {TRACK_LIMIT * 1000:.0f}mm)")
        if before is not None and apple_xyz(node) is not None:
            REC["apple_moved"] = float(np.linalg.norm(apple_xyz(node) - before))
        print(f"  arrived at the grasp pose; worst deviation from the approach line "
              f"{worst_track * 1000:.0f}mm, apple undisturbed")

    # If the apple is no longer in front of the hand, say so plainly and stop. Measuring
    # fingertip gaps, thumb opposition or "centring" against an apple 0.6-1.9m away
    # produces confident-looking numbers that mean nothing -- a "3deg" thumb opposition
    # was reported for exactly that reason.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    gone_by = float(np.linalg.norm(live_apple_local(node, apple_local) - apple_local))
    if gone_by > APPLE_RADIUS.get(target_name, 0.0555) + 0.05:
        culprit = (max(watch["knocks"], key=lambda kd: kd[1])
                   if watch["knocks"] else None)
        print(f"  APPLE GONE: it is {gone_by:.3f}m from where the hand is aiming. "
              + (f"Knocked hardest by {culprit[0]} ({culprit[1]:.3f}m)."
                 if culprit else "No single arm move moved it past the threshold."))
        print("  The grasp was never attempted -- skipping closing and lift.")
        return {"preshape": preshape, "palm_tilt": palm_tilt,
                "thumb_yaw": THUMB_GRASP_YAW, "palm_offset": palm_offset,
                "lateral": lateral, "ok": False, "knocked": True,
                "knocked_by": culprit[0] if culprit else "unattributed"}

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
        drift = abs(real_yaw - THUMB_GRASP_YAW)
        print(f"  thumb yaw commanded {THUMB_GRASP_YAW:+.2f}, "
              f"actually at {real_yaw:+.2f}"
              + ("" if drift < 0.05 else f"  <-- OFF BY {drift:.2f} rad, being pushed back"))
    print_finger_spread(node, "at the grasp pose, before closing")

    # Which fingers are actually within reach of the apple BEFORE closing starts?
    # The run that prompted this had 5/5 "contacts" and yet finished with index, middle
    # and ring all reading 0.04-0.05Nm -- the idle noise floor, i.e. touching nothing.
    # Only one finger per attempt was ever loaded. That is the signature of the apple
    # sitting off to one side of the hand rather than in the middle of the closing arc,
    # so measure each fingertip's own gap instead of trusting the centroid.
    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.1)
    apple_now = live_apple_local(node, apple_local)
    shifted = float(np.linalg.norm(apple_now - apple_local))
    if shifted > 0.01:
        print(f"  the apple is now {shifted:.3f}m from where the hand aimed -- measuring "
              f"against where it actually is")
    gaps = fingertip_gaps(node, apple_now,
                          APPLE_RADIUS.get(target_name, 0.0555))
    reach = []
    valid_gaps = [v for v in gaps.values() if v is not None] if gaps else []
    if valid_gaps:
        REC["nearest_tip"] = min(valid_gaps)
    if gaps:
        print("  fingertip gap to the apple surface before closing:")
        print("    " + ", ".join(
            f"{g.replace('R_', '')}={v * 1000:+.0f}mm" if v is not None else f"{g}=?"
            for g, v in gaps.items()))
        four = [v for g, v in gaps.items() if g != "R_Thumb" and v is not None]
        if len(four) == 4:
            # Kept so other layers can score "is the hand centred?" against a measurement
            # (vlm_predict_verify.py) instead of re-deriving it from the printed line.
            REC["gap_spread"] = float(max(four) - min(four))
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

    if empty:
        # Control: exactly the same approach and grasp pose, but with the apple taken
        # away, so the lift that follows carries no apple. In two grasped attempts the
        # apple stayed within 5mm of the hand for the whole lift, yet the arm stopped
        # after 2.3-2.4cm with wrist_1 pinned at its 28Nm limit; an attempt that held
        # nothing lifted 14.4cm with wrist_1 at 9Nm. The apple's weight adds under 2Nm
        # at the wrist, so weight does not explain it -- this separates "holding an apple
        # stalls the arm" from "this arm stalls at this pose anyway".
        wx_, wy_ = APPLE_HOME_WORLD_XY[target_name]
        node.teleport_model(target_name, wx_, wy_ + 0.9, 0.06, settle_sec=1.0)
        print("  CONTROL: apple moved off the table -- the hand will close and lift empty")
    squeeze_extra = SQUEEZE_EXTRA
    if before_close is not None:
        overrides = before_close(node) or {}
        if overrides.get("squeeze_extra") is not None:
            squeeze_extra = float(overrides["squeeze_extra"])
    REC["squeeze"] = squeeze_extra
    if squeeze_extra != SQUEEZE_EXTRA:
        print(f"  squeezing {squeeze_extra:.3f} rad past first contact "
              f"(tested value {SQUEEZE_EXTRA:.3f})")
    contacted, peak = close_and_measure(
        node, start_pitch=preshape, apple_local=apple_now,
        radius=APPLE_RADIUS.get(target_name, 0.0555), thumb_yaw=THUMB_GRASP_YAW,
        squeeze_extra=squeeze_extra, after_contact=after_contact,
        during_squeeze=during_squeeze, observe_rad=observe_rad)
    n = sum(contacted.values())
    REC["contacts"] = n
    print(f"  fingers contacted: {n}/5")
    opp = thumb_opposition(node, live_apple_local(node, apple_now)) if n > 0 else None
    if opp is not None:
        angle, nearest = opp
        REC["opposition"] = angle
        print(f"  thumb vs fingers around the apple: {angle:.0f}deg apart "
              f"(180 = directly opposite), thumb {nearest * 1000:.0f}mm from the "
              f"nearest finger"
              + ("" if angle >= OPPOSITION_MIN_DEG
                 else "  <-- OPEN SIDE: squeezing pushes the apple out between them"))
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
    here = arm_now(node)
    lift = solve_ik(node.chain, lift_target, target_rotation=rot, current=here)
    if lift is not None and joint_jump(lift[0], here) > COARSE_MOVE_MAX_JUMP:
        print(f"  lift REFUSED: it needs a {joint_jump(lift[0], here):.2f} rad joint "
              f"swing, which would fling the apple")
        lift = None
    if lift is None:
        print("  lift target UNREACHABLE -- cannot test the hold")
    else:
        if relax and LAST_FINGER_CMD:
            _, _, teff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
            before_relax = abs(teff or 0.0)
            eased = 0.0
            while (abs(teff or 0.0) > THUMB_LIFT_NM and eased < THUMB_RELAX_MAX):
                eased += THUMB_RELAX_STEP
                LAST_FINGER_CMD["R_Thumb"] = max(LAST_FINGER_CMD["R_Thumb"] - THUMB_RELAX_STEP,
                                                 0.0)
                node.command_fingers(LAST_FINGER_CMD, STEP_COMMAND_TIME,
                                     thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_ROLL)
                for _ in range(CHECKS_PER_STEP):
                    rclpy.spin_once(node, timeout_sec=0.08)
                _, _, teff = node.latest_joint_state.get("R_Thumb_Pitch", (0, 0, 0))
            REC["thumb_eased_from"] = before_relax
            print(f"  eased the thumb from {before_relax:.2f}Nm to {abs(teff or 0.0):.2f}Nm "
                  f"({eased:.3f} rad) before lifting")
        print(f"  lifting {LIFT_HEIGHT:.2f}m...")
        # Watch the lift itself. Seven attempts with near-identical grasp geometry --
        # fingers 9-12mm below the apple's middle, thumb 12-15mm above -- lifted it
        # anywhere from +1.6cm to +8.5cm, so nothing measured before the lift decides
        # the outcome; it is decided during the lift, where nothing was measured. Record
        # how far the hand and the apple have each risen, which fingers are still loaded,
        # and the moment the apple starts falling behind the hand.
        for _ in range(5):
            rclpy.spin_once(node, timeout_sec=0.05)
        w0 = node.real_wrist_position()
        a0 = apple_xyz(node)
        loads0 = finger_loads(node)
        REC["fingers_at_lift"] = sum(1 for g in FINGER_GROUPS
                                     if g != "R_Thumb" and loads0[g] > GRIP_HOLD_NM)
        REC["thumb_at_lift"] = loads0["R_Thumb"]
        print("  at lift start, load (strongest joint): "
              + ", ".join(f"{g.replace('R_', '')}={loads0[g]:.2f}" for g in FINGER_GROUPS))
        stamp = getattr(node, "joint_stamp", None)
        if stamp is not None:
            REC["sim_lift_start"] = stamp
            print(f"  [phase] lift started at sim t={stamp:.3f}s")
        node.send_arm_trajectory(lift[0], LIFT_SECONDS)
        trace = []
        slip_at = None
        # Lists, so the watch loop can update them without needing `nonlocal`.
        regrip_done, regrip_last, regrip_count = [0.0], [None], [0]
        arm_peak = {jn: 0.0 for jn in ARM_JOINTS}
        t_start = time.time()
        sim_start = getattr(node, "joint_stamp", None)
        last_rise_t, last_rise_h = t_start, 0.0
        last_rise_sim = 0.0
        last_clock_move, last_sim_seen = t_start, None
        use_sim = sim_start is not None
        while (time.time() - t_start < (LIFT_WALL_CAP_S if use_sim else LIFT_WATCH_S)
               and w0 is not None and a0 is not None):
            for _ in range(2):
                rclpy.spin_once(node, timeout_sec=LIFT_SAMPLE_S / 2)
            w = node.real_wrist_position()
            a = apple_xyz(node)
            if w is None or a is None:
                continue
            now_t = time.time()
            hand_rise = (w[2] - w0[2]) * 1000.0
            apple_rise = (a[2] - a0[2]) * 1000.0
            lagging = hand_rise - apple_rise
            loads = finger_loads(node)
            n_loaded = sum(1 for g in FINGER_GROUPS
                           if g != "R_Thumb" and loads[g] > GRIP_HOLD_NM)
            for jn in ARM_JOINTS:
                eff = abs(node.latest_joint_state.get(jn, (0, 0, 0))[2] or 0.0)
                arm_peak[jn] = max(arm_peak[jn], eff)
            lift_eff = abs(node.latest_joint_state.get('shoulder_lift_joint', (0, 0, 0))[2] or 0.0)
            w1_eff = abs(node.latest_joint_state.get('wrist_1_joint', (0, 0, 0))[2] or 0.0)
            sim_now = getattr(node, "joint_stamp", None)
            sim_t = (sim_now - sim_start) if (sim_now is not None and sim_start is not None) else None
            trace.append((now_t - t_start, hand_rise, apple_rise, lagging,
                          n_loaded, loads["R_Thumb"], lift_eff, w1_eff, sim_t))
            if slip_at is None and lagging > SLIP_MM:
                slip_at = (hand_rise, n_loaded, loads["R_Thumb"], now_t - t_start)
            # Re-tighten a finger that has lost its load, while the apple is still in the
            # hand. Position-held fingers cannot recover contact on their own.
            if (REGRIP and LAST_FINGER_CMD and lagging < SLIP_MM
                    and regrip_done[0] < REGRIP_MAX
                    and (sim_t is None or regrip_last[0] is None
                         or sim_t - regrip_last[0] >= REGRIP_MIN_GAP_S)):
                slack = [g for g in FINGER_GROUPS
                         if contacted.get(g) and loads[g] < (GRIP_MIN_THUMB_NM
                                                             if g == "R_Thumb"
                                                             else GRIP_HOLD_NM)]
                if slack:
                    for g in slack:
                        LAST_FINGER_CMD[g] = min(LAST_FINGER_CMD[g] + REGRIP_STEP,
                                                 MAX_PITCH_CEILING)
                    node.command_fingers(LAST_FINGER_CMD, STEP_COMMAND_TIME,
                                         thumb_yaw=THUMB_GRASP_YAW, thumb_roll=THUMB_ROLL)
                    regrip_done[0] += REGRIP_STEP
                    regrip_last[0] = sim_t
                    regrip_count[0] += 1
            if use_sim and sim_t is not None:
                if last_sim_seen is None or sim_t > last_sim_seen + 1e-6:
                    last_sim_seen, last_clock_move = sim_t, now_t
                elif now_t - last_clock_move > LIFT_FROZEN_WALL_S:
                    break           # the simulation clock has stopped
                if hand_rise - last_rise_h > 2.0:
                    last_rise_h, last_rise_sim = hand_rise, sim_t
                elif (sim_t > LIFT_SIM_DONE_S
                      and sim_t - max(last_rise_sim, LIFT_SIM_DONE_S - LIFT_SIM_STILL_S)
                      > LIFT_SIM_STILL_S):
                    break
            elif hand_rise - last_rise_h > 2.0:
                last_rise_t, last_rise_h = now_t, hand_rise
            elif now_t - last_rise_t > LIFT_STILL_S and now_t - t_start > 4.0:
                break
        settle(node, 5.0)
        if trace:
            print("    time  sim time   hand up  apple up  apple behind  fingers loaded   thumb  "
                  "shoulder  wrist_1")
            shown_t, shown_sim = -99.0, -99.0
            for row in trace:
                t_, h_, a_, lag_, nl_, th_, se_, w1_, st_ = row
                # One row per 0.25s of simulated time (or 2s of wall time without a clock),
                # so a slow simulation does not print hundreds of identical rows.
                due = (st_ - shown_sim >= 0.25) if st_ is not None else (t_ - shown_t >= 2.0)
                if due or row is trace[-1]:
                    st_txt = "-" if st_ is None else f"{st_:.2f}s"
                    print(f"    {t_:5.1f}s  {st_txt:>8}  {h_:+6.0f}mm  {a_:+6.0f}mm  {lag_:+8.0f}mm  "
                          f"{nl_:>10}/4  {th_:5.2f}Nm  {se_:5.0f}Nm  {w1_:5.1f}Nm")
                    shown_t, shown_sim = t_, (st_ if st_ is not None else shown_sim)
            final = trace[-1]
            REC["hand_rise"] = final[1] / 1000.0
            REC["lift_time"] = final[0]
            REC["final_lag"] = final[3] / 1000.0
            REC["lift_sim_time"] = final[8]
            # How much of the lift wrist_1 spent pinned at its limit: a brief spike and a
            # joint held at its limit throughout are different problems.
            REC["wrist1_pinned"] = (sum(1 for row in trace if row[7] >= 0.98 * 28.0)
                                    / float(len(trace)))
            if final[8] is not None and final[0] > 0:
                REC["rtf"] = final[8] / final[0]
            sim_txt = ("" if final[8] is None else
                       f" ({final[8]:.1f}s of simulated time; the simulation ran at "
                       f"{final[8] / max(final[0], 1e-6):.3f}x real time)")
            REC["regrip_count"] = regrip_count[0]
        REC["regrip_rad"] = regrip_done[0]
        if REGRIP:
            print(f"  re-gripped {regrip_count[0]} times during the lift "
                  f"({regrip_done[0]:.3f} rad of extra closing in total)")
        print(f"  hand rose {final[1]:.0f}mm of the commanded {LIFT_HEIGHT * 1000:.0f}mm "
                  f"in {final[0]:.1f}s{sim_txt}, commanded to take 3.0s; wrist_1 at its limit "
                  f"in {REC['wrist1_pinned'] * 100:.0f}% of samples")
        saturated = [jn for jn in ARM_JOINTS
                     if arm_peak[jn] >= 0.98 * ARM_EFFORT_LIMIT[jn]]
        REC["shoulder_peak"] = arm_peak['shoulder_lift_joint']
        REC["wrist1_peak"] = arm_peak['wrist_1_joint']
        REC["arm_saturated"] = ", ".join(jn.replace("_joint", "") for jn in saturated)
        print("  arm joint peak effort during the lift: "
              + ", ".join(f"{jn.replace('_joint', '')}={arm_peak[jn]:.0f}Nm"
                          f"{' (AT LIMIT)' if jn in saturated else ''}" for jn in ARM_JOINTS))
        if slip_at is not None:
            REC["slip_at"] = slip_at[0] / 1000.0
            REC["fingers_at_slip"] = slip_at[1]
            print(f"  the apple started falling behind the hand {slip_at[3]:.1f}s into the "
                  f"lift, when the hand had risen {slip_at[0]:.0f}mm, with {slip_at[1]}/4 "
                  f"fingers loaded and the thumb at {slip_at[2]:.2f}Nm")
        else:
            print("  the apple never fell more than "
                  f"{SLIP_MM:.0f}mm behind the hand during the whole lift")
        for _ in range(10):
            rclpy.spin_once(node, timeout_sec=0.1)
        after_lift = apple_xyz(node)
        wrist_after = node.real_wrist_position()
        if before is not None and after_lift is not None:
            lifted = float(after_lift[2] - before[2])
            REC["lift"] = lifted
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
            "thumb_yaw": THUMB_GRASP_YAW, "palm_offset": palm_offset, "lateral": lateral,
            "ok": True, "contacts": n, "peak": peak,
            "moved": moved, "lifted": lifted}


def main():
    target_name = sys.argv[1] if len(sys.argv) > 1 else "apple_06"
    if target_name not in APPLE_HOME_WORLD_XY:
        print(f"Unknown target {target_name}")
        return
    # Optional second argument: how many of the CASES to run (e.g. 2 for a quicker check
    # across many apples). Default: all of them.
    cases = CASES
    squeeze_override = None
    recentre_mode = "off"
    thumb_arg = "spike"
    spread_arg = "hold"
    lateral_list = None
    pace_list = None
    lift_list = None
    regrip_mode = "off"
    variants = None
    for arg in sys.argv[2:]:
        # lift=6 or lift=3,6: how long the lift is commanded to take (s); with several
        # values the attempts cycle through them.
        if arg.startswith("lift="):
            try:
                lift_list = [float(v) for v in arg.split("=", 1)[1].split(",") if v]
            except ValueError:
                print(f"lift= needs seconds, e.g. lift=3,6, got {arg!r}")
                return
            if not lift_list:
                print(f"lift= needs at least one value, got {arg!r}")
                return
            continue
        # regrip=on|off|ab: re-tighten a finger that loses its load during the lift.
        if arg.startswith("regrip="):
            regrip_mode = arg.split("=", 1)[1]
            if regrip_mode not in ("on", "off", "ab"):
                print(f"regrip= must be on, off or ab, got {arg!r}")
                return
            continue
        # variants=drop:0.016/preshape:0.3/drop:0.016+tilt:55 -- one grasp change per
        # attempt, cycling in order. Keys: drop (m lower), preshape (rad), lateral (m),
        # offset (palm back-off, m), tilt (deg), relax (1: ease the thumb to THUMB_LIFT_NM
        # before lifting). Anything not named keeps its tested value.
        if arg.startswith("variants="):
            keys = {"drop", "preshape", "lateral", "offset", "tilt", "relax"}
            variants = []
            try:
                for chunk in arg.split("=", 1)[1].split("/"):
                    v = {}
                    for kv in chunk.split("+"):
                        k, val = kv.split(":")
                        if k not in keys:
                            raise ValueError(k)
                        v[k] = float(val)
                    variants.append(v)
            except ValueError as e:
                print(f"variants= format is key:value+key:value/..., keys {sorted(keys)}; "
                      f"problem: {e}")
                return
            continue
        # pace=0.04 or pace=0.04,0.02 or pace=msgs,0.04: how long each closing step waits,
        # in seconds of simulated time ("msgs" = the tested message-count pacing); with
        # several values the attempts cycle through them.
        if arg.startswith("pace="):
            try:
                pace_list = [None if v == "msgs" else float(v)
                             for v in arg.split("=", 1)[1].split(",") if v]
            except ValueError:
                print(f"pace= needs seconds or msgs, e.g. pace=0.04,0.02, got {arg!r}")
                return
            if not pace_list:
                print(f"pace= needs at least one value, got {arg!r}")
                return
            continue
        # lateral=0.025 or lateral=0.025,0.035: aim this far across the hand (m) instead of
        # LATERAL; with several values the attempts cycle through them in order.
        if arg.startswith("lateral="):
            try:
                lateral_list = [float(v) for v in arg.split("=", 1)[1].split(",") if v]
            except ValueError:
                print(f"lateral= needs metres, e.g. lateral=0.025,0.035, got {arg!r}")
                return
            if not lateral_list:
                print(f"lateral= needs at least one value, got {arg!r}")
                return
            continue
        # spread=hold|zero|ab: leave the finger spread joints uncommanded (hold, the tested
        # behaviour) or command them to 0.0. "ab" runs the first half of the attempts on
        # hold and the second half on zero -- once commanded they cannot go back to hold.
        if arg.startswith("spread="):
            spread_arg = arg.split("=", 1)[1]
            if spread_arg not in ("hold", "zero", "ab"):
                print(f"spread= must be hold, zero or ab, got {arg!r}")
                return
            continue
        # thumb=spike|steady|ab: how the thumb preload decides it is pressing (see
        # THUMB_MODE); "ab" alternates steady/spike attempt by attempt.
        if arg.startswith("thumb="):
            thumb_arg = arg.split("=", 1)[1]
            if thumb_arg not in ("spike", "steady", "ab"):
                print(f"thumb= must be spike, steady or ab, got {arg!r}")
                return
            continue
        # recentre=on|off|ab: re-centre the hand on the apple before closing; "ab"
        # alternates on/off attempt by attempt so both are compared in one run.
        if arg.startswith("recentre="):
            recentre_mode = arg.split("=", 1)[1]
            if recentre_mode not in ("on", "off", "ab"):
                print(f"recentre= must be on, off or ab, got {arg!r}")
                return
            continue
        # squeeze=0.08: squeeze this far (rad) past first contact instead of the tested
        # SQUEEZE_EXTRA. Used to find the range the vision layer may choose from.
        if arg.startswith("squeeze="):
            try:
                squeeze_override = float(arg.split("=", 1)[1])
            except ValueError:
                print(f"squeeze= needs a number in radians, got {arg!r}")
                return
            continue
        try:
            n_cases = max(1, int(arg))
            cases = [CASES[i % len(CASES)] for i in range(n_cases)]
        except ValueError:
            print(f"Arguments: [number of attempts] [squeeze=RAD], got {arg!r}")
            return
    before_close = None
    if squeeze_override is not None:
        print(f"Squeeze past first contact set to {squeeze_override:.3f} rad "
              f"(tested value {SQUEEZE_EXTRA:.3f})")
        before_close = lambda node: {"squeeze_extra": squeeze_override}  # noqa: E731

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

    global CONTACT_THRESHOLD_BY_FINGER, CLOSE_STEP_SIM_S, PALM_OFFSET
    global LIFT_SECONDS, REGRIP
    measured = calibrate_contact_threshold(node)
    if measured is not None:
        CONTACT_THRESHOLD_BY_FINGER = measured
    print_finger_spread(node, "at startup")

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
    records = []
    for case_i, (pt, ps, lat, po, dr, cap, emp, rel, fin) in enumerate(cases):
        rc = recentre_mode == "on" or (recentre_mode == "ab" and case_i % 2 == 0)
        tm = thumb_arg if thumb_arg != "ab" else ("steady" if case_i % 2 == 0 else "spike")
        zero = spread_arg == "zero" or (spread_arg == "ab" and case_i >= len(cases) // 2)
        node.finger_yaw = 0.0 if zero else None
        if lateral_list is not None:
            lat = lateral_list[case_i % len(lateral_list)]
        PALM_OFFSET = TESTED_PALM_OFFSET
        variant_s = "tested"
        if variants:
            var = variants[case_i % len(variants)]
            dr = var.get("drop", dr)
            ps = var.get("preshape", ps)
            lat = var.get("lateral", lat)
            if "tilt" in var:
                pt = np.radians(var["tilt"])
            PALM_OFFSET = var.get("offset", TESTED_PALM_OFFSET)
            rel = bool(var.get("relax", rel))
            variant_s = "+".join(f"{k}:{v:g}" for k, v in var.items())
        print(f"\n(grasp variant: {variant_s} -> tilt {np.degrees(pt):.0f}deg, "
              f"lowered {dr * 1000:.0f}mm, pre-shape {ps:.2f}, lateral {lat * 1000:+.0f}mm, "
              f"back-off {PALM_OFFSET:.3f}m, thumb eased before lift: {'yes' if rel else 'no'})")
        if lift_list is not None:
            LIFT_SECONDS = lift_list[case_i % len(lift_list)]
        REGRIP = regrip_mode == "on" or (regrip_mode == "ab" and case_i % 2 == 0)
        print(f"\n(lift commanded over {LIFT_SECONDS:.1f}s; re-grip during the lift: "
              f"{'ON' if REGRIP else 'off'})")
        if pace_list is not None:
            CLOSE_STEP_SIM_S = pace_list[case_i % len(pace_list)]
        pace_s = "msgs" if CLOSE_STEP_SIM_S is None else f"{CLOSE_STEP_SIM_S * 1000:.0f}ms"
        print(f"\n(closing step pace: {pace_s}; "
              f"re-centre before closing: {'ON' if rc else 'off'}; thumb preload: {tm}; "
              f"finger spread: {'commanded 0.0' if zero else 'uncommanded'}; "
              f"lateral aim: {lat * 1000:+.0f}mm)")
        r = attempt(node, target_name, pt, ps, lat, po, dr, cap, emp, rel, fin,
                    before_close=before_close, recentre=rc, thumb_mode=tm)
        REC["spread"] = "zero" if zero else "hold"
        REC["lateral"] = lat
        REC["variant"] = variant_s
        REC["pace"] = pace_s
        REC["lift_s"] = LIFT_SECONDS
        REC["regrip"] = "on" if REGRIP else "off"
        results.append(r)
        records.append(dict(REC))
        if r.get("dead_sim"):
            print()
            print("Stopping the remaining attempts -- the simulation died mid-run.")
            break

    def mm(v):
        return "-" if v is None else f"{v * 1000:.0f}mm"

    rows = []
    for idx, (r, rec) in enumerate(zip(results, records), 1):
        if r.get("unsettled"):
            result, why = "SKIPPED", "the apple would not stop moving at reset"
        elif r.get("dead_sim"):
            result, why = "DEAD SIM", "the simulation stopped publishing joint states"
        elif r.get("knocked"):
            result, why = "KNOCKED", f"apple knocked away by {r['knocked_by']}"
        elif r.get("sim_frozen"):
            result, why = "SIM FROZE", ("the simulation clock stopped during the grasp move; "
                                        "restart Gazebo and rerun this apple")
        elif r.get("not_positioned"):
            result, why = "HELD BACK", r["reason"]
        elif not r.get("ok"):
            result, why = "UNREACHABLE", "the arm cannot reach the target"
        elif (rec.get("lift_sim_time") is not None and rec["lift_sim_time"] < LIFT_SIM_DONE_S
              and rec.get("lift_sim_time") >= 0.5 and r.get("lifted") is None):
            result = "SIM TOO SLOW"
            why = (f"the arm had only {rec['lift_sim_time']:.1f}s of its 3.0s lift in "
                   f"{rec.get('lift_time', 0):.0f}s of watching (the simulation ran at "
                   f"{rec.get('rtf', 0):.3f}x real time); apple rose "
                   f"{(rec.get('lift') or 0) * 100:+.1f}cm so far, "
                   f"{'still in the hand' if (rec.get('final_lag') or 1) * 1000 <= FOLLOW_MM else 'falling behind'}")
        elif rec.get("lift_sim_time") is not None and rec["lift_sim_time"] < 0.5:
            # The simulation's own clock did not advance during the lift: Gazebo froze.
            # One attempt was scored ARM STALLED with 0.0s of simulated time over 8s of
            # watching, and the reset after it then waited out a full timeout.
            result = "SIM FROZE"
            why = (f"the simulation clock advanced only {rec['lift_sim_time']:.1f}s during "
                   f"{rec.get('lift_time', 0):.0f}s of watching the lift -- nothing about "
                   f"the grasp or the arm can be concluded")
        elif rec.get("empty"):
            hr, w1 = rec.get("hand_rise"), rec.get("wrist1_peak")
            result = "CONTROL"
            why = (f"empty hand rose {'-' if hr is None else f'{hr * 100:.1f}cm'} in "
                   f"{rec.get('lift_time', 0):.1f}s, wrist_1 peak "
                   f"{'-' if w1 is None else f'{w1:.0f}Nm'} (limit 28Nm)")
        elif r.get("lifted") is not None:
            result, why = "PICKED", f"apple held through the {LIFT_HEIGHT:.2f}m lift"
        elif (rec.get("final_lag") is not None and rec.get("contacts")
              and rec["final_lag"] * 1000.0 <= FOLLOW_MM
              and (rec.get("hand_rise") or 0.0) < LIFT_HEIGHT * 0.5):
            # The apple stayed in the hand, but the hand did not go up far enough. This
            # used to be reported as NOT HELD, blaming the grasp for the arm's failure.
            w1 = rec.get("wrist1_peak")
            result = "ARM STALLED"
            why = (f"the apple stayed in the hand the whole lift (ended "
                   f"{rec['final_lag'] * 1000:.0f}mm behind it), but the arm stopped after "
                   f"raising it {rec['hand_rise'] * 100:.1f}cm; wrist_1 peak "
                   f"{'-' if w1 is None else f'{w1:.0f}Nm'} (limit 28Nm)")
        else:
            touched = rec.get("contacts")
            gripping = rec.get("holding")
            lift = rec.get("lift")
            parts = []
            if touched is not None:
                parts.append(f"{touched}/5 fingers touched the apple")
            if rec.get("below"):
                parts.append(f"{rec['below']} of those under its widest part")
            if gripping is not None:
                parts.append(f"{gripping}/5 still gripping after the squeeze")
            if lift is not None:
                parts.append(f"apple rose {lift * 100:+.1f}cm (needs "
                             f"+{LIFT_HEIGHT * 50:.1f}cm)")
            result, why = "NOT HELD", ", ".join(parts) or "the grip did not hold"
        rows.append((idx, rec, result, why))

    bar = "=" * 96
    print(f"\n{bar}\nRESULTS\n{bar}")
    print("\n1) GETTING THE HAND TO THE APPLE")
    print(f"{'#':>2}  {'lift':>5}  {'regrip':>6}  {'variant':<24}  {'pace':>5}  "
          f"{'lateral':>7}  {'spread':>6}  {'thumb':>6}  {'thumb deg':>9}  "
          f"{'recentre':>8}  {'apple still':>11}  {'1st move off':>12}  "
          f"{'after fixing':>12}  {'approach':>9}  {'apple moved':>11}  {'RESULT':<10}")
    for idx, rec, result, why in rows:
        opp = rec.get("opposition")
        opp_s = f"{opp:.0f}" if isinstance(opp, (int, float)) else "-"
        lat_v = rec.get("lateral")
        lat_s = f"{lat_v * 1000:+.0f}mm" if isinstance(lat_v, (int, float)) else "-"
        lift_s = rec.get("lift_s")
        print(f"{idx:>2}  {('-' if lift_s is None else f'{lift_s:.1f}s'):>5}  "
              f"{rec.get('regrip', '-'):>6}  "
              f"{rec.get('variant', '-'):<24}  {rec.get('pace', '-'):>5}  {lat_s:>7}  {rec.get('spread', '-'):>6}  {rec.get('thumb', '-'):>6}  "
              f"{opp_s:>9}  {rec.get('recentre', '-'):>8}  "
              f"{rec.get('settled', '-'):>11}  {mm(rec.get('first_err')):>12}  "
              f"{mm(rec.get('pre_err')):>12}  {rec.get('steps', '-'):>9}  "
              f"{mm(rec.get('apple_moved')):>11}  {result:<10}")

    print("\n2) THE GRASP AND THE LIFT (only filled in when the hand reached the apple)")
    print(f"{'#':>2}  {'touched':>7}  {'under widest':>12}  {'hand rose':>9}  "
          f"{'took':>6}  {'shoulder':>8}  {'wrist_1':>7}  {'slips after':>11}  "
          f"{'apple rose':>10}  {'RESULT':<11}")
    for idx, rec, result, why in rows:
        touched = rec.get("contacts")
        hr = rec.get("hand_rise")
        lt = rec.get("lift_time")
        sp = rec.get("shoulder_peak")
        w1 = rec.get("wrist1_peak")
        sl = rec.get("slip_at")
        lift = rec.get("lift")
        slip_txt = ("-" if touched is None
                    else "never" if sl is None else f"{sl * 100:.1f}cm up")
        print(f"{idx:>2}  {('-' if touched is None else f'{touched}/5'):>7}  "
              f"{rec.get('below', '-'):>12}  "
              f"{('-' if hr is None else f'{hr * 100:.1f}cm'):>9}  "
              f"{('-' if lt is None else f'{lt:.1f}s'):>6}  "
              f"{('-' if sp is None else f'{sp:.0f}Nm'):>8}  "
              f"{('-' if w1 is None else f'{w1:.0f}Nm'):>7}  "
              f"{slip_txt:>11}  "
              f"{('-' if lift is None else f'{lift * 100:+.1f}cm'):>10}  {result:<11}")

    print("\nWHAT HAPPENED IN EACH ATTEMPT")
    for idx, rec, result, why in rows:
        print(f"  {idx}. [thumb cap {rec.get('cap', '-')}] {result}: {why}")

    print("\nHOW TO READ THIS")
    print("  recentre         -- on: the hand was shifted onto the apple's measured position "
          "before closing")
    print("  thumb cap        -- on: the thumb is backed off until it pushes no more than "
          f"{THUMB_PRELOAD_FORCE:.1f}Nm")
    print(f"  hand rose / took -- how far the hand actually rose, and how long it took "
          f"(commanded: {LIFT_HEIGHT * 100:.0f}cm in {LIFT_SECONDS:.1f}s)")
    print("  sim time         -- how long the lift took in simulated time (wall time is longer "
          "when Gazebo runs slower than real time)")
    print("  shoulder/wrist_1 -- peak effort on those arm joints during the lift "
          "(limits 150Nm and 28Nm)")
    print("  ARM STALLED      -- the apple stayed in the hand, but the arm stopped lifting")
    print("  SIM TOO SLOW     -- the watch ran out before the arm had its whole lift in "
          "simulated time")
    print("  SIM FROZE        -- Gazebo's clock stopped during the lift; the attempt tells us "
          "nothing -- restart the simulation")
    print("  CONTROL          -- same attempt with the apple removed, to see if the arm "
          "stalls anyway")
    print("  slips after      -- how far the hand had risen when the apple started falling "
          f"behind it by {SLIP_MM:.0f}mm")
    print("  apple still      -- the apple was resting still before the attempt began")
    print(f"  1st move off     -- how far the first rough move missed (normal: 50-90mm)")
    print("  after fixing     -- how far off after correction. pick: from the grasp pose "
          f"(stops under {PICK_WRIST_TOLERANCE * 1000:.0f}mm); servo: from the pre-grasp "
          f"pose (must be under {APPROACH_GATE * 1000:.0f}mm to approach)")
    print(f"  approach         -- how many of the {APPROACH_STEPS} final 10mm steps were "
          f"completed")
    print("  apple moved      -- how far the apple moved during the whole attempt")
    print("  under widest     -- fingers that met the apple BELOW its middle "
          "(needed to lift it)")
    print(f"  apple rose       -- how far the apple rose; +{LIFT_HEIGHT * 50:.1f}cm or "
          f"more counts as held")

    picked = sum(1 for _, _, result, _ in rows if result == "PICKED")
    squeeze_note = ("" if squeeze_override is None
                    else f" (squeeze {squeeze_override:.3f} rad)")
    print(f"\nPICKED {picked} OF {len(rows)} ATTEMPTS ON {target_name}{squeeze_note}.")
    lifts = [rec["lift"] for _, rec, _, _ in rows
             if rec.get("lift") is not None and not rec.get("empty")]
    if lifts:
        print(f"  grasp attempts lifted {', '.join(f'{x * 100:+.1f}cm' for x in lifts)} "
              f"-- a spread of {(max(lifts) - min(lifts)) * 100:.1f}cm")
    for setting in ("on", "off"):
        grp = [rec for _, rec, _, _ in rows
               if not rec.get("empty") and rec.get("relax") == setting
               and rec.get("hand_rise") is not None]
        if grp:
            rises = ", ".join("%.1fcm" % (r_["hand_rise"] * 100) for r_ in grp)
            wrists = ", ".join("%.0fNm" % r_.get("wrist1_peak", 0.0) for r_ in grp)
            apples = ", ".join("%+.1fcm" % (r_.get("lift", 0.0) * 100) for r_ in grp)
            print(f"  thumb eased {setting:>3}: wrist_1 peak {wrists}; hand rose {rises}; "
                  f"apple rose {apples}")
    for _, rec, _, _ in sorted(rows, key=lambda r: -(r[1].get("lift") or -9)):
        if rec.get("lift") is None or rec.get("empty"):
            continue
        hr, lt, sl = rec.get("hand_rise"), rec.get("lift_time"), rec.get("slip_at")
        st, pin = rec.get("lift_sim_time"), rec.get("wrist1_pinned")
        print(f"    apple rose {rec['lift'] * 100:+.1f}cm: hand rose "
              f"{'-' if hr is None else f'{hr * 100:.1f}cm'} in "
              f"{'-' if lt is None else f'{lt:.1f}s'}"
              f"{'' if st is None else f' ({st:.1f}s simulated)'}, wrist_1 at limit "
              f"{'-' if pin is None else f'{pin * 100:.0f}%'} of the lift, shoulder peak "
              f"{rec.get('shoulder_peak', 0):.0f}Nm"
              f"{' (saturated: ' + rec['arm_saturated'] + ')' if rec.get('arm_saturated') else ''}, "
              f"slipped {'never' if sl is None else f'after {sl * 100:.1f}cm of hand rise'}")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
