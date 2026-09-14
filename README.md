# UR5e + DexHand v2 Apple Picking

A UR5e arm fitted with a 21-joint DexHand v2 picks apples off a table in Gazebo
(Ignition Fortress) with ROS 2 Humble, on WSL2. The apple is held by **real friction
only**: no attach plugin or "glue". A gripper camera feeds a local Qwen2.5-VL model
(through Ollama), which estimates the apple's fragility for the grip-force layer.

## Current status (14 Sep 2026)

**All 10 apples have been picked with one set of grasp settings: 29 of 30 attempts.**
The one miss was a Gazebo freeze, not a failed grasp. No apple slipped in any attempt.

| Apple | Diameter | Mass | Friction | Picked | Apple rose (hand rose 15.0 cm) |
|---|---|---|---|---|---|
| apple_01 | 10.0 cm | 0.41 kg | 0.80 | 4 / 4 | +15.1 to +15.2 cm |
| apple_02 | 10.2 cm | 0.40 kg | 0.85 | 2 / 2 | +15.1 to +15.2 cm |
| apple_03 | 10.4 cm | 0.48 kg | 0.75 | 1 / 2 (miss = Gazebo froze) | +15.2 cm |
| apple_04 | 10.7 cm | 0.47 kg | 0.90 | 2 / 2 | +14.9 to +15.3 cm |
| apple_05 | 10.9 cm | 0.47 kg | 0.70 | 2 / 2 | +15.2 to +15.3 cm |
| apple_06 | 11.1 cm | 0.54 kg | 0.85 | 8 / 8 | +15.1 to +15.2 cm |
| apple_07 | 11.3 cm | 0.55 kg | 0.95 | 2 / 2 | +15.3 cm |
| apple_08 | 11.5 cm | 0.65 kg | 0.78 | 2 / 2 | +14.4 to +14.8 cm |
| apple_09 | 11.8 cm | 0.59 kg | 0.90 | 2 / 2 | +14.8 to +15.1 cm |
| apple_10 | 12.0 cm | 0.70 kg | 0.82 | 4 / 4 | +14.2 to +14.4 cm |

These results come from `pocket_grasp_test.py` (section 2). The working settings
have **not yet been moved into** the main pipeline, `full_layer_grasp.py`, so that
script does not reproduce these results yet.

### The grasp that works
- **Aim:** the centre of the *closed* fingertips goes to the apple's live centre,
  8 mm below it, with a +15 mm sideways offset. The hand follows each apple's size.
- **Palm** tilted 45° from straight down. **Approach:** from 0.16 m back along the hand,
  then in, so the fingers arrive beside the apple rather than on top of it.
- **Wait for the grasp move to finish** (in simulated time) before measuring the
  wrist. The arm then lands on target and no corrections are needed.
- **Fingers** pre-shaped to 0.4 rad. The thumb (yaw −0.50 rad) closes first as a
  backstop, with its preload capped at 1.2 Nm. The four fingers then close to contact
  and squeeze 0.12 rad further. Finger joints are capped at 1.5 Nm.
- **Lift** 15 cm and watch until the arm has had the whole move in simulated time.

### Things that matter on this machine
- **Gazebo runs at about 1.5–9% of real time**, and slower still while the hand
  squeezes hard. A 3 s arm move can take 40–240 s on the clock. Everything that
  waits for the arm uses the simulation clock (`/joint_states` timestamps), never the
  wall clock. Wall-clock waits caused the false "arm stalled" results and the
  first-attempt failures found on 13–14 Sep.
- **Gazebo sometimes freezes** (the simulation clock stops). The test detects this
  and reports `SIM FROZE` instead of blaming the grasp. Restart Gazebo when you see it.

The full debugging history, measurements and novelty discussion are in
`Apple_Gripper_Progress_Report.pdf`.

---

## How to run

All commands run inside WSL2, in `~/ur_gz_ws`.

**Every new terminal** needs ROS sourced before any `ros2 ...` command will see
anything. An empty `ros2 topic list` almost always means this step was skipped:
```bash
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
```

### 1. Start the simulation (Terminal 1)

```bash
pkill -9 -f "ign gazebo"; pkill -9 -f "gz sim"; pkill -9 -f ros2_control_node; pkill -9 -f parameter_bridge; pkill -9 -f "ros2 launch"
cd ~/ur_gz_ws && git pull
bash start_everything.sh          # headless: faster
bash start_everything.sh gui      # with Gazebo's 3D window: slower
```

This kills leftover processes and regenerates `/tmp/real_robot_exact.urdf` (the static
URDF the IK code loads; WSL2 clears `/tmp` on reboot). It then launches Gazebo,
activates the hand controller, and starts the camera and apple-pose bridges.

Wait for `=== SETUP COMPLETE. Verifying... ===`, then check:
- `dexhand_controller` and `joint_trajectory_controller` are `active`
- all 10 `/model/apple_XX/pose` topics, `/gripper_camera` and `/overhead_camera` are listed

If topics are missing, wait a few seconds and re-check. It is usually a startup race:
```bash
ros2 topic list | grep -E "model/apple|gripper_camera|overhead_camera"
```

### 2. Grasp test on one apple (Terminal 2)

```bash
cd ~/ur_gz_ws && source /opt/ros/humble/setup.bash && source install/setup.bash
cd src/my_pick_and_place/scripts
python3 pocket_grasp_test.py apple_06        # 4 attempts
python3 pocket_grasp_test.py apple_06 2      # 2 attempts
```

Each attempt resets the robot and apple, approaches, closes the hand, lifts 15 cm and
watches the lift. The run ends with two results tables (reaching the apple; grasp and
lift), a verdict and reason per attempt, and a legend. An attempt takes about 2–4
minutes; a slow lift can take several. **Don't press Ctrl+C** just because the terminal
looks still.

| Verdict | Meaning |
|---|---|
| `PICKED` | apple rose at least 7.5 cm and ended within 0.20 m of the wrist |
| `NOT HELD` | the grasp did not carry the apple |
| `ARM STALLED` | apple stayed in the hand but the arm stopped rising |
| `SIM TOO SLOW` | the watch ran out before the arm had its whole lift in simulated time |
| `SIM FROZE` | Gazebo's clock stopped; the attempt means nothing, so restart Gazebo |
| `SKIPPED` | the apple would not settle at reset |

### 3. Grasp test on several apples

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
bash test_all_apples.sh                      # apples 02-05 and 07-09, 2 attempts each
bash test_all_apples.sh 4 apple_03 apple_05  # chosen apples, 4 attempts each
```

Prints one summary line per apple at the end. Full logs are saved to
`/tmp/pocket_<apple>.log`.

### 4. Full pipeline on one apple

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 full_layer_grasp.py apple_06
```

VLM analysis → arm positioning → finger closing → lift → place in crate. This does not
yet use the settings from `pocket_grasp_test.py` (see Current status).

### 5. VLM layer (optional, separate terminal)

```bash
cd ~/vlm_scripts
source /opt/ros/humble/setup.bash
python3 vlm_fragility_node.py
ros2 topic echo /gripper_camera/fragility_analysis
```

### 6. Other tools

| Script | What it does |
|---|---|
| `diagnostic_grasp_attempt.py apple_06` | One attempt with forgiving settings: is a grasp possible at all? |
| `train_closing_policy.py apple_06` | Small (1+λ) evolution-strategy search over per-finger closing parameters, scored by real lift height. Logs to `~/ur_gz_ws/closing_policy_runs.json`. Flags: `--generations N`, `--lambda N` |
| `rviz_world_markers.py` | Publishes the table and apples as RViz markers on `/world_markers`. It doesn't slow Gazebo down the way its GUI does |
| `thumb_opposition_check.py`, `finger_geometry_check.py` | Measure the hand's real fingertip geometry from TF |
| `tools/check_scopes.py` | Pre-commit check for names used but not defined in a function, and for local variables that hide a module-level function (both reached live runs before) |

Before committing changes to the scripts:
```bash
python3 tools/check_scopes.py src/my_pick_and_place/scripts/*.py
```

---

## Repository layout

| Path | Contents |
|---|---|
| `src/my_pick_and_place/scripts/` | All grasp, test and diagnostic code (original to this project) |
| `src/my_pick_and_place/urdf/` | UR5e + DexHand xacro and the merged controller config |
| `src/apple_gripper_sim/` | Apple world and the 10 apple models (each with a flat base so it cannot roll) |
| `src/dexhandv2_description/` | DexHand v2 URDF (finger joint effort capped at 1.5 Nm) |
| `src/ur_simulation_gz/` | UR Gazebo simulation launch files |
| `vlm_scripts/` | Qwen2.5-VL fragility node |
| `tools/` | Development checks |
| `start_everything.sh` | One-command simulation startup |
| `Apple_Gripper_Progress_Report.pdf` | Project report: design, full debugging timeline, results, novelty |

## Troubleshooting

**`SIM FROZE` / the terminal shows no progress for many minutes**: Gazebo has stalled.
Restart Terminal 1 (section 1) and rerun that apple.

**Apple flung far off the table**: the apple's physics got corrupted. The test respawns
an apple that keeps moving after teleports; if that fails too, restart Gazebo.

**Topics missing right after `start_everything.sh`**: startup race. Re-run the
`ros2 topic list` check before assuming something is broken.

**`FileNotFoundError` for `/tmp/real_robot_exact.urdf`**: `start_everything.sh` has not
run in this WSL session. Run it, or regenerate the file directly:
```bash
xacro /home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro > /tmp/real_robot_exact.urdf
```

**`git pull` says "Already up to date" but the output looks old**: check with
`git log --oneline -1` that you are on the latest commit.
