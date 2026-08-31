# UR5e + DexHand v2 Apple Picking — How to Run

All commands run inside WSL2, in `~/ur_gz_ws`.

**Every new terminal** needs ROS sourced before any `ros2 ...` command will
see anything (an empty `ros2 topic list` almost always means this step was
skipped, not that something is broken):
```bash
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
```

## 1. Start everything (normal way)

```bash
cd ~/ur_gz_ws
bash start_everything.sh
```

This kills any leftover processes, regenerates `/tmp/real_robot_exact.urdf`
(the static URDF the IK code loads — WSL2 clears `/tmp` on every reboot, so
this must happen every fresh session or the grasp/diagnostic/training scripts
fail immediately with `FileNotFoundError`), launches Gazebo **headless** (no
GUI window — see the GUI note below), activates the hand controller, and
starts the camera + apple pose bridges.

Wait for `=== SETUP COMPLETE. Verifying... ===` at the end, then confirm:
- `dexhand_controller` shows `ACTIVE`
- all 10 `/model/apple_01/pose` … `/model/apple_10/pose` topics are listed
- `/gripper_camera` and `/overhead_camera` are listed

If any topics are missing, just wait a few seconds and re-check — it's usually
a startup timing race, not a real failure:

```bash
ros2 topic list | grep -E "model/apple|gripper_camera|overhead_camera"
```

## 2. Run a single pick-and-place

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 full_layer_grasp.py apple_06
```

Runs the full pipeline (VLM analysis → arm positioning → finger closing →
lift → place in crate) on one apple.

## 3. Diagnostic: is a grasp even achievable right now?

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 diagnostic_grasp_attempt.py apple_06
```

One attempt with generous, slow, forgiving closing settings. Use this to
sanity-check the hand/positioning setup before trusting the learned policy
below to find good numbers on its own.

## 4. Train the learned finger-closing policy (small verification run)

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 train_closing_policy.py apple_06
```

Small search (1 baseline + a few mutated attempts) over per-finger closing
speed and contact-force threshold, scored by a reward function. Logs every
attempt to `~/ur_gz_ws/closing_policy_runs.json`. Stops itself immediately if
the target apple's position looks corrupted (see Troubleshooting).

Optional flags: `--generations N` (default 3), `--lambda N` (default 3).

## 5. VLM layer (optional, separate terminal)

```bash
cd ~/vlm_scripts
source /opt/ros/humble/setup.bash
python3 vlm_fragility_node.py
```

Check its output:
```bash
ros2 topic echo /gripper_camera/fragility_analysis
```

## Troubleshooting

**Apple flung far off the table / "Sanity check ... way outside the table
area" errors**: the apple's physics got corrupted (usually from an earlier
attempt). Restart everything from step 1 — apples only respawn cleanly on a
full Gazebo restart, nothing in-process can fix this.

**Topics missing right after `start_everything.sh` finishes**: startup race
condition — the bridge processes can take a few seconds longer than the
script's own wait. Re-run the `ros2 topic list` check above before assuming
something is broken.

**Missing `/tmp/real_robot_exact.urdf` (`FileNotFoundError`)**: only happens
if a grasp/diagnostic/training script is run without `start_everything.sh`
having run first in this WSL session (it regenerates this file — see step 1).
Fix directly if needed:
```bash
xacro /home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro > /tmp/real_robot_exact.urdf
```

**Want to see Gazebo visually**: pass `gazebo_gui:=true` to the `ros2 launch`
command instead of using `start_everything.sh` directly (see that script for
the full command with all required env vars/paths). Confirmed working, but
the first load can take up to ~60s (downloading/caching the Fuel models) —
give it that long before assuming it's broken. Headless mode is still the
default since nothing the pick-and-place logic does needs a GUI (it only
talks to Gazebo over ROS topics/services); use the GUI only when you actually
want to watch it.
