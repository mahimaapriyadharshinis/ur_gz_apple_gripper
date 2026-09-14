# UR5e + DexHand v2 Apple Picking

A simulated robot that picks apples from a table: a UR5e arm with a DexHand v2
five-fingered hand, running in Gazebo (Ignition Fortress) with ROS 2 Humble on WSL2.

## How to run

All commands run inside WSL2, in `~/ur_gz_ws`. In every new terminal, source ROS first:
```bash
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
```

### 1. Start the simulation (Terminal 1)

```bash
cd ~/ur_gz_ws
bash start_everything.sh          # headless
bash start_everything.sh gui      # with the Gazebo window
```

Wait for `=== SETUP COMPLETE ===` and check that `dexhand_controller` is active and the
apple pose topics are listed.

### 2. Run a grasp test (Terminal 2)

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 pocket_grasp_test.py apple_06        # one apple, 4 attempts
python3 pocket_grasp_test.py apple_06 2      # one apple, 2 attempts
bash test_all_apples.sh                      # several apples in sequence
```

Each run prints a results table at the end. The simulation runs slower than real time,
so an attempt can take a few minutes.

### 3. Full pick-and-place

```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 full_layer_grasp.py apple_06
```

## Troubleshooting

- **Nothing moves for many minutes / `SIM FROZE`**: restart Terminal 1 and run again.
- **Topics missing right after startup**: wait a few seconds and check again with
  `ros2 topic list`.
- **`FileNotFoundError` for `/tmp/real_robot_exact.urdf`**: run `start_everything.sh` first.
