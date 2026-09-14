# UR5e + DexHand v2 Apple Picking

A simulated robot that picks apples from a table using a UR5e arm and a DexHand v2
five-fingered hand, built with ROS 2 Humble and Gazebo.

## How to run

All commands run inside WSL2. In every new terminal, source ROS first:
```bash
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
```

**Terminal 1: start the simulation**
```bash
cd ~/ur_gz_ws
bash start_everything.sh          # add "gui" to see the Gazebo window
```
Wait for `SETUP COMPLETE` before continuing.

**Terminal 2: run the robot**
```bash
cd ~/ur_gz_ws/src/my_pick_and_place/scripts
python3 pocket_grasp_test.py apple_06     # grasp test on one apple
bash test_all_apples.sh                   # grasp test on several apples
python3 full_layer_grasp.py apple_06      # full pick-and-place
bash test_all_apples.sh full              # full pick-and-place on all apples
```

If nothing moves for several minutes, restart Terminal 1 and run again.
