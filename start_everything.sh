#!/bin/bash
set -e

# Pass "gui" as the first argument to launch Gazebo's GUI (bash start_everything.sh
# gui). Previously this required copy-pasting a long separate manual command every
# time -- folding it in here means it's one command either way. wait_for_settled's
# timeouts were raised specifically so GUI mode (which runs well below real-time on
# this VM) still produces correct, trustworthy results instead of stale ones.
GUI_FLAG=false
if [ "$1" = "gui" ]; then
    GUI_FLAG=true
fi

source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share:~/ur_gz_ws/src/apple_gripper_sim/models
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/opt/ros/humble/lib

echo "=== Killing any leftover processes ==="
# "ign gazebo server" never actually matched anything -- the literal word "server"
# isn't in the real command line (headless uses a "-s" flag, not that word), so old
# Gazebo/controller_manager processes were never reliably killed here. That's why a
# stale dexhand_controller could still be "already loaded" against a supposedly-fresh
# launch. Broadened to match both binary names Ignition/Gazebo uses.
pkill -9 -f "ign gazebo" 2>/dev/null || true
pkill -9 -f "gz sim" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "ros2 launch" 2>/dev/null || true
pkill -9 -f "ros2_control_node" 2>/dev/null || true
sleep 2

# full_layer_grasp.py's IK chain loads this static URDF from /tmp -- WSL2 clears /tmp
# on every reboot, so it must be regenerated each fresh session or the grasp/diagnostic/
# training scripts fail immediately with FileNotFoundError. Doing it here means it's
# always fresh and this never has to be a separate manual step again.
echo "=== Regenerating /tmp/real_robot_exact.urdf (cleared on WSL2 reboot) ==="
xacro /home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro > /tmp/real_robot_exact.urdf

# gazebo_gui:=false runs `ign gazebo -s` (server only, no GUI process) -- the default,
# since the GUI's 3D view runs well below real-time on this VM (confirmed directly:
# joints still moving fast after generous timeouts). Pass "gui" as this script's first
# argument to launch the GUI anyway; wait_for_settled's timeouts were raised
# specifically to tolerate that slowdown, so GUI-mode runs are now trustworthy, just
# slower wall-clock than headless.
if [ "$GUI_FLAG" = true ]; then
    echo "=== Launching simulation with apple_world, GUI enabled (background) ==="
else
    echo "=== Launching simulation with apple_world, headless (background) ==="
fi
setsid ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=/home/mahimaa/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world \
    gazebo_gui:=$GUI_FLAG \
    > /tmp/sim_launch.log 2>&1 < /dev/null &
disown

echo "Waiting for simulation to be ready..."
if [ "$GUI_FLAG" = true ]; then
    sleep 60
else
    sleep 12
fi

echo "=== Activating hand controller ==="
ros2 run controller_manager spawner dexhand_controller --controller-manager /controller_manager

echo "=== Starting gripper camera bridge (background) ==="
ros2 run ros_gz_bridge parameter_bridge \
    /gripper_camera@sensor_msgs/msg/Image[ignition.msgs.Image \
    > /tmp/camera_bridge.log 2>&1 &
disown

echo "=== Starting overhead camera bridge (background) ==="
ros2 run ros_gz_bridge parameter_bridge \
    /overhead_camera@sensor_msgs/msg/Image[ignition.msgs.Image \
    > /tmp/overhead_camera_bridge.log 2>&1 &
disown

echo "=== Starting apple pose bridges (background) ==="
for i in 01 02 03 04 05 06 07 08 09 10; do
    ros2 run ros_gz_bridge parameter_bridge \
        "/model/apple_${i}/pose@geometry_msgs/msg/Pose[ignition.msgs.Pose" \
        >> /tmp/pose_bridges.log 2>&1 &
    disown
done

sleep 3
echo ""
echo "=== SETUP COMPLETE. Verifying... ==="
ros2 control list_controllers
echo ""
ros2 topic list | grep -E "model/apple|gripper_camera|overhead_camera"
echo ""
echo "If dexhand_controller is ACTIVE and all 10 apple pose topics + gripper_camera +"
echo "overhead_camera are listed above, everything is ready."
echo ""
echo "Single apple (also runs the new place-in-crate step):"
echo "  python3 ~/ur_gz_ws/src/my_pick_and_place/scripts/full_layer_grasp.py apple_06"
echo ""
echo "Overhead apple detector (color-based, projects pixel positions to world"
echo "coordinates -- run with --calibrate to compare against ground truth):"
echo "  python3 ~/ur_gz_ws/src/my_pick_and_place/scripts/detect_apples.py --calibrate"
echo ""
echo "For the VLM layer (Layer 1), in a separate terminal:"
echo "  cd ~/vlm_scripts && source /opt/ros/humble/setup.bash && python3 vlm_fragility_node.py"
