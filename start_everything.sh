#!/bin/bash
set -e
source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash
export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share:~/ur_gz_ws/src/apple_gripper_sim/models
export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/opt/ros/humble/lib

echo "=== Killing any leftover processes ==="
pkill -9 -f "ign gazebo server" 2>/dev/null || true
pkill -9 -f "parameter_bridge" 2>/dev/null || true
pkill -9 -f "ros2 launch" 2>/dev/null || true
sleep 2

# gazebo_gui:=false runs `ign gazebo -s` (server only, no GUI process). The Sensors
# system (physics/gripper_camera/ros2_control) is a separate subsystem from the GUI
# and works fine on this VM under ogre1 (see apple_world.world) -- the GUI's own 3D
# view is still ogre2 and crashes (Ogre::UnimplementedException in GL3PlusTextureGpu)
# when apple meshes load. Skipping the GUI avoids that crash entirely; nothing the
# grasp scripts do needs it, they only talk to Gazebo over ROS topics/services.
echo "=== Launching simulation with apple_world, headless (background) ==="
setsid ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=/home/mahimaa/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world \
    gazebo_gui:=false \
    > /tmp/sim_launch.log 2>&1 < /dev/null &
disown

echo "Waiting for simulation to be ready..."
sleep 12

echo "=== Activating hand controller ==="
ros2 run controller_manager spawner dexhand_controller --controller-manager /controller_manager

echo "=== Starting gripper camera bridge (background) ==="
ros2 run ros_gz_bridge parameter_bridge \
    /gripper_camera@sensor_msgs/msg/Image[ignition.msgs.Image \
    > /tmp/camera_bridge.log 2>&1 &
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
ros2 topic list | grep -E "model/apple|gripper_camera"
echo ""
echo "If dexhand_controller is ACTIVE and all 10 apple pose topics + gripper_camera are"
echo "listed above, everything is ready."
echo ""
echo "Single apple (also runs the new place-in-crate step):"
echo "  python3 ~/ur_gz_ws/src/my_pick_and_place/scripts/full_layer_grasp.py apple_06"
echo ""
echo "For the VLM layer (Layer 1), in a separate terminal:"
echo "  cd ~/vlm_scripts && source /opt/ros/humble/setup.bash && python3 vlm_fragility_node.py"
