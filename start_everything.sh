#!/bin/bash
set -e

# Pass "gui" as the first argument to launch Gazebo's GUI (bash start_everything.sh
# gui). Previously this required copy-pasting a long separate manual command every
# time -- folding it in here means it's one command either way. wait_for_settled's
# timeouts were raised specifically so GUI mode (which runs well below real-time on
# this VM) still produces correct, trustworthy results instead of stale ones.
GUI_FLAG=false
TACTILE=false
FRAGILITY=false
for arg in "$@"; do
    case "$arg" in
        gui) GUI_FLAG=true ;;
        # "tactile" builds the robot with a force_torque sensor in each fingertip and
        # bridges the five sensor topics. Off by default: without it the robot, the
        # world and every topic are exactly what picked all ten apples, so the tested
        # result cannot be affected by sensors that are not there.
        tactile) TACTILE=true ;;
        # "fragility" loads apple_world_fragility.world instead: the same world with
        # each apple's contact softness and colour set from a true fragility score
        # (tools/make_fragility_world.py). The normal world is untouched.
        fragility) FRAGILITY=true ;;
    esac
done

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

DESCRIPTION=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro
CONTROLLERS=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml
WORLD=/home/mahimaa/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world
if [ "$FRAGILITY" = true ]; then
    WORLD=/home/mahimaa/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world_fragility.world
    if [ ! -f "$WORLD" ]; then
        echo "ABORT: $WORLD is missing. Run: python3 tools/make_fragility_world.py"
        exit 1
    fi
    echo "=== Using the FRAGILITY world (apples differ in softness and colour) ==="
fi
if [ "$TACTILE" = true ]; then
    echo "=== Building the robot WITH fingertip force sensors (tactile mode) ==="
    # simulation_controllers MUST be passed here. The launch file normally supplies
    # it while expanding the xacro itself; pre-expanding without it left
    # <parameters></parameters> empty, so ign_ros2_control started no
    # controller_manager and the hand controller spawner waited forever.
    xacro "$DESCRIPTION" tactile:=true \
        simulation_controllers:="$CONTROLLERS" > /tmp/real_robot_tactile.urdf
    if ! grep -q "<parameters>$CONTROLLERS</parameters>" /tmp/real_robot_tactile.urdf; then
        echo "ABORT: /tmp/real_robot_tactile.urdf has no controller parameters -- the"
        echo "       hand controller would never start. Not launching."
        exit 1
    fi
    echo "    sensors in the built robot: $(grep -c force_torque /tmp/real_robot_tactile.urdf)"
    DESCRIPTION=/tmp/real_robot_tactile.urdf
fi

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
    description_file:=$DESCRIPTION \
    controllers_file:=/home/mahimaa/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=$WORLD \
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
# Retried: once (17 Sep, tactile + fragility) the controller manager existed but did
# not answer within the spawner's 10s, the script stopped, and the exact same launch
# worked on the next try. A start-up race, so wait and try again rather than abort.
# When the first try succeeds -- the normal case -- this behaves exactly as before.
HAND_OK=false
for try in 1 2 3 4 5; do
    if ros2 run controller_manager spawner dexhand_controller \
        --controller-manager /controller_manager --service-call-timeout 30; then
        HAND_OK=true
        break
    fi
    echo "    hand controller not ready yet (try $try of 5) -- waiting 20s"
    sleep 20
done
if [ "$HAND_OK" != true ]; then
    echo "ABORT: the hand controller never activated. Last lines of /tmp/sim_launch.log:"
    tail -20 /tmp/sim_launch.log
    exit 1
fi

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

if [ "$TACTILE" = true ]; then
    echo "=== Starting finger sensor bridges (background, one process) ==="
    # One bridge process for all 20 topics: 5 fingertip force_torque sensors and
    # 15 finger-segment contact sensors. Twenty separate processes would each cost
    # memory and CPU on a machine already running Gazebo at about 8% of real time.
    TOPICS=()
    for pair in "R_Index_DIP index_tip_ft" "R_Middle_DIP middle_tip_ft" \
                "R_Ring_DIP ring_tip_ft" "R_Pinky_DIP pinky_tip_ft" \
                "R_Thumb_DIP thumb_tip_ft"; do
        set -- $pair
        TOPICS+=("/world/apple_world/model/ur/joint/$1/sensor/$2/forcetorque@geometry_msgs/msg/Wrench[ignition.msgs.Wrench")
    done
    # Contact sensors: Ignition Fortress ignores the <topic> given in the xacro and
    # publishes on /world/<world>/model/ur/link/<link>/sensor/<name>/contact
    # (checked with ign topic -l), so those exact paths are bridged.
    for pair in "Index_Knuckle_1 index_proximal" "Index_Middle_1 index_middle" \
                "Index_Tip_1 index_tip" "Middle_Knuckle_1 middle_proximal" \
                "Middle_Middle_1 middle_middle" "Midle_Tip_1 middle_tip" \
                "Ring_Knuckle_1 ring_proximal" "Ring_Middle_1 ring_middle" \
                "Ring_Tip_1 ring_tip" "Pinky_Knuckle_1 pinky_proximal" \
                "Pinky_Middle_1 pinky_middle" "Pinky_Tip_1 pinky_tip" \
                "Thumb_Knuckle_1 thumb_proximal" "Thumb_Middle_1 thumb_middle" \
                "Thumb_Tip_1 thumb_tip"; do
        set -- $pair
        TOPICS+=("/world/apple_world/model/ur/link/$1/sensor/$2_contact/contact@ros_gz_interfaces/msg/Contacts[ignition.msgs.Contacts")
    done
    ros2 run ros_gz_bridge parameter_bridge "${TOPICS[@]}" \
        > /tmp/tactile_bridges.log 2>&1 &
    disown
    echo "    bridging ${#TOPICS[@]} sensor topics"
fi

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
