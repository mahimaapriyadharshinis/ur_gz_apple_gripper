
```bash
pkill -9 -f "ign gazebo" 2>/dev/null
pkill -9 -f "parameter_bridge" 2>/dev/null
sleep 2

source /opt/ros/humble/setup.bash
source ~/ur_gz_ws/install/setup.bash

export LIBGL_ALWAYS_SOFTWARE=1
export OGRE_RTT_MODE=Copy
export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:~/ur_gz_ws/src:~/ur_gz_ws/src/dexhandv2_description:~/ur_gz_ws/install/dexhandv2_description/share

ros2 launch ur_simulation_gz ur_sim_control.launch.py \
    ur_type:=ur5e \
    description_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro \
    controllers_file:=/home/tt501/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml \
    world_file:=/home/tt501/ur_gz_ws/sorting_environment.world
```
```bach
ros2 launch apple_gripper_sim spawn_apples.launch.py
```

```bash
tt501@tt501-ThinkCentre-neo-50s-Gen-4:~/gripper_project/ur_gz_ws/src/my_pick_and_place$ ros2 launch ur_simulation_gz ur_sim_control.launch.py     ur_type:=ur5e     description_file:=/home/tt501/gripper_project/ur_gz_ws/src/my_pick_and_place/urdf/ur5e_dexhand.xacro     controllers_file:=/home/tt501/gripper_project/ur_gz_ws/src/my_pick_and_place/urdf/merged_controllers.yaml     world_file:=/home/tt501/gripper_project/ur_gz_ws/src/apple_gripper_sim/worlds/apple_world.world
```
