<img width="795" height="566" alt="image" src="https://github.com/user-attachments/assets/9f526342-1ba9-4eb2-bbd3-c2f0ff904b17" /><img width="795" height="566" alt="image" src="https://github.com/user-attachments/assets/b285b497-03be-4d3c-a7b4-95d0e756f9c4" /><img width="795" height="566" alt="image" src="https://github.com/user-attachments/assets/b7a8af23-5abc-4c3b-95e6-e760ce575773" />## Run

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
ros2 launch apple_gripper_sim spawn_apples.launch.py
