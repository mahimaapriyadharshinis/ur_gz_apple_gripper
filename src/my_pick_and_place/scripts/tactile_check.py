#!/usr/bin/env python3
"""Print fingertip force next to joint effort, so the new sensors can be trusted before
anything is built on them.

Phase 2 of adding tactile sensing: the grasp still reads joint effort (a torque that
saturates at 1.5Nm). This script reads both at once and puts them side by side, which is
the check that matters -- a sensor reading 0N while a finger is clearly pressing at 1.5Nm
means the sensor frame or joint is wrong, and nothing should use it yet.

It only listens. No commands are sent, so it can be left running during a normal grasp
run in another terminal.

Usage (simulation started with fingertip sensors):
    bash start_everything.sh tactile        # terminal 1
    python3 tactile_check.py                # terminal 2, prints until Ctrl+C
    python3 tactile_check.py --once         # one reading, for a quick check
"""

import sys
import time

import rclpy
from geometry_msgs.msg import Wrench
from rclpy.node import Node
from sensor_msgs.msg import JointState

WORLD = "apple_world"
MODEL = "ur"

# finger -> (fingertip joint, sensor name declared in ur5e_dexhand.xacro)
FINGERTIP_SENSORS = {
    "Index": ("R_Index_DIP", "index_tip_ft"),
    "Middle": ("R_Middle_DIP", "middle_tip_ft"),
    "Ring": ("R_Ring_DIP", "ring_tip_ft"),
    "Pinky": ("R_Pinky_DIP", "pinky_tip_ft"),
    "Thumb": ("R_Thumb_DIP", "thumb_tip_ft"),
}


def topic_for(joint, sensor):
    return f"/world/{WORLD}/model/{MODEL}/joint/{joint}/sensor/{sensor}/forcetorque"


class TactileCheck(Node):
    def __init__(self):
        super().__init__("tactile_check")
        self.force = {}
        self.effort = {}
        for finger, (joint, sensor) in FINGERTIP_SENSORS.items():
            self.create_subscription(
                Wrench, topic_for(joint, sensor),
                lambda msg, f=finger: self._wrench_cb(f, msg), 10)
        self.create_subscription(JointState, "/joint_states", self._joint_cb, 10)

    def _wrench_cb(self, finger, msg):
        f = msg.force
        self.force[finger] = (f.x, f.y, f.z)

    def _joint_cb(self, msg):
        for name, eff in zip(msg.name, msg.effort):
            if name.endswith("_Pitch") or name.endswith("_DIP"):
                self.effort[name] = eff

    def line(self):
        parts = []
        for finger, (joint, _) in FINGERTIP_SENSORS.items():
            vec = self.force.get(finger)
            if vec is None:
                parts.append(f"{finger}: sensor -")
            else:
                mag = sum(v * v for v in vec) ** 0.5
                parts.append(f"{finger}: {mag:5.2f}N")
            eff = self.effort.get(joint)
            parts[-1] += f" / {'-' if eff is None else f'{abs(eff):4.2f}Nm'}"
        return "  ".join(parts)


def main():
    once = "--once" in sys.argv
    rclpy.init()
    node = TactileCheck()
    print("Waiting for the fingertip sensor topics (start the simulation with "
          "'bash start_everything.sh tactile')...")
    deadline = time.time() + 30.0
    while rclpy.ok() and time.time() < deadline and not node.force:
        rclpy.spin_once(node, timeout_sec=0.2)
    if not node.force:
        print("No fingertip force data arrived in 30s. Check, in order:")
        print("  ign topic -l | grep forcetorque      # does Gazebo publish it at all?")
        print("  cat /tmp/tactile_bridges.log          # did the ROS bridges start?")
        print("  grep -c force_torque /tmp/real_robot_tactile.urdf   # was the robot built "
              "with sensors?")
        node.destroy_node()
        rclpy.shutdown()
        return 1

    print("force per fingertip (N) / joint effort at the same joint (Nm)")
    print("   sensor and effort should rise and fall together when a finger touches "
          "something.\n")
    try:
        while rclpy.ok():
            for _ in range(10):
                rclpy.spin_once(node, timeout_sec=0.1)
            print(node.line(), flush=True)
            if once:
                break
    except KeyboardInterrupt:
        pass
    node.destroy_node()
    rclpy.shutdown()
    return 0


if __name__ == "__main__":
    sys.exit(main())
