#!/usr/bin/env python3
"""
Publishes the apples and table as RViz markers, so the scene is visible in RViz
without needing Gazebo's own GUI -- which has repeatedly caused the simulation to
run in slow motion on this VM, corrupting real diagnostic/training runs. RViz just
subscribes to topics; it doesn't share a rendering loop with the physics engine the
way Gazebo's GUI does, so watching here doesn't risk slowing anything down.

Apple positions come from the same /model/apple_XX/pose topics the grasp code
itself uses (real, live Gazebo ground truth, not a guess), converted into the
robot's local frame with the exact same world_to_local() math full_layer_grasp.py
uses for IK -- so what you see here lines up with what the grasp code is actually
aiming at. The table is drawn from its known static pose in apple_world.world
(model pose (1.125, 0, 0) + its collision box's local offset (0, 0, 0.20)).

Usage: python3 rviz_world_markers.py
Then in RViz: Add -> By display type -> MarkerArray, set topic to /world_markers.
Fixed Frame should already be base_footprint (same frame real_wrist_position()
already looks up successfully).
"""
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from visualization_msgs.msg import Marker, MarkerArray

from full_layer_grasp import world_to_local, DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW

APPLE_NAMES = [f"apple_{i:02d}" for i in range(1, 11)]
APPLE_DIAMETER = 0.08  # matches the ~0.04m radius used elsewhere in the grasp code

# From apple_world.world: model pose (1.125, 0, 0) + collision box's local pose
# (0, 0, 0.20) -> world center (1.125, 0.0, 0.20), size 2.60 x 0.50 x 0.40.
TABLE_WORLD_XYZ = (1.125, 0.0, 0.20)
TABLE_SIZE = (2.60, 0.50, 0.40)


class WorldMarkers(Node):
    def __init__(self):
        super().__init__('rviz_world_markers')
        self.apple_pose = {name: None for name in APPLE_NAMES}
        for name in APPLE_NAMES:
            self.create_subscription(
                Pose, f'/model/{name}/pose',
                lambda msg, n=name: self.apple_pose.__setitem__(n, msg), 10)
        self.pub = self.create_publisher(MarkerArray, '/world_markers', 10)
        self.create_timer(0.5, self.publish_markers)

    def publish_markers(self):
        arr = MarkerArray()

        lx, ly = world_to_local(*TABLE_WORLD_XYZ[:2], DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y,
                                 DELIVERY_ROBOT_YAW)
        table = Marker()
        table.header.frame_id = 'base_footprint'
        table.header.stamp = self.get_clock().now().to_msg()
        table.ns = 'table'
        table.id = 0
        table.type = Marker.CUBE
        table.action = Marker.ADD
        table.pose.position.x = lx
        table.pose.position.y = ly
        table.pose.position.z = TABLE_WORLD_XYZ[2]
        # The table's box is world-axis-aligned (long side along world X), but its
        # POSITION above was rotated into the robot's local frame (which is rotated by
        # the station yaw relative to world) -- the box's ORIENTATION needs the same
        # rotation, or it stays drawn as if unrotated while the apples (whose positions
        # go through the same rotation) end up spread along the axis actually
        # perpendicular to how the table is drawn.
        table.pose.orientation.z = np.sin(-DELIVERY_ROBOT_YAW / 2.0)
        table.pose.orientation.w = np.cos(-DELIVERY_ROBOT_YAW / 2.0)
        table.scale.x, table.scale.y, table.scale.z = TABLE_SIZE
        table.color.r, table.color.g, table.color.b, table.color.a = 0.55, 0.42, 0.30, 1.0
        arr.markers.append(table)

        for i, name in enumerate(APPLE_NAMES):
            pose = self.apple_pose[name]
            m = Marker()
            m.header.frame_id = 'base_footprint'
            m.header.stamp = self.get_clock().now().to_msg()
            m.ns = 'apples'
            m.id = i + 1
            m.type = Marker.SPHERE
            if pose is None:
                m.action = Marker.DELETE
            else:
                m.action = Marker.ADD
                ax, ay = world_to_local(pose.position.x, pose.position.y,
                                         DELIVERY_ROBOT_X, DELIVERY_ROBOT_Y, DELIVERY_ROBOT_YAW)
                m.pose.position.x = ax
                m.pose.position.y = ay
                m.pose.position.z = pose.position.z
                m.pose.orientation.w = 1.0
                m.scale.x = m.scale.y = m.scale.z = APPLE_DIAMETER
                m.color.r, m.color.g, m.color.b, m.color.a = 0.8, 0.05, 0.05, 1.0
            arr.markers.append(m)

        self.pub.publish(arr)


def main():
    rclpy.init()
    node = WorldMarkers()
    print("Publishing table + apple markers to /world_markers. In RViz: Add -> "
          "By display type -> MarkerArray -> set topic to /world_markers.")
    rclpy.spin(node)


if __name__ == '__main__':
    main()
