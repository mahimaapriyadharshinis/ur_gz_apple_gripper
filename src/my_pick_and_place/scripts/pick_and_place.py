#!/usr/bin/env python3
import time
import numpy as np
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Pose
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Empty

DEXHAND_TRAJ_TOPIC = '/dexhand_controller/joint_trajectory'
DEXHAND_JOINT_NAMES = [
    'R_Index_Pitch', 'R_Middle_Pitch', 'R_Ring_Pitch', 'R_Pinky_Pitch',
    'R_Index_Flexor', 'R_Middle_Flexor', 'R_Ring_Flexor', 'R_Pinky_Flexor',
    'R_Index_DIP', 'R_Middle_DIP', 'R_Ring_DIP', 'R_Pinky_DIP',
    'R_Thumb_Yaw', 'R_Thumb_Roll', 'R_Thumb_Flexor', 'R_Thumb_DIP',
    'R_Index_Yaw', 'R_Middle_Yaw', 'R_Ring_Yaw', 'R_Pinky_Yaw', 'R_Thumb_Pitch',
]
DEXHAND_OPEN = [0.0] * 21
DEXHAND_CLOSED = [0.0, 0.0, 0.0, 0.0, 0.9, 0.9, 0.9, 0.9, 0.5, 0.5, 0.5, 0.5,
                  0.3, 0.0, 0.9, 0.5, 0.0, 0.0, 0.0, 0.0, 0.6]

class PickAndPlaceNode(Node):
    def __init__(self):
        super().__init__('pick_and_place_node')
        self.pose_sub = self.create_subscription(
            Pose, '/model/red_block/pose', self.pose_callback, 10)
        self.arm_pub = self.create_publisher(
            JointTrajectory, '/joint_trajectory_controller/joint_trajectory', 10)
        self.hand_pub = self.create_publisher(
            JointTrajectory, DEXHAND_TRAJ_TOPIC, 10)
        self.attach_pub = self.create_publisher(Empty, '/attach', 10)
        self.detach_pub = self.create_publisher(Empty, '/detach', 10)
        self.a2 = 0.425
        self.a3 = 0.3922
        self.z_shoulder = 0.6075
        self.hand_length = 0.18
        self.wrist_offset = 0.109
        self.target_pose = None

    def pose_callback(self, msg):
        self.target_pose = msg

    def wait_for_pose(self, timeout=3.0):
        self.target_pose = None
        start = time.time()
        while rclpy.ok() and self.target_pose is None and (time.time() - start) < timeout:
            rclpy.spin_once(self, timeout_sec=0.1)
        return self.target_pose

    def compute_3d_palm_down_ik(self, x, y, z_top):
        pan = np.arctan2(y, x)
        r = np.sqrt(x**2 + y**2)
        r_corrected = np.sqrt(max(r**2 - self.wrist_offset**2, 0.0))
        z_wrist = z_top + self.hand_length
        dx = r_corrected
        dz = z_wrist - self.z_shoulder
        D = np.sqrt(dx**2 + dz**2)
        max_reach = self.a2 + self.a3
        if D > max_reach or D < abs(self.a2 - self.a3):
            self.get_logger().error(f"UNREACHABLE: ({x:.3f},{y:.3f},{z_top:.3f}) D={D:.3f}m")
            return None
        cos_elbow = np.clip((D**2 - self.a2**2 - self.a3**2) / (2 * self.a2 * self.a3), -1.0, 1.0)
        elbow = np.arccos(cos_elbow)
        gamma = np.arctan2(dz, dx)
        cos_alpha = np.clip((self.a2**2 + D**2 - self.a3**2) / (2 * self.a2 * D), -1.0, 1.0)
        alpha = np.arccos(cos_alpha)
        phi_upper = gamma - alpha
        shoulder_lift = phi_upper - (np.pi / 2.0)
        phi_forearm = phi_upper + elbow
        wrist_1 = -np.pi / 2.0 - phi_forearm
        return [float(pan), float(shoulder_lift), float(elbow), float(wrist_1), -1.5708, 0.0]

    def send_arm_trajectory(self, joint_positions, duration_sec):
        msg = JointTrajectory()
        msg.joint_names = ['shoulder_pan_joint', 'shoulder_lift_joint', 'elbow_joint',
                            'wrist_1_joint', 'wrist_2_joint', 'wrist_3_joint']
        point = JointTrajectoryPoint()
        point.positions = joint_positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.arm_pub.publish(msg)

    def set_hand(self, positions, duration_sec=1.0):
        msg = JointTrajectory()
        msg.joint_names = DEXHAND_JOINT_NAMES
        point = JointTrajectoryPoint()
        point.positions = positions
        point.time_from_start.sec = int(duration_sec)
        point.time_from_start.nanosec = int((duration_sec - int(duration_sec)) * 1e9)
        msg.points = [point]
        self.hand_pub.publish(msg)

    def run(self):
        pose = self.wait_for_pose()
        if pose is not None:
            x, y, z_center = pose.position.x, pose.position.y, pose.position.z
            self.get_logger().info(f"Live block pose: X={x:.3f} Y={y:.3f} Z={z_center:.3f}")
        else:
            x, y, z_center = 0.50, 0.00, 0.07
            self.get_logger().warn(f"Pose bridge timed out, using fallback ({x},{y},{z_center})")

        z_top = z_center + 0.07
        approach = self.compute_3d_palm_down_ik(x, y, z_top + 0.15)
        grasp = self.compute_3d_palm_down_ik(x, y, z_top)

        if approach is None or grasp is None:
            self.get_logger().error("ABORTED: target unreachable.")
            return

        self.get_logger().info("1. Opening hand, moving overhead...")
        self.set_hand(DEXHAND_OPEN, 1.0)
        self.send_arm_trajectory(approach, 3.5)
        time.sleep(4.0)

        self.get_logger().info("2. Lowering onto block...")
        self.send_arm_trajectory(grasp, 2.5)
        time.sleep(3.0)

        self.get_logger().info("3. Closing hand around block...")
        self.set_hand(DEXHAND_CLOSED, 1.0)
        time.sleep(1.5)

        self.get_logger().info("4. Locking grasp joint (/attach)...")
        self.attach_pub.publish(Empty())
        time.sleep(0.5)

        self.get_logger().info("5. Lifting...")
        self.send_arm_trajectory(approach, 2.5)
        time.sleep(3.0)

        self.get_logger().info("TEST COMPLETE - check Gazebo viewport for actual result.")


def main():
    rclpy.init()
    node = PickAndPlaceNode()
    node.run()
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
