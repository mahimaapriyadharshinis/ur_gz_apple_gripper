#!/usr/bin/env python3
"""
Overhead camera-based apple detector: finds apples in the fixed overhead
camera's image via color thresholding, and projects their pixel positions to
real-world (x, y) coordinates using the camera's known, fixed pose (see
apple_world.world's overhead_camera_rig).

This is a real, working color-based detector -- not a placeholder -- but the
pixel-to-world projection math is UNVERIFIED against the real sim (no way to
run Gazebo/OpenCV outside the user's VM). Run with --calibrate: it also
subscribes to the apples' live ground-truth poses and prints both side by
side, so any axis/sign error in the projection shows up immediately as a
clear mismatch instead of a silent wrong answer.

Requires the overhead camera to be bridged first:
  ros2 run ros_gz_bridge parameter_bridge /overhead_camera@sensor_msgs/msg/Image[ignition.msgs.Image
"""
import sys
import time

import numpy as np
import cv2
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from geometry_msgs.msg import Pose
from cv_bridge import CvBridge

# Must match apple_world.world's overhead_camera_rig pose exactly.
CAMERA_WORLD_XYZ = (1.125, 0.0, 2.5)
CAMERA_HORIZONTAL_FOV = 1.047  # radians, must match the <horizontal_fov> in the world file
IMAGE_WIDTH = 800
IMAGE_HEIGHT = 600
GROUND_Z = 0.04  # apple center height -- ground_dist = camera_z - this

# HSV thresholds for "reddish apple" -- red wraps around hue 0/180 in OpenCV,
# so two ranges are combined. UNVERIFIED against the real rendered apple
# colors/lighting -- tune these if detection misses apples or picks up noise.
RED_HSV_RANGES = [
    ((0, 80, 50), (10, 255, 255)),
    ((170, 80, 50), (180, 255, 255)),
]
MIN_BLOB_AREA = 20  # pixels -- filters out noise


def pixel_to_world(px, py, ground_z=GROUND_Z):
    """Project an image pixel to world (x, y), assuming the object is at
    ground_z and the camera looks straight down from CAMERA_WORLD_XYZ."""
    cam_x, cam_y, cam_z = CAMERA_WORLD_XYZ
    ground_dist = cam_z - ground_z
    aspect = IMAGE_WIDTH / IMAGE_HEIGHT
    half_hfov = CAMERA_HORIZONTAL_FOV / 2.0
    half_vfov = np.arctan(np.tan(half_hfov) / aspect)

    norm_x = (px - IMAGE_WIDTH / 2.0) / (IMAGE_WIDTH / 2.0)
    norm_y = (py - IMAGE_HEIGHT / 2.0) / (IMAGE_HEIGHT / 2.0)

    # Best-guess axis mapping for a camera pitched straight down (see world file
    # comments) -- UNVERIFIED. If --calibrate shows a consistent swap or sign
    # error against ground truth, this is the first place to correct.
    x_offset = norm_y * ground_dist * np.tan(half_vfov)
    y_offset = -norm_x * ground_dist * np.tan(half_hfov)

    return cam_x + x_offset, cam_y + y_offset


class ApplePoseSub:
    def __init__(self, node, name):
        self.pose = None
        node.create_subscription(Pose, f'/model/{name}/pose', self._cb, 10)

    def _cb(self, msg):
        self.pose = msg


class OverheadDetector(Node):
    def __init__(self, calibrate=False):
        super().__init__('overhead_apple_detector')
        self.bridge = CvBridge()
        self.latest_frame = None
        self.create_subscription(Image, '/overhead_camera', self._image_cb, 10)
        self.gt_subs = {}
        if calibrate:
            for i in range(1, 11):
                name = f"apple_{i:02d}"
                self.gt_subs[name] = ApplePoseSub(self, name)

    def _image_cb(self, msg):
        try:
            self.latest_frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().warn(f"Camera frame conversion failed: {e}")

    def detect(self):
        if self.latest_frame is None:
            return []
        hsv = cv2.cvtColor(self.latest_frame, cv2.COLOR_BGR2HSV)
        mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
        for lo, hi in RED_HSV_RANGES:
            mask |= cv2.inRange(hsv, np.array(lo), np.array(hi))

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        detections = []
        for c in contours:
            area = cv2.contourArea(c)
            if area < MIN_BLOB_AREA:
                continue
            m = cv2.moments(c)
            if m['m00'] == 0:
                continue
            px = m['m10'] / m['m00']
            py = m['m01'] / m['m00']
            world_x, world_y = pixel_to_world(px, py)
            detections.append({'pixel': (px, py), 'area': area, 'world': (world_x, world_y)})
        return detections


def wait_for_frame(node, timeout=5.0):
    start = time.time()
    while rclpy.ok() and node.latest_frame is None and (time.time() - start) < timeout:
        rclpy.spin_once(node, timeout_sec=0.1)
    return node.latest_frame is not None


def main():
    calibrate = '--calibrate' in sys.argv
    rclpy.init()
    node = OverheadDetector(calibrate=calibrate)

    print("Waiting for /overhead_camera frame...")
    if not wait_for_frame(node):
        print("No camera frame received in 5s -- is the overhead camera bridged? "
              "ros2 run ros_gz_bridge parameter_bridge "
              "/overhead_camera@sensor_msgs/msg/Image[ignition.msgs.Image")
        node.destroy_node()
        rclpy.shutdown()
        sys.exit(1)

    for _ in range(10):
        rclpy.spin_once(node, timeout_sec=0.2)

    detections = node.detect()
    print(f"\nDetected {len(detections)} apple-colored blob(s):\n")
    for d in detections:
        px, py = d['pixel']
        wx, wy = d['world']
        print(f"  pixel=({px:.0f}, {py:.0f}) area={d['area']:.0f} -> world=({wx:.3f}, {wy:.3f})")

    if calibrate:
        print("\n=== Calibration: detected vs ground-truth ===")
        for name, sub in node.gt_subs.items():
            rclpy.spin_once(node, timeout_sec=0.1)
            if sub.pose is None:
                print(f"  {name}: no ground-truth pose received")
                continue
            gt_x, gt_y = sub.pose.position.x, sub.pose.position.y
            if detections:
                best = min(detections, key=lambda d: (d['world'][0] - gt_x) ** 2 + (d['world'][1] - gt_y) ** 2)
                dist = ((best['world'][0] - gt_x) ** 2 + (best['world'][1] - gt_y) ** 2) ** 0.5
                print(f"  {name}: ground_truth=({gt_x:.3f},{gt_y:.3f}) "
                      f"closest_detection=({best['world'][0]:.3f},{best['world'][1]:.3f}) "
                      f"error={dist:.3f}m")
            else:
                print(f"  {name}: ground_truth=({gt_x:.3f},{gt_y:.3f}) -- no detections to compare")

    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
