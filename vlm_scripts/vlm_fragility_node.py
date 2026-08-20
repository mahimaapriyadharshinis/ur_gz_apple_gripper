#!/usr/bin/env python3
import base64
import json
import time

import cv2
import requests
import rclpy
from cv_bridge import CvBridge
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import String

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5vl:3b"
IMAGE_TOPIC = "/gripper_camera"
RESULT_TOPIC = "/gripper_camera/fragility_analysis"
QUERY_INTERVAL_SEC = 2.0

PROMPT = """You are a robotic grasping assistant analyzing an object seen by a gripper-mounted camera.
Look at the image and respond with ONLY a valid JSON object (no markdown, no extra text) with these exact keys:

{
  "object_name": "best guess at what the object is",
  "fragility_score": 0-10 integer, where 0 is indestructible and 10 is extremely fragile,
  "estimated_material": "e.g. glass, plastic, fruit skin, metal, ceramic, fabric",
  "recommended_grip_force": "low, medium, or high",
  "surface_texture": "e.g. smooth, rough, bumpy, soft",
  "confidence": 0.0-1.0 float representing how confident you are in this analysis,
  "notes": "one short sentence with any additional relevant observation"
}

Respond with ONLY the JSON object, nothing else."""


class VLMFragilityNode(Node):
    def __init__(self):
        super().__init__("vlm_fragility_node")
        self.bridge = CvBridge()
        self.last_query_time = 0.0

        self.subscription = self.create_subscription(
            Image, IMAGE_TOPIC, self.image_callback, 10
        )
        self.publisher = self.create_publisher(String, RESULT_TOPIC, 10)

        self.get_logger().info(f"Subscribed to {IMAGE_TOPIC}")
        self.get_logger().info(f"Publishing analysis to {RESULT_TOPIC}")
        self.get_logger().info(f"Using Ollama model: {MODEL_NAME}")

    def image_callback(self, msg: Image):
        now = time.time()
        if now - self.last_query_time < QUERY_INTERVAL_SEC:
            return
        self.last_query_time = now

        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="bgr8")
        except Exception as e:
            self.get_logger().error(f"cv_bridge conversion failed: {e}")
            return

        success, buffer = cv2.imencode(".jpg", cv_image)
        if not success:
            self.get_logger().error("Failed to encode image as JPEG")
            return
        image_b64 = base64.b64encode(buffer).decode("utf-8")

        analysis = self.query_vlm(image_b64)
        if analysis is not None:
            result_msg = String()
            result_msg.data = json.dumps(analysis)
            self.publisher.publish(result_msg)
            self.get_logger().info(f"Analysis: {analysis}")

    def query_vlm(self, image_b64: str):
        payload = {
            "model": MODEL_NAME,
            "prompt": PROMPT,
            "images": [image_b64],
            "stream": False,
            "format": "json",
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=120)
            response.raise_for_status()
        except requests.exceptions.RequestException as e:
            self.get_logger().error(f"Ollama request failed: {e}")
            return None

        raw_text = response.json().get("response", "").strip()

        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            self.get_logger().warning(f"Could not parse VLM response as JSON: {raw_text}")
            return {"raw_response": raw_text}


def main(args=None):
    rclpy.init(args=args)
    node = VLMFragilityNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
