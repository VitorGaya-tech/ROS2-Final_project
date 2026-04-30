"""
sign_detection_node.py
----------------------
Runs the best.pt YOLO model on the forward camera and publishes the detected
sign class to /sign/detected.

Published labels:
  stop
  road_closed
  one_way
  one_way_left   (heuristic)
  one_way_right  (heuristic)

The node is intentionally conservative: if the model or YOLO runtime is not
available, it stays alive and logs the missing dependency instead of crashing.
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np

import rclpy
from rclpy.node import Node

from ament_index_python.packages import PackageNotFoundError, get_package_share_directory
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
from std_msgs.msg import String

try:
    from ultralytics import YOLO
except Exception:  # pragma: no cover - depends on local ROS/Python env
    YOLO = None


DEFAULT_IMAGE_TOPIC = '/camera_2/image_raw'
DEFAULT_CONFIDENCE = 0.45
DEFAULT_PUBLISH_COOLDOWN = 0.75


class SignDetectionNode(Node):

    def __init__(self):
        super().__init__('sign_detection_node')

        self.declare_parameter('image_topic', DEFAULT_IMAGE_TOPIC)
        self.declare_parameter('model_path', '')
        self.declare_parameter('confidence_threshold', DEFAULT_CONFIDENCE)
        self.declare_parameter('publish_cooldown', DEFAULT_PUBLISH_COOLDOWN)
        self.declare_parameter('debug_image', False)

        self.bridge = CvBridge()
        self.model = self._load_model()

        self.last_label = ''
        self.last_publish_time = 0.0

        self.pub_sign = self.create_publisher(String, '/sign/detected', 10)
        self.pub_debug = self.create_publisher(Image, '/sign/debug_image', 10)

        image_topic = self.get_parameter('image_topic').value
        self.create_subscription(Image, image_topic, self._image_cb, 10)

        self.get_logger().info(f'Sign detection node started — topic: {image_topic}')

    def _load_model(self):
        if YOLO is None:
            self.get_logger().error('ultralytics is not installed, sign detection is disabled.')
            return None

        candidates = []
        model_path = str(self.get_parameter('model_path').value).strip()
        if model_path:
            candidates.append(model_path)

        try:
            share_dir = get_package_share_directory('autonomous_car_pkg')
            candidates.append(os.path.join(share_dir, 'best.pt'))
        except PackageNotFoundError:
            pass

        repo_root = Path(__file__).resolve().parents[1]
        candidates.append(str(repo_root / 'best.pt'))

        for candidate in candidates:
            if candidate and os.path.exists(candidate):
                try:
                    self.get_logger().info(f'Loading sign model: {candidate}')
                    return YOLO(candidate)
                except Exception as exc:
                    self.get_logger().error(f'Failed to load model {candidate}: {exc}')

        self.get_logger().error('best.pt not found; sign detection is disabled.')
        return None

    def _image_cb(self, msg: Image):
        if self.model is None:
            return

        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as exc:
            self.get_logger().error(f'cv_bridge error: {exc}')
            return

        conf_thresh = float(self.get_parameter('confidence_threshold').value)
        try:
            results = self.model.predict(frame, verbose=False, conf=conf_thresh, imgsz=640)
        except Exception as exc:
            self.get_logger().error(f'YOLO inference failed: {exc}')
            return

        if not results:
            return

        result = results[0]
        if result.boxes is None or len(result.boxes) == 0:
            return

        best_box = max(result.boxes, key=lambda box: float(box.conf[0]))
        confidence = float(best_box.conf[0])
        class_id = int(best_box.cls[0])
        raw_label = str(result.names.get(class_id, class_id)).lower().strip().replace(' ', '_').replace('-', '_')
        label = self._normalize_label(raw_label)

        if label == 'one_way':
            direction = self._infer_one_way_direction(frame, best_box.xyxy[0].cpu().numpy())
            if direction in ('left', 'right'):
                label = f'one_way_{direction}'

        if not label:
            return

        now = time.time()
        cooldown = float(self.get_parameter('publish_cooldown').value)
        if label == self.last_label and (now - self.last_publish_time) < cooldown:
            return

        self.last_label = label
        self.last_publish_time = now
        self.pub_sign.publish(String(data=label))
        self.get_logger().info(f'Sign detected: {label} ({confidence:.2f})')

        if self.get_parameter('debug_image').value:
            self._publish_debug(frame, best_box.xyxy[0].cpu().numpy(), label, confidence)

    def _normalize_label(self, raw_label: str):
        if 'stop' in raw_label:
            return 'stop'
        if 'road' in raw_label or 'closed' in raw_label:
            return 'road_closed'
        if 'one' in raw_label or 'way' in raw_label:
            return 'one_way'
        return ''

    def _infer_one_way_direction(self, frame, xyxy):
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        h, w = frame.shape[:2]
        x1 = max(0, min(w - 1, x1))
        x2 = max(0, min(w, x2))
        y1 = max(0, min(h - 1, y1))
        y2 = max(0, min(h, y2))

        if x2 <= x1 or y2 <= y1:
            return None

        crop = frame[y1:y2, x1:x2]
        if crop.size == 0:
            return None

        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        bright_mask = cv2.inRange(hsv, np.array([0, 0, 170]), np.array([180, 85, 255]))
        if cv2.countNonZero(bright_mask) < 100:
            return None

        ys, xs = np.where(bright_mask > 0)
        if len(xs) == 0:
            return None

        cx = float(xs.mean())
        center = crop.shape[1] / 2.0
        normalized = (cx - center) / max(center, 1.0)
        if abs(normalized) < 0.08:
            return None
        return 'right' if normalized > 0.0 else 'left'

    def _publish_debug(self, frame, xyxy, label, confidence):
        debug = frame.copy()
        x1, y1, x2, y2 = [int(v) for v in xyxy]
        cv2.rectangle(debug, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.putText(debug, f'{label} {confidence:.2f}', (x1, max(20, y1 - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2, cv2.LINE_AA)
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))


def main(args=None):
    rclpy.init(args=args)
    node = SignDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()