"""
lane_detection_node.py
----------------------
Subscribes to the downward-facing camera and detects:
  - Solid white line  (right boundary  → robot must stay LEFT of it)
  - Dashed yellow line (center divider → robot can cross to avoid obstacles)
  - Orange end-of-road line            → triggers 180° turn

Publishes:
  /lane/error          (std_msgs/Float32)  — lateral error for the controller
                        positive = robot too far LEFT  (white line too close)
                        negative = robot too far RIGHT (white line too far)
  /lane/markers        (visualization_msgs/MarkerArray) — RViz visualisation
  /lane/end_of_road    (std_msgs/Bool)     — True when orange line detected
  /lane/debug_image    (sensor_msgs/Image) — annotated image for tuning

HOW IT WORKS
------------
1. Crop the bottom third of the image (closest to ground, most stable).
2. Convert to HSV and threshold for each colour.
3. Find the largest contour for each colour.
4. Compute the X centroid of the white line contour.
5. Error = (centroid_x - target_x), where target_x places the white line
   ~20 % from the right edge (robot slightly left of the solid line).
6. Publish the error so navigation_node can do PD control.

TUNING
------
All HSV thresholds are exposed as ROS parameters so you can tune them
at runtime with:
  ros2 param set /lane_detection_node white_h_min 0
"""

import rclpy
from rclpy.node import Node
from rclpy.parameter import Parameter

import cv2
import numpy as np

from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from cv_bridge import CvBridge


# ──────────────────────────────────────────────────────────────────
#  Default HSV thresholds  (tune via ROS parameters)
# ──────────────────────────────────────────────────────────────────
WHITE_HSV_LOW  = (0,   0,  220)
WHITE_HSV_HIGH = (180, 30, 255)

YELLOW_HSV_LOW  = (25,  100, 120)
YELLOW_HSV_HIGH = (60, 255, 255)

ORANGE_HSV_LOW  = (5,  150, 150)
ORANGE_HSV_HIGH = (15, 255, 255)

# Minimum contour area to be considered a real line (px²)
MIN_CONTOUR_AREA = 2500

# How far from the right edge (0–1) the white line centroid should sit
TARGET_WHITE_X_RATIO = 0.20

# How far from the left edge (0–1) the yellow line centroid should sit
TARGET_YELLOW_X_RATIO = 0.20


class LaneDetectionNode(Node):

    def __init__(self):
        super().__init__('lane_detection_node')

        # ── Parameters (tunable at runtime) ──────────────────────
        self.declare_parameter('white_h_min',  int(WHITE_HSV_LOW[0]))
        self.declare_parameter('white_s_max',  int(WHITE_HSV_HIGH[1]))
        self.declare_parameter('white_v_min',  int(WHITE_HSV_LOW[2]))
        self.declare_parameter('yellow_h_min', int(YELLOW_HSV_LOW[0]))
        self.declare_parameter('yellow_h_max', int(YELLOW_HSV_HIGH[0]))
        self.declare_parameter('orange_h_min', int(ORANGE_HSV_LOW[0]))
        self.declare_parameter('orange_h_max', int(ORANGE_HSV_HIGH[0]))
        self.declare_parameter('crop_top_ratio', 0.5)
        self.declare_parameter('crop_bottom_ratio', 0.1)
        self.declare_parameter('yellow_target_x_ratio', TARGET_YELLOW_X_RATIO)
        self.declare_parameter('yellow_weight', 0.5)   # blend when both visible [0=white only, 1=yellow only]
        self.declare_parameter('debug_image', True)

        # ── Subscribers ──────────────────────────────────────────
        self.sub_img = self.create_subscription(
            Image,
            '/camera_2/image_raw',
            self.image_callback,
            10)

        # ── Publishers ───────────────────────────────────────────
        self.pub_error   = self.create_publisher(Float32,      '/lane/error',       10)
        self.pub_eor     = self.create_publisher(Bool,         '/lane/end_of_road', 10)
        self.pub_markers = self.create_publisher(MarkerArray,  '/lane/markers',     10)
        self.pub_debug   = self.create_publisher(Image,        '/lane/debug_image', 10)

        self.bridge = CvBridge()
        self.get_logger().info('Lane detection node started.')

    # ── Main callback ─────────────────────────────────────────────
    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        
        h, w = frame.shape[:2]

        # 1. Crop: keep only the bottom (1 - crop_ratio) fraction
        crop_top = self.get_parameter('crop_top_ratio').value
        crop_bottom = self.get_parameter('crop_bottom_ratio').value
        
        roi_y = int(h * crop_top)
        roi_y_bottom = int(h * (1.0 - crop_bottom)) # Onde a imagem termina agora
        
        # Crop y_top to y_bottom
        roi = frame[roi_y:roi_y_bottom, 0:w]

        # 2. Convert to HSV
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # 3. Build masks
        white_mask  = self._white_mask(hsv)
        yellow_mask = self._yellow_mask(hsv)
        orange_mask = self._orange_mask(hsv)

        # 4. Detect white line centroid → primary lateral error
        white_error = 0.0
        white_cx = None
        white_cnt = self._largest_contour(white_mask)
        if white_cnt is not None:
            M = cv2.moments(white_cnt)
            if M['m00'] > 0:
                white_cx = int(M['m10'] / M['m00'])
                target_white_x = int(w * (1.0 - TARGET_WHITE_X_RATIO))
                white_error = (white_cx - target_white_x) / float(w / 2)

        # 4b. Detect yellow line centroid → fallback lateral error
        yellow_error = 0.0
        yellow_cx = None
        yellow_cnt = self._largest_contour(yellow_mask)
        if yellow_cnt is not None:
            M = cv2.moments(yellow_cnt)
            if M['m00'] > 0:
                yellow_cx = int(M['m10'] / M['m00'])
                target_yellow_x = int(w * self.get_parameter('yellow_target_x_ratio').value)
                yellow_error = (yellow_cx - target_yellow_x) / float(w / 2)

        # Both visible → weighted blend; one missing → use what's available
        yw = self.get_parameter('yellow_weight').value
        if white_cnt is not None and yellow_cnt is not None:
            final_error = (1.0 - yw) * white_error + yw * yellow_error
        elif white_cnt is not None:
            final_error = white_error
        elif yellow_cnt is not None:
            final_error = yellow_error
        else:
            final_error = 0.0

        # 5. Detect end-of-road orange line
        orange_cnt = self._largest_contour(orange_mask)
        end_of_road = orange_cnt is not None

        # 6. Publish
        self.pub_error.publish(Float32(data=float(final_error)))
        self.pub_eor.publish(Bool(data=end_of_road))
        self._publish_markers(w, roi_y, white_cnt, yellow_cnt, orange_cnt)

        # 7. Debug image
        if self.get_parameter('debug_image').value:
            self._publish_debug(roi, white_mask, yellow_mask, orange_mask,
                                white_cx, yellow_cx, w, roi_y,
                                white_cnt is not None)

    # ── Colour masks ──────────────────────────────────────────────
    def _white_mask(self, hsv):
        h_min = self.get_parameter('white_h_min').value
        s_max = self.get_parameter('white_s_max').value
        v_min = self.get_parameter('white_v_min').value
        lo = np.array([h_min, 10,     v_min])
        hi = np.array([180,   s_max, 255])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))

    def _yellow_mask(self, hsv):
        h_min = self.get_parameter('yellow_h_min').value
        h_max = self.get_parameter('yellow_h_max').value
        lo = np.array([h_min, 80, 80])
        hi = np.array([h_max, 255, 255])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((5, 5), np.uint8))

    def _orange_mask(self, hsv):
        h_min = self.get_parameter('orange_h_min').value
        h_max = self.get_parameter('orange_h_max').value
        lo = np.array([h_min, 150, 150])
        hi = np.array([h_max, 255, 255])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN,
                                np.ones((7, 7), np.uint8))

    # ── Contour helper ────────────────────────────────────────────
    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < MIN_CONTOUR_AREA:
            return None
        return best

    # ── Marker publisher (for RViz) ───────────────────────────────
    def _publish_markers(self, img_w, roi_y_offset, white_cnt,
                          yellow_cnt, orange_cnt):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def make_marker(mid, r, g, b):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp
            m.ns = 'lanes'
            m.id = mid
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.05
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            return m

        if white_cnt is not None:
            M = cv2.moments(white_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                m = make_marker(0, 1.0, 1.0, 1.0)
                m.pose.position.x = 0.3
                m.pose.position.y = (img_w / 2 - cx) * 0.001
                m.pose.position.z = 0.0
                arr.markers.append(m)

        if yellow_cnt is not None:
            M = cv2.moments(yellow_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                m = make_marker(1, 1.0, 1.0, 0.0)
                m.pose.position.x = 0.3
                m.pose.position.y = (img_w / 2 - cx) * 0.001
                m.pose.position.z = 0.0
                arr.markers.append(m)

        if orange_cnt is not None:
            m = make_marker(2, 1.0, 0.5, 0.0)
            m.pose.position.x = 0.2
            m.pose.position.y = 0.0
            m.pose.position.z = 0.0
            arr.markers.append(m)

        self.pub_markers.publish(arr)

    # ── Debug image ───────────────────────────────────────────────
    def _publish_debug(self, roi, white_mask, yellow_mask, orange_mask,
                        white_cx, yellow_cx, img_w, roi_y, white_active):
        debug = roi.copy()
        debug[white_mask  > 0] = (200, 200, 255)   # blue tint
        debug[yellow_mask > 0] = (0,   220, 220)   # yellow tint
        debug[orange_mask > 0] = (0,   120, 255)   # orange tint

        # White target line (green)
        target_white_x = int(img_w * (1.0 - TARGET_WHITE_X_RATIO))
        cv2.line(debug, (target_white_x, 0), (target_white_x, debug.shape[0]),
                 (0, 255, 0), 1)

        # Yellow target line (dark green, dashed feel via thinner line)
        target_yellow_x = int(img_w * self.get_parameter('yellow_target_x_ratio').value)
        cv2.line(debug, (target_yellow_x, 0), (target_yellow_x, debug.shape[0]),
                 (0, 180, 0), 1)

        # Detected white centroid (cyan) — thicker when active source
        if white_cx is not None:
            thickness = 3 if white_active else 1
            cv2.line(debug, (white_cx, 0), (white_cx, debug.shape[0]),
                     (255, 255, 0), thickness)

        # Detected yellow centroid (magenta) — thicker when active source
        if yellow_cx is not None:
            thickness = 3 if not white_active else 1
            cv2.line(debug, (yellow_cx, 0), (yellow_cx, debug.shape[0]),
                     (255, 0, 255), thickness)

        # Source label
        if white_cx is not None and yellow_cx is not None:
            label = 'SRC: BLEND'
        elif white_cx is not None:
            label = 'SRC: WHITE'
        elif yellow_cx is not None:
            label = 'SRC: YELLOW'
        else:
            label = 'SRC: NONE'
        cv2.putText(debug, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 1, cv2.LINE_AA)

        msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self.pub_debug.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
