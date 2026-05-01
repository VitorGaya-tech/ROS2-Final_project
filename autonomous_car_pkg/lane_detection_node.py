"""
lane_detection_node.py
----------------------
Detecta líneas y extrae SEGMENTOS de línea para un mapeo limpio en RViz.
"""

import math
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
#  Default HSV thresholds
# ──────────────────────────────────────────────────────────────────
WHITE_H_MIN,  WHITE_H_MAX  =   0, 180
WHITE_S_MIN,  WHITE_S_MAX  =  10,  30
WHITE_V_MIN,  WHITE_V_MAX  = 220, 255

YELLOW_H_MIN, YELLOW_H_MAX =  25,  60
YELLOW_S_MIN, YELLOW_S_MAX =  80, 255
YELLOW_V_MIN, YELLOW_V_MAX =  80, 255

ORANGE_H_MIN, ORANGE_H_MAX =   5,  30
ORANGE_S_MIN, ORANGE_S_MAX = 70, 255
ORANGE_V_MIN, ORANGE_V_MAX = 150, 255

MIN_CONTOUR_AREA = 2500
TARGET_WHITE_X_RATIO = 0.20
TARGET_YELLOW_X_RATIO = 0.10

class LaneDetectionNode(Node):

    def __init__(self):
        super().__init__('lane_detection_node')

        # ── Parameters ──────────────────────
        self.declare_parameter('white_h_min',  WHITE_H_MIN)
        self.declare_parameter('white_h_max',  WHITE_H_MAX)
        self.declare_parameter('white_s_min',  WHITE_S_MIN)
        self.declare_parameter('white_s_max',  WHITE_S_MAX)
        self.declare_parameter('white_v_min',  WHITE_V_MIN)
        self.declare_parameter('white_v_max',  WHITE_V_MAX)

        self.declare_parameter('yellow_h_min', YELLOW_H_MIN)
        self.declare_parameter('yellow_h_max', YELLOW_H_MAX)
        self.declare_parameter('yellow_s_min', YELLOW_S_MIN)
        self.declare_parameter('yellow_s_max', YELLOW_S_MAX)
        self.declare_parameter('yellow_v_min', YELLOW_V_MIN)
        self.declare_parameter('yellow_v_max', YELLOW_V_MAX)

        self.declare_parameter('orange_h_min', ORANGE_H_MIN)
        self.declare_parameter('orange_h_max', ORANGE_H_MAX)
        self.declare_parameter('orange_s_min', ORANGE_S_MIN)
        self.declare_parameter('orange_s_max', ORANGE_S_MAX)
        self.declare_parameter('orange_v_min', ORANGE_V_MIN)
        self.declare_parameter('orange_v_max', ORANGE_V_MAX)

        self.declare_parameter('crop_top_ratio', 0.5)
        self.declare_parameter('crop_bottom_ratio', 0.1)
        self.declare_parameter('crop_left_ratio', 0.1)
        self.declare_parameter('crop_top_orange_ratio', 0.1)
        self.declare_parameter('yellow_target_x_ratio', TARGET_YELLOW_X_RATIO)
        self.declare_parameter('yellow_weight', 0.5)
        self.declare_parameter('right_bias', 0.3)
        self.declare_parameter('yellow_memory_secs', 0.5)
        self.declare_parameter('min_orange_pixels', 4500)
        self.declare_parameter('debug_image', True)

        self.last_yellow_cx    = None
        self.last_yellow_stamp = 0.0

        # ── Subscribers ──────────────────────────────────────────
        self.sub_img = self.create_subscription(Image, '/camera_2/image_raw', self.image_callback, 10)

        # ── Publishers ───────────────────────────────────────────
        self.pub_error      = self.create_publisher(Float32,     '/lane/error',          10)
        self.pub_eor        = self.create_publisher(Bool,        '/lane/end_of_road',    10)
        self.pub_white_det  = self.create_publisher(Bool,        '/lane/white_detected', 10)
        self.pub_markers    = self.create_publisher(MarkerArray, '/lane/markers',        10)
        self.pub_debug        = self.create_publisher(Image, '/lane/debug_image',  10)
        self.pub_debug_orange = self.create_publisher(Image, '/lane/debug_orange', 10)

        self.bridge = CvBridge()
        self.get_logger().info('Lane detection node started.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'cv_bridge error: {e}')
            return
        
        h, w = frame.shape[:2]

        crop_top    = self.get_parameter('crop_top_ratio').value
        crop_bottom = self.get_parameter('crop_bottom_ratio').value
        crop_left   = self.get_parameter('crop_left_ratio').value

        roi_y      = int(h * crop_top)
        roi_y_bot  = int(h * (1.0 - crop_bottom))
        roi_x      = int(w * crop_left)

        roi = frame[roi_y:roi_y_bot, roi_x:w]
        roi_h, roi_w = roi.shape[:2] 

        orange_top = int(h * self.get_parameter('crop_top_orange_ratio').value)
        roi_orange = frame[orange_top:roi_y_bot, roi_x:]

        hsv        = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        hsv_orange = cv2.cvtColor(roi_orange, cv2.COLOR_BGR2HSV)

        white_mask        = self._white_mask(hsv)
        yellow_mask       = self._yellow_mask(hsv)
        orange_mask       = self._orange_mask(hsv_orange)
        orange_mask_debug = self._orange_mask(hsv)

        # ── NUEVO: Extraer segmentos de línea en lugar de centroides ──
        white_error = 0.0
        white_cx = None
        white_cnt = self._largest_contour(white_mask)
        white_segment = None # Tupla (p1, p2)
        
        if white_cnt is not None:
            white_segment, white_cx = self._fit_line_segment(white_cnt, roi_h)
            if white_cx is not None:
                target_white_x = int(roi_w * (1.0 - TARGET_WHITE_X_RATIO))
                white_error = (white_cx - target_white_x) / float(roi_w / 2)

        yellow_error      = 0.0
        yellow_cx         = None
        yellow_available  = False
        yellow_from_memory = False
        yellow_segment = None
        
        now_sec   = self.get_clock().now().nanoseconds * 1e-9
        yellow_cnt = self._largest_contour(yellow_mask)

        if yellow_cnt is not None:
            yellow_segment, yellow_cx = self._fit_line_segment(yellow_cnt, roi_h)
            if yellow_cx is not None:
                target_yellow_x = int(roi_w * self.get_parameter('yellow_target_x_ratio').value)
                yellow_error = (yellow_cx - target_yellow_x) / float(roi_w / 2)
                self.last_yellow_cx    = yellow_cx
                self.last_yellow_stamp = now_sec
                yellow_available = True
        else:
            memory_secs = self.get_parameter('yellow_memory_secs').value
            if (self.last_yellow_cx is not None and (now_sec - self.last_yellow_stamp) < memory_secs):
                yellow_cx = self.last_yellow_cx
                target_yellow_x = int(roi_w * self.get_parameter('yellow_target_x_ratio').value)
                yellow_error = (yellow_cx - target_yellow_x) / float(roi_w / 2)
                yellow_available  = True
                yellow_from_memory = True

        white_available = white_cx is not None
        yw = self.get_parameter('yellow_weight').value
        
        if white_available and yellow_available:
            final_error = (1.0 - yw) * white_error + yw * yellow_error
        elif white_available:
            final_error = white_error
        elif yellow_available:
            bias = self.get_parameter('right_bias').value
            final_error = yellow_error + bias
        else:
            final_error = 0.0

        final_error = max(-1.5, min(1.5, final_error))

        orange_cnt    = self._largest_contour(orange_mask)
        orange_pixels = int(cv2.countNonZero(orange_mask))
        min_orange    = self.get_parameter('min_orange_pixels').value
        end_of_road   = orange_pixels >= min_orange

        self.pub_error.publish(Float32(data=float(final_error)))
        self.pub_eor.publish(Bool(data=end_of_road))
        self.pub_white_det.publish(Bool(data=white_available))
        
        # Publicamos los marcadores usando los SEGMENTOS
        self._publish_markers(roi_w, white_segment, yellow_segment, orange_cnt)

        if self.get_parameter('debug_image').value:
            self._publish_debug(roi, white_mask, yellow_mask, orange_mask_debug,
                                white_cx, yellow_cx, white_segment, yellow_segment, roi_w,
                                white_available, orange_pixels, yellow_from_memory)
            self._publish_debug_orange(roi_orange, orange_mask, orange_pixels, end_of_road)

    def _white_mask(self, hsv):
        lo = np.array([self.get_parameter('white_h_min').value, self.get_parameter('white_s_min').value, self.get_parameter('white_v_min').value])
        hi = np.array([self.get_parameter('white_h_max').value, self.get_parameter('white_s_max').value, self.get_parameter('white_v_max').value])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def _yellow_mask(self, hsv):
        lo = np.array([self.get_parameter('yellow_h_min').value, self.get_parameter('yellow_s_min').value, self.get_parameter('yellow_v_min').value])
        hi = np.array([self.get_parameter('yellow_h_max').value, self.get_parameter('yellow_s_max').value, self.get_parameter('yellow_v_max').value])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))

    def _orange_mask(self, hsv):
        lo = np.array([self.get_parameter('orange_h_min').value, self.get_parameter('orange_s_min').value, self.get_parameter('orange_v_min').value])
        hi = np.array([self.get_parameter('orange_h_max').value, self.get_parameter('orange_s_max').value, self.get_parameter('orange_v_max').value])
        mask = cv2.inRange(hsv, lo, hi)
        return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < MIN_CONTOUR_AREA: return None
        return best

    # ── NUEVO: Ajuste de línea ────────────────────────────────────
    def _fit_line_segment(self, contour, roi_h):
        """Ajusta una línea al contorno y devuelve (segmento, centroide_x)"""
        # Ajustamos una línea 2D usando mínimos cuadrados
        [vx, vy, x, y] = cv2.fitLine(contour, cv2.DIST_L2, 0, 0.01, 0.01)
        
        # Evitar división por cero si la línea es completamente horizontal
        if vy == 0: 
            return None, None
            
        m = vy / vx
        b = y - m * x
        
        # Proyectamos la línea hasta el borde superior (y=0) e inferior (y=roi_h) del ROI
        y1 = roi_h
        x1 = int((y1 - b) / m)
        y2 = 0
        x2 = int((y2 - b) / m)
        
        segment = ((x1, y1), (x2, y2))
        
        # Calculamos el centroide tradicional para el control PD (a mitad del ROI)
        cx = int(( (roi_h/2) - b) / m)
        
        return segment, cx

    # ── Marker publisher modificado para segmentos ────────────────
    def _publish_markers(self, img_w, white_segment, yellow_segment, orange_cnt):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        def make_line_marker(mid, r, g, b, segment):
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp
            m.ns = 'lanes'
            m.id = mid
            m.type = Marker.LINE_LIST # ¡Cambiado de SPHERE a LINE_LIST!
            m.action = Marker.ADD
            m.scale.x = 0.05 # Grosor de la línea
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            
            # Mapeo simple píxeles -> metros (¡Ajustar según la calibración de tu cámara!)
            px_to_m = 0.002 
            
            # Punto 1 (más cerca del robot)
            p1 = Point()
            p1.x = 0.4 # Distancia fija hacia adelante para el inicio del ROI
            p1.y = (img_w / 2 - segment[0][0]) * px_to_m
            p1.z = 0.0
            
            # Punto 2 (más lejos)
            p2 = Point()
            p2.x = 0.8 # Distancia hasta el final del ROI
            p2.y = (img_w / 2 - segment[1][0]) * px_to_m
            p2.z = 0.0
            
            m.points = [p1, p2]
            return m

        if white_segment is not None:
            arr.markers.append(make_line_marker(0, 1.0, 1.0, 1.0, white_segment))

        if yellow_segment is not None:
            arr.markers.append(make_line_marker(1, 1.0, 1.0, 0.0, yellow_segment))

        if orange_cnt is not None:
            # El naranja lo dejamos como un punto (obstáculo/parada)
            m = Marker()
            m.header.frame_id = 'base_link'
            m.header.stamp = stamp
            m.ns = 'lanes'
            m.id = 2
            m.type = Marker.SPHERE
            m.action = Marker.ADD
            m.scale.x = m.scale.y = m.scale.z = 0.2
            m.color.r = 1.0; m.color.g = 0.5; m.color.b = 0.0; m.color.a = 1.0
            m.pose.position.x = 0.5
            m.pose.position.y = 0.0
            m.pose.position.z = 0.0
            arr.markers.append(m)

        self.pub_markers.publish(arr)

    # ── Debug image ───────────────────────────────────────────────
    def _publish_debug(self, roi, white_mask, yellow_mask, orange_mask,
                        white_cx, yellow_cx, white_segment, yellow_segment, img_w, white_active,
                        orange_pixels=0, yellow_from_memory=False):
        debug = roi.copy()
        debug[white_mask  > 0] = (200, 200, 255)
        debug[yellow_mask > 0] = (0,   220, 220)
        debug[orange_mask > 0] = (0,   120, 255)

        target_white_x = int(img_w * (1.0 - TARGET_WHITE_X_RATIO))
        cv2.line(debug, (target_white_x, 0), (target_white_x, debug.shape[0]), (0, 255, 0), 1)

        target_yellow_x = int(img_w * self.get_parameter('yellow_target_x_ratio').value)
        cv2.line(debug, (target_yellow_x, 0), (target_yellow_x, debug.shape[0]), (0, 180, 0), 1)

        # Dibujar segmentos de línea ajustados
        if white_segment is not None:
            cv2.line(debug, white_segment[0], white_segment[1], (255, 255, 0), 3 if white_active else 1)
        if yellow_segment is not None:
            color = (180, 0, 180) if yellow_from_memory else (255, 0, 255)
            cv2.line(debug, yellow_segment[0], yellow_segment[1], color, 3 if not white_active else 1)

        if white_cx is not None and yellow_cx is not None: label = 'SRC: BLEND'
        elif white_cx is not None: label = 'SRC: WHITE'
        elif yellow_cx is not None: label = 'SRC: YEL+BIAS (MEM)' if yellow_from_memory else 'SRC: YEL+BIAS'
        else: label = 'SRC: NONE'
        cv2.putText(debug, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)

        min_px = self.get_parameter('min_orange_pixels').value
        color  = (0, 80, 255) if orange_pixels >= min_px else (180, 180, 180)
        cv2.putText(debug, f'ORANGE: {orange_pixels}px', (5, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

        msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self.pub_debug.publish(msg)

    def _publish_debug_orange(self, roi_orange, orange_mask, orange_pixels, triggered):
        debug = roi_orange.copy()
        debug[orange_mask > 0] = (0, 100, 255)
        cv2.line(debug, (0, 0), (debug.shape[1], 0), (0, 255, 255), 3)
        min_px = self.get_parameter('min_orange_pixels').value
        color  = (0, 255, 0) if triggered else (0, 0, 255)
        status = 'TRIGGERED' if triggered else 'NOT TRIGGERED'
        cv2.putText(debug, f'{status}  {orange_pixels}/{min_px}px',
                    (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        msg = self.bridge.cv2_to_imgmsg(debug, encoding='bgr8')
        self.pub_debug_orange.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()