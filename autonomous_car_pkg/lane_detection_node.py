import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from visualization_msgs.msg import Marker, MarkerArray
from cv_bridge import CvBridge

# --- Ajuste de Filtros para Cámara Inferior ---
WHITE_H_MIN,  WHITE_H_MAX  =  0, 180
WHITE_S_MIN,  WHITE_S_MAX  =  0, 40    # Muy poca saturación para blanco
WHITE_V_MIN,  WHITE_V_MAX  = 200, 255  # Muy alto brillo

YELLOW_H_MIN, YELLOW_H_MAX =  20, 40   # Rango más estrecho para amarillo
YELLOW_S_MIN, YELLOW_S_MAX =  100, 255
YELLOW_V_MIN, YELLOW_V_MAX =  100, 255

MIN_CONTOUR_AREA = 1500 # Bajamos un poco para la cámara inferior
TARGET_WHITE_X_RATIO = 0.25 # Distancia de la línea blanca al borde derecho

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Parámetros básicos
        self.declare_parameter('white_v_min', WHITE_V_MIN)
        self.declare_parameter('yellow_weight', 0.2) # Bajamos el peso de la amarilla
        self.declare_parameter('crop_top_ratio', 0.6) # Mirar solo el suelo

        self.sub_img = self.create_subscription(Image, '/camera_1/image_raw', self.image_callback, 10)
        self.pub_error = self.create_publisher(Float32, '/lane/error', 10)
        self.pub_debug = self.create_publisher(Image, '/lane/debug_image', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info('Detección de carril optimizada para Cámara Inferior.')

    def image_callback(self, msg: Image):
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            return

        h, w = frame.shape[:2]
        # Cortar para ver solo el suelo (evita ver luces del techo o el horizonte)
        roi_y = int(h * self.get_parameter('crop_top_ratio').value)
        roi = frame[roi_y:h, :]
        h_roi, w_roi = roi.shape[:2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Máscaras
        white_mask = cv2.inRange(hsv, np.array([0, 0, 200]), np.array([180, 50, 255]))
        yellow_mask = cv2.inRange(hsv, np.array([20, 100, 100]), np.array([40, 255, 255]))

        white_cnt = self._largest_contour(white_mask)
        yellow_cnt = self._largest_contour(yellow_mask)

        error = 0.0
        active_line = "NONE"

        # LÓGICA DE CONTROL PRIORITARIA
        if white_cnt is not None:
            # Si vemos la blanca, mandamos nosotros
            M = cv2.moments(white_cnt)
            cx = int(M['m10'] / M['m00'])
            target_x = int(w_roi * (1.0 - TARGET_WHITE_X_RATIO))
            error = (cx - target_x) / float(w_roi / 2)
            active_line = "WHITE"
        elif yellow_cnt is not None:
            # Si no hay blanca, usamos la amarilla pero con cuidado
            M = cv2.moments(yellow_cnt)
            cx = int(M['m10'] / M['m00'])
            target_x = int(w_roi * 0.2) # La amarilla debería estar a la izquierda
            error = (cx - target_x) / float(w_roi / 2) + 0.4 # Offset para no cruzar
            active_line = "YELLOW_ONLY"
        
        # Publicar error
        self.pub_error.publish(Float32(data=float(error)))

        # Imagen de Debug
        debug = roi.copy()
        if white_cnt is not None: cv2.drawContours(debug, [white_cnt], -1, (0, 255, 0), 2)
        if yellow_cnt is not None: cv2.drawContours(debug, [yellow_cnt], -1, (0, 255, 255), 2)
        cv2.putText(debug, f"LINE: {active_line} ERR: {error:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug, encoding='bgr8'))

    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts: return None
        best = max(cnts, key=cv2.contourArea)
        return best if cv2.contourArea(best) > MIN_CONTOUR_AREA else None

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()