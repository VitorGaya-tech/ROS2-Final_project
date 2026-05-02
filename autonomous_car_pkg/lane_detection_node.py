import rclpy
from rclpy.node import Node
import cv2
import numpy as np
from sensor_msgs.msg import Image
from std_msgs.msg import Float32, Bool
from cv_bridge import CvBridge

# --- Ajuste de Filtros para Cámara Inferior (Cámara 2) ---
WHITE_V_MIN = 200  # Brillo mínimo para considerar algo como blanco
MIN_CONTOUR_AREA = 1200  # Área mínima de píxeles
TARGET_WHITE_X_RATIO = 0.25 # El objetivo es que la línea blanca esté al 25% del borde derecho

class LaneDetectionNode(Node):
    def __init__(self):
        super().__init__('lane_detection_node')

        # Parámetros para ajustar en tiempo real si fuera necesario
        self.declare_parameter('crop_top_ratio', 0.6) # Ignoramos el 60% superior de la imagen
        self.declare_parameter('yellow_weight', 0.2)

        # --- CAMBIADO A CAMERA_2 (Inferior) ---
        self.sub_img = self.create_subscription(
            Image, 
            '/camera_2/image_raw', 
            self.image_callback, 
            10)

        self.pub_error = self.create_publisher(Float32, '/lane/error', 10)
        self.pub_debug = self.create_publisher(Image, '/lane/debug_image', 10)
        self.bridge = CvBridge()
        
        self.get_logger().info('Lane Detection iniciado en CÁMARA 2 (Inferior).')

    def image_callback(self, msg: Image):
        try:
            # Convertir imagen de ROS a OpenCV
            frame = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'Error CV Bridge: {e}')
            return

        h, w = frame.shape[:2]
        
        # 1. CORTAR IMAGEN (ROI)
        # Miramos solo la parte inferior para evitar reflejos lejanos
        roi_y = int(h * self.get_parameter('crop_top_ratio').value)
        roi = frame[roi_y:h, :]
        h_roi, w_roi = roi.shape[:2]

        # 2. PROCESAMIENTO DE COLOR (HSV)
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        # Máscara Blanca: Buscamos cosas muy brillantes con poca saturación
        lower_white = np.array([0, 0, WHITE_V_MIN])
        upper_white = np.array([180, 60, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)

        # Máscara Amarilla: Color chillón en el centro
        lower_yellow = np.array([15, 100, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        # 3. ENCONTRAR LÍNEAS (Contornos)
        white_cnt = self._largest_contour(white_mask)
        yellow_cnt = self._largest_contour(yellow_mask)

        error = 0.0
        label = "PERDIDO"

        # 4. LÓGICA DE NAVEGACIÓN PRIORITARIA
        # Si ve la línea blanca (derecha), manda ella.
        if white_cnt is not None:
            M = cv2.moments(white_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                # Queremos que la línea blanca esté a la derecha del centro
                target_x = int(w_roi * (1.0 - TARGET_WHITE_X_RATIO))
                error = (cx - target_x) / float(w_roi / 2)
                label = "SIGUIENDO BLANCA"
        
        # Si NO ve la blanca, intenta seguir la amarilla (izquierda) con un margen de seguridad
        elif yellow_cnt is not None:
            M = cv2.moments(yellow_cnt)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                # La amarilla debería estar en el lado izquierdo
                target_x = int(w_roi * 0.25)
                # Sumamos un offset (+0.5) para forzar al coche a alejarse de la amarilla hacia la derecha
                error = ((cx - target_x) / float(w_roi / 2)) + 0.5
                label = "SIGUIENDO AMARILLA (PRECAUCIÓN)"

        # 5. PUBLICAR RESULTADOS
        self.pub_error.publish(Float32(data=float(error)))

        # 6. IMAGEN DE DEBUG PARA RQT_IMAGE_VIEW
        debug_img = roi.copy()
        if white_cnt is not None:
            cv2.drawContours(debug_img, [white_cnt], -1, (0, 255, 0), 3) # Verde para blanca
        if yellow_cnt is not None:
            cv2.drawContours(debug_img, [yellow_cnt], -1, (0, 255, 255), 3) # Amarillo para amarilla
        
        cv2.putText(debug_img, f"{label} ERR: {error:.2f}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
        
        self.pub_debug.publish(self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8'))

    def _largest_contour(self, mask):
        cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return None
        best = max(cnts, key=cv2.contourArea)
        if cv2.contourArea(best) < MIN_CONTOUR_AREA:
            return None
        return best

def main(args=None):
    rclpy.init(args=args)
    node = LaneDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()