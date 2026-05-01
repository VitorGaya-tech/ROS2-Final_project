"""
lane_map_node.py
----------------
Toma los segmentos de línea de lane_detection_node y los republica en el
frame /map (usando TF para transformar de base_link → map).
"""

import math
import rclpy
from rclpy.node import Node
from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point

try:
    from tf2_ros import Buffer, TransformListener
    from tf2_ros import LookupException, ConnectivityException, ExtrapolationException
    import tf2_geometry_msgs  # noqa
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


MAX_MARKERS   = 2000   # max stored waypoints per trail (en puntos, es decir, 1000 segmentos)
DEDUPE_DIST   = 0.10   # metros — saltar si el segmento es casi idéntico al anterior


class LaneMapNode(Node):

    def __init__(self):
        super().__init__('lane_map_node')

        self.white_trail:  list[Point] = []
        self.yellow_trail: list[Point] = []

        if TF_AVAILABLE:
            self.tf_buffer   = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.get_logger().warn('tf2 no está disponible. No se puede mapear globalmente.')

        self.create_subscription(MarkerArray, '/lane/markers', self._markers_cb, 10)

        self.pub_markers = self.create_publisher(MarkerArray, '/lane_map/markers', 10)
        self.pub_path    = self.create_publisher(Path,        '/lane_map/path',    10)

        self.create_timer(0.5, self._republish_map)
        self.get_logger().info('Lane map node started (Segment Mode).')

    # ── Incoming marker callback ───────────────────────────────────
    def _markers_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns != 'lanes':
                continue
            if m.id not in (0, 1):   # 0=white, 1=yellow; skip orange (2)
                continue

            # Ahora esperamos recibir LINE_LIST con al menos 2 puntos (un segmento)
            if m.type == Marker.LINE_LIST and len(m.points) >= 2:
                p1_local = m.points[0]
                p2_local = m.points[1]

                p1_map = self._to_map_frame_point(p1_local)
                p2_map = self._to_map_frame_point(p2_local)

                if p1_map is None or p2_map is None:
                    continue # El SLAM falló en este instante, ignoramos

                trail = self.white_trail if m.id == 0 else self.yellow_trail
                self._append_segment_deduped(trail, p1_map, p2_map)

    def _append_segment_deduped(self, trail: list, p1: Point, p2: Point):
        # Comprobar si el inicio de este segmento está muy cerca del anterior
        if len(trail) >= 2:
            last_p1 = trail[-2]
            dx = p1.x - last_p1.x
            dy = p1.y - last_p1.y
            if (dx * dx + dy * dy) < DEDUPE_DIST ** 2:
                return # Segmento redundante, lo ignoramos

        # Añadimos ambos puntos al trail
        trail.append(p1)
        trail.append(p2)

        # Mantenemos el límite borrando de 2 en 2 (para no romper segmentos)
        while len(trail) > MAX_MARKERS:
            trail.pop(0)
            trail.pop(0)

    # ── TF helper (ahora procesa Points directos) ─────────────────
    def _to_map_frame_point(self, p_local: Point):
        """Transform local point from base_link to map."""
        if not TF_AVAILABLE:
            return None

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            q  = transform.transform.rotation
            
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)

            p_map = Point()
            p_map.x = tx + p_local.x * math.cos(yaw) - p_local.y * math.sin(yaw)
            p_map.y = ty + p_local.x * math.sin(yaw) + p_local.y * math.cos(yaw)
            p_map.z = 0.0
            return p_map

        except (LookupException, ConnectivityException, ExtrapolationException):
            return None
        except Exception as e:
            self.get_logger().debug(f'Error inesperado de TF: {e}')
            return None

    # ── Periodic republish ─────────────────────────────────────────
    def _republish_map(self):
        if not self.white_trail and not self.yellow_trail:
            return

        stamp = self.get_clock().now().to_msg()
        arr   = MarkerArray()

        def make_line_list(trail, mid, r, g, b):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = stamp
            m.ns      = 'lane_map'
            m.id      = mid
            m.type    = Marker.LINE_LIST # Dibujará segmentos independientes
            m.action  = Marker.ADD
            m.scale.x = 0.02 # Grosor de la línea
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.8
            m.points  = list(trail)
            return m

        if self.white_trail:
            arr.markers.append(make_line_list(self.white_trail,  0, 1.0, 1.0, 1.0))
        if self.yellow_trail:
            arr.markers.append(make_line_list(self.yellow_trail, 1, 1.0, 0.9, 0.0))

        self.pub_markers.publish(arr)

        # Path para navegación usando el centro de los segmentos blancos
        if self.white_trail:
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp    = stamp
            # Iteramos de 2 en 2 para sacar el centro de cada segmento
            for i in range(0, len(self.white_trail) - 1, 2):
                p1 = self.white_trail[i]
                p2 = self.white_trail[i+1]
                
                mid_p = Point()
                mid_p.x = (p1.x + p2.x) / 2.0
                mid_p.y = (p1.y + p2.y) / 2.0
                mid_p.z = 0.0

                ps = PoseStamped()
                ps.header = path.header
                ps.pose.position = mid_p
                ps.pose.orientation.w = 1.0
                path.poses.append(ps)
                
            self.pub_path.publish(path)


def main(args=None):
    rclpy.init(args=args)
    node = LaneMapNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()