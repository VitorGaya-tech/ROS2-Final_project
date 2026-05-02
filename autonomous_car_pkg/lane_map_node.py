"""
lane_map_node.py
----------------
Toma la parte más cercana de las líneas detectadas y las guarda 
como "miguitas de pan" para dibujar el circuito de forma progresiva.
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

MAX_MARKERS   = 5000   # Más puntos para dibujar el circuito completo
DEDUPE_DIST   = 0.05   # Cada 5 cm dejamos un punto nuevo (miguita de pan)

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
        self.get_logger().info('Lane map node started (Breadcrumb Mode con frame odom).')

    def _markers_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns != 'lanes' or m.id not in (0, 1):
                continue

            # Extraemos SOLO el punto más cercano al robot (p1_local)
            if m.type == Marker.LINE_LIST and len(m.points) >= 2:
                p1_local = m.points[0] # Este es el punto que está pegado al coche

                p1_odom = self._to_odom_frame_point(p1_local)

                if p1_odom is None:
                    continue # Ignorar si falla la odometría (TF)

                trail = self.white_trail if m.id == 0 else self.yellow_trail
                self._append_point_deduped(trail, p1_odom)

    def _append_point_deduped(self, trail: list, p: Point):
        # Evitar guardar puntos en el mismo sitio si el robot está parado
        if len(trail) > 0:
            last_p = trail[-1]
            dx = p.x - last_p.x
            dy = p.y - last_p.y
            if (dx * dx + dy * dy) < DEDUPE_DIST ** 2:
                return # Está muy cerca del anterior, lo ignoramos

        trail.append(p)

        if len(trail) > MAX_MARKERS:
            trail.pop(0) # Borramos los más viejos si llegamos al límite

    def _to_odom_frame_point(self, p_local: Point):
        if not TF_AVAILABLE:
            return None

        try:
            # CAMBIO: Usamos 'odom' como referencia base
            transform = self.tf_buffer.lookup_transform(
                'odom', 'base_link',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))

            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            q  = transform.transform.rotation
            
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)

            p_odom = Point()
            p_odom.x = tx + p_local.x * math.cos(yaw) - p_local.y * math.sin(yaw)
            p_odom.y = ty + p_local.x * math.sin(yaw) + p_local.y * math.cos(yaw)
            p_odom.z = 0.0
            return p_odom

        except (LookupException, ConnectivityException, ExtrapolationException):
            return None

    def _republish_map(self):
        if not self.white_trail and not self.yellow_trail:
            return

        stamp = self.get_clock().now().to_msg()
        arr   = MarkerArray()

        def make_points_marker(trail, mid, r, g, b):
            m = Marker()
            # CAMBIO: Publicamos los puntos en 'odom'
            m.header.frame_id = 'odom'
            m.header.stamp    = stamp
            m.ns      = 'lane_map'
            m.id      = mid
            m.type    = Marker.POINTS 
            m.action  = Marker.ADD
            m.scale.x = 0.04 
            m.scale.y = 0.04
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            m.points  = list(trail)
            return m

        if self.white_trail:
            arr.markers.append(make_points_marker(self.white_trail,  0, 1.0, 1.0, 1.0))
        if self.yellow_trail:
            arr.markers.append(make_points_marker(self.yellow_trail, 1, 1.0, 0.9, 0.0))

        self.pub_markers.publish(arr)

def main(args=None):
    rclpy.init(args=args)
    node = LaneMapNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()

if __name__ == '__main__':
    main()