"""
lane_map_node.py
----------------
Takes the lane markers from lane_detection_node and re-publishes them in the
/map frame (using TF to transform from base_link → map).

This allows RViz to accumulate a "painted" map of where lane lines have
been seen throughout the run — satisfying the mapping requirement.

Subscriptions:
  /lane/markers       (visualization_msgs/MarkerArray)  — in base_link frame

Publishes:
  /map/lane_markers   (visualization_msgs/MarkerArray)  — in map frame
  /map/lane_path      (nav_msgs/Path)                   — trajectory of white line

The node accumulates up to MAX_MARKERS waypoints and periodically publishes
them as a persistent marker array so they appear as a "breadcrumb trail" in RViz.
"""

import rclpy
from rclpy.node import Node

from visualization_msgs.msg import Marker, MarkerArray
from nav_msgs.msg import Path
from geometry_msgs.msg import PoseStamped, Point

try:
    from tf2_ros import Buffer, TransformListener
    import tf2_geometry_msgs  # noqa: needed for do_transform_point
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


MAX_MARKERS = 2000   # max stored lane waypoints
DEDUPE_DIST = 0.05   # metres — skip if last point is closer than this


class LaneMapNode(Node):

    def __init__(self):
        super().__init__('lane_map_node')

        self.white_trail: list[Point] = []
        self.marker_id_counter = 0

        # TF listener to convert base_link → map
        if TF_AVAILABLE:
            self.tf_buffer   = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.get_logger().warn('tf2 not available — using base_link frame directly')

        self.create_subscription(
            MarkerArray, '/lane/markers', self._markers_cb, 10)

        self.pub_lane_markers = self.create_publisher(
            MarkerArray, '/map/lane_markers', 10)
        self.pub_path = self.create_publisher(
            Path, '/map/lane_path', 10)

        # Republish accumulated map every 0.5 s
        self.create_timer(0.5, self._republish_map)

        self.get_logger().info('Lane map node started.')

    def _markers_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns != 'lanes':
                continue

            # Try to transform to map frame
            point_map = self._to_map_frame(m)
            if point_map is None:
                continue

            # De-duplicate: skip if too close to the last stored point
            if self.white_trail:
                last = self.white_trail[-1]
                dx = point_map.x - last.x
                dy = point_map.y - last.y
                if (dx*dx + dy*dy) < DEDUPE_DIST**2:
                    continue

            self.white_trail.append(point_map)
            if len(self.white_trail) > MAX_MARKERS:
                self.white_trail.pop(0)

    def _to_map_frame(self, marker: Marker):
        """Transform marker pose from base_link to map. Falls back to base_link."""
        p = Point()
        p.x = marker.pose.position.x
        p.y = marker.pose.position.y
        p.z = 0.0

        if not TF_AVAILABLE:
            return p

        try:
            transform = self.tf_buffer.lookup_transform(
                'map', 'base_footprint',
                rclpy.time.Time(),
                timeout=rclpy.duration.Duration(seconds=0.05))

            # Manual transform application (translation + yaw only for ground plane)
            import math
            tx = transform.transform.translation.x
            ty = transform.transform.translation.y
            q  = transform.transform.rotation
            # Yaw from quaternion
            siny = 2.0 * (q.w * q.z + q.x * q.y)
            cosy = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
            yaw  = math.atan2(siny, cosy)

            p_map = Point()
            p_map.x = tx + p.x * math.cos(yaw) - p.y * math.sin(yaw)
            p_map.y = ty + p.x * math.sin(yaw) + p.y * math.cos(yaw)
            p_map.z = 0.0
            return p_map

        except Exception:
            # TF not yet available (SLAM still initialising)
            return p

    def _republish_map(self):
        if not self.white_trail:
            return

        stamp = self.get_clock().now().to_msg()

        # ── MarkerArray ───────────────────────────────────────────
        arr = MarkerArray()
        # One LINE_STRIP marker with all points is far more efficient
        m = Marker()
        m.header.frame_id = 'map'
        m.header.stamp    = stamp
        m.ns    = 'lane_map'
        m.id    = 0
        m.type  = Marker.LINE_STRIP
        m.action = Marker.ADD
        m.scale.x = 0.02          # line width in metres
        m.color.r = 1.0
        m.color.g = 1.0
        m.color.b = 1.0
        m.color.a = 0.8
        m.points  = list(self.white_trail)
        arr.markers.append(m)
        self.pub_lane_markers.publish(arr)

        # ── Path (also useful in RViz) ────────────────────────────
        path = Path()
        path.header.frame_id = 'map'
        path.header.stamp    = stamp
        for pt in self.white_trail:
            ps = PoseStamped()
            ps.header = path.header
            ps.pose.position = pt
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
