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
  /lane_map/markers   (visualization_msgs/MarkerArray)  — white + yellow trails in map frame
  /lane_map/path      (nav_msgs/Path)                   — trajectory of white line

Marker IDs from lane_detection_node:
  id=0 → white line
  id=1 → yellow line
  id=2 → orange (ignored here, not mapped)

Each colour has its own independent LINE_STRIP so they never draw a
connecting segment between a white point and a yellow point.
"""

import math

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


MAX_MARKERS   = 2000   # max stored waypoints per trail
DEDUPE_DIST   = 0.10   # metres — skip if last point is closer than this
SMOOTH_WINDOW = 5      # points averaged for display (raw data unchanged)


class LaneMapNode(Node):

    def __init__(self):
        super().__init__('lane_map_node')

        self.white_trail:  list[Point] = []
        self.yellow_trail: list[Point] = []

        if TF_AVAILABLE:
            self.tf_buffer   = Buffer()
            self.tf_listener = TransformListener(self.tf_buffer, self)
        else:
            self.get_logger().warn('tf2 not available — using base_link frame directly')

        self.create_subscription(
            MarkerArray, '/lane/markers', self._markers_cb, 10)

        self.pub_markers = self.create_publisher(MarkerArray, '/lane_map/markers', 10)
        self.pub_path    = self.create_publisher(Path,        '/lane_map/path',    10)

        self.create_timer(0.5, self._republish_map)

        self.get_logger().info('Lane map node started.')

    # ── Incoming marker callback ───────────────────────────────────
    def _markers_cb(self, msg: MarkerArray):
        for m in msg.markers:
            if m.ns != 'lanes':
                continue
            if m.id not in (0, 1):   # 0=white, 1=yellow; skip orange (2)
                continue

            point_map = self._to_map_frame(m)
            if point_map is None:
                continue

            trail = self.white_trail if m.id == 0 else self.yellow_trail
            self._append_deduped(trail, point_map)

    def _append_deduped(self, trail: list, pt: Point):
        if trail:
            last = trail[-1]
            dx = pt.x - last.x
            dy = pt.y - last.y
            if (dx * dx + dy * dy) < DEDUPE_DIST ** 2:
                return
        trail.append(pt)
        if len(trail) > MAX_MARKERS:
            trail.pop(0)

    # ── TF helper ─────────────────────────────────────────────────
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
            p_map.x = tx + p.x * math.cos(yaw) - p.y * math.sin(yaw)
            p_map.y = ty + p.x * math.sin(yaw) + p.y * math.cos(yaw)
            p_map.z = 0.0
            return p_map

        except Exception:
            # TF not yet available (SLAM still initialising)
            return p

    # ── Smoothing helper ──────────────────────────────────────────
    def _smooth(self, trail: list) -> list:
        n = len(trail)
        if n < SMOOTH_WINDOW:
            return list(trail)
        half = SMOOTH_WINDOW // 2
        result = []
        for i in range(n):
            lo = max(0, i - half)
            hi = min(n, i + half + 1)
            pts = trail[lo:hi]
            p = Point()
            p.x = sum(pt.x for pt in pts) / len(pts)
            p.y = sum(pt.y for pt in pts) / len(pts)
            p.z = 0.0
            result.append(p)
        return result

    # ── Periodic republish ─────────────────────────────────────────
    def _republish_map(self):
        if not self.white_trail and not self.yellow_trail:
            return

        stamp = self.get_clock().now().to_msg()
        arr   = MarkerArray()

        def make_strip(trail, mid, r, g, b):
            m = Marker()
            m.header.frame_id = 'map'
            m.header.stamp    = stamp
            m.ns      = 'lane_map'
            m.id      = mid
            m.type    = Marker.LINE_STRIP
            m.action  = Marker.ADD
            m.scale.x = 0.02
            m.color.r = r
            m.color.g = g
            m.color.b = b
            m.color.a = 0.8
            m.points  = list(trail)
            return m

        if self.white_trail:
            arr.markers.append(make_strip(self._smooth(self.white_trail),  0, 1.0, 1.0, 1.0))
        if self.yellow_trail:
            arr.markers.append(make_strip(self._smooth(self.yellow_trail), 1, 1.0, 0.9, 0.0))

        self.pub_markers.publish(arr)

        # Path uses white trail (primary navigation reference)
        if self.white_trail:
            path = Path()
            path.header.frame_id = 'map'
            path.header.stamp    = stamp
            for pt in self._smooth(self.white_trail):
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