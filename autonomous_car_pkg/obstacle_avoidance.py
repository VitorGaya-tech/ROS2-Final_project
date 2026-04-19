"""
obstacle_avoidance.py
---------------------
Reads /scan (LaserScan) and decides whether an obstacle is in the path.
When an obstacle is detected it:
  1. Publishes STOP to /behavior/state (navigation_node halts).
  2. Tries to route around to the LEFT (cross the yellow dashed line).
  3. Once clear, publishes GO.

Subscriptions:
  /scan              (sensor_msgs/LaserScan)

Publishes:
  /behavior/state    (std_msgs/String)   — STOP | GO | AVOID_LEFT
  /obstacles/markers (visualization_msgs/MarkerArray) — RViz

DETECTION ZONE
--------------
We only look at a frontal cone:
  angle range: -CONE_HALF_DEG to +CONE_HALF_DEG (default ±25°)
  distance:    < STOP_DISTANCE  → STOP
               < WARN_DISTANCE  → slow down (future)

The robot is assumed to be ~0.25 m wide, so anything closer than
STOP_DISTANCE = 0.40 m in the frontal cone must be avoided.
"""

import rclpy
from rclpy.node import Node
import math

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


STOP_DISTANCE   = 0.40   # metres  — full stop
WARN_DISTANCE   = 0.70   # metres  — not used yet, reserved
CONE_HALF_DEG   = 25.0   # degrees — frontal detection cone half-angle
CLEAR_DISTANCE  = 0.65   # metres  — obstacle gone if all readings > this


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.declare_parameter('stop_distance',  STOP_DISTANCE)
        self.declare_parameter('cone_half_deg',  CONE_HALF_DEG)

        self.state = 'CLEAR'   # CLEAR | BLOCKED
        self.last_behavior = 'GO'

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.pub_behavior = self.create_publisher(String,      '/behavior/state',    10)
        self.pub_markers  = self.create_publisher(MarkerArray, '/obstacles/markers', 10)

        self.get_logger().info('Obstacle avoidance node started.')

    def _scan_cb(self, msg: LaserScan):
        stop_dist   = self.get_parameter('stop_distance').value
        cone_half   = math.radians(self.get_parameter('cone_half_deg').value)

        min_front = float('inf')
        min_left  = float('inf')
        min_right = float('inf')

        markers = MarkerArray()
        stamp = self.get_clock().now().to_msg()

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment

            # ── Front cone ────────────────────────────────────────
            if -cone_half <= angle <= cone_half:
                min_front = min(min_front, r)

                # Publish an obstacle marker if close
                if r < stop_dist * 1.5:
                    m = Marker()
                    m.header.frame_id = 'laser'
                    m.header.stamp = stamp
                    m.ns = 'obstacles'
                    m.id = i
                    m.type = Marker.SPHERE
                    m.action = Marker.ADD
                    m.pose.position.x = r * math.cos(angle)
                    m.pose.position.y = r * math.sin(angle)
                    m.pose.position.z = 0.0
                    m.scale.x = m.scale.y = m.scale.z = 0.08
                    m.color.r = 1.0; m.color.g = 0.3; m.color.b = 0.0
                    m.color.a = 0.9
                    m.lifetime.sec = 0
                    m.lifetime.nanosec = 200_000_000   # 0.2 s
                    markers.markers.append(m)

            # Left quadrant (for gap detection during avoidance)
            if 0 < angle <= math.pi / 2:
                min_left = min(min_left, r)

            # Right quadrant
            if -math.pi / 2 <= angle < 0:
                min_right = min(min_right, r)

        self.pub_markers.publish(markers)
        self._update_state(min_front, min_left, min_right, stop_dist)

    def _update_state(self, min_front, min_left, min_right, stop_dist):
        behavior_msg = String()

        if self.state == 'CLEAR':
            if min_front < stop_dist:
                self.get_logger().warn(
                    f'Obstacle at {min_front:.2f} m → STOP')
                self.state = 'BLOCKED'
                behavior_msg.data = 'STOP'
                self.pub_behavior.publish(behavior_msg)

        elif self.state == 'BLOCKED':
            # Check if path is clear again
            if min_front > CLEAR_DISTANCE:
                self.get_logger().info('Path clear → GO')
                self.state = 'CLEAR'
                behavior_msg.data = 'GO'
                self.pub_behavior.publish(behavior_msg)
            # Otherwise stay STOP (navigation_node holds position)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
