"""
obstacle_avoidance.py
---------------------
Reads /scan (LaserScan) and decides whether an obstacle is in the path.

Uses ONLY the frontal cone (±CONE_HALF_DEG) — lateral readings are ignored
because walls / lane barriers are always close on the sides.

States & published values on /behavior/state:
  CLEAR   → GO          (no obstacle)
  WARN    → AVOID_RIGHT (obstacle 0.30–0.70 m ahead → change lane)
  BLOCKED → STOP        (obstacle < 0.30 m → no room to manoeuvre)

Subscriptions:
  /scan              (sensor_msgs/LaserScan)

Publishes:
  /behavior/state    (std_msgs/String)   — GO | AVOID_RIGHT | STOP
  /obstacles/markers (visualization_msgs/MarkerArray) — RViz
"""

import rclpy
from rclpy.node import Node
import math

from sensor_msgs.msg import LaserScan
from std_msgs.msg import String
from visualization_msgs.msg import Marker, MarkerArray


STOP_DISTANCE   = 0.30   # metres — full stop (obstacle too close)
AVOID_DISTANCE  = 0.70   # metres — start lane change
CLEAR_DISTANCE  = 0.80   # metres — hysteresis: obstacle considered gone
CONE_HALF_DEG   = 25.0   # degrees — frontal detection cone (only front!)


class ObstacleAvoidanceNode(Node):

    def __init__(self):
        super().__init__('obstacle_avoidance')

        self.declare_parameter('stop_distance',  STOP_DISTANCE)
        self.declare_parameter('avoid_distance', AVOID_DISTANCE)
        self.declare_parameter('cone_half_deg',  CONE_HALF_DEG)

        self.state = 'CLEAR'   # CLEAR | WARN | BLOCKED

        self.create_subscription(LaserScan, '/scan', self._scan_cb, 10)

        self.pub_behavior = self.create_publisher(String,      '/behavior/state',    10)
        self.pub_markers  = self.create_publisher(MarkerArray, '/obstacles/markers', 10)

        self.get_logger().info('Obstacle avoidance node started.')

    def _scan_cb(self, msg: LaserScan):
        stop_dist  = self.get_parameter('stop_distance').value
        avoid_dist = self.get_parameter('avoid_distance').value
        cone_half  = math.radians(self.get_parameter('cone_half_deg').value)

        min_front = float('inf')
        markers   = MarkerArray()
        stamp     = self.get_clock().now().to_msg()

        for i, r in enumerate(msg.ranges):
            if not math.isfinite(r) or r < msg.range_min or r > msg.range_max:
                continue
            angle = msg.angle_min + i * msg.angle_increment

            if -cone_half <= angle <= cone_half:
                min_front = min(min_front, r)

                if r < avoid_dist:
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
                    m.lifetime.nanosec = 200_000_000   # 0.2 s
                    markers.markers.append(m)

        self.pub_markers.publish(markers)
        self._update_state(min_front, stop_dist, avoid_dist)

    def _update_state(self, min_front, stop_dist, avoid_dist):
        prev_state = self.state

        if min_front < stop_dist:
            self.state = 'BLOCKED'
        elif min_front < avoid_dist:
            self.state = 'WARN'
        elif min_front > CLEAR_DISTANCE:
            self.state = 'CLEAR'
        # else: stay in current state (hysteresis band between avoid_dist and CLEAR_DISTANCE)

        if self.state == prev_state:
            return   # no change → don't flood the topic

        msg = String()
        if self.state == 'BLOCKED':
            msg.data = 'STOP'
            self.get_logger().warn(f'Obstacle {min_front:.2f} m → STOP')
        elif self.state == 'WARN':
            msg.data = 'AVOID_RIGHT'
            self.get_logger().info(f'Obstacle {min_front:.2f} m → AVOID_RIGHT (lane change)')
        else:  # CLEAR
            msg.data = 'GO'
            self.get_logger().info('Path clear → GO')

        self.pub_behavior.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = ObstacleAvoidanceNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
