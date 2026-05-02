"""
behavior_controller.py
-----------------------
Top-level state machine for the autonomous car.  It listens to all
high-level events and publishes a single /behavior/state that navigation_node
and other nodes obey.

STATES
------
FOLLOWING       → normal lane following
STOP_SIGN       → stopped at stop sign (5 s pause)
ROAD_CLOSED     → turning around (road_closed sign detected)
ONE_WAY_WRONG   → blocked wrong-way street
OBSTACLE_STOP   → obstacle blocking, waiting for clear
INTERSECTION    → intersection logic (turn right unless signed otherwise)
END_OF_ROAD     → turning around at orange line

Subscriptions:
  /lane/end_of_road   (std_msgs/Bool)
  /obstacles/state    → via /behavior/state echo from obstacle_avoidance
  /sign/detected      (std_msgs/String)  — stop | road_closed | one_way

Publishes:
  /behavior/state     (std_msgs/String)
  /behavior/state_display (std_msgs/String)  — human-readable for RViz text
"""

import rclpy
from rclpy.node import Node
import time

from std_msgs.msg import Bool, String


STOP_SIGN_WAIT = 5.0   # seconds to pause at stop sign


class BehaviorController(Node):

    def __init__(self):
        super().__init__('behavior_controller')

        self.state = 'FOLLOWING'
        self.state_start = time.time()

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Bool,   '/lane/end_of_road', self._eor_cb,      10)
        self.create_subscription(String, '/sign/detected',    self._sign_cb,     10)
        self.create_subscription(String, '/behavior/state',   self._obstacle_cb, 10)

        # ── Publishers ────────────────────────────────────────────
        self.pub_state   = self.create_publisher(String, '/behavior/state',         10)
        self.pub_display = self.create_publisher(String, '/behavior/state_display',  10)

        # ── Timer: check timed states every 100 ms ────────────────
        self.create_timer(0.1, self._tick)

        self._publish_state('GO')
        self.get_logger().info('Behavior controller ready — state: FOLLOWING')

    # ── Callbacks ─────────────────────────────────────────────────
    def _eor_cb(self, msg: Bool):
        if msg.data and self.state == 'FOLLOWING':
            self.get_logger().info('End of road → turning around')
            self._transition('END_OF_ROAD')

    def _sign_cb(self, msg: String):
        sign = msg.data.lower()
        if sign == 'stop' and self.state == 'FOLLOWING':
            self.get_logger().info('Stop sign detected')
            self._transition('STOP_SIGN')
            self._publish_state('STOP')

        elif sign == 'road_closed' and self.state == 'FOLLOWING':
            self.get_logger().info('Road closed sign → turning around')
            self._transition('ROAD_CLOSED')
            self._publish_state('TURN_AROUND')

        elif sign == 'one_way' and self.state == 'FOLLOWING':
            self.get_logger().warn('One-way wrong direction → stopping')
            self._transition('ONE_WAY_WRONG')
            self._publish_state('STOP')

    def _obstacle_cb(self, msg: String):
        """Relay obstacle state but don't override sign states."""
        if msg.data == 'STOP' and self.state == 'FOLLOWING':
            self._transition('OBSTACLE_STOP')
        elif msg.data == 'GO' and self.state == 'OBSTACLE_STOP':
            self._transition('FOLLOWING')
            self._publish_state('GO')

    # ── Timed state logic ─────────────────────────────────────────
    def _tick(self):
        elapsed = time.time() - self.state_start

        if self.state == 'STOP_SIGN':
            if elapsed >= STOP_SIGN_WAIT:
                self.get_logger().info('Stop sign wait done → FOLLOWING')
                self._transition('FOLLOWING')
                self._publish_state('GO')

        # Publish display string for debugging / RViz
        self.pub_display.publish(String(data=self.state))

    # ── Helpers ───────────────────────────────────────────────────
    def _transition(self, new_state: str):
        self.get_logger().info(f'State: {self.state} → {new_state}')
        self.state = new_state
        self.state_start = time.time()

    def _publish_state(self, cmd: str):
        self.pub_state.publish(String(data=cmd))


def main(args=None):
    rclpy.init(args=args)
    node = BehaviorController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()