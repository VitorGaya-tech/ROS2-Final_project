"""
navigation_node.py
------------------
Reads the lateral error from lane_detection_node and converts it into
/cmd_vel commands using a PD controller.

Subscriptions:
  /lane/error        (std_msgs/Float32)  — normalised lateral error [-1, 1]
  /lane/end_of_road  (std_msgs/Bool)     — True  → start 180° turn sequence
  /behavior/state    (std_msgs/String)   — behaviour controller override

Publishes:
  /cmd_vel           (geometry_msgs/Twist)

STATE MACHINE (simple, inside this node)
-----------------------------------------
FOLLOWING  → normal PD lane following
TURNING_RIGHT → intersection: fixed right turn for N seconds
TURNING_AROUND → end-of-road: 180° spin
STOPPED    → obstacle, sign, or behaviour override

CONTROL LAW
-----------
  angular_z = -(Kp * error + Kd * d_error/dt)
  linear_x  = BASE_SPEED * (1 - 0.5 * |error|)   # slow down on big curves

The negative sign because:
  error > 0  → white line is left of target → robot too far right → turn left (negative z)
  error < 0  → white line is right of target → robot too far left  → turn right (positive z)
"""

import rclpy
from rclpy.node import Node
import time

from std_msgs.msg import Float32, Bool, String
from geometry_msgs.msg import Twist


# ── Default gains (tune via ROS parameters) ───────────────────────
DEFAULT_KP        = 0.8    # proportional gain
DEFAULT_KD        = 0.15   # derivative gain
DEFAULT_BASE_SPEED = 0.15  # m/s forward
DEFAULT_MAX_TURN  = 1.2    # rad/s maximum angular velocity

# Duration of manoeuvres (seconds)
RIGHT_TURN_DURATION    = 2.2   # timed right turn at intersection
TURNAROUND_DURATION    = 3.5   # 180° spin (~π / ω  where ω≈0.9)
TURNAROUND_SPEED       = 0.9   # rad/s for 180° spin


class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('kp',         DEFAULT_KP)
        self.declare_parameter('kd',         DEFAULT_KD)
        self.declare_parameter('base_speed', DEFAULT_BASE_SPEED)
        self.declare_parameter('max_turn',   DEFAULT_MAX_TURN)

        # ── State ─────────────────────────────────────────────────
        self.state = 'FOLLOWING'       # FOLLOWING | TURNING_RIGHT | TURNING_AROUND | STOPPED
        self.manoeuvre_start = None
        self.last_error = 0.0
        self.last_error_time = time.time()
        self.current_error = 0.0
        self.external_override = 'NONE'

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Float32, '/lane/error',       self._error_cb,    10)
        self.create_subscription(Bool,    '/lane/end_of_road', self._eor_cb,      10)
        self.create_subscription(String,  '/behavior/state',   self._behavior_cb, 10)

        # ── Publisher ─────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Control loop at 20 Hz ─────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('Navigation node started — state: FOLLOWING')

    # ── Callbacks ─────────────────────────────────────────────────
    def _error_cb(self, msg: Float32):
        self.current_error = msg.data

    def _eor_cb(self, msg: Bool):
        if msg.data and self.state == 'FOLLOWING':
            self.get_logger().info('End of road detected → TURNING_AROUND')
            self._start_manoeuvre('TURNING_AROUND')

    def _behavior_cb(self, msg: String):
        """Behaviour controller can override: STOP, GO, TURN_RIGHT"""
        self.external_override = msg.data
        if msg.data == 'STOP':
            self.state = 'STOPPED'
        elif msg.data == 'GO' and self.state == 'STOPPED':
            self.state = 'FOLLOWING'
        elif msg.data == 'TURN_RIGHT' and self.state == 'FOLLOWING':
            self._start_manoeuvre('TURNING_RIGHT')

    # ── Main control loop ─────────────────────────────────────────
    def _control_loop(self):
        cmd = Twist()

        if self.state == 'STOPPED':
            self.pub_cmd.publish(cmd)   # zero velocity
            return

        if self.state == 'TURNING_RIGHT':
            elapsed = time.time() - self.manoeuvre_start
            if elapsed < RIGHT_TURN_DURATION:
                cmd.linear.x  =  self.get_parameter('base_speed').value * 0.6
                cmd.angular.z = -self.get_parameter('max_turn').value * 0.8
            else:
                self.get_logger().info('Right turn done → FOLLOWING')
                self.state = 'FOLLOWING'
            self.pub_cmd.publish(cmd)
            return

        if self.state == 'TURNING_AROUND':
            elapsed = time.time() - self.manoeuvre_start
            if elapsed < TURNAROUND_DURATION:
                cmd.linear.x  = 0.0
                cmd.angular.z = TURNAROUND_SPEED
            else:
                self.get_logger().info('Turnaround done → FOLLOWING')
                self.state = 'FOLLOWING'
            self.pub_cmd.publish(cmd)
            return

        # ── FOLLOWING: PD controller ──────────────────────────────
        kp = self.get_parameter('kp').value
        kd = self.get_parameter('kd').value
        base_speed = self.get_parameter('base_speed').value
        max_turn   = self.get_parameter('max_turn').value

        now = time.time()
        dt = now - self.last_error_time
        if dt <= 0:
            dt = 0.05
        d_error = (self.current_error - self.last_error) / dt

        raw_turn = -(kp * self.current_error + kd * d_error)
        # Clamp angular velocity
        cmd.angular.z = max(-max_turn, min(max_turn, raw_turn))
        # Slow down proportionally to how much we are turning
        cmd.linear.x  = base_speed * (1.0 - 0.5 * abs(self.current_error))
        cmd.linear.x  = max(0.05, cmd.linear.x)   # never stop completely

        self.last_error = self.current_error
        self.last_error_time = now

        self.pub_cmd.publish(cmd)

    # ── Helper ────────────────────────────────────────────────────
    def _start_manoeuvre(self, state: str):
        self.state = state
        self.manoeuvre_start = time.time()


def main(args=None):
    rclpy.init(args=args)
    node = NavigationNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
