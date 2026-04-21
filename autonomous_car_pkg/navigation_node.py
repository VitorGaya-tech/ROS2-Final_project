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
FORK_RIGHT_TIMEOUT     = 4.0   # max seconds seeking right fork before giving up
RECOVERY_DURATION      = 2.0   # seconds of post-turnaround realignment
RECOVERY_ANGULAR       = 0.4   # rad/s left curve during LANE_RECOVERY
WHITE_ABSENT_THRESH    = 30    # control cycles (~0.75 s at 20 Hz) without white → FORK_RIGHT

# U-turn manoeuvre: 90° spin → forward → 90° spin
UTURN_SPIN_SPEED     = 0.9    # rad/s for each 90° turn
UTURN_SPIN_DURATION  = 1.75   # seconds per 90° (~π/2 / 0.9)
UTURN_FWD_DURATION   = 1    # seconds driving forward between the two spins


class NavigationNode(Node):

    def __init__(self):
        super().__init__('navigation_node')

        # ── Parameters ────────────────────────────────────────────
        self.declare_parameter('kp',         DEFAULT_KP)
        self.declare_parameter('kd',         DEFAULT_KD)
        self.declare_parameter('base_speed', DEFAULT_BASE_SPEED)
        self.declare_parameter('max_turn',   DEFAULT_MAX_TURN)

        # ── State ─────────────────────────────────────────────────
        self.state = 'FOLLOWING'
        self.manoeuvre_start = None
        self.last_error = 0.0
        self.last_error_time = time.time()
        self.current_error = 0.0
        self.external_override = 'NONE'

        self.white_detected      = True
        self.white_absent_cycles = 0
        self.turn_phase          = 0      # sub-phase for TURNING_AROUND (0, 1, 2)

        # ── Subscribers ───────────────────────────────────────────
        self.create_subscription(Float32, '/lane/error',          self._error_cb,     10)
        self.create_subscription(Bool,    '/lane/end_of_road',    self._eor_cb,       10)
        self.create_subscription(Bool,    '/lane/white_detected', self._white_det_cb, 10)
        self.create_subscription(String,  '/behavior/state',      self._behavior_cb,  10)

        # ── Publisher ─────────────────────────────────────────────
        self.pub_cmd = self.create_publisher(Twist, '/cmd_vel', 10)

        # ── Control loop at 20 Hz ─────────────────────────────────
        self.create_timer(0.05, self._control_loop)

        self.get_logger().info('Navigation node started — state: FOLLOWING')

    # ── Callbacks ─────────────────────────────────────────────────
    def _error_cb(self, msg: Float32):
        self.current_error = msg.data

    def _white_det_cb(self, msg: Bool):
        self.white_detected = msg.data

    def _eor_cb(self, msg: Bool):
        if msg.data and self.state in ('FOLLOWING', 'FORK_RIGHT'):
            self.get_logger().info('End of road detected → TURNING_AROUND (phase 0)')
            self.turn_phase = 0
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
            if self.turn_phase == 0:                        # first 90°
                if elapsed < UTURN_SPIN_DURATION:
                    cmd.angular.z = UTURN_SPIN_SPEED
                else:
                    self.get_logger().info('U-turn phase 1 → forward')
                    self.turn_phase = 1
                    self.manoeuvre_start = time.time()
            elif self.turn_phase == 1:                      # forward
                if elapsed < UTURN_FWD_DURATION:
                    cmd.linear.x = self.get_parameter('base_speed').value
                else:
                    self.get_logger().info('U-turn phase 2 → second 90°')
                    self.turn_phase = 2
                    self.manoeuvre_start = time.time()
            elif self.turn_phase == 2:                      # second 90°
                if elapsed < UTURN_SPIN_DURATION:
                    cmd.angular.z = UTURN_SPIN_SPEED
                else:
                    self.get_logger().info('U-turn done → LANE_RECOVERY')
                    self.turn_phase = 0
                    self._start_manoeuvre('LANE_RECOVERY')
            self.pub_cmd.publish(cmd)
            return

        if self.state == 'LANE_RECOVERY':
            # After 180° turn white line is on the wrong side; drive forward
            # with a left curve until the robot finds the lane again.
            elapsed = time.time() - self.manoeuvre_start
            if elapsed < RECOVERY_DURATION:
                cmd.linear.x  = self.get_parameter('base_speed').value * 0.5
                cmd.angular.z = RECOVERY_ANGULAR
            else:
                self.get_logger().info('Lane recovery done → FOLLOWING')
                self.state = 'FOLLOWING'
                self.white_absent_cycles = 0
            self.pub_cmd.publish(cmd)
            return

        if self.state == 'FORK_RIGHT':
            # White line disappeared at a fork — turn right until it reappears
            elapsed = time.time() - self.manoeuvre_start
            if self.white_detected or elapsed > FORK_RIGHT_TIMEOUT:
                self.get_logger().info('Fork resolved → FOLLOWING')
                self.state = 'FOLLOWING'
                self.white_absent_cycles = 0
                return
            cmd.linear.x  = self.get_parameter('base_speed').value * 0.5
            cmd.angular.z = -self.get_parameter('max_turn').value * 0.6
            self.pub_cmd.publish(cmd)
            return

        # ── FOLLOWING: PD controller ──────────────────────────────
        # Track white line absence to detect right fork
        if not self.white_detected:
            self.white_absent_cycles += 1
        else:
            self.white_absent_cycles = 0

        if self.white_absent_cycles >= WHITE_ABSENT_THRESH:
            self.get_logger().info('White absent → FORK_RIGHT')
            self.white_absent_cycles = 0
            self._start_manoeuvre('FORK_RIGHT')
            return

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
