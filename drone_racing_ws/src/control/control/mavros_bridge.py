"""
mavros_bridge.py
----------------
ROS2 node that manages the full MAVROS ↔ PX4 connection.

Responsibilities:
  1. Arm the drone
  2. Switch to OFFBOARD mode (required for autonomous flight)
  3. Send continuous setpoint stream (PX4 requires >2Hz to stay in OFFBOARD)
  4. Monitor failsafe conditions
  5. Handle emergency landing

CRITICAL PX4 OFFBOARD rules:
  - Must send setpoints at >2Hz BEFORE switching to OFFBOARD mode
  - If setpoint stream stops for >500ms, PX4 exits OFFBOARD → failsafe
  - Always have a kill-switch / RC override ready

ROS2:
  Publishes: /mavros/setpoint_raw/attitude  [mavros_msgs/AttitudeTarget]
  Calls:     /mavros/cmd/arming             [mavros_msgs/CommandBool]
  Calls:     /mavros/set_mode              [mavros_msgs/SetMode]
  Subscribes:/mavros/state                 [mavros_msgs/State]
"""

import rclpy
from rclpy.node import Node
import numpy as np
import time

from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg  import State, AttitudeTarget
from mavros_msgs.srv  import CommandBool, SetMode
from nav_msgs.msg     import Odometry


class MAVROSBridge(Node):
    """
    Manages PX4 arming, mode switching, and setpoint streaming.
    This node coordinates the full autonomous flight sequence.
    """

    def __init__(self):
        super().__init__('mavros_bridge')

        self.declare_parameter('takeoff_altitude',   1.5)
        self.declare_parameter('setpoint_rate_hz',   20.0)
        self.declare_parameter('arm_timeout_sec',    10.0)
        self.declare_parameter('offboard_timeout_sec', 5.0)

        self.takeoff_alt     = self.get_parameter('takeoff_altitude').value
        self.sp_rate         = self.get_parameter('setpoint_rate_hz').value
        self.arm_timeout     = self.get_parameter('arm_timeout_sec').value

        # State tracking
        self.mavros_state    = State()
        self.current_pos     = np.zeros(3)
        self.is_armed        = False
        self.in_offboard     = False
        self.race_active     = False

        # Current setpoint to stream (updated by PID controller)
        self.current_setpoint = self._hover_setpoint()

        # ── Subscribers ───────────────────────────────────────────────────────
        self.state_sub = self.create_subscription(
            State,
            '/mavros/state',
            self.state_callback,
            10
        )
        self.odom_sub = self.create_subscription(
            Odometry,
            '/state_estimation/odom',
            self.odom_callback,
            10
        )
        # Listen to PID controller output
        self.attitude_sub = self.create_subscription(
            AttitudeTarget,
            '/control/pid_attitude_target',
            self.attitude_target_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )
        # Local position setpoint (used during takeoff)
        self.local_pos_pub = self.create_publisher(
            PoseStamped,
            '/mavros/setpoint_position/local',
            10
        )

        # ── Service clients ───────────────────────────────────────────────────
        self.arming_client = self.create_client(
            CommandBool,
            '/mavros/cmd/arming'
        )
        self.set_mode_client = self.create_client(
            SetMode,
            '/mavros/set_mode'
        )

        # ── Setpoint stream timer (MUST run before OFFBOARD switch) ───────────
        stream_dt = 1.0 / self.sp_rate
        self.stream_timer = self.create_timer(
            stream_dt,
            self.stream_setpoint
        )

        # ── Startup sequence timer ────────────────────────────────────────────
        self.startup_timer = self.create_timer(0.5, self.startup_sequence)
        self.startup_phase = 'wait_connection'
        self.startup_count = 0

        self.get_logger().info('MAVROSBridge ready')

    # ──────────────────────────────────────────────────────────────────────────
    # CALLBACKS
    # ──────────────────────────────────────────────────────────────────────────

    def state_callback(self, msg: State):
        self.mavros_state = msg
        self.is_armed     = msg.armed
        self.in_offboard  = (msg.mode == 'OFFBOARD')

    def odom_callback(self, msg: Odometry):
        self.current_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])

    def attitude_target_callback(self, msg: AttitudeTarget):
        """Receive attitude target from PID controller, pass to PX4."""
        if self.in_offboard and self.is_armed:
            self.current_setpoint = msg

    # ──────────────────────────────────────────────────────────────────────────
    # SETPOINT STREAM
    # ──────────────────────────────────────────────────────────────────────────

    def stream_setpoint(self):
        """
        Continuously stream setpoint to PX4.
        CRITICAL: PX4 exits OFFBOARD mode if this stops for >500ms.
        """
        self.current_setpoint.header.stamp = self.get_clock().now().to_msg()
        self.attitude_pub.publish(self.current_setpoint)

    def _hover_setpoint(self) -> AttitudeTarget:
        """Create a neutral hover attitude setpoint."""
        msg = AttitudeTarget()
        msg.type_mask = (AttitudeTarget.IGNORE_ROLL_RATE |
                         AttitudeTarget.IGNORE_PITCH_RATE |
                         AttitudeTarget.IGNORE_YAW_RATE)
        msg.orientation.w = 1.0   # identity quaternion = level
        msg.thrust        = 0.5   # ~50% = hover
        return msg

    # ──────────────────────────────────────────────────────────────────────────
    # STARTUP SEQUENCE
    # ──────────────────────────────────────────────────────────────────────────

    def startup_sequence(self):
        """
        State machine for safe startup:
          1. Wait for MAVROS connection
          2. Pre-stream setpoints for 2 seconds
          3. Arm
          4. Switch to OFFBOARD
          5. Takeoff to hover altitude
          6. Start race
        """
        phase = self.startup_phase
        self.startup_count += 1

        if phase == 'wait_connection':
            if self.mavros_state.connected:
                self.get_logger().info('✅ MAVROS connected to PX4')
                self.startup_phase = 'pre_stream'
                self.startup_count = 0
            else:
                if self.startup_count % 4 == 0:
                    self.get_logger().info('⏳ Waiting for MAVROS connection...')

        elif phase == 'pre_stream':
            # Stream for 2 seconds before requesting OFFBOARD
            if self.startup_count >= 4:  # 4 * 0.5s = 2s
                self.get_logger().info('Pre-streaming done. Requesting OFFBOARD...')
                self.startup_phase = 'request_offboard'

        elif phase == 'request_offboard':
            if not self.in_offboard:
                self._set_mode('OFFBOARD')
            elif not self.is_armed:
                self.startup_phase = 'arm'

        elif phase == 'arm':
            if not self.is_armed:
                self._arm()
            else:
                self.get_logger().info('✅ Armed!')
                self.startup_phase = 'takeoff'
                self._set_takeoff_setpoint()

        elif phase == 'takeoff':
            alt_err = abs(self.current_pos[2] - self.takeoff_alt)
            if alt_err < 0.15:
                self.get_logger().info(
                    f'✅ At hover altitude {self.takeoff_alt}m — race ready!'
                )
                self.startup_phase  = 'race'
                self.race_active    = True
                self.startup_timer.cancel()

        elif phase == 'race':
            pass  # PID controller takes over

    def _set_takeoff_setpoint(self):
        """Switch to position setpoint for takeoff."""
        msg = PoseStamped()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.pose.position.x = float(self.current_pos[0])
        msg.pose.position.y = float(self.current_pos[1])
        msg.pose.position.z = float(self.takeoff_alt)
        msg.pose.orientation.w = 1.0
        self.local_pos_pub.publish(msg)

    # ──────────────────────────────────────────────────────────────────────────
    # SERVICE CALLS
    # ──────────────────────────────────────────────────────────────────────────

    def _arm(self):
        """Send arm command to PX4."""
        if not self.arming_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('Arming service not available')
            return
        req = CommandBool.Request()
        req.value = True
        future = self.arming_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'Arm result: {f.result().success}'
            )
        )

    def _set_mode(self, mode: str):
        """Request PX4 flight mode change."""
        if not self.set_mode_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().error('SetMode service not available')
            return
        req = SetMode.Request()
        req.custom_mode = mode
        future = self.set_mode_client.call_async(req)
        future.add_done_callback(
            lambda f: self.get_logger().info(
                f'SetMode {mode} result: {f.result().mode_sent}'
            )
        )

    def emergency_land(self):
        """
        Emergency: switch to LAND mode immediately.
        Call this if any safety check fails.
        """
        self.get_logger().error('⚠️  EMERGENCY LAND TRIGGERED')
        self._set_mode('AUTO.LAND')
        self.race_active = False


def main(args=None):
    rclpy.init(args=args)
    node = MAVROSBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()