"""
pid_controller.py
-----------------
Cascaded PID controller for quadrotor position + attitude control.

Architecture (standard quadrotor cascade):
    Position error
         ↓  [Position PID]
    Desired acceleration
         ↓  [Attitude from acceleration]
    Desired roll/pitch/yawrate
         ↓  [Attitude PID]
    Desired thrust + torques
         ↓
    Motor commands (via PX4)

This outer-loop controller runs at ~50Hz.
PX4's inner-loop (rate controller) runs at ~1000Hz onboard.

ROS2:
    Subscribes: /planning/next_waypoint     [geometry_msgs/PoseStamped]
    Subscribes: /state_estimation/odom      [nav_msgs/Odometry]
    Publishes:  /mavros/setpoint_raw/attitude [mavros_msgs/AttitudeTarget]
"""

import rclpy
from rclpy.node import Node
import numpy as np
from dataclasses import dataclass


@dataclass
class PIDGains:
    kp: float
    ki: float
    kd: float
    integral_limit: float = 5.0
    output_limit:   float = np.inf


class PIDController:
    """Single-axis PID with anti-windup and derivative filtering."""

    def __init__(self, gains: PIDGains, dt: float = 0.02):
        self.kp = gains.kp
        self.ki = gains.ki
        self.kd = gains.kd
        self.int_limit = gains.integral_limit
        self.out_limit = gains.output_limit

        self._integral    = 0.0
        self._prev_error  = 0.0
        self._initialized = False

        # Derivative low-pass filter (cuts sensor noise)
        self._derivative_filter = 0.0
        self._alpha = 0.6   # filter coefficient (tune per axis)

    def reset(self):
        self._integral    = 0.0
        self._prev_error  = 0.0
        self._initialized = False
        self._derivative_filter = 0.0

    def compute(self, error: float, dt: float) -> float:
        """
        Compute PID output.

        Args:
            error: setpoint - measurement
            dt:    time step (seconds)
        Returns:
            control output
        """
        if not self._initialized:
            self._prev_error  = error
            self._initialized = True

        # Proportional
        P = self.kp * error

        # Integral with anti-windup clamp
        self._integral += error * dt
        self._integral  = np.clip(self._integral,
                                  -self.int_limit, self.int_limit)
        I = self.ki * self._integral

        # Derivative with low-pass filter (avoids derivative kick)
        raw_deriv            = (error - self._prev_error) / max(dt, 1e-6)
        self._derivative_filter = (self._alpha * self._derivative_filter +
                                   (1 - self._alpha) * raw_deriv)
        D = self.kd * self._derivative_filter

        self._prev_error = error
        output = np.clip(P + I + D, -self.out_limit, self.out_limit)
        return output


class CascadedPIDNode(Node):
    """
    Cascaded PID controller: position → attitude target.

    Outer loop (position PID, 50Hz):
        Input:  position error [ex, ey, ez]
        Output: desired acceleration vector [ax, ay, az]

    Attitude from acceleration:
        Compute desired roll/pitch from desired horizontal acceleration
        (standard quadrotor differential flatness mapping)

    Inner loop (attitude PID, 100Hz):
        PX4 handles this onboard — we send attitude setpoints via MAVROS
    """

    def __init__(self):
        super().__init__('pid_controller')

        # ── PID gains (tune these for your drone) ──────────────────────────
        # Start conservative, increase kp until oscillation, then back off
        pos_x_gains = PIDGains(kp=2.0,  ki=0.1, kd=0.5,  output_limit=8.0)
        pos_y_gains = PIDGains(kp=2.0,  ki=0.1, kd=0.5,  output_limit=8.0)
        pos_z_gains = PIDGains(kp=3.0,  ki=0.2, kd=0.8,  output_limit=5.0)
        yaw_gains   = PIDGains(kp=2.0,  ki=0.0, kd=0.3,  output_limit=2.0)

        self.pid_x   = PIDController(pos_x_gains)
        self.pid_y   = PIDController(pos_y_gains)
        self.pid_z   = PIDController(pos_z_gains)
        self.pid_yaw = PIDController(yaw_gains)

        # Physical parameters
        self.declare_parameter('mass_kg',         0.8)
        self.declare_parameter('hover_thrust',    0.5)   # normalized [0-1]
        self.declare_parameter('max_tilt_deg',    35.0)
        self.declare_parameter('control_freq_hz', 50.0)

        self.mass          = self.get_parameter('mass_kg').value
        self.hover_thrust  = self.get_parameter('hover_thrust').value
        self.max_tilt      = np.radians(self.get_parameter('max_tilt_deg').value)
        freq               = self.get_parameter('control_freq_hz').value

        self.dt            = 1.0 / freq
        self.g             = 9.81

        # State
        self.drone_pos     = np.zeros(3)
        self.drone_vel     = np.zeros(3)
        self.drone_yaw     = 0.0
        self.target_pos    = np.zeros(3)
        self.target_vel    = np.zeros(3)   # feedforward velocity
        self.target_yaw    = 0.0

        # ── Subscribers ───────────────────────────────────────────────────────
        self.wp_sub = self.create_subscription(
            PoseStamped if True else None,
            '/planning/next_waypoint',
            self.waypoint_callback,
            10
        )
        from nav_msgs.msg import Odometry
        self.odom_sub = self.create_subscription(
            Odometry,
            '/state_estimation/odom',
            self.odom_callback,
            10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        from mavros_msgs.msg import AttitudeTarget
        self.attitude_pub = self.create_publisher(
            AttitudeTarget,
            '/mavros/setpoint_raw/attitude',
            10
        )

        # ── Control timer ─────────────────────────────────────────────────────
        self.create_timer(self.dt, self.control_loop)
        self.get_logger().info(
            f'CascadedPIDNode ready at {freq}Hz'
        )

    def waypoint_callback(self, msg):
        self.target_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        # Extract desired yaw from orientation
        q  = msg.pose.orientation
        self.target_yaw = self._quat_to_yaw(q.w, q.x, q.y, q.z)

    def odom_callback(self, msg):
        self.drone_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.drone_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        q = msg.pose.pose.orientation
        self.drone_yaw = self._quat_to_yaw(q.w, q.x, q.y, q.z)

    def control_loop(self):
        """
        Main control loop: compute attitude target from position error.
        """
        pos_error = self.target_pos - self.drone_pos

        # ── Outer loop: position → desired acceleration ────────────────────
        ax_des = self.pid_x.compute(pos_error[0], self.dt)
        ay_des = self.pid_y.compute(pos_error[1], self.dt)
        az_des = self.pid_z.compute(pos_error[2], self.dt)

        # ── Differential flatness mapping: acc → roll/pitch/thrust ─────────
        # Standard quadrotor model:
        #   Total thrust direction must equal (gravity + desired_acc)
        thrust_vec  = np.array([ax_des, ay_des, az_des + self.g])
        thrust_mag  = np.linalg.norm(thrust_vec)

        # Normalize thrust for unit [0-1] scale
        thrust_norm = np.clip(
            thrust_mag / (self.g * 2.0),
            0.1, 1.0
        )

        # Desired roll and pitch from thrust vector
        # (small angle approximation; for large angles use full rotation)
        roll_des  = np.arcsin(
            np.clip(
                (thrust_vec[1] * np.cos(self.drone_yaw) -
                 thrust_vec[0] * np.sin(self.drone_yaw)) / thrust_mag,
                -1.0, 1.0
            )
        )
        pitch_des = np.arctan2(
            thrust_vec[0] * np.cos(self.drone_yaw) +
            thrust_vec[1] * np.sin(self.drone_yaw),
            thrust_vec[2]
        )

        # Clamp roll/pitch to safe limits
        roll_des  = np.clip(roll_des,  -self.max_tilt, self.max_tilt)
        pitch_des = np.clip(pitch_des, -self.max_tilt, self.max_tilt)

        # Yaw control
        yaw_error    = self._wrap_angle(self.target_yaw - self.drone_yaw)
        yaw_rate_des = self.pid_yaw.compute(yaw_error, self.dt)

        # ── Build AttitudeTarget message ───────────────────────────────────
        from mavros_msgs.msg import AttitudeTarget
        msg = AttitudeTarget()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'

        # Type mask: ignore body rates, use attitude + thrust
        # Bit 0: body roll rate    (1 = ignore)
        # Bit 1: body pitch rate   (1 = ignore)
        # Bit 2: body yaw rate     (0 = use yaw rate below)
        # Bit 6: attitude          (0 = use attitude)
        # Bit 7: thrust            (0 = use thrust)
        msg.type_mask = AttitudeTarget.IGNORE_ROLL_RATE | \
                        AttitudeTarget.IGNORE_PITCH_RATE

        # Convert roll/pitch/yaw to quaternion
        q = self._rpy_to_quat(roll_des, pitch_des, self.target_yaw)
        msg.orientation.w = float(q[0])
        msg.orientation.x = float(q[1])
        msg.orientation.y = float(q[2])
        msg.orientation.z = float(q[3])

        msg.body_rate.z = float(yaw_rate_des)
        msg.thrust      = float(thrust_norm)

        self.attitude_pub.publish(msg)

    def _quat_to_yaw(self, qw, qx, qy, qz) -> float:
        return float(np.arctan2(
            2.0 * (qw*qz + qx*qy),
            1.0 - 2.0 * (qy**2 + qz**2)
        ))

    def _rpy_to_quat(self, r, p, y) -> np.ndarray:
        """Roll, pitch, yaw (rad) → quaternion [qw, qx, qy, qz]."""
        cr, sr = np.cos(r/2), np.sin(r/2)
        cp, sp = np.cos(p/2), np.sin(p/2)
        cy, sy = np.cos(y/2), np.sin(y/2)
        return np.array([
            cr*cp*cy + sr*sp*sy,
            sr*cp*cy - cr*sp*sy,
            cr*sp*cy + sr*cp*sy,
            cr*cp*sy - sr*sp*cy
        ])

    def _wrap_angle(self, angle: float) -> float:
        """Wrap angle to [-π, π]."""
        return float((angle + np.pi) % (2 * np.pi) - np.pi)


def main(args=None):
    rclpy.init(args=args)
    node = CascadedPIDNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()