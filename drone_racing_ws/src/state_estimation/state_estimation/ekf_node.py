"""
ekf_node.py
-----------
Extended Kalman Filter for drone state estimation.

Fuses:
  - IMU (accelerometer + gyroscope) at ~200Hz  → prediction step
  - Visual gate pose estimates at ~30Hz         → update step
  - Barometer (optional) at ~50Hz              → altitude update

State: [px,py,pz, vx,vy,vz, qw,qx,qy,qz, bax,bay,baz, bgx,bgy,bgz]
       position  velocity   orientation(quat)  accel_bias  gyro_bias

ROS2:
  Subscribes: /mavros/imu/data          [sensor_msgs/Imu]
  Subscribes: /perception/gate_pose     [geometry_msgs/PoseStamped]
  Publishes:  /state_estimation/odom    [nav_msgs/Odometry]
  Publishes:  /state_estimation/state   [drone_racing_msgs/DroneState]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import numpy as np
from scipy.spatial.transform import Rotation
import threading
import time

from sensor_msgs.msg import Imu
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry


class QuaternionEKF:
    """
    15-DOF EKF with quaternion orientation representation.

    IMU model:
        a_measured = R_body_to_world^T * (a_true - g) + b_a + n_a
        w_measured = w_true + b_g + n_g

    where:
        g   = [0, 0, 9.81] gravity in world frame
        b_a = accelerometer bias (slowly varying)
        b_g = gyroscope bias (slowly varying)
        n_a = accelerometer noise (white)
        n_g = gyroscope noise (white)
    """

    # State indices for clean access
    IDX_POS  = slice(0, 3)   # px, py, pz
    IDX_VEL  = slice(3, 6)   # vx, vy, vz
    IDX_QUAT = slice(6, 10)  # qw, qx, qy, qz
    IDX_BACC = slice(10, 13) # bax, bay, baz
    IDX_BGYR = slice(13, 16) # bgx, bgy, bgz
    STATE_DIM = 16

    # Error-state dimension (15, not 16 — rotation uses 3-param error)
    ERR_DIM = 15

    def __init__(self, config: dict):
        self.g_world = np.array([0.0, 0.0, 9.81])  # gravity (world, Z-up)

        # ── Initial state ─────────────────────────────────────────────────────
        self.x = np.zeros(self.STATE_DIM)
        self.x[6] = 1.0  # qw = 1 (identity quaternion)

        # ── Initial covariance (error-state, 15x15) ───────────────────────────
        self.P = np.diag([
            0.1,  0.1,  0.1,     # position variance (m²)
            0.01, 0.01, 0.01,    # velocity variance (m/s)²
            0.01, 0.01, 0.01,    # orientation variance (rad²)
            0.01, 0.01, 0.01,    # accel bias variance
            0.001, 0.001, 0.001  # gyro bias variance
        ])

        # ── Process noise Q ───────────────────────────────────────────────────
        self.sigma_a    = config.get('accel_noise',     0.1)    # m/s² / sqrt(Hz)
        self.sigma_g    = config.get('gyro_noise',      0.01)   # rad/s / sqrt(Hz)
        self.sigma_ba   = config.get('accel_bias_noise', 0.001) # m/s³ / sqrt(Hz)
        self.sigma_bg   = config.get('gyro_bias_noise',  0.0001)# rad/s² / sqrt(Hz)

        # ── Measurement noise R ───────────────────────────────────────────────
        self.R_vision_pos  = np.eye(3) * config.get('vision_pos_noise',  0.05)**2
        self.R_vision_quat = np.eye(3) * config.get('vision_quat_noise', 0.1)**2

        self.last_imu_time = None
        self.initialized   = False
        self.lock          = threading.Lock()

    # ──────────────────────────────────────────────────────────────────────────
    # PREDICTION STEP (IMU)
    # ──────────────────────────────────────────────────────────────────────────

    def predict(self, accel_body: np.ndarray,
                      gyro_body:  np.ndarray,
                      dt:         float):
        """
        IMU-based prediction step.
        Called at full IMU rate (~200Hz).

        Args:
            accel_body: [ax, ay, az] accelerometer reading (body frame, m/s²)
            gyro_body:  [gx, gy, gz] gyroscope reading (body frame, rad/s)
            dt:         time since last prediction (seconds)
        """
        with self.lock:
            if dt <= 0 or dt > 0.1:
                return   # Reject bad dt

            # Extract state
            pos  = self.x[self.IDX_POS].copy()
            vel  = self.x[self.IDX_VEL].copy()
            quat = self.x[self.IDX_QUAT].copy()   # [qw, qx, qy, qz]
            b_a  = self.x[self.IDX_BACC].copy()
            b_g  = self.x[self.IDX_BGYR].copy()

            # Current rotation: body → world
            R = self._quat_to_rotation_matrix(quat)

            # ── Correct IMU readings with bias ────────────────────────────────
            a_corrected = accel_body - b_a
            w_corrected = gyro_body  - b_g

            # ── Compute acceleration in world frame ───────────────────────────
            a_world = R @ a_corrected - self.g_world

            # ── Integrate position and velocity ───────────────────────────────
            pos_new = pos + vel * dt + 0.5 * a_world * dt**2
            vel_new = vel + a_world * dt

            # ── Integrate orientation (quaternion integration) ─────────────────
            quat_new = self._integrate_quaternion(quat, w_corrected, dt)

            # ── Biases: random walk (no update until measurement) ─────────────
            b_a_new = b_a.copy()
            b_g_new = b_g.copy()

            # ── Update state ───────────────────────────────────────────────────
            self.x[self.IDX_POS]  = pos_new
            self.x[self.IDX_VEL]  = vel_new
            self.x[self.IDX_QUAT] = quat_new
            self.x[self.IDX_BACC] = b_a_new
            self.x[self.IDX_BGYR] = b_g_new

            # ── Propagate covariance P = F·P·Fᵀ + Q ──────────────────────────
            F = self._compute_F_matrix(R, a_corrected, w_corrected, dt)
            Q = self._compute_Q_matrix(dt)
            self.P = F @ self.P @ F.T + Q

    # ──────────────────────────────────────────────────────────────────────────
    # UPDATE STEP (Vision)
    # ──────────────────────────────────────────────────────────────────────────

    def update_vision(self, pos_meas: np.ndarray, quat_meas: np.ndarray):
        """
        Vision-based update using gate pose measurement.
        Called at camera rate (~30Hz).

        Args:
            pos_meas:  [x, y, z] gate-derived position in world frame
            quat_meas: [qw, qx, qy, qz] orientation measurement
        """
        with self.lock:
            # ── Position update ────────────────────────────────────────────────
            H_pos = np.zeros((3, self.ERR_DIM))
            H_pos[:3, :3] = np.eye(3)   # measurement = position directly

            z_pos    = pos_meas
            h_pos    = self.x[self.IDX_POS]
            innov    = z_pos - h_pos

            S = H_pos @ self.P @ H_pos.T + self.R_vision_pos
            K = self.P @ H_pos.T @ np.linalg.inv(S)

            delta_x = K @ innov
            self._apply_error_state(delta_x)
            self.P  = (np.eye(self.ERR_DIM) - K @ H_pos) @ self.P

            # ── Orientation update ─────────────────────────────────────────────
            H_quat = np.zeros((3, self.ERR_DIM))
            H_quat[:3, 6:9] = np.eye(3)  # measurement = orientation error

            q_est  = self.x[self.IDX_QUAT]
            q_err  = self._quaternion_error(q_est, quat_meas)  # 3-vector error

            S2 = H_quat @ self.P @ H_quat.T + self.R_vision_quat
            K2 = self.P @ H_quat.T @ np.linalg.inv(S2)

            delta_x2 = K2 @ q_err
            self._apply_error_state(delta_x2)
            self.P   = (np.eye(self.ERR_DIM) - K2 @ H_quat) @ self.P

    def update_altitude(self, alt_meas: float, noise_sigma: float = 0.05):
        """
        Barometer altitude update (scalar measurement).
        """
        with self.lock:
            H = np.zeros((1, self.ERR_DIM))
            H[0, 2] = 1.0   # Z position

            z   = np.array([alt_meas])
            h   = np.array([self.x[2]])
            R_z = np.array([[noise_sigma**2]])

            S = H @ self.P @ H.T + R_z
            K = self.P @ H.T @ np.linalg.inv(S)

            delta_x = (K @ (z - h)).flatten()
            self._apply_error_state(delta_x)
            self.P  = (np.eye(self.ERR_DIM) - K @ H) @ self.P

    # ──────────────────────────────────────────────────────────────────────────
    # HELPER METHODS
    # ──────────────────────────────────────────────────────────────────────────

    def _integrate_quaternion(self,
                               q: np.ndarray,
                               omega: np.ndarray,
                               dt: float) -> np.ndarray:
        """
        Integrate quaternion with angular velocity using zeroth-order integration.
        q_new = q ⊗ exp(ω·dt/2)

        This is the standard approach used in VIO systems (VINS-Mono, etc.)
        """
        angle = np.linalg.norm(omega) * dt
        if angle < 1e-10:
            return q / np.linalg.norm(q)

        axis = omega / np.linalg.norm(omega)
        dq = np.array([
            np.cos(angle / 2),
            *(axis * np.sin(angle / 2))
        ])   # [qw, qx, qy, qz]

        # Quaternion multiplication q ⊗ dq
        qw, qx, qy, qz       = q
        dqw, dqx, dqy, dqz   = dq

        q_new = np.array([
            qw*dqw - qx*dqx - qy*dqy - qz*dqz,
            qw*dqx + qx*dqw + qy*dqz - qz*dqy,
            qw*dqy - qx*dqz + qy*dqw + qz*dqx,
            qw*dqz + qx*dqy - qy*dqx + qz*dqw
        ])

        return q_new / np.linalg.norm(q_new)

    def _quat_to_rotation_matrix(self, q: np.ndarray) -> np.ndarray:
        """[qw, qx, qy, qz] → 3x3 rotation matrix (body → world)."""
        qw, qx, qy, qz = q
        return np.array([
            [1-2*(qy**2+qz**2),   2*(qx*qy-qz*qw),   2*(qx*qz+qy*qw)],
            [  2*(qx*qy+qz*qw), 1-2*(qx**2+qz**2),   2*(qy*qz-qx*qw)],
            [  2*(qx*qz-qy*qw),   2*(qy*qz+qx*qw), 1-2*(qx**2+qy**2)]
        ])

    def _skew(self, v: np.ndarray) -> np.ndarray:
        """3-vector → 3x3 skew-symmetric matrix."""
        return np.array([
            [ 0,    -v[2],  v[1]],
            [ v[2],  0,    -v[0]],
            [-v[1],  v[0],  0   ]
        ])

    def _compute_F_matrix(self,
                           R:  np.ndarray,
                           a:  np.ndarray,
                           w:  np.ndarray,
                           dt: float) -> np.ndarray:
        """
        15x15 error-state Jacobian F.
        Derived from the IMU kinematic model.
        """
        F = np.eye(self.ERR_DIM)

        # dp/dv
        F[0:3, 3:6]   = np.eye(3) * dt
        # dp/dtheta
        F[0:3, 6:9]   = -0.5 * R @ self._skew(a) * dt**2
        # dp/dba
        F[0:3, 9:12]  = -0.5 * R * dt**2

        # dv/dtheta
        F[3:6, 6:9]   = -R @ self._skew(a) * dt
        # dv/dba
        F[3:6, 9:12]  = -R * dt

        # dtheta/dtheta
        F[6:9, 6:9]   = np.eye(3) - self._skew(w) * dt
        # dtheta/dbg
        F[6:9, 12:15] = -np.eye(3) * dt

        return F

    def _compute_Q_matrix(self, dt: float) -> np.ndarray:
        """
        15x15 process noise covariance Q.
        """
        Q = np.zeros((self.ERR_DIM, self.ERR_DIM))

        # Velocity noise (from accel noise)
        Q[3:6, 3:6]   = np.eye(3) * (self.sigma_a * dt)**2
        # Orientation noise (from gyro noise)
        Q[6:9, 6:9]   = np.eye(3) * (self.sigma_g * dt)**2
        # Accel bias random walk
        Q[9:12, 9:12]  = np.eye(3) * (self.sigma_ba * dt)**2
        # Gyro bias random walk
        Q[12:15, 12:15] = np.eye(3) * (self.sigma_bg * dt)**2

        return Q

    def _quaternion_error(self, q_est: np.ndarray,
                                q_meas: np.ndarray) -> np.ndarray:
        """
        Compute 3-vector orientation error between estimated and measured quat.
        Uses the error quaternion approach: δq = q_meas ⊗ q_est⁻¹
        """
        # q_est inverse = [qw, -qx, -qy, -qz]
        q_est_inv = np.array([q_est[0], -q_est[1], -q_est[2], -q_est[3]])

        qw, qx, qy, qz         = q_meas
        dqw, dqx, dqy, dqz     = q_est_inv

        dq = np.array([
            qw*dqw - qx*dqx - qy*dqy - qz*dqz,
            qw*dqx + qx*dqw + qy*dqz - qz*dqy,
            qw*dqy - qx*dqz + qy*dqw + qz*dqx,
            qw*dqz + qx*dqy - qy*dqx + qz*dqw
        ])

        if dq[0] < 0:
            dq = -dq

        # Extract 3-vector (small angle approx: 2*[qx, qy, qz])
        return 2.0 * dq[1:]

    def _apply_error_state(self, delta_x: np.ndarray):
        """
        Apply 15-vector error state correction to the nominal state.
        """
        self.x[self.IDX_POS]  += delta_x[0:3]
        self.x[self.IDX_VEL]  += delta_x[3:6]

        # Apply orientation correction via quaternion product
        dtheta = delta_x[6:9]
        angle  = np.linalg.norm(dtheta)
        if angle > 1e-10:
            axis  = dtheta / angle
            dq    = np.array([np.cos(angle/2), *(axis * np.sin(angle/2))])
            q     = self.x[self.IDX_QUAT]
            qw, qx, qy, qz       = q
            dqw, dqx, dqy, dqz   = dq
            q_new = np.array([
                qw*dqw - qx*dqx - qy*dqy - qz*dqz,
                qw*dqx + qx*dqw + qy*dqz - qz*dqy,
                qw*dqy - qx*dqz + qy*dqw + qz*dqx,
                qw*dqz + qx*dqy - qy*dqx + qz*dqw
            ])
            self.x[self.IDX_QUAT] = q_new / np.linalg.norm(q_new)

        self.x[self.IDX_BACC] += delta_x[9:12]
        self.x[self.IDX_BGYR] += delta_x[12:15]

    def get_state(self) -> dict:
        """Return current state as a clean dictionary."""
        with self.lock:
            return {
                'position':    self.x[self.IDX_POS].copy(),
                'velocity':    self.x[self.IDX_VEL].copy(),
                'quaternion':  self.x[self.IDX_QUAT].copy(),
                'accel_bias':  self.x[self.IDX_BACC].copy(),
                'gyro_bias':   self.x[self.IDX_BGYR].copy(),
                'covariance':  self.P.copy()
            }


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 EKF Node
# ─────────────────────────────────────────────────────────────────────────────

class EKFNode(Node):

    def __init__(self):
        super().__init__('ekf_node')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('accel_noise',      0.1)
        self.declare_parameter('gyro_noise',       0.01)
        self.declare_parameter('accel_bias_noise', 0.001)
        self.declare_parameter('gyro_bias_noise',  0.0001)
        self.declare_parameter('vision_pos_noise', 0.05)
        self.declare_parameter('vision_quat_noise',0.1)

        config = {
            'accel_noise':      self.get_parameter('accel_noise').value,
            'gyro_noise':       self.get_parameter('gyro_noise').value,
            'accel_bias_noise': self.get_parameter('accel_bias_noise').value,
            'gyro_bias_noise':  self.get_parameter('gyro_bias_noise').value,
            'vision_pos_noise': self.get_parameter('vision_pos_noise').value,
            'vision_quat_noise':self.get_parameter('vision_quat_noise').value,
        }

        self.ekf = QuaternionEKF(config)

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.imu_sub = self.create_subscription(
            Imu,
            '/mavros/imu/data',
            self.imu_callback,
            sensor_qos
        )

        self.vision_sub = self.create_subscription(
            PoseStamped,
            '/perception/gate_pose',
            self.vision_callback,
            10
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.odom_pub = self.create_publisher(
            Odometry,
            '/state_estimation/odom',
            10
        )

        # ── Timer: publish state at 100Hz ─────────────────────────────────────
        self.pub_timer = self.create_timer(0.01, self.publish_state)

        self.last_imu_stamp = None
        self.get_logger().info('EKFNode ready — waiting for IMU data')

    def imu_callback(self, msg: Imu):
        """Prediction step from IMU."""
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9

        if self.last_imu_stamp is None:
            self.last_imu_stamp = stamp
            return

        dt = stamp - self.last_imu_stamp
        self.last_imu_stamp = stamp

        accel = np.array([
            msg.linear_acceleration.x,
            msg.linear_acceleration.y,
            msg.linear_acceleration.z
        ])
        gyro = np.array([
            msg.angular_velocity.x,
            msg.angular_velocity.y,
            msg.angular_velocity.z
        ])

        self.ekf.predict(accel, gyro, dt)

    def vision_callback(self, msg: PoseStamped):
        """Update step from gate pose."""
        pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])
        q = msg.pose.orientation
        quat = np.array([q.w, q.x, q.y, q.z])

        self.ekf.update_vision(pos, quat)

    def publish_state(self):
        """Publish current EKF state as Odometry message."""
        state = self.ekf.get_state()

        msg          = Odometry()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.child_frame_id  = 'base_link'

        p = state['position']
        v = state['velocity']
        q = state['quaternion']   # [qw, qx, qy, qz]

        msg.pose.pose.position.x    = float(p[0])
        msg.pose.pose.position.y    = float(p[1])
        msg.pose.pose.position.z    = float(p[2])
        msg.pose.pose.orientation.w = float(q[0])
        msg.pose.pose.orientation.x = float(q[1])
        msg.pose.pose.orientation.y = float(q[2])
        msg.pose.pose.orientation.z = float(q[3])

        msg.twist.twist.linear.x = float(v[0])
        msg.twist.twist.linear.y = float(v[1])
        msg.twist.twist.linear.z = float(v[2])

        self.odom_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = EKFNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()