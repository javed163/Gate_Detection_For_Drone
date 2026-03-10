"""
vio_node.py
-----------
Lightweight Visual-Inertial Odometry using optical flow + IMU.

Architecture:
    1. Detect + track feature points (Shi-Tomasi + Lucas-Kanade)
    2. Compute frame-to-frame camera motion (Essential matrix + RANSAC)
    3. Scale recovery using IMU pre-integrated velocity
    4. Feed estimated pose into EKF as visual measurement

NOTE: For competition use, consider integrating OpenVINS or VINS-Mono
instead of this custom implementation. This code explains the principles
clearly so you understand what those systems do internally.

ROS2:
    Subscribes: /camera/image_raw     [sensor_msgs/Image]
    Subscribes: /mavros/imu/data      [sensor_msgs/Imu]
    Publishes:  /vio/pose             [geometry_msgs/PoseStamped]
    Publishes:  /vio/debug_image      [sensor_msgs/Image]
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

import cv2
import numpy as np
from collections import deque

from sensor_msgs.msg import Imu, Image
from geometry_msgs.msg import PoseStamped
from cv_bridge import CvBridge


class IMUPreintegrator:
    """
    IMU pre-integration between two camera frames.

    Instead of running full EKF at camera rate, we pre-integrate
    all IMU measurements between frames to get a delta-velocity
    estimate. This is used to recover the scale of visual motion.

    Reference: Forster et al. "IMU Preintegration on Manifold" (RSS 2015)
    """

    def __init__(self, gravity: float = 9.81):
        self.g = np.array([0.0, 0.0, gravity])
        self.reset()

    def reset(self):
        """Reset pre-integration accumulators."""
        self.delta_p   = np.zeros(3)   # integrated position change
        self.delta_v   = np.zeros(3)   # integrated velocity change
        self.delta_R   = np.eye(3)     # integrated rotation change
        self.delta_t   = 0.0           # accumulated time
        self.last_time = None

    def integrate(self, accel: np.ndarray,
                        gyro:  np.ndarray,
                        timestamp: float):
        """Add one IMU measurement to pre-integration."""
        if self.last_time is None:
            self.last_time = timestamp
            return

        dt = timestamp - self.last_time
        self.last_time = timestamp

        if dt <= 0 or dt > 0.05:
            return

        # Integrate rotation
        angle     = np.linalg.norm(gyro) * dt
        if angle > 1e-10:
            axis  = gyro / np.linalg.norm(gyro)
            K     = self._skew(axis)
            dR    = np.eye(3) + np.sin(angle)*K + (1-np.cos(angle))*(K@K)
            self.delta_R = self.delta_R @ dR

        # Integrate velocity (in IMU frame — no gravity compensation here)
        self.delta_v += self.delta_R @ accel * dt

        # Integrate position
        self.delta_p += self.delta_v * dt + 0.5 * self.delta_R @ accel * dt**2

        self.delta_t += dt

    def _skew(self, v):
        return np.array([
            [0, -v[2], v[1]],
            [v[2], 0, -v[0]],
            [-v[1], v[0], 0]
        ])

    def get_delta(self) -> dict:
        return {
            'delta_p':  self.delta_p.copy(),
            'delta_v':  self.delta_v.copy(),
            'delta_R':  self.delta_R.copy(),
            'delta_t':  self.delta_t
        }


class FeatureTracker:
    """
    Sparse optical flow feature tracker using Lucas-Kanade.
    Tracks corner features across consecutive frames.
    """

    def __init__(self,
                 max_features:    int   = 200,
                 min_features:    int   = 50,
                 quality_level:   float = 0.01,
                 min_distance:    int   = 20):

        self.max_features  = max_features
        self.min_features  = min_features
        self.quality_level = quality_level
        self.min_distance  = min_distance

        # Lucas-Kanade parameters
        self.lk_params = dict(
            winSize=(21, 21),
            maxLevel=3,   # pyramid levels
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )

        self.prev_gray     = None
        self.prev_pts      = None
        self.track_ids     = []
        self.next_id       = 0

    def detect_features(self, gray: np.ndarray) -> np.ndarray:
        """Detect new Shi-Tomasi corner features."""
        mask = np.ones_like(gray, dtype=np.uint8) * 255

        # Exclude regions near existing tracks (avoid crowding)
        if self.prev_pts is not None:
            for pt in self.prev_pts:
                x, y = int(pt[0, 0]), int(pt[0, 1])
                cv2.circle(mask, (x, y), self.min_distance, 0, -1)

        pts = cv2.goodFeaturesToTrack(
            gray,
            maxCorners=self.max_features,
            qualityLevel=self.quality_level,
            minDistance=self.min_distance,
            mask=mask,
            blockSize=7
        )
        return pts

    def track(self, curr_gray: np.ndarray) -> tuple:
        """
        Track features from previous frame to current frame.

        Returns:
            pts_prev: (N, 2) feature points in previous frame
            pts_curr: (N, 2) corresponding feature points in current frame
            ids:      (N,)   track IDs
        """
        if self.prev_gray is None or self.prev_pts is None:
            self.prev_gray = curr_gray
            self.prev_pts  = self.detect_features(curr_gray)
            if self.prev_pts is not None:
                self.track_ids = list(range(len(self.prev_pts)))
                self.next_id   = len(self.prev_pts)
            return None, None, None

        if len(self.prev_pts) < self.min_features:
            new_pts = self.detect_features(curr_gray)
            if new_pts is not None:
                n = len(new_pts)
                self.prev_pts  = np.vstack([self.prev_pts, new_pts]) if len(self.prev_pts) > 0 else new_pts
                self.track_ids += list(range(self.next_id, self.next_id + n))
                self.next_id   += n

        # Lucas-Kanade tracking
        curr_pts, status, _ = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            curr_gray,
            self.prev_pts,
            None,
            **self.lk_params
        )

        # Forward-backward error check
        prev_pts_back, status_back, _ = cv2.calcOpticalFlowPyrLK(
            curr_gray,
            self.prev_gray,
            curr_pts,
            None,
            **self.lk_params
        )

        fb_error = np.abs(self.prev_pts - prev_pts_back).max(axis=2).flatten()
        good_mask = (status.flatten() == 1) & \
                    (status_back.flatten() == 1) & \
                    (fb_error < 1.0)

        pts_prev = self.prev_pts[good_mask].reshape(-1, 2)
        pts_curr = curr_pts[good_mask].reshape(-1, 2)
        ids      = [self.track_ids[i] for i, g in enumerate(good_mask) if g]

        # Update state for next frame
        self.prev_gray = curr_gray.copy()
        self.prev_pts  = pts_curr.reshape(-1, 1, 2)
        self.track_ids = ids

        return pts_prev, pts_curr, ids

    def draw_tracks(self, image: np.ndarray,
                          pts_prev, pts_curr) -> np.ndarray:
        """Draw optical flow tracks on image."""
        vis = image.copy()
        if pts_prev is None or pts_curr is None:
            return vis
        for p, c in zip(pts_prev, pts_curr):
            p = tuple(p.astype(int))
            c = tuple(c.astype(int))
            cv2.line(vis, p, c, (0, 255, 0), 1)
            cv2.circle(vis, c, 3, (0, 200, 255), -1)
        return vis


class VIONode(Node):
    """
    Visual-Inertial Odometry ROS2 Node.

    Pipeline per frame:
      1. Track features with LK optical flow
      2. Estimate relative rotation via Essential matrix + RANSAC
      3. Recover translation direction from 5-point algorithm
      4. Recover scale from IMU pre-integration
      5. Accumulate pose and publish
    """

    def __init__(self):
        super().__init__('vio_node')

        self.declare_parameter('camera_params', 'config/camera_params.yaml')
        self.declare_parameter('publish_debug', True)

        import yaml
        cam_path = self.get_parameter('camera_params').value
        with open(cam_path, 'r') as f:
            cam = yaml.safe_load(f)

        self.K = np.array(cam['camera_matrix'], dtype=np.float64)
        self.D = np.array(cam['dist_coefficients'], dtype=np.float64)

        self.tracker       = FeatureTracker(max_features=200)
        self.imu_integrator = IMUPreintegrator()

        # Accumulated pose in world frame
        self.T_world = np.eye(4)   # 4x4 homogeneous transform

        self.bridge    = CvBridge()
        self.pub_debug = self.get_parameter('publish_debug').value

        # ── QoS ───────────────────────────────────────────────────────────────
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw', self.image_callback, sensor_qos
        )
        self.imu_sub = self.create_subscription(
            Imu, '/mavros/imu/data', self.imu_callback, sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────
        self.pose_pub = self.create_publisher(
            PoseStamped, '/vio/pose', 10
        )
        if self.pub_debug:
            self.debug_pub = self.create_publisher(
                Image, '/vio/debug_image', 10
            )

        self.frame_count = 0
        self.get_logger().info('VIONode ready')

    def imu_callback(self, msg: Imu):
        """Accumulate IMU for pre-integration."""
        t = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        a = np.array([msg.linear_acceleration.x,
                      msg.linear_acceleration.y,
                      msg.linear_acceleration.z])
        g = np.array([msg.angular_velocity.x,
                      msg.angular_velocity.y,
                      msg.angular_velocity.z])
        self.imu_integrator.integrate(a, g, t)

    def image_callback(self, msg: Image):
        """Process frame: track → estimate motion → publish pose."""
        try:
            frame = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
        except Exception as e:
            self.get_logger().error(f'Bridge error: {e}')
            return

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Track features
        pts_prev, pts_curr, ids = self.tracker.track(gray)

        if pts_prev is None or len(pts_prev) < 8:
            self.imu_integrator.reset()
            return

        # Undistort feature points
        pts_prev_u = cv2.undistortPoints(
            pts_prev.reshape(-1,1,2), self.K, self.D, P=self.K
        ).reshape(-1, 2)
        pts_curr_u = cv2.undistortPoints(
            pts_curr.reshape(-1,1,2), self.K, self.D, P=self.K
        ).reshape(-1, 2)

        # Estimate Essential matrix with RANSAC
        E, mask = cv2.findEssentialMat(
            pts_prev_u, pts_curr_u,
            self.K,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None:
            self.imu_integrator.reset()
            return

        # Recover relative pose [R, t] — t is unit vector (scale unknown)
        _, R, t, mask_pose = cv2.recoverPose(E, pts_prev_u, pts_curr_u, self.K)

        # ── Scale recovery from IMU ────────────────────────────────────────────
        imu_delta = self.imu_integrator.get_delta()
        self.imu_integrator.reset()

        scale = np.linalg.norm(imu_delta['delta_p'])

        if scale < 0.001:
            # Drone barely moved — use small default scale
            scale = 0.01

        # Scale the translation
        t_scaled = t.flatten() * scale

        # ── Accumulate pose ───────────────────────────────────────────────────
        T_rel = np.eye(4)
        T_rel[:3, :3] = R
        T_rel[:3,  3] = t_scaled

        self.T_world = self.T_world @ np.linalg.inv(T_rel)

        # ── Publish pose ───────────────────────────────────────────────────────
        self._publish_pose(msg.header)

        # ── Debug visualization ────────────────────────────────────────────────
        if self.pub_debug:
            vis = self.tracker.draw_tracks(frame, pts_prev, pts_curr)
            inliers = int(mask_pose.sum()) if mask_pose is not None else 0
            cv2.putText(vis,
                f'Tracks: {len(pts_curr)} | Inliers: {inliers} | Scale: {scale:.3f}',
                (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 255), 1)
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(vis, encoding='bgr8')
            )

        self.frame_count += 1

    def _publish_pose(self, header):
        """Convert accumulated T_world to PoseStamped and publish."""
        R  = self.T_world[:3, :3]
        t  = self.T_world[:3, 3]

        # R → quaternion
        r  = Rotation.from_matrix(R)
        q  = r.as_quat()  # [qx, qy, qz, qw]

        msg = PoseStamped()
        msg.header       = header
        msg.header.frame_id = 'map'

        msg.pose.position.x = float(t[0])
        msg.pose.position.y = float(t[1])
        msg.pose.position.z = float(t[2])

        msg.pose.orientation.x = float(q[0])
        msg.pose.orientation.y = float(q[1])
        msg.pose.orientation.z = float(q[2])
        msg.pose.orientation.w = float(q[3])

        self.pose_pub.publish(msg)


# Need this import inside the class method
from scipy.spatial.transform import Rotation


def main(args=None):
    rclpy.init(args=args)
    node = VIONode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()