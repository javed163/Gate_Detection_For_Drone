"""
pose_estimator.py
-----------------
Computes the 6DoF pose of the racing gate relative to the drone camera
using OpenCV solvePnP (Perspective-n-Point).

The PnP problem:
    Given:
        - 4 known 3D gate corner points (from gate_geometry.yaml)
        - 4 observed 2D image corner points (from corner_extractor.py)
        - Camera intrinsic matrix K
    Find:
        - Rotation vector rvec  → rotation of gate in camera frame
        - Translation vector tvec → position of gate in camera frame

This gives us the FULL 6DoF pose of the gate.

ROS2 Node:
    Subscribes:  /perception/gate_detections
    Subscribes:  /camera/camera_info
    Publishes:   /perception/gate_pose  [GatePose]
"""

import cv2
import numpy as np
import yaml
from pathlib import Path
from dataclasses import dataclass
from typing import Optional, Tuple

import rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped, Point, Quaternion
from std_msgs.msg import Header

from .corner_extractor import GateCorners


@dataclass
class GatePose6DoF:
    """Full 6DoF pose of gate in camera coordinate frame."""
    # Translation: position of gate center relative to camera (meters)
    position: np.ndarray        # [x, y, z] in camera frame

    # Rotation
    rvec: np.ndarray            # Rodrigues rotation vector (3,)
    tvec: np.ndarray            # Translation vector (3,)
    rotation_matrix: np.ndarray # 3x3 rotation matrix

    # Derived useful quantities
    distance: float             # Euclidean distance to gate center (meters)
    bearing_angle: float        # Horizontal angle to gate (radians)
    elevation_angle: float      # Vertical angle to gate (radians)

    # Quality
    reprojection_error: float   # Mean pixel reprojection error
    is_valid: bool


class PoseEstimator:
    """
    Gate pose estimation using OpenCV solvePnP.

    Coordinate frames:
        Camera frame: X=right, Y=down, Z=forward (standard OpenCV convention)
        Gate frame:   X=right, Y=up, Z=out of gate face

    The 3D-2D point correspondence MUST match corner_extractor.py ordering:
        Index 0: Top-Left
        Index 1: Top-Right
        Index 2: Bottom-Right
        Index 3: Bottom-Left
    """

    def __init__(self, camera_params_path: str, gate_geometry_path: str):

        # ── Load camera intrinsics ────────────────────────────────────────────
        with open(camera_params_path, 'r') as f:
            cam_data = yaml.safe_load(f)

        K_list = cam_data['camera_matrix']
        self.K = np.array(K_list, dtype=np.float64)
        self.D = np.array(cam_data['dist_coefficients'], dtype=np.float64)

        print(f'[PoseEstimator] Camera K loaded: fx={self.K[0,0]:.1f}, fy={self.K[1,1]:.1f}')

        # ── Load gate geometry ────────────────────────────────────────────────
        with open(gate_geometry_path, 'r') as f:
            gate_data = yaml.safe_load(f)

        gate = gate_data['gate']
        corners = gate['corners_3d']

        # 3D gate corners in gate coordinate frame
        # CRITICAL: order matches corner_extractor.py TL→TR→BR→BL
        self.gate_corners_3d = np.array([
            corners['top_left'],
            corners['top_right'],
            corners['bottom_right'],
            corners['bottom_left'],
        ], dtype=np.float64)

        print(f'[PoseEstimator] Gate corners 3D:\n{self.gate_corners_3d}')

        # ── solvePnP method selection ─────────────────────────────────────────
        # SOLVEPNP_ITERATIVE: Most stable for 4-point coplanar problem
        # SOLVEPNP_IPPE_SQUARE: Designed for square targets — USE THIS for gates
        # SOLVEPNP_EPNP: Good for N>4 points
        self.pnp_method = cv2.SOLVEPNP_IPPE_SQUARE  # Best for square gates

    def estimate_pose(self, corners: GateCorners) -> Optional[GatePose6DoF]:
        """
        Estimate 6DoF gate pose from 2D corner observations.

        Args:
            corners: GateCorners from corner_extractor (TL, TR, BR, BL)

        Returns:
            GatePose6DoF or None if estimation fails
        """
        # 2D image points (observed)
        image_points = corners.as_array().astype(np.float64)  # shape (4, 2)

        # ── Undistort image points first ─────────────────────────────────────
        # This removes lens distortion from the 2D observations
        image_points_undist = cv2.undistortPoints(
            image_points.reshape(-1, 1, 2),
            self.K,
            self.D,
            P=self.K   # Re-project onto same K after undistortion
        ).reshape(-1, 2)

        # ── Run solvePnP ──────────────────────────────────────────────────────
        try:
            success, rvec, tvec = cv2.solvePnP(
                self.gate_corners_3d,      # 3D points in gate frame
                image_points_undist,       # 2D points in image
                self.K,                    # Camera intrinsics
                np.zeros(4),               # Distortion (already undistorted)
                flags=self.pnp_method
            )
        except cv2.error as e:
            print(f'[PoseEstimator] solvePnP failed: {e}')
            return None

        if not success:
            return None

        # ── Convert Rodrigues vector → rotation matrix ────────────────────────
        R, _ = cv2.Rodrigues(rvec)

        # ── Compute gate center position in camera frame ──────────────────────
        # tvec is the position of the gate's ORIGIN in the camera frame
        position = tvec.flatten()  # [x, y, z] in camera frame

        # ── Compute derived quantities ─────────────────────────────────────────
        distance       = np.linalg.norm(position)
        bearing_angle  = np.arctan2(position[0], position[2])   # horizontal
        elevation_angle = np.arctan2(-position[1], position[2]) # vertical (flip Y)

        # ── Compute reprojection error ─────────────────────────────────────────
        projected_pts, _ = cv2.projectPoints(
            self.gate_corners_3d,
            rvec, tvec,
            self.K,
            self.D
        )
        projected_pts = projected_pts.reshape(-1, 2)
        original_pts  = corners.as_array()
        reprojection_error = float(
            np.mean(np.linalg.norm(projected_pts - original_pts, axis=1))
        )

        # Reject if reprojection error is too high
        is_valid = reprojection_error < 5.0  # pixels

        return GatePose6DoF(
            position=position,
            rvec=rvec.flatten(),
            tvec=tvec.flatten(),
            rotation_matrix=R,
            distance=distance,
            bearing_angle=bearing_angle,
            elevation_angle=elevation_angle,
            reprojection_error=reprojection_error,
            is_valid=is_valid
        )

    def compute_3d_gate_position_world(
        self,
        pose: GatePose6DoF,
        T_camera_to_body: np.ndarray,
        drone_pose_world: np.ndarray
    ) -> np.ndarray:
        """
        Transform gate position from camera frame → body frame → world frame.

        This gives us the gate's absolute position in the world (map) frame,
        which is what the trajectory planner needs.

        Args:
            pose:              Gate pose in camera frame (from estimate_pose)
            T_camera_to_body:  4x4 homogeneous transform: camera → body frame
                               (from your drone's URDF / manual measurement)
            drone_pose_world:  [x, y, z, qx, qy, qz, qw] drone in world frame
                               (from EKF / VIO)

        Returns:
            gate_pos_world: [x, y, z] gate center in world frame
        """
        # Gate position in camera frame (homogeneous)
        p_camera = np.array([*pose.position, 1.0])

        # Transform to body frame
        p_body = T_camera_to_body @ p_camera

        # Build world transform from drone pose
        T_body_to_world = self._pose_to_transform(drone_pose_world)

        # Transform to world frame
        p_world = T_body_to_world @ p_body

        return p_world[:3]

    def _pose_to_transform(self, pose_7d: np.ndarray) -> np.ndarray:
        """
        Convert [x, y, z, qx, qy, qz, qw] to 4x4 homogeneous transform.
        """
        x, y, z, qx, qy, qz, qw = pose_7d

        # Quaternion → rotation matrix
        R = np.array([
            [1 - 2*(qy**2 + qz**2),     2*(qx*qy - qz*qw),     2*(qx*qz + qy*qw)],
            [    2*(qx*qy + qz*qw), 1 - 2*(qx**2 + qz**2),     2*(qy*qz - qx*qw)],
            [    2*(qx*qz - qy*qw),     2*(qy*qz + qx*qw), 1 - 2*(qx**2 + qy**2)]
        ])

        T = np.eye(4)
        T[:3, :3] = R
        T[:3,  3] = [x, y, z]
        return T

    def get_gate_normal_vector(self, pose: GatePose6DoF) -> np.ndarray:
        """
        Get the unit normal vector of the gate plane in camera frame.
        This tells us which direction the drone should fly through.
        The gate normal is the Z-axis of the gate's coordinate frame.
        """
        # Gate's Z-axis in camera frame = third column of rotation matrix
        gate_normal_cam = pose.rotation_matrix[:, 2]
        return gate_normal_cam / np.linalg.norm(gate_normal_cam)

    def draw_pose_axes(
        self,
        image: np.ndarray,
        pose: GatePose6DoF,
        axis_length: float = 0.3
    ) -> np.ndarray:
        """
        Draw 3D coordinate axes on the image at the gate center.
        Red=X, Green=Y, Blue=Z (forward through gate).
        """
        axis_points_3d = np.float32([
            [0, 0, 0],                      # Origin (gate center)
            [axis_length, 0, 0],            # X axis (right)
            [0, axis_length, 0],            # Y axis (up in gate frame)
            [0, 0, axis_length],            # Z axis (normal, through gate)
        ])

        projected, _ = cv2.projectPoints(
            axis_points_3d,
            pose.rvec, pose.tvec,
            self.K, self.D
        )
        projected = projected.reshape(-1, 2).astype(int)

        origin = tuple(projected[0])
        cv2.arrowedLine(image, origin, tuple(projected[1]), (0, 0, 255),   2)  # X = red
        cv2.arrowedLine(image, origin, tuple(projected[2]), (0, 255, 0),   2)  # Y = green
        cv2.arrowedLine(image, origin, tuple(projected[3]), (255, 0, 0),   2)  # Z = blue

        # Overlay distance and error info
        dist_text = f'd={pose.distance:.2f}m  err={pose.reprojection_error:.1f}px'
        cv2.putText(image, dist_text, (origin[0] + 10, origin[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        return image


# ─────────────────────────────────────────────────────────────────────────────
# ROS2 Node wrapper
# ─────────────────────────────────────────────────────────────────────────────

class PoseEstimatorNode(Node):
    """
    ROS2 node that wraps PoseEstimator.

    Subscribes:  /perception/gate_detections  (custom msg or republish)
    Publishes:   /perception/gate_pose        [geometry_msgs/PoseStamped]
    """

    def __init__(self):
        super().__init__('pose_estimator')

        self.declare_parameter('camera_params', 'config/camera_params.yaml')
        self.declare_parameter('gate_geometry', 'config/gate_geometry.yaml')
        self.declare_parameter('max_reprojection_error', 5.0)

        cam_path  = self.get_parameter('camera_params').value
        gate_path = self.get_parameter('gate_geometry').value

        self.estimator = PoseEstimator(cam_path, gate_path)
        self.max_err   = self.get_parameter('max_reprojection_error').value

        self.pose_pub = self.create_publisher(
            PoseStamped,
            '/perception/gate_pose',
            10
        )

        self.get_logger().info('PoseEstimatorNode ready')

    def process_corners(self, corners: GateCorners, header: Header):
        """Call from GateDetectorNode after corner extraction."""
        pose = self.estimator.estimate_pose(corners)

        if pose is None or not pose.is_valid:
            return

        msg = PoseStamped()
        msg.header = header
        msg.header.frame_id = 'camera_optical_frame'

        msg.pose.position.x = float(pose.position[0])
        msg.pose.position.y = float(pose.position[1])
        msg.pose.position.z = float(pose.position[2])

        # Convert rotation to quaternion
        quat = self._rvec_to_quaternion(pose.rvec)
        msg.pose.orientation.x = float(quat[0])
        msg.pose.orientation.y = float(quat[1])
        msg.pose.orientation.z = float(quat[2])
        msg.pose.orientation.w = float(quat[3])

        self.pose_pub.publish(msg)

        self.get_logger().debug(
            f'Gate pose: dist={pose.distance:.2f}m '
            f'bearing={np.degrees(pose.bearing_angle):.1f}deg '
            f'reproj_err={pose.reprojection_error:.2f}px'
        )

    def _rvec_to_quaternion(self, rvec: np.ndarray) -> np.ndarray:
        """Convert Rodrigues vector to quaternion [x, y, z, w]."""
        R, _ = cv2.Rodrigues(rvec)
        trace = R[0,0] + R[1,1] + R[2,2]

        if trace > 0:
            s  = 0.5 / np.sqrt(trace + 1.0)
            qw = 0.25 / s
            qx = (R[2,1] - R[1,2]) * s
            qy = (R[0,2] - R[2,0]) * s
            qz = (R[1,0] - R[0,1]) * s
        elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
            s  = 2.0 * np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
            qw = (R[2,1] - R[1,2]) / s
            qx = 0.25 * s
            qy = (R[0,1] + R[1,0]) / s
            qz = (R[0,2] + R[2,0]) / s
        elif R[1,1] > R[2,2]:
            s  = 2.0 * np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
            qw = (R[0,2] - R[2,0]) / s
            qx = (R[0,1] + R[1,0]) / s
            qy = 0.25 * s
            qz = (R[1,2] + R[2,1]) / s
        else:
            s  = 2.0 * np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
            qw = (R[1,0] - R[0,1]) / s
            qx = (R[0,2] + R[2,0]) / s
            qy = (R[1,2] + R[2,1]) / s
            qz = 0.25 * s

        return np.array([qx, qy, qz, qw])


def main(args=None):
    rclpy.init(args=args)
    node = PoseEstimatorNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()