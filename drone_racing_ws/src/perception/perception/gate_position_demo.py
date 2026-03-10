"""
gate_position_demo.py
---------------------
End-to-end demonstration:
  Camera frame → 3D gate position → World frame

Run this to verify your calibration + pose pipeline is working
before integrating into the full ROS2 system.
"""

import cv2
import numpy as np
from pose_estimator import PoseEstimator
from corner_extractor import CornerExtractor, GateCorners


def demo_full_pipeline():
    """
    Simulate the full perception pipeline on a test image.
    Replace test_image_path with a real image of your gate.
    """

    # ── Setup ─────────────────────────────────────────────────────────────────
    estimator = PoseEstimator(
        camera_params_path='config/camera_params.yaml',
        gate_geometry_path='config/gate_geometry.yaml'
    )
    extractor = CornerExtractor(640, 480, refine_corners=True)

    # ── Load test image ────────────────────────────────────────────────────────
    # Swap with your real gate image
    image = np.zeros((480, 640, 3), dtype=np.uint8)

    # Draw a simulated gate rectangle (white frame on black background)
    # Represents a gate at roughly 3m distance, slightly off-center
    cv2.rectangle(image, (180, 120), (460, 380), (255, 255, 255), 8)

    # ── Simulate YOLO bbox ─────────────────────────────────────────────────────
    # Gate spans pixels x: 180-460, y: 120-380 on 640x480 image
    cx = (180 + 460) / 2 / 640
    cy = (120 + 380) / 2 / 480
    bw = (460 - 180) / 640
    bh = (380 - 120) / 480
    bbox = np.array([cx, cy, bw, bh])

    print(f'YOLO bbox (normalized): cx={cx:.3f} cy={cy:.3f} w={bw:.3f} h={bh:.3f}')

    # ── Extract corners ────────────────────────────────────────────────────────
    corners = extractor.extract(image, bbox, confidence=0.92)
    print(f'\nCorner extraction method: {corners.method}')
    print(f'Top-Left:     {corners.top_left}')
    print(f'Top-Right:    {corners.top_right}')
    print(f'Bottom-Right: {corners.bottom_right}')
    print(f'Bottom-Left:  {corners.bottom_left}')

    # ── Estimate 6DoF pose ─────────────────────────────────────────────────────
    pose = estimator.estimate_pose(corners)

    if pose is None:
        print('\n[ERROR] Pose estimation failed')
        return

    print(f'\n--- Gate Pose in Camera Frame ---')
    print(f'Position (camera frame):')
    print(f'  X (right):   {pose.position[0]:+.3f} m')
    print(f'  Y (down):    {pose.position[1]:+.3f} m')
    print(f'  Z (forward): {pose.position[2]:+.3f} m')
    print(f'Distance to gate: {pose.distance:.3f} m')
    print(f'Bearing angle:    {np.degrees(pose.bearing_angle):+.2f} deg')
    print(f'Elevation angle:  {np.degrees(pose.elevation_angle):+.2f} deg')
    print(f'Reprojection err: {pose.reprojection_error:.3f} px')
    print(f'Valid:            {pose.is_valid}')

    # ── Gate normal vector ─────────────────────────────────────────────────────
    normal = estimator.get_gate_normal_vector(pose)
    print(f'\nGate normal (camera frame): {normal.round(3)}')
    print('  (Drone should approach along this direction)')

    # ── Transform to world frame ───────────────────────────────────────────────
    # Example: camera is 0.05m forward, 0.02m below drone center, no rotation
    T_cam_to_body = np.array([
        [ 1,  0,  0,  0.05],  # camera 5cm forward of body center
        [ 0,  1,  0,  0.00],
        [ 0,  0,  1,  0.02],  # camera 2cm below body center
        [ 0,  0,  0,  1.00]
    ])

    # Example drone pose in world frame
    # [x, y, z, qx, qy, qz, qw] — drone hovering at z=1.5m, facing forward
    drone_pose_world = np.array([0.0, 0.0, 1.5, 0.0, 0.0, 0.0, 1.0])

    gate_world_pos = estimator.compute_3d_gate_position_world(
        pose, T_cam_to_body, drone_pose_world
    )

    print(f'\n--- Gate Position in World Frame ---')
    print(f'X: {gate_world_pos[0]:+.3f} m (East)')
    print(f'Y: {gate_world_pos[1]:+.3f} m (North)')
    print(f'Z: {gate_world_pos[2]:+.3f} m (Up)')

    # ── Visualize ──────────────────────────────────────────────────────────────
    vis = image.copy()

    # Draw corners
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    labels = ['TL', 'TR', 'BR', 'BL']
    for pt, color, label in zip(corners.as_array(), colors, labels):
        cv2.circle(vis, tuple(pt.astype(int)), 8, color, -1)
        cv2.putText(vis, label, tuple((pt + 10).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

    # Draw pose axes
    vis = estimator.draw_pose_axes(vis, pose, axis_length=0.5)

    # Info overlay
    info_lines = [
        f'Distance: {pose.distance:.2f}m',
        f'Bearing:  {np.degrees(pose.bearing_angle):.1f}deg',
        f'Reproj:   {pose.reprojection_error:.2f}px',
        f'World:  ({gate_world_pos[0]:.2f}, {gate_world_pos[1]:.2f}, {gate_world_pos[2]:.2f})',
    ]
    for i, line in enumerate(info_lines):
        cv2.putText(vis, line, (10, 30 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)

    cv2.imwrite('/tmp/pose_estimation_demo.png', vis)
    print('\nVisualization saved to /tmp/pose_estimation_demo.png')


if __name__ == '__main__':
    demo_full_pipeline()