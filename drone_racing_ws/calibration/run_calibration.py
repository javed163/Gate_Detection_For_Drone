"""
run_calibration.py
------------------
Performs camera calibration using a checkerboard pattern.
Outputs the camera intrinsic matrix K and distortion coefficients D.

Usage:
    # Collect images first:
    python run_calibration.py --collect --camera 0 --output calibration/checkerboard_images/

    # Then calibrate:
    python run_calibration.py --calibrate --images calibration/checkerboard_images/ --output calibration/calibration_results.yaml

    # Or do both at once:
    python run_calibration.py --collect --calibrate --camera 0

Physical requirement:
    Print a checkerboard pattern (9x6 inner corners recommended).
    Measure the EXACT square size in meters (e.g., 0.025 = 2.5cm).
    Hold it flat — any warping ruins calibration.
"""

import cv2
import numpy as np
import yaml
import argparse
import os
import glob
from pathlib import Path
from datetime import datetime


class CameraCalibrator:
    """
    Standard Zhang's method camera calibration using OpenCV.

    The camera model:
        [u]   [fx  0  cx] [X/Z]
        [v] = [ 0 fy  cy] [Y/Z]
        [1]   [ 0  0   1] [ 1 ]

    Where:
        fx, fy = focal lengths in pixels
        cx, cy = principal point (optical center)
        K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
        D = [k1, k2, p1, p2, k3]  (radial + tangential distortion)
    """

    def __init__(self,
                 checkerboard_size: tuple = (9, 6),
                 square_size_meters: float = 0.025):
        """
        Args:
            checkerboard_size: (cols, rows) of INNER corners — not squares.
                               A 10x7 square board has 9x6 inner corners.
            square_size_meters: Physical size of each square in meters.
        """
        self.board_size   = checkerboard_size
        self.square_size  = square_size_meters
        self.cols, self.rows = checkerboard_size

        # Termination criteria for corner sub-pixel refinement
        self.criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001
        )

        # Prepare 3D object points for one checkerboard view
        # Shape: (rows*cols, 3) — Z=0 since board is flat
        self.objp = np.zeros((self.rows * self.cols, 3), dtype=np.float32)
        self.objp[:, :2] = np.mgrid[0:self.cols, 0:self.rows].T.reshape(-1, 2)
        self.objp *= self.square_size

        # Storage for calibration data
        self.obj_points  = []  # 3D points in world space
        self.img_points  = []  # 2D points in image space
        self.image_size  = None
        self.good_images = []

    def collect_images_live(self,
                            camera_index: int = 0,
                            output_dir: str = 'checkerboard_images',
                            target_count: int = 30):
        """
        Collect calibration images from live camera feed.
        Press SPACE to capture, Q to quit.
        Minimum 20 images recommended; 30+ for accuracy.
        """
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            raise RuntimeError(f'Cannot open camera {camera_index}')

        captured = 0
        print(f'\n[Calibration] Target: {target_count} images')
        print('[Calibration] SPACE = capture | Q = quit\n')

        while captured < target_count:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Try to find checkerboard in current frame
            found, corners = cv2.findChessboardCorners(
                gray,
                (self.cols, self.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_FAST_CHECK +
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            display = frame.copy()
            if found:
                cv2.drawChessboardCorners(display, (self.cols, self.rows), corners, found)
                status_color = (0, 255, 0)
                status_text  = f'BOARD DETECTED — SPACE to capture ({captured}/{target_count})'
            else:
                status_color = (0, 0, 255)
                status_text  = f'No board detected ({captured}/{target_count})'

            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
            cv2.imshow('Camera Calibration', display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' ') and found:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
                fname = os.path.join(output_dir, f'calib_{timestamp}.png')
                cv2.imwrite(fname, frame)
                captured += 1
                print(f'  Captured image {captured}/{target_count}: {fname}')

        cap.release()
        cv2.destroyAllWindows()
        print(f'\n[Calibration] Collected {captured} images in {output_dir}')

    def calibrate_from_images(self, image_dir: str) -> dict:
        """
        Run calibration from saved checkerboard images.

        Returns dict with K, D, reprojection error, and per-image stats.
        """
        image_paths = sorted(
            glob.glob(os.path.join(image_dir, '*.png')) +
            glob.glob(os.path.join(image_dir, '*.jpg'))
        )

        if len(image_paths) < 10:
            raise ValueError(
                f'Only {len(image_paths)} images found. Need at least 10. '
                f'Recommend 25-30 for competition use.'
            )

        print(f'\n[Calibration] Processing {len(image_paths)} images...')

        rejected = 0
        for path in image_paths:
            img  = cv2.imread(path)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            if self.image_size is None:
                self.image_size = gray.shape[::-1]  # (width, height)

            found, corners = cv2.findChessboardCorners(
                gray,
                (self.cols, self.rows),
                cv2.CALIB_CB_ADAPTIVE_THRESH +
                cv2.CALIB_CB_NORMALIZE_IMAGE
            )

            if found:
                # Refine to sub-pixel accuracy
                corners_refined = cv2.cornerSubPix(
                    gray, corners, (11, 11), (-1, -1), self.criteria
                )
                self.obj_points.append(self.objp)
                self.img_points.append(corners_refined)
                self.good_images.append(path)
            else:
                rejected += 1
                print(f'  [SKIP] No corners found: {os.path.basename(path)}')

        print(f'  Good images: {len(self.good_images)} / {len(image_paths)}')
        print(f'  Rejected:    {rejected}')

        if len(self.good_images) < 10:
            raise ValueError('Not enough valid images for calibration.')

        # ── Run OpenCV calibration ──────────────────────────────────────────
        print('\n[Calibration] Running calibration...')
        rms_error, K, D, rvecs, tvecs = cv2.calibrateCamera(
            self.obj_points,
            self.img_points,
            self.image_size,
            None,   # initial K (None = estimate from scratch)
            None    # initial D
        )

        print(f'\n[Calibration] ✅ RMS Reprojection Error: {rms_error:.4f} pixels')
        print('  (Good: < 0.5px | Acceptable: < 1.0px | Bad: > 1.5px)\n')

        # ── Compute per-image reprojection errors ───────────────────────────
        per_image_errors = []
        for i, (objp, imgp, rvec, tvec) in enumerate(
            zip(self.obj_points, self.img_points, rvecs, tvecs)
        ):
            projected, _ = cv2.projectPoints(objp, rvec, tvec, K, D)
            err = cv2.norm(imgp, projected, cv2.NORM_L2) / len(projected)
            per_image_errors.append(err)

        # Flag high-error images
        mean_err = np.mean(per_image_errors)
        std_err  = np.std(per_image_errors)
        outliers = [
            (self.good_images[i], e)
            for i, e in enumerate(per_image_errors)
            if e > mean_err + 2 * std_err
        ]
        if outliers:
            print('[Warning] High-error images (consider recapturing):')
            for path, err in outliers:
                print(f'  {os.path.basename(path)}: {err:.4f}px')

        result = {
            'camera_matrix':        K.tolist(),
            'dist_coefficients':    D.flatten().tolist(),
            'image_size':           list(self.image_size),
            'rms_error':            float(rms_error),
            'num_images_used':      len(self.good_images),
            'square_size_meters':   self.square_size,
            'checkerboard_size':    list(self.board_size),
            'per_image_errors':     per_image_errors,
            # Useful derived values
            'fx': float(K[0, 0]),
            'fy': float(K[1, 1]),
            'cx': float(K[0, 2]),
            'cy': float(K[1, 2]),
        }

        self._print_results(K, D)
        return result

    def _print_results(self, K: np.ndarray, D: np.ndarray):
        print('Camera Matrix K:')
        print(f'  fx={K[0,0]:.2f}  fy={K[1,1]:.2f}')
        print(f'  cx={K[0,2]:.2f}  cy={K[1,2]:.2f}')
        print(f'\nDistortion D: {D.flatten().round(6)}')

    def save_results(self, result: dict, output_path: str):
        """Save calibration to YAML — readable by ROS2 camera_info."""
        # Remove non-serializable items
        save_data = {k: v for k, v in result.items()
                     if k != 'per_image_errors'}

        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(save_data, f, default_flow_style=False)

        print(f'\n[Calibration] Results saved to: {output_path}')

    def compute_optimal_matrix(self,
                               K: np.ndarray,
                               D: np.ndarray,
                               image_size: tuple,
                               alpha: float = 0.0) -> tuple:
        """
        Compute the optimal new camera matrix for undistortion.

        Args:
            alpha: 0.0 = crop all black pixels (full FOV usable)
                   1.0 = keep all pixels (some black borders)

        Returns:
            new_K: Optimal camera matrix for undistorted images
            roi:   Valid pixel region after undistortion
        """
        new_K, roi = cv2.getOptimalNewCameraMatrix(
            K, D, image_size, alpha, image_size
        )
        return new_K, roi


def main():
    parser = argparse.ArgumentParser(description='Camera Calibration Tool')
    parser.add_argument('--collect',   action='store_true')
    parser.add_argument('--calibrate', action='store_true')
    parser.add_argument('--camera',    type=int,   default=0)
    parser.add_argument('--images',    type=str,   default='calibration/checkerboard_images')
    parser.add_argument('--output',    type=str,   default='calibration/calibration_results.yaml')
    parser.add_argument('--cols',      type=int,   default=9,     help='Inner corner columns')
    parser.add_argument('--rows',      type=int,   default=6,     help='Inner corner rows')
    parser.add_argument('--square',    type=float, default=0.025, help='Square size in meters')
    parser.add_argument('--count',     type=int,   default=30,    help='Images to collect')
    args = parser.parse_args()

    calibrator = CameraCalibrator(
        checkerboard_size=(args.cols, args.rows),
        square_size_meters=args.square
    )

    if args.collect:
        calibrator.collect_images_live(
            camera_index=args.camera,
            output_dir=args.images,
            target_count=args.count
        )

    if args.calibrate:
        result = calibrator.calibrate_from_images(args.images)
        calibrator.save_results(result, args.output)


if __name__ == '__main__':
    main()