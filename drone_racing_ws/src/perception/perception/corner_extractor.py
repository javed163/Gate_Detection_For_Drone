"""
corner_extractor.py
-------------------
Converts YOLO bounding box detections into refined 2D gate corner points.

YOLO gives us: [x_center, y_center, width, height] (normalized 0-1)
We need:       4 corner points of the gate in pixel coordinates

Strategy:
  1. Convert bbox to pixel corners (rough estimate)
  2. Refine using edge detection + contour fitting (Shi-Tomasi / Harris)
  3. Order corners consistently (TL, TR, BR, BL)
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class GateCorners:
    """Ordered 2D gate corner points in pixel coordinates."""
    top_left:     np.ndarray   # [u, v]
    top_right:    np.ndarray
    bottom_right: np.ndarray
    bottom_left:  np.ndarray
    confidence:   float
    method:       str          # 'bbox' or 'refined'

    def as_array(self) -> np.ndarray:
        """Returns shape (4, 2) array ordered TL→TR→BR→BL."""
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left
        ], dtype=np.float32)


class CornerExtractor:
    """
    Extracts refined 4-corner representation of a racing gate from:
      - YOLO bounding box (fast, less accurate)
      - Sub-pixel corner refinement via goodFeaturesToTrack (more accurate)
    
    For competition use: always attempt refinement, fall back to bbox method.
    """

    def __init__(self,
                 image_width: int,
                 image_height: int,
                 refine_corners: bool = True,
                 corner_quality: float = 0.01,
                 min_corner_distance: float = 10.0):
        
        self.W = image_width
        self.H = image_height
        self.refine_corners = refine_corners
        self.corner_quality = corner_quality
        self.min_corner_distance = min_corner_distance

        # Sub-pixel refinement termination criteria
        self.subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,    # max iterations
            0.001  # epsilon
        )

    def bbox_to_pixel_corners(
        self,
        bbox_normalized: np.ndarray,
        image_shape: Tuple[int, int]
    ) -> np.ndarray:
        """
        Convert YOLO normalized bbox to 4 pixel corner coordinates.
        
        Args:
            bbox_normalized: [x_center, y_center, width, height] in [0, 1]
            image_shape: (height, width)
        
        Returns:
            corners: shape (4, 2) — [TL, TR, BR, BL] in pixels
        """
        H, W = image_shape
        cx, cy, w, h = bbox_normalized

        # Convert to pixel coordinates
        cx_px = cx * W
        cy_px = cy * H
        w_px  = w  * W
        h_px  = h  * H

        # Compute corners
        x1 = cx_px - w_px / 2.0   # left
        x2 = cx_px + w_px / 2.0   # right
        y1 = cy_px - h_px / 2.0   # top
        y2 = cy_px + h_px / 2.0   # bottom

        corners = np.array([
            [x1, y1],  # Top-Left
            [x2, y1],  # Top-Right
            [x2, y2],  # Bottom-Right
            [x1, y2],  # Bottom-Left
        ], dtype=np.float32)

        return corners

    def refine_with_harris(
        self,
        image: np.ndarray,
        bbox_normalized: np.ndarray,
        padding_factor: float = 0.1
    ) -> Optional[np.ndarray]:
        """
        Refine gate corners using Harris corner detection within the gate ROI.
        
        We search for exactly 4 dominant corners inside the bounding box region.
        The gate structure (rectangular frame) produces 4 strong Harris responses
        at the inner corners of the gate opening.
        
        Args:
            image: Full grayscale image (H, W)
            bbox_normalized: [cx, cy, w, h] normalized
            padding_factor: Expand ROI slightly beyond bbox
        
        Returns:
            corners: shape (4, 2) ordered [TL, TR, BR, BL], or None if failed
        """
        H, W = image.shape[:2]
        cx, cy, bw, bh = bbox_normalized

        # Expand ROI
        pad_x = bw * padding_factor
        pad_y = bh * padding_factor

        x1 = int(max(0, (cx - bw/2 - pad_x) * W))
        y1 = int(max(0, (cy - bh/2 - pad_y) * H))
        x2 = int(min(W, (cx + bw/2 + pad_x) * W))
        y2 = int(min(H, (cy + bh/2 + pad_y) * H))

        if x2 - x1 < 20 or y2 - y1 < 20:
            return None  # ROI too small

        # Extract ROI
        roi = image[y1:y2, x1:x2]
        if len(roi.shape) == 3:
            roi_gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            roi_gray = roi.copy()

        # Detect strong corners using Shi-Tomasi (more stable than Harris for this)
        corners = cv2.goodFeaturesToTrack(
            roi_gray,
            maxCorners=20,
            qualityLevel=self.corner_quality,
            minDistance=self.min_corner_distance,
            blockSize=7
        )

        if corners is None or len(corners) < 4:
            return None

        # Sub-pixel refinement
        corners_subpix = cv2.cornerSubPix(
            roi_gray,
            corners,
            winSize=(5, 5),
            zeroZone=(-1, -1),
            criteria=self.subpix_criteria
        )

        # Convert from ROI-local to full image coordinates
        corners_global = corners_subpix.reshape(-1, 2)
        corners_global[:, 0] += x1
        corners_global[:, 1] += y1

        # Select the 4 best corners forming the gate rectangle
        gate_corners = self._select_gate_corners(corners_global)
        return gate_corners

    def _select_gate_corners(self, points: np.ndarray) -> Optional[np.ndarray]:
        """
        From many candidate corners, select the 4 that best form
        the gate rectangle using convex hull + quadrilateral fitting.
        
        Args:
            points: shape (N, 2) candidate corners
        
        Returns:
            corners: shape (4, 2) ordered [TL, TR, BR, BL], or None
        """
        if len(points) < 4:
            return None

        # Use convex hull to find outermost points
        hull_idx = cv2.convexHull(points.astype(np.float32), returnPoints=False)
        hull_points = points[hull_idx.flatten()]

        # If hull has exactly 4 points, use directly
        if len(hull_points) == 4:
            return self._order_corners(hull_points)

        # Otherwise approximate to quadrilateral using Douglas-Peucker
        hull_closed = hull_points.reshape((-1, 1, 2)).astype(np.float32)
        epsilon = 0.05 * cv2.arcLength(hull_closed, True)
        approx = cv2.approxPolyDP(hull_closed, epsilon, True)

        if len(approx) == 4:
            return self._order_corners(approx.reshape(-1, 2))

        # Fallback: pick 4 extreme points (top, bottom, left, right → form quad)
        top    = points[np.argmin(points[:, 1])]
        bottom = points[np.argmax(points[:, 1])]
        left   = points[np.argmin(points[:, 0])]
        right  = points[np.argmax(points[:, 0])]
        
        quad = np.array([top, right, bottom, left], dtype=np.float32)
        return self._order_corners(quad)

    def _order_corners(self, corners: np.ndarray) -> np.ndarray:
        """
        Order 4 corners as [Top-Left, Top-Right, Bottom-Right, Bottom-Left].
        
        This consistent ordering is CRITICAL for solvePnP — the 2D-3D
        point correspondence must match exactly.
        
        Method: 
          - Sum  (x+y): smallest = TL, largest = BR
          - Diff (x-y): smallest = BL, largest = TR
        """
        corners = corners.reshape(4, 2).astype(np.float32)
        
        ordered = np.zeros((4, 2), dtype=np.float32)
        
        s    = corners.sum(axis=1)
        diff = np.diff(corners, axis=1).flatten()

        ordered[0] = corners[np.argmin(s)]     # Top-Left:     min(x+y)
        ordered[2] = corners[np.argmax(s)]     # Bottom-Right: max(x+y)
        ordered[1] = corners[np.argmax(diff)]  # Top-Right:    max(x-y)
        ordered[3] = corners[np.argmin(diff)]  # Bottom-Left:  min(x-y)

        return ordered

    def extract(
        self,
        image: np.ndarray,
        bbox_normalized: np.ndarray,
        confidence: float
    ) -> GateCorners:
        """
        Main extraction function. Tries refined method, falls back to bbox.
        
        Args:
            image: Full color or grayscale image
            bbox_normalized: YOLO output [cx, cy, w, h] in [0,1]
            confidence: YOLO detection confidence
        
        Returns:
            GateCorners with consistent TL→TR→BR→BL ordering
        """
        H, W = image.shape[:2]
        
        # Always compute bbox corners as fallback
        bbox_corners = self.bbox_to_pixel_corners(bbox_normalized, (H, W))
        
        if self.refine_corners:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
            refined = self.refine_with_harris(gray, bbox_normalized)
            
            if refined is not None:
                return GateCorners(
                    top_left=refined[0],
                    top_right=refined[1],
                    bottom_right=refined[2],
                    bottom_left=refined[3],
                    confidence=confidence,
                    method='refined'
                )

        # Fallback to bbox corners
        ordered = self._order_corners(bbox_corners)
        return GateCorners(
            top_left=ordered[0],
            top_right=ordered[1],
            bottom_right=ordered[2],
            bottom_left=ordered[3],
            confidence=confidence,
            method='bbox'
        )


# ──────────────────────────────────────────────────────────────────────────────
# Quick visual test (run directly: python corner_extractor.py)
# ──────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    import sys

    # Create a synthetic test image with a bright rectangle (simulated gate)
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.rectangle(img, (200, 150), (440, 330), (255, 255, 255), 6)

    # Simulated YOLO bbox (normalized)
    # Gate spans x: 200-440, y: 150-330 on 640x480 image
    cx = (200 + 440) / 2 / 640   # 0.5
    cy = (150 + 330) / 2 / 480   # 0.5
    bw = (440 - 200) / 640        # 0.375
    bh = (330 - 150) / 480        # 0.375
    bbox = np.array([cx, cy, bw, bh])

    extractor = CornerExtractor(640, 480, refine_corners=True)
    corners = extractor.extract(img, bbox, confidence=0.95)

    print(f"Method: {corners.method}")
    print(f"Top-Left:     {corners.top_left}")
    print(f"Top-Right:    {corners.top_right}")
    print(f"Bottom-Right: {corners.bottom_right}")
    print(f"Bottom-Left:  {corners.bottom_left}")

    # Visualize
    vis = img.copy()
    colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
    labels = ['TL', 'TR', 'BR', 'BL']
    arr = corners.as_array()
    for i, (pt, color, label) in enumerate(zip(arr, colors, labels)):
        cv2.circle(vis, tuple(pt.astype(int)), 8, color, -1)
        cv2.putText(vis, label, tuple((pt + 10).astype(int)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    cv2.imwrite('/tmp/corner_test.png', vis)
    print("Saved visualization to /tmp/corner_test.png")