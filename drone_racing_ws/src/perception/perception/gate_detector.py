"""
gate_detector.py
----------------
ROS2 node that runs YOLO inference on camera frames and publishes
GateDetection messages with bounding boxes and refined 2D corners.

Subscribes:  /camera/image_raw  [sensor_msgs/Image]
Publishes:   /perception/gate_detections [drone_racing_msgs/GateDetection]
             /perception/debug_image     [sensor_msgs/Image]  (visualization)
"""

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy

from sensor_msgs.msg import Image
from cv_bridge import CvBridge

import cv2
import numpy as np
import torch
from pathlib import Path

# Import your custom messages
# from drone_racing_msgs.msg import GateDetection

from .corner_extractor import CornerExtractor


class GateDetectorNode(Node):
    
    def __init__(self):
        super().__init__('gate_detector')

        # ── Parameters ────────────────────────────────────────────────────────
        self.declare_parameter('model_path', 'models/gate_detector.pt')
        self.declare_parameter('confidence_threshold', 0.5)
        self.declare_parameter('iou_threshold', 0.45)
        self.declare_parameter('image_size', 640)
        self.declare_parameter('refine_corners', True)
        self.declare_parameter('publish_debug', True)
        self.declare_parameter('camera_width', 640)
        self.declare_parameter('camera_height', 480)

        model_path    = self.get_parameter('model_path').value
        self.conf_thr = self.get_parameter('confidence_threshold').value
        self.iou_thr  = self.get_parameter('iou_threshold').value
        self.img_size = self.get_parameter('image_size').value
        refine        = self.get_parameter('refine_corners').value
        self.pub_dbg  = self.get_parameter('publish_debug').value
        cam_w         = self.get_parameter('camera_width').value
        cam_h         = self.get_parameter('camera_height').value

        # ── Load YOLO model ───────────────────────────────────────────────────
        self.get_logger().info(f'Loading YOLO model from: {model_path}')
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.get_logger().info(f'Using device: {self.device}')
        
        # Using ultralytics YOLOv8 API
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.model.to(self.device)
        self.get_logger().info('YOLO model loaded successfully')

        # ── Corner extractor ──────────────────────────────────────────────────
        self.corner_extractor = CornerExtractor(
            image_width=cam_w,
            image_height=cam_h,
            refine_corners=refine
        )

        # ── ROS2 QoS ──────────────────────────────────────────────────────────
        # Use BEST_EFFORT for camera (sensor data — drop old frames, get latest)
        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # ── Subscribers ───────────────────────────────────────────────────────
        self.image_sub = self.create_subscription(
            Image,
            '/camera/image_raw',
            self.image_callback,
            sensor_qos
        )

        # ── Publishers ────────────────────────────────────────────────────────
        # Using geometry for now; replace with custom msg in full build
        from geometry_msgs.msg import PoseArray
        self.detection_pub = self.create_publisher(
            PoseArray,   # placeholder — use GateDetection in full build
            '/perception/gate_detections',
            10
        )
        
        if self.pub_dbg:
            self.debug_pub = self.create_publisher(
                Image,
                '/perception/debug_image',
                10
            )

        self.bridge = CvBridge()
        self.frame_count = 0
        self.get_logger().info('GateDetectorNode ready')

    def image_callback(self, msg: Image):
        """Process incoming camera frame."""
        try:
            # Convert ROS Image → OpenCV BGR
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f'CV Bridge error: {e}')
            return

        # Run YOLO inference
        detections = self.run_yolo(cv_image)

        # Process each detection
        processed = []
        for det in detections:
            bbox_norm = det['bbox_normalized']
            conf      = det['confidence']

            # Extract refined 2D corners
            corners = self.corner_extractor.extract(cv_image, bbox_norm, conf)
            
            processed.append({
                'bbox': bbox_norm,
                'confidence': conf,
                'corners': corners,
                'gate_id': det.get('gate_id', 0)
            })

        # Publish detections
        self.publish_detections(processed, msg.header)

        # Publish debug visualization
        if self.pub_dbg:
            debug_img = self.draw_detections(cv_image.copy(), processed)
            self.debug_pub.publish(
                self.bridge.cv2_to_imgmsg(debug_img, encoding='bgr8')
            )

        self.frame_count += 1
        if self.frame_count % 30 == 0:
            self.get_logger().info(
                f'Processed {self.frame_count} frames, '
                f'{len(processed)} gates detected'
            )

    def run_yolo(self, image: np.ndarray) -> list:
        """
        Run YOLOv8 inference.
        
        Returns list of dicts with:
          bbox_normalized: [cx, cy, w, h] in [0,1]
          confidence: float
          gate_id: int (class index, or track ID if using tracker)
        """
        results = self.model.predict(
            image,
            conf=self.conf_thr,
            iou=self.iou_thr,
            imgsz=self.img_size,
            verbose=False
        )

        detections = []
        H, W = image.shape[:2]

        for result in results:
            if result.boxes is None:
                continue
            
            for box in result.boxes:
                # xywhn = normalized [cx, cy, w, h]
                bbox_norm = box.xywhn[0].cpu().numpy()
                conf = float(box.conf[0].cpu())
                cls  = int(box.cls[0].cpu())

                detections.append({
                    'bbox_normalized': bbox_norm,
                    'confidence': conf,
                    'gate_id': cls
                })

        return detections

    def publish_detections(self, detections: list, header):
        """Publish detections. Replace with custom GateDetection msg."""
        # In full build: publish drone_racing_msgs/GateDetection
        # For now log the data
        for det in detections:
            corners = det['corners']
            self.get_logger().debug(
                f"Gate detected: conf={det['confidence']:.2f}, "
                f"method={corners.method}, "
                f"TL={corners.top_left}"
            )

    def draw_detections(self, image: np.ndarray, detections: list) -> np.ndarray:
        """Draw bounding boxes and corners for debug visualization."""
        H, W = image.shape[:2]
        colors = [(0,255,0), (255,0,0), (0,0,255), (255,255,0)]
        labels = ['TL', 'TR', 'BR', 'BL']

        for det in detections:
            cx, cy, bw, bh = det['bbox']
            
            # Draw bounding box
            x1 = int((cx - bw/2) * W)
            y1 = int((cy - bh/2) * H)
            x2 = int((cx + bw/2) * W)
            y2 = int((cy + bh/2) * H)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            conf_text = f"Gate {det['confidence']:.2f}"
            cv2.putText(image, conf_text, (x1, y1 - 8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

            # Draw corners
            corners_arr = det['corners'].as_array()
            for i, (pt, color, label) in enumerate(zip(corners_arr, colors, labels)):
                cv2.circle(image, tuple(pt.astype(int)), 6, color, -1)
                cv2.putText(image, label, tuple((pt + np.array([5, -5])).astype(int)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw corner polygon
            poly = corners_arr.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(image, [poly], True, (0, 200, 255), 1)

            # Show refinement method
            method_color = (0, 255, 0) if det['corners'].method == 'refined' else (0, 165, 255)
            cv2.putText(image, det['corners'].method,
                        (x1, y2 + 15), cv2.FONT_HERSHEY_SIMPLEX, 0.4, method_color, 1)

        return image


def main(args=None):
    rclpy.init(args=args)
    node = GateDetectorNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()