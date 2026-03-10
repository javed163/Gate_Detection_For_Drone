"""
airsim_bridge.py
----------------
Bridges AirSim simulation ↔ ROS2 topics.
Publishes simulated camera images and IMU as ROS2 messages.
Subscribes to control commands and applies them in AirSim.

Requires: pip install airsim msgpack-rpc-python

ROS2:
    Publishes: /camera/image_raw   [sensor_msgs/Image]
    Publishes: /mavros/imu/data    [sensor_msgs/Imu]
    Publishes: /ground_truth/odom  [nav_msgs/Odometry]  (for training eval)
    Subscribes:/mavros/setpoint_raw/attitude [mavros_msgs/AttitudeTarget]
"""

import rclpy
from rclpy.node import Node
import numpy as np

from sensor_msgs.msg import Image, Imu
from nav_msgs.msg    import Odometry
from cv_bridge       import CvBridge

try:
    import airsim
    AIRSIM_AVAILABLE = True
except ImportError:
    AIRSIM_AVAILABLE = False
    print('[AirSimBridge] WARNING: airsim package not installed')


class AirSimBridge(Node):

    def __init__(self):
        super().__init__('airsim_bridge')

        if not AIRSIM_AVAILABLE:
            self.get_logger().error(
                'AirSim not installed. Run: pip install airsim'
            )
            return

        self.declare_parameter('camera_name',   'front_center')
        self.declare_parameter('vehicle_name',  'Drone1')
        self.declare_parameter('image_rate_hz', 30.0)
        self.declare_parameter('imu_rate_hz',   200.0)

        cam_name  = self.get_parameter('camera_name').value
        veh_name  = self.get_parameter('vehicle_name').value

        # ── Connect to AirSim ──────────────────────────────────────────────
        self.get_logger().info('Connecting to AirSim...')
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        self.client.enableApiControl(True, veh_name)
        self.client.armDisarm(True, veh_name)
        self.get_logger().info('✅ AirSim connected')

        self.cam_name = cam_name
        self.veh_name = veh_name
        self.bridge   = CvBridge()

        # ── Publishers ────────────────────────────────────────────────────────
        self.img_pub   = self.create_publisher(Image,    '/camera/image_raw',  10)
        self.imu_pub   = self.create_publisher(Imu,      '/mavros/imu/data',   10)
        self.odom_pub  = self.create_publisher(Odometry, '/ground_truth/odom', 10)

        # ── Timers ────────────────────────────────────────────────────────────
        img_dt = 1.0 / self.get_parameter('image_rate_hz').value
        imu_dt = 1.0 / self.get_parameter('imu_rate_hz').value

        self.create_timer(img_dt, self.publish_image)
        self.create_timer(imu_dt, self.publish_imu_and_odom)

        self.get_logger().info('AirSimBridge publishing sensor data')

    def publish_image(self):
        """Capture and publish front camera image."""
        try:
            responses = self.client.simGetImages([
                airsim.ImageRequest(
                    self.cam_name,
                    airsim.ImageType.Scene,
                    False,   # not pixels as float
                    False    # not compressed
                )
            ], vehicle_name=self.veh_name)

            if not responses or responses[0].width == 0:
                return

            resp = responses[0]
            img  = np.frombuffer(resp.image_data_uint8, dtype=np.uint8)
            img  = img.reshape(resp.height, resp.width, 3)

            msg = self.bridge.cv2_to_imgmsg(img, encoding='rgb8')
            msg.header.stamp    = self.get_clock().now().to_msg()
            msg.header.frame_id = 'camera_optical_frame'
            self.img_pub.publish(msg)

        except Exception as e:
            self.get_logger().warn(f'Image capture error: {e}')

    def publish_imu_and_odom(self):
        """Publish IMU data and ground truth odometry."""
        try:
            imu_data  = self.client.getImuData(vehicle_name=self.veh_name)
            state     = self.client.getMultirotorState(vehicle_name=self.veh_name)

            # ── IMU message ────────────────────────────────────────────────
            imu_msg = Imu()
            imu_msg.header.stamp    = self.get_clock().now().to_msg()
            imu_msg.header.frame_id = 'imu_link'

            imu_msg.linear_acceleration.x = imu_data.linear_acceleration.x_val
            imu_msg.linear_acceleration.y = imu_data.linear_acceleration.y_val
            imu_msg.linear_acceleration.z = imu_data.linear_acceleration.z_val

            imu_msg.angular_velocity.x = imu_data.angular_velocity.x_val
            imu_msg.angular_velocity.y = imu_data.angular_velocity.y_val
            imu_msg.angular_velocity.z = imu_data.angular_velocity.z_val

            self.imu_pub.publish(imu_msg)

            # ── Ground truth odometry ──────────────────────────────────────
            pos = state.kinematics_estimated.position
            vel = state.kinematics_estimated.linear_velocity
            ori = state.kinematics_estimated.orientation

            odom_msg = Odometry()
            odom_msg.header.stamp    = imu_msg.header.stamp
            odom_msg.header.frame_id = 'map'
            odom_msg.child_frame_id  = 'base_link'

            odom_msg.pose.pose.position.x    = pos.x_val
            odom_msg.pose.pose.position.y    = pos.y_val
            odom_msg.pose.pose.position.z    = pos.z_val
            odom_msg.pose.pose.orientation.w = ori.w_val
            odom_msg.pose.pose.orientation.x = ori.x_val
            odom_msg.pose.pose.orientation.y = ori.y_val
            odom_msg.pose.pose.orientation.z = ori.z_val

            odom_msg.twist.twist.linear.x = vel.x_val
            odom_msg.twist.twist.linear.y = vel.y_val
            odom_msg.twist.twist.linear.z = vel.z_val

            self.odom_pub.publish(odom_msg)

        except Exception as e:
            self.get_logger().warn(f'IMU/state error: {e}')


def main(args=None):
    rclpy.init(args=args)
    node = AirSimBridge()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()