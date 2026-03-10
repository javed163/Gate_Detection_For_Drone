from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():
    pkg = get_package_share_directory('state_estimation')

    ekf_params  = os.path.join(pkg, 'config', 'ekf_params.yaml')
    cam_params  = os.path.join(pkg, '..', 'perception', 'config', 'camera_params.yaml')

    use_vio_arg = DeclareLaunchArgument(
        'use_vio', default_value='true',
        description='Enable Visual-Inertial Odometry'
    )

    ekf_node = Node(
        package='state_estimation',
        executable='ekf_node',
        name='ekf_node',
        parameters=[ekf_params],
        remappings=[
            ('/mavros/imu/data', '/mavros/imu/data'),
            ('/perception/gate_pose', '/perception/gate_pose'),
        ],
        output='screen'
    )

    vio_node = Node(
        package='state_estimation',
        executable='vio_node',
        name='vio_node',
        parameters=[{
            'camera_params': cam_params,
            'publish_debug': True
        }],
        output='screen'
    )

    return LaunchDescription([
        use_vio_arg,
        ekf_node,
        vio_node,
    ])