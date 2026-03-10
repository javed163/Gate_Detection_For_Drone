from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os


def generate_launch_description():

    perception_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('perception'),
            'launch', 'perception.launch.py'
        ))
    )

    state_estimation_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory('state_estimation'),
            'launch', 'state_estimation.launch.py'
        ))
    )

    gate_sequencer_node = Node(
        package='planning',
        executable='gate_sequencer',
        name='gate_sequencer',
        output='screen'
    )

    pid_node = Node(
        package='control',
        executable='pid_controller',
        name='pid_controller',
        parameters=[os.path.join(
            get_package_share_directory('control'),
            'config', 'pid_params.yaml'
        )],
        output='screen'
    )

    mavros_bridge_node = Node(
        package='control',
        executable='mavros_bridge',
        name='mavros_bridge',
        parameters=[os.path.join(
            get_package_share_directory('control'),
            'config', 'pid_params.yaml'
        )],
        output='screen'
    )

    return LaunchDescription([
        perception_launch,
        state_estimation_launch,
        gate_sequencer_node,
        # Delay PID + MAVROS bridge by 3s — let state estimation initialize
        TimerAction(period=3.0, actions=[pid_node]),
        TimerAction(period=4.0, actions=[mavros_bridge_node]),
    ])