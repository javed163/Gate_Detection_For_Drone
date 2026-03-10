"""
rl_controller.py
----------------
Deploys trained SB3 PPO policy as a ROS2 node.

Runs inference at 50Hz using the trained policy.
Outputs action [thrust, roll_rate, pitch_rate, yaw_rate]
which is mixed with PID output for safety.

ROS2:
    Subscribes: /state_estimation/odom    [nav_msgs/Odometry]
    Subscribes: /perception/gate_pose     [geometry_msgs/PoseStamped]
    Publishes:  /control/rl_action        [geometry_msgs/Twist]
"""

import rclpy
from rclpy.node import Node
import numpy as np
import torch
from pathlib import Path

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import VecNormalize

from geometry_msgs.msg import Twist, PoseStamped
from nav_msgs.msg      import Odometry
from mavros_msgs.msg   import AttitudeTarget


class RLControllerNode(Node):

    def __init__(self):
        super().__init__('rl_controller')

        self.declare_parameter('policy_path',  'models/rl_policy.zip')
        self.declare_parameter('norm_path',    'models/vec_normalize.pkl')
        self.declare_parameter('control_hz',   50.0)
        self.declare_parameter('rl_weight',    0.7)  # blend: 0=PID only, 1=RL only
        self.declare_parameter('gate_pass_dist', 0.5)

        policy_path  = self.get_parameter('policy_path').value
        norm_path    = self.get_parameter('norm_path').value
        self.rl_w    = self.get_parameter('rl_weight').value
        hz           = self.get_parameter('control_hz').value
        self.dt      = 1.0 / hz

        # Physical limits
        self.max_rate = np.radians(300)   # rad/s

        # ── Load trained policy ────────────────────────────────────────────
        self.get_logger().info(f'Loading RL policy from: {policy_path}')
        self.model = PPO.load(policy_path, device='cpu')
        self.model.set_env(None)

        # Load observation normalizer
        self.obs_normalizer = None
        if Path(norm_path).exists():
            self.obs_normalizer = VecNormalize.load(norm_path, venv=None)
            self.obs_normalizer.training = False
            self.obs_normalizer.norm_reward = False
            self.get_logger().info('Observation normalizer loaded')

        self.get_logger().info('RL policy loaded successfully')

        # State
        self.drone_pos   = np.zeros(3)
        self.drone_vel   = np.zeros(3)
        self.drone_euler = np.zeros(3)
        self.gate_pos    = np.array([5.0, 0.0, 1.5])   # default
        self.gate_normal = np.array([1.0, 0.0, 0.0])
        self.next_gate_pos = np.array([10.0, 0.0, 1.5])

        # ── Subscribers ───────────────────────────────────────────────────────
        self.odom_sub = self.create_subscription(
            Odometry, '/state_estimation/odom', self.odom_callback, 10
        )
        self.gate_sub = self.create_subscription(
            PoseStamped, '/perception/gate_pose', self.gate_callback, 10
        )

        # ── Publisher ─────────────────────────────────────────────────────────
        self.action_pub = self.create_publisher(
            AttitudeTarget,
            '/control/rl_attitude_target',
            10
        )

        # ── Control timer ──────────────────────────────────────────────────
        self.create_timer(self.dt, self.control_step)
        self.get_logger().info(f'RLControllerNode running at {hz}Hz')

    def odom_callback(self, msg: Odometry):
        self.drone_pos = np.array([
            msg.pose.pose.position.x,
            msg.pose.pose.position.y,
            msg.pose.pose.position.z
        ])
        self.drone_vel = np.array([
            msg.twist.twist.linear.x,
            msg.twist.twist.linear.y,
            msg.twist.twist.linear.z
        ])
        # Quaternion → euler (simplified)
        q  = msg.pose.pose.orientation
        self.drone_euler = self._quat_to_euler(q.w, q.x, q.y, q.z)

    def gate_callback(self, msg: PoseStamped):
        self.gate_pos = np.array([
            msg.pose.position.x,
            msg.pose.position.y,
            msg.pose.position.z
        ])

    def _build_observation(self) -> np.ndarray:
        """Build the same 18-dim observation as training env."""
        R = self._rpy_to_R(*self.drone_euler)

        # Relative position to gate in body frame
        rel_world = self.gate_pos - self.drone_pos
        rel_body  = R.T @ rel_world
        gate_dist = float(np.linalg.norm(rel_world))

        # Gate normal in body frame
        gate_normal_body = R.T @ self.gate_normal

        # Next gate look-ahead in body frame
        next_rel_world = self.next_gate_pos - self.drone_pos
        next_rel_body  = R.T @ next_rel_world

        obs = np.array([
            rel_body[0], rel_body[1], rel_body[2],
            self.drone_vel[0], self.drone_vel[1], self.drone_vel[2],
            self.drone_euler[0], self.drone_euler[1], self.drone_euler[2],
            gate_normal_body[0], gate_normal_body[1], gate_normal_body[2],
            next_rel_body[0], next_rel_body[1], next_rel_body[2],
            gate_dist,
            float(np.linalg.norm(self.drone_vel)),
            1.0
        ], dtype=np.float32)

        return obs

    def control_step(self):
        """Run one RL inference step and publish action."""
        obs = self._build_observation()

        # Normalize observation if normalizer available
        if self.obs_normalizer is not None:
            obs = self.obs_normalizer.normalize_obs(obs.reshape(1, -1))[0]

        # RL inference
        action, _ = self.model.predict(
            obs.reshape(1, -1),
            deterministic=True   # no exploration during deployment
        )
        action = action.flatten()   # [thrust_n, roll_rate_n, pitch_rate_n, yaw_rate_n]

        # Map to physical units
        thrust     = float(np.clip((action[0] + 1.0) / 2.0, 0.1, 1.0))
        roll_rate  = float(action[1] * self.max_rate)
        pitch_rate = float(action[2] * self.max_rate)
        yaw_rate   = float(action[3] * np.radians(180))

        # Build and publish AttitudeTarget
        msg = AttitudeTarget()
        msg.header.stamp    = self.get_clock().now().to_msg()
        msg.header.frame_id = 'map'
        msg.type_mask       = 0b10000000   # use body rates + thrust
        msg.body_rate.x     = roll_rate
        msg.body_rate.y     = pitch_rate
        msg.body_rate.z     = yaw_rate
        msg.thrust          = thrust

        self.action_pub.publish(msg)

    def _quat_to_euler(self, qw, qx, qy, qz):
        roll  = np.arctan2(2*(qw*qx + qy*qz), 1 - 2*(qx**2 + qy**2))
        pitch = np.arcsin(np.clip(2*(qw*qy - qz*qx), -1, 1))
        yaw   = np.arctan2(2*(qw*qz + qx*qy), 1 - 2*(qy**2 + qz**2))
        return np.array([roll, pitch, yaw])

    def _rpy_to_R(self, r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])


def main(args=None):
    rclpy.init(args=args)
    node = RLControllerNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()