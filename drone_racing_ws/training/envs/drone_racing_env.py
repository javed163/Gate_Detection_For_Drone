"""
drone_racing_env.py
-------------------
OpenAI Gym environment for drone racing RL training.

Wraps AirSim (primary) or a simple physics model (fast training).
The RL agent learns to fly through gates as fast as possible.

Observation space (18-dim):
    [0:3]   relative position to next gate (x,y,z) in body frame
    [3:6]   drone velocity (vx,vy,vz) in world frame
    [6:9]   drone orientation as euler angles (roll,pitch,yaw)
    [9:12]  gate normal vector in body frame
    [12:15] relative position to gate after next (look-ahead)
    [15]    distance to current gate
    [16]    current speed
    [17]    time remaining in episode

Action space (4-dim continuous):
    [0]     collective thrust   [-1, 1] → mapped to [0, 1]
    [1]     roll rate           [-1, 1] → mapped to [-max_rate, max_rate]
    [2]     pitch rate          [-1, 1] → mapped to [-max_rate, max_rate]
    [3]     yaw rate            [-1, 1] → mapped to [-max_rate, max_rate]
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from typing import Optional, Tuple, Dict, Any
import yaml


class SimpleDronePhysics:
    """
    Fast analytical drone physics model for initial RL training.
    Use this before moving to AirSim — 100x faster, great for
    policy bootstrapping.

    Model: rigid body with drag, no propeller dynamics.
    """

    def __init__(self, dt: float = 0.02):
        self.dt   = dt
        self.mass = 0.8       # kg
        self.g    = 9.81      # m/s²
        self.max_thrust = self.mass * self.g * 2.0   # N

        # Rate limits (rad/s)
        self.max_roll_rate  = np.radians(300)
        self.max_pitch_rate = np.radians(300)
        self.max_yaw_rate   = np.radians(180)

        # Drag coefficients
        self.kd_lin  = 0.2    # linear drag
        self.kd_ang  = 0.5    # angular drag

        self.reset()

    def reset(self, pos=None, vel=None):
        self.pos   = pos if pos is not None else np.zeros(3)
        self.vel   = vel if vel is not None else np.zeros(3)
        self.euler = np.zeros(3)   # roll, pitch, yaw
        self.omega = np.zeros(3)   # angular rates

    def step(self, action: np.ndarray):
        """
        Apply action and integrate dynamics.

        action: [thrust_norm, roll_rate_norm, pitch_rate_norm, yaw_rate_norm]
                all in [-1, 1]
        """
        thrust_norm = (action[0] + 1.0) / 2.0   # [-1,1] → [0,1]
        roll_rate   = action[1] * self.max_roll_rate
        pitch_rate  = action[2] * self.max_pitch_rate
        yaw_rate    = action[3] * self.max_yaw_rate

        thrust = thrust_norm * self.max_thrust

        # Rotation matrix body → world
        r, p, y = self.euler
        R = self._rpy_to_R(r, p, y)

        # Thrust force in world frame (along body Z-up)
        thrust_world = R @ np.array([0, 0, thrust])

        # Net force = thrust - gravity - drag
        gravity   = np.array([0, 0, -self.mass * self.g])
        drag      = -self.kd_lin * self.vel
        F_net     = thrust_world + gravity + drag

        # Integrate velocity and position
        acc      = F_net / self.mass
        self.vel = self.vel + acc * self.dt
        self.pos = self.pos + self.vel * self.dt

        # Integrate angular rates (simple Euler, good enough for RL training)
        target_omega = np.array([roll_rate, pitch_rate, yaw_rate])
        ang_drag     = -self.kd_ang * self.omega
        self.omega   = self.omega + (target_omega - self.omega + ang_drag) * self.dt * 10
        self.euler   = self.euler + self.omega * self.dt

        # Wrap yaw
        self.euler[2] = (self.euler[2] + np.pi) % (2*np.pi) - np.pi

    def _rpy_to_R(self, r, p, y):
        cr, sr = np.cos(r), np.sin(r)
        cp, sp = np.cos(p), np.sin(p)
        cy, sy = np.cos(y), np.sin(y)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp,   cp*sr,            cp*cr           ]
        ])

    def get_state(self):
        return {
            'position':    self.pos.copy(),
            'velocity':    self.vel.copy(),
            'euler':       self.euler.copy(),
            'speed':       float(np.linalg.norm(self.vel))
        }


class DroneRacingEnv(gym.Env):
    """
    Drone racing Gymnasium environment.

    Compatible with Stable-Baselines3 out of the box.
    """

    metadata = {'render_modes': ['human', 'rgb_array']}

    def __init__(self, config: dict = None):
        super().__init__()

        config = config or {}

        # ── Environment config ─────────────────────────────────────────────
        self.dt              = config.get('dt',              0.02)
        self.max_episode_sec = config.get('max_episode_sec', 30.0)
        self.n_gates         = config.get('n_gates',          5)
        self.gate_radius     = config.get('gate_radius',      0.6)   # m — half-width
        self.gate_pass_dist  = config.get('gate_pass_dist',   0.5)   # m threshold
        self.crash_height    = config.get('crash_height',     0.15)  # m above ground
        self.use_airsim      = config.get('use_airsim',       False)
        self.randomize_gates = config.get('randomize_gates',  True)

        # ── Physics ────────────────────────────────────────────────────────
        self.drone = SimpleDronePhysics(dt=self.dt)

        # ── Spaces ─────────────────────────────────────────────────────────
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf,
            shape=(18,), dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=-1.0, high=1.0,
            shape=(4,), dtype=np.float32
        )

        # ── Race track ─────────────────────────────────────────────────────
        self.gate_positions  = None
        self.gate_normals    = None
        self.current_gate    = 0
        self.gates_passed    = 0
        self.step_count      = 0
        self.max_steps       = int(self.max_episode_sec / self.dt)
        self.prev_gate_dist  = np.inf
        self.total_reward    = 0.0

    # ──────────────────────────────────────────────────────────────────────────
    # TRACK GENERATION
    # ──────────────────────────────────────────────────────────────────────────

    def _generate_track(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a random race track with n_gates gates.

        Gates are placed along a curved path with random offsets.
        Returns gate positions (N,3) and normal vectors (N,3).
        """
        positions = []
        normals   = []

        # Start with a base circular track, add noise
        angle_step = 2 * np.pi / self.n_gates

        for i in range(self.n_gates):
            if self.randomize_gates:
                radius  = np.random.uniform(4.0, 8.0)
                height  = np.random.uniform(1.0, 2.5)
                angle   = i * angle_step + np.random.uniform(-0.2, 0.2)
            else:
                radius  = 6.0
                height  = 1.5
                angle   = i * angle_step

            x = radius * np.cos(angle)
            y = radius * np.sin(angle)
            z = height

            positions.append([x, y, z])

            # Normal vector: tangent to the circle (direction of travel)
            nx = -np.sin(angle)
            ny =  np.cos(angle)
            normals.append([nx, ny, 0.0])

        return np.array(positions), np.array(normals)

    # ──────────────────────────────────────────────────────────────────────────
    # GYM INTERFACE
    # ──────────────────────────────────────────────────────────────────────────

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Generate new track
        self.gate_positions, self.gate_normals = self._generate_track()
        self.current_gate    = 0
        self.gates_passed    = 0
        self.step_count      = 0
        self.total_reward    = 0.0

        # Spawn drone near first gate
        start_pos = self.gate_positions[0] - self.gate_normals[0] * 2.0
        start_pos[2] = self.gate_positions[0][2]
        self.drone.reset(pos=start_pos)

        self.prev_gate_dist = np.linalg.norm(
            self.drone.get_state()['position'] - self.gate_positions[0]
        )

        obs  = self._get_observation()
        info = {'gates_passed': 0, 'total_reward': 0.0}
        return obs.astype(np.float32), info

    def step(self, action: np.ndarray):
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)

        # Step physics
        self.drone.step(action)
        self.step_count += 1

        state = self.drone.get_state()

        # Compute reward
        reward, reward_info = self._compute_reward(state, action)
        self.total_reward += reward

        # Check termination
        terminated, term_reason = self._check_terminated(state)
        truncated = self.step_count >= self.max_steps

        obs  = self._get_observation()
        info = {
            'gates_passed': self.gates_passed,
            'total_reward': self.total_reward,
            'speed':        state['speed'],
            **reward_info
        }
        if terminated:
            info['termination_reason'] = term_reason

        return obs.astype(np.float32), reward, terminated, truncated, info

    # ──────────────────────────────────────────────────────────────────────────
    # OBSERVATION
    # ──────────────────────────────────────────────────────────────────────────

    def _get_observation(self) -> np.ndarray:
        """Build 18-dim observation vector."""
        state    = self.drone.get_state()
        pos      = state['position']
        vel      = state['velocity']
        euler    = state['euler']

        # Current gate relative position in body frame
        gate_pos = self.gate_positions[self.current_gate]
        rel_pos_world = gate_pos - pos
        R_body_to_world = SimpleDronePhysics()._rpy_to_R(*euler)
        rel_pos_body  = R_body_to_world.T @ rel_pos_world
        gate_dist     = float(np.linalg.norm(rel_pos_world))

        # Gate normal in body frame
        gate_normal_world = self.gate_normals[self.current_gate]
        gate_normal_body  = R_body_to_world.T @ gate_normal_world

        # Look-ahead: next gate
        next_gate_idx = (self.current_gate + 1) % len(self.gate_positions)
        next_gate_pos = self.gate_positions[next_gate_idx]
        next_rel_world = next_gate_pos - pos
        next_rel_body  = R_body_to_world.T @ next_rel_world

        obs = np.array([
            # [0:3] relative position to current gate (body frame)
            rel_pos_body[0], rel_pos_body[1], rel_pos_body[2],
            # [3:6] drone velocity (world frame)
            vel[0], vel[1], vel[2],
            # [6:9] euler angles
            euler[0], euler[1], euler[2],
            # [9:12] gate normal (body frame)
            gate_normal_body[0], gate_normal_body[1], gate_normal_body[2],
            # [12:15] next gate relative position (look-ahead)
            next_rel_body[0], next_rel_body[1], next_rel_body[2],
            # [15] distance to gate
            gate_dist,
            # [16] current speed
            state['speed'],
            # [17] time remaining (normalized)
            1.0 - self.step_count / self.max_steps
        ], dtype=np.float32)

        return obs

    # ──────────────────────────────────────────────────────────────────────────
    # REWARD
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_reward(self, state, action) -> Tuple[float, dict]:
        """
        Reward function for drone racing.

        Design principles:
          1. Reward progress toward gate (dense — shapes learning)
          2. Big reward for passing through gate
          3. Speed bonus (faster = better in racing)
          4. Penalize crashes hard
          5. Penalize large deviations from gate center
          6. Penalize large tilt angles (instability)
        """
        pos  = state['position']
        vel  = state['velocity']
        euler = state['euler']

        gate_pos  = self.gate_positions[self.current_gate]
        gate_dist = float(np.linalg.norm(pos - gate_pos))

        reward_info = {}

        # ── 1. Progress reward (dense) ─────────────────────────────────────
        dist_improvement  = self.prev_gate_dist - gate_dist
        progress_reward   = dist_improvement * 2.0
        self.prev_gate_dist = gate_dist
        reward_info['progress'] = progress_reward

        # ── 2. Gate passage reward ─────────────────────────────────────────
        gate_reward = 0.0
        if gate_dist < self.gate_pass_dist:
            # Check alignment: drone must pass through, not just be nearby
            gate_normal = self.gate_normals[self.current_gate]
            vel_norm    = np.linalg.norm(vel)
            if vel_norm > 0.1:
                alignment = float(np.dot(vel / vel_norm, gate_normal))
                if alignment > 0.5:   # flying in correct direction
                    gate_reward = 20.0 + state['speed'] * 2.0   # speed bonus
                    self.gates_passed    += 1
                    self.current_gate     = (self.current_gate + 1) % len(self.gate_positions)
                    self.prev_gate_dist   = np.linalg.norm(
                        pos - self.gate_positions[self.current_gate]
                    )
        reward_info['gate'] = gate_reward

        # ── 3. Speed reward (encourage fast flight) ────────────────────────
        speed_reward = state['speed'] * 0.05
        reward_info['speed'] = speed_reward

        # ── 4. Lateral deviation penalty ──────────────────────────────────
        # Penalize flying far off the line between gates
        gate_normal   = self.gate_normals[self.current_gate]
        rel_to_gate   = gate_pos - pos
        axial_dist    = float(np.dot(rel_to_gate, gate_normal))
        lateral_vec   = rel_to_gate - axial_dist * gate_normal
        lateral_dist  = float(np.linalg.norm(lateral_vec))
        lateral_penalty = -lateral_dist * 0.1 if lateral_dist > self.gate_radius else 0.0
        reward_info['lateral'] = lateral_penalty

        # ── 5. Attitude penalty (penalize extreme tilt) ────────────────────
        roll_deg  = abs(np.degrees(euler[0]))
        pitch_deg = abs(np.degrees(euler[1]))
        attitude_penalty = 0.0
        if roll_deg > 60 or pitch_deg > 60:
            attitude_penalty = -0.5
        reward_info['attitude'] = attitude_penalty

        # ── 6. Crash penalty ───────────────────────────────────────────────
        crash_penalty = 0.0
        if pos[2] < self.crash_height:
            crash_penalty = -50.0
        reward_info['crash'] = crash_penalty

        total = (progress_reward + gate_reward + speed_reward +
                 lateral_penalty + attitude_penalty + crash_penalty)

        return float(total), reward_info

    # ──────────────────────────────────────────────────────────────────────────
    # TERMINATION
    # ──────────────────────────────────────────────────────────────────────────

    def _check_terminated(self, state) -> Tuple[bool, str]:
        pos   = state['position']
        euler = state['euler']

        if pos[2] < self.crash_height:
            return True, 'ground_crash'

        if abs(pos[0]) > 30 or abs(pos[1]) > 30 or pos[2] > 15:
            return True, 'out_of_bounds'

        if abs(np.degrees(euler[0])) > 80 or abs(np.degrees(euler[1])) > 80:
            return True, 'flipped'

        if self.gates_passed >= len(self.gate_positions):
            return True, 'track_complete'

        return False, ''

    def render(self):
        pass   # Implement matplotlib visualization if needed

    def close(self):
        pass