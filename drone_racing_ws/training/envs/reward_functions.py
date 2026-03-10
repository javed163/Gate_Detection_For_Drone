"""
reward_functions.py
-------------------
Modular reward components. Swap these in DroneRacingEnv
to experiment with different training strategies.

Research shows shaped rewards + curriculum learning
converge much faster than sparse gate rewards alone.
"""

import numpy as np


class GateProgressReward:
    """Dense reward for making progress toward the gate."""

    def __init__(self, scale: float = 2.0):
        self.scale    = scale
        self.prev_dist = np.inf

    def reset(self, initial_dist: float):
        self.prev_dist = initial_dist

    def __call__(self, current_dist: float) -> float:
        improvement    = self.prev_dist - current_dist
        self.prev_dist = current_dist
        return float(improvement * self.scale)


class GatePassReward:
    """Sparse reward for passing through a gate."""

    def __init__(self,
                 base_reward:      float = 20.0,
                 speed_multiplier: float = 2.0,
                 alignment_thresh: float = 0.5):
        self.base        = base_reward
        self.speed_mult  = speed_multiplier
        self.align_thr   = alignment_thresh

    def __call__(self,
                 dist_to_gate: float,
                 pass_threshold: float,
                 velocity: np.ndarray,
                 gate_normal: np.ndarray) -> float:

        if dist_to_gate > pass_threshold:
            return 0.0

        speed     = float(np.linalg.norm(velocity))
        if speed < 0.1:
            return 0.0

        alignment = float(np.dot(velocity / speed, gate_normal))
        if alignment < self.align_thr:
            return 0.0

        return self.base + speed * self.speed_mult


class CurriculumManager:
    """
    Curriculum learning: gradually increase difficulty.

    Stage 0: Single gate, large passage threshold, slow speed
    Stage 1: 3 gates, medium threshold
    Stage 2: Full track, strict threshold, fast speed required
    Stage 3: Random tracks, wind disturbances
    """

    STAGES = [
        {'n_gates': 1, 'pass_dist': 1.5, 'avg_speed': 1.0, 'randomize': False},
        {'n_gates': 3, 'pass_dist': 1.0, 'avg_speed': 2.0, 'randomize': False},
        {'n_gates': 5, 'pass_dist': 0.6, 'avg_speed': 3.5, 'randomize': True},
        {'n_gates': 8, 'pass_dist': 0.4, 'avg_speed': 5.0, 'randomize': True},
    ]

    def __init__(self,
                 success_threshold:   float = 0.8,
                 window_size:         int   = 100):
        self.stage             = 0
        self.success_threshold = success_threshold
        self.window_size       = window_size
        self.recent_results    = []

    def record_episode(self, gates_passed: int, n_gates: int):
        success = (gates_passed >= n_gates)
        self.recent_results.append(float(success))
        if len(self.recent_results) > self.window_size:
            self.recent_results.pop(0)

    def maybe_advance(self) -> bool:
        """Check if we should advance to next stage."""
        if len(self.recent_results) < self.window_size:
            return False
        if self.stage >= len(self.STAGES) - 1:
            return False

        success_rate = np.mean(self.recent_results)
        if success_rate >= self.success_threshold:
            self.stage += 1
            self.recent_results.clear()
            print(f'\n🎓 Curriculum advance → Stage {self.stage}: '
                  f'{self.STAGES[self.stage]}')
            return True
        return False

    def get_config(self) -> dict:
        return self.STAGES[self.stage].copy()