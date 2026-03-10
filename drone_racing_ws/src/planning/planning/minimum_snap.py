"""
minimum_snap.py
---------------
Minimum snap trajectory generation for drone racing.

Given a sequence of gate waypoints, generates smooth polynomial
trajectories that minimize snap (4th derivative of position).

Reference: Mellinger & Kumar, "Minimum Snap Trajectory Generation
and Control for Quadrotors" (ICRA 2011) — the foundational paper
for quadrotor racing trajectory planning.

Usage:
    planner = MinimumSnapPlanner()
    waypoints = np.array([
        [0, 0, 1.5],    # start
        [3, 0, 1.5],    # gate 1 center
        [6, 2, 1.5],    # gate 2 center
        [9, 0, 1.5],    # gate 3 center
    ])
    times = planner.allocate_time(waypoints, avg_speed=3.0)
    traj  = planner.generate(waypoints, times)
    pos, vel, acc = traj.evaluate(t=1.5)
"""

import numpy as np
from scipy.linalg import block_diag
from dataclasses import dataclass
from typing import List, Tuple, Optional
import warnings


@dataclass
class TrajectorySegment:
    """One polynomial segment between two consecutive waypoints."""
    coeffs_x: np.ndarray   # 8 polynomial coefficients
    coeffs_y: np.ndarray
    coeffs_z: np.ndarray
    t_start:  float        # segment start time
    t_end:    float        # segment end time

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start


class PolynomialTrajectory:
    """
    Piecewise polynomial trajectory — the output of MinimumSnapPlanner.

    Stores one 7th-order polynomial per segment per axis.
    Provides evaluation at any time t with derivatives.
    """

    def __init__(self, segments: List[TrajectorySegment],
                       total_time: float):
        self.segments   = segments
        self.total_time = total_time

    def _get_segment(self, t: float) -> Tuple[TrajectorySegment, float]:
        """Get segment and local time for global time t."""
        t = np.clip(t, 0.0, self.total_time)
        for seg in self.segments:
            if t <= seg.t_end + 1e-9:
                t_local = t - seg.t_start
                return seg, t_local
        return self.segments[-1], self.segments[-1].duration

    @staticmethod
    def _eval_poly(coeffs: np.ndarray, t: float, deriv: int = 0) -> float:
        """
        Evaluate polynomial or its derivative at time t.

        coeffs: [c0, c1, c2, ..., c7] for p(t) = c0 + c1*t + ... + c7*t^7
        deriv:  0=position, 1=velocity, 2=acceleration, 3=jerk, 4=snap
        """
        n = len(coeffs)
        result = 0.0
        for i in range(deriv, n):
            # Coefficient of t^(i-deriv) after taking deriv-th derivative
            c = coeffs[i]
            for j in range(deriv):
                c *= (i - j)
            result += c * (t ** (i - deriv))
        return result

    def evaluate(self, t: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Evaluate trajectory at time t.

        Returns:
            position:     [x, y, z] in meters
            velocity:     [vx, vy, vz] in m/s
            acceleration: [ax, ay, az] in m/s²
        """
        seg, t_local = self._get_segment(t)

        pos = np.array([
            self._eval_poly(seg.coeffs_x, t_local, 0),
            self._eval_poly(seg.coeffs_y, t_local, 0),
            self._eval_poly(seg.coeffs_z, t_local, 0),
        ])
        vel = np.array([
            self._eval_poly(seg.coeffs_x, t_local, 1),
            self._eval_poly(seg.coeffs_y, t_local, 1),
            self._eval_poly(seg.coeffs_z, t_local, 1),
        ])
        acc = np.array([
            self._eval_poly(seg.coeffs_x, t_local, 2),
            self._eval_poly(seg.coeffs_y, t_local, 2),
            self._eval_poly(seg.coeffs_z, t_local, 2),
        ])
        return pos, vel, acc

    def evaluate_yaw(self, t: float) -> float:
        """
        Compute desired yaw from velocity direction.
        Drone always faces direction of travel.
        """
        _, vel, _ = self.evaluate(t)
        if np.linalg.norm(vel[:2]) < 0.1:
            return 0.0
        return float(np.arctan2(vel[1], vel[0]))

    def sample_path(self, dt: float = 0.05) -> np.ndarray:
        """Sample full trajectory at fixed time intervals for visualization."""
        times = np.arange(0, self.total_time, dt)
        path  = np.array([self.evaluate(t)[0] for t in times])
        return path


class MinimumSnapPlanner:
    """
    Generates minimum-snap piecewise polynomial trajectories.

    Polynomial order: 7 (8 coefficients per segment per axis)
    Minimizes: integral of snap² over full trajectory
    Solver: closed-form QP via unconstrained optimization
            (Richter et al. 2016 formulation)
    """

    POLY_ORDER  = 7    # polynomial degree
    N_COEFFS    = 8    # coefficients per segment (order + 1)
    DERIV_SNAP  = 4    # minimize 4th derivative (snap)

    def __init__(self,
                 max_velocity:     float = 8.0,   # m/s competition speed
                 max_acceleration: float = 15.0,  # m/s²
                 corridor_radius:  float = 0.4):  # gate passage margin (m)
        self.v_max   = max_velocity
        self.a_max   = max_acceleration
        self.r_corr  = corridor_radius

    # ──────────────────────────────────────────────────────────────────────────
    # TIME ALLOCATION
    # ──────────────────────────────────────────────────────────────────────────

    def allocate_time(self,
                      waypoints:  np.ndarray,
                      avg_speed:  float = 3.0) -> np.ndarray:
        """
        Allocate time for each segment based on distance.

        Simple trapezoidal speed profile:
          - accelerate to avg_speed
          - cruise
          - decelerate

        Args:
            waypoints: (N, 3) array of waypoints
            avg_speed: target cruising speed (m/s)

        Returns:
            times: (N,) cumulative time array [0, t1, t2, ..., T]
        """
        n_segments = len(waypoints) - 1
        distances  = np.array([
            np.linalg.norm(waypoints[i+1] - waypoints[i])
            for i in range(n_segments)
        ])

        # Time per segment proportional to distance
        segment_times = distances / avg_speed

        # Enforce minimum segment time (avoid numerical issues)
        segment_times = np.maximum(segment_times, 0.5)

        # Cumulative times starting from 0
        times = np.concatenate([[0.0], np.cumsum(segment_times)])
        return times

    # ──────────────────────────────────────────────────────────────────────────
    # COST MATRIX
    # ──────────────────────────────────────────────────────────────────────────

    def _compute_Q(self, T: float, deriv: int) -> np.ndarray:
        """
        Compute the (N_COEFFS x N_COEFFS) cost matrix Q for one segment.

        Q[i,j] = integral_0^T d^deriv/dt^deriv(t^i) * d^deriv/dt^deriv(t^j) dt

        This is the standard minimum-snap cost matrix derivation.
        """
        n = self.N_COEFFS
        Q = np.zeros((n, n))

        for i in range(deriv, n):
            for j in range(deriv, n):
                # Coefficient from taking deriv-th derivative of t^i and t^j
                ci = 1.0
                cj = 1.0
                for k in range(deriv):
                    ci *= (i - k)
                    cj *= (j - k)

                # Integral of t^(i+j-2*deriv) from 0 to T
                power       = i + j - 2 * deriv + 1
                Q[i, j]     = ci * cj * (T ** power) / power

        return Q

    # ──────────────────────────────────────────────────────────────────────────
    # CONSTRAINT MATRICES
    # ──────────────────────────────────────────────────────────────────────────

    def _constraint_row(self, t: float, deriv: int) -> np.ndarray:
        """
        One row of the constraint matrix A.
        Constrains the deriv-th derivative of the polynomial at time t.

        Returns: (N_COEFFS,) row vector
        """
        row = np.zeros(self.N_COEFFS)
        for i in range(deriv, self.N_COEFFS):
            c = 1.0
            for k in range(deriv):
                c *= (i - k)
            row[i] = c * (t ** (i - deriv))
        return row

    # ──────────────────────────────────────────────────────────────────────────
    # MAIN SOLVER
    # ──────────────────────────────────────────────────────────────────────────

    def _solve_axis(self,
                    waypoints_1d: np.ndarray,
                    times:        np.ndarray) -> np.ndarray:
        """
        Solve minimum snap QP for one axis (x, y, or z).

        Formulation:
            min  cᵀ Q c
            s.t. A c = b   (boundary + continuity constraints)

        Returns: coefficients for all segments flattened
        """
        n_wp   = len(waypoints_1d)
        n_seg  = n_wp - 1
        n_vars = n_seg * self.N_COEFFS  # total decision variables

        # ── Build block-diagonal cost matrix ─────────────────────────────────
        Q_blocks = []
        for i in range(n_seg):
            T_seg = times[i+1] - times[i]
            Q_blocks.append(self._compute_Q(T_seg, self.DERIV_SNAP))
        Q_full = block_diag(*Q_blocks)

        # ── Build constraint matrix A and rhs b ───────────────────────────────
        # Constraints:
        # (1) Position at start/end of each segment = waypoint
        # (2) Velocity = 0 at start and end of full trajectory
        # (3) Continuity of vel, acc, jerk at interior waypoints

        constraints_A = []
        constraints_b = []

        for seg_i in range(n_seg):
            T_seg  = times[seg_i+1] - times[seg_i]
            offset = seg_i * self.N_COEFFS

            # Position at t=0 of segment = waypoint[seg_i]
            row         = np.zeros(n_vars)
            row[offset:offset+self.N_COEFFS] = self._constraint_row(0.0, 0)
            constraints_A.append(row)
            constraints_b.append(waypoints_1d[seg_i])

            # Position at t=T of segment = waypoint[seg_i+1]
            row         = np.zeros(n_vars)
            row[offset:offset+self.N_COEFFS] = self._constraint_row(T_seg, 0)
            constraints_A.append(row)
            constraints_b.append(waypoints_1d[seg_i+1])

        # Velocity = 0 at very start
        row         = np.zeros(n_vars)
        row[0:self.N_COEFFS] = self._constraint_row(0.0, 1)
        constraints_A.append(row)
        constraints_b.append(0.0)

        # Velocity = 0 at very end
        row         = np.zeros(n_vars)
        T_last      = times[-1] - times[-2]
        offset_last = (n_seg - 1) * self.N_COEFFS
        row[offset_last:offset_last+self.N_COEFFS] = self._constraint_row(T_last, 1)
        constraints_A.append(row)
        constraints_b.append(0.0)

        # Continuity at interior waypoints: vel, acc, jerk must match
        for seg_i in range(n_seg - 1):
            T_seg    = times[seg_i+1] - times[seg_i]
            offset_L = seg_i * self.N_COEFFS
            offset_R = (seg_i + 1) * self.N_COEFFS

            for deriv in [1, 2, 3]:   # velocity, acceleration, jerk
                row = np.zeros(n_vars)
                row[offset_L:offset_L+self.N_COEFFS] =  self._constraint_row(T_seg, deriv)
                row[offset_R:offset_R+self.N_COEFFS] = -self._constraint_row(0.0,   deriv)
                constraints_A.append(row)
                constraints_b.append(0.0)

        A = np.array(constraints_A)
        b = np.array(constraints_b)

        # ── Solve via Lagrangian: [Q Aᵀ; A 0][c; λ] = [0; b] ─────────────────
        n_c  = len(b)
        M    = np.zeros((n_vars + n_c, n_vars + n_c))
        M[:n_vars, :n_vars] = Q_full * 2
        M[:n_vars, n_vars:] = A.T
        M[n_vars:, :n_vars] = A

        rhs             = np.zeros(n_vars + n_c)
        rhs[n_vars:]    = b

        try:
            solution = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            warnings.warn('Minimum snap solver singular — using lstsq fallback')
            solution, _, _, _ = np.linalg.lstsq(M, rhs, rcond=None)

        return solution[:n_vars]

    def generate(self,
                 waypoints: np.ndarray,
                 times:     np.ndarray) -> PolynomialTrajectory:
        """
        Generate minimum snap trajectory through waypoints.

        Args:
            waypoints: (N, 3) gate center waypoints in world frame
            times:     (N,) cumulative time array

        Returns:
            PolynomialTrajectory object
        """
        assert waypoints.shape[0] == len(times), \
            f'Waypoints ({waypoints.shape[0]}) must match times ({len(times)})'
        assert waypoints.shape[1] == 3, 'Waypoints must be (N, 3)'

        n_seg = len(waypoints) - 1

        # Solve for each axis independently
        coeffs_x = self._solve_axis(waypoints[:, 0], times)
        coeffs_y = self._solve_axis(waypoints[:, 1], times)
        coeffs_z = self._solve_axis(waypoints[:, 2], times)

        # Pack into segment objects
        segments = []
        for i in range(n_seg):
            seg_slice = slice(i * self.N_COEFFS, (i+1) * self.N_COEFFS)
            segments.append(TrajectorySegment(
                coeffs_x = coeffs_x[seg_slice],
                coeffs_y = coeffs_y[seg_slice],
                coeffs_z = coeffs_z[seg_slice],
                t_start  = float(times[i]),
                t_end    = float(times[i+1])
            ))

        total_time = float(times[-1])
        traj = PolynomialTrajectory(segments, total_time)

        print(f'[MinSnap] Generated {n_seg}-segment trajectory')
        print(f'[MinSnap] Total time: {total_time:.2f}s')
        print(f'[MinSnap] Avg speed:  {self._compute_avg_speed(waypoints, total_time):.2f} m/s')

        return traj

    def _compute_avg_speed(self, waypoints, T):
        total_dist = sum(
            np.linalg.norm(waypoints[i+1] - waypoints[i])
            for i in range(len(waypoints)-1)
        )
        return total_dist / T if T > 0 else 0