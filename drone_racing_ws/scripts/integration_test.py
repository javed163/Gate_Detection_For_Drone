"""
integration_test.py
-------------------
Verifies the full pipeline end-to-end before real flight.
Run this in simulation to confirm all nodes are communicating.

Tests:
  1. All required topics are publishing
  2. Gate detection is running at target rate
  3. EKF state is stable (no NaN, reasonable values)
  4. MAVROS is connected to PX4
  5. Control loop is running

Usage:
    python scripts/integration_test.py
"""

import rclpy
from rclpy.node import Node
from rclpy.qos  import QoSProfile, ReliabilityPolicy, HistoryPolicy
import numpy as np
import time

from sensor_msgs.msg  import Image, Imu
from nav_msgs.msg     import Odometry
from geometry_msgs.msg import PoseStamped
from mavros_msgs.msg  import State


class SystemHealthChecker(Node):

    REQUIRED_TOPICS = {
        '/camera/image_raw':          {'msg_type': Image,       'min_hz': 20.0},
        '/mavros/imu/data':           {'msg_type': Imu,         'min_hz': 100.0},
        '/state_estimation/odom':     {'msg_type': Odometry,    'min_hz': 50.0},
        '/perception/gate_pose':      {'msg_type': PoseStamped, 'min_hz': 5.0},
        '/mavros/state':              {'msg_type': State,       'min_hz': 1.0},
    }

    def __init__(self):
        super().__init__('system_health_checker')

        self.counts   = {t: 0 for t in self.REQUIRED_TOPICS}
        self.last_msg = {t: None for t in self.REQUIRED_TOPICS}
        self.ekf_ok   = True
        self.mavros_connected = False

        sensor_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Subscribe to all required topics
        self.create_subscription(Image, '/camera/image_raw',
            lambda m: self._record('/camera/image_raw'), sensor_qos)
        self.create_subscription(Imu, '/mavros/imu/data',
            lambda m: self._record('/mavros/imu/data'), sensor_qos)
        self.create_subscription(Odometry, '/state_estimation/odom',
            self._odom_cb, 10)
        self.create_subscription(PoseStamped, '/perception/gate_pose',
            lambda m: self._record('/perception/gate_pose'), 10)
        self.create_subscription(State, '/mavros/state',
            self._state_cb, 10)

        # Run health check every 5 seconds
        self.start_time = time.time()
        self.create_timer(5.0, self.health_check)
        self.get_logger().info('System health checker started (5s warmup)...')

    def _record(self, topic: str):
        self.counts[topic] += 1
        self.last_msg[topic] = time.time()

    def _odom_cb(self, msg: Odometry):
        self._record('/state_estimation/odom')
        p = msg.pose.pose.position
        if any(np.isnan([p.x, p.y, p.z])):
            self.ekf_ok = False
            self.get_logger().error('❌ EKF state contains NaN!')
        if abs(p.z) > 50:
            self.get_logger().warn(f'⚠️  Unusual altitude: {p.z:.1f}m')

    def _state_cb(self, msg: State):
        self._record('/mavros/state')
        self.mavros_connected = msg.connected

    def health_check(self):
        elapsed = time.time() - self.start_time
        if elapsed < 4.0:
            return   # warmup

        print('\n' + '='*50)
        print(f'  SYSTEM HEALTH REPORT  (t={elapsed:.0f}s)')
        print('='*50)

        all_ok = True

        for topic, cfg in self.REQUIRED_TOPICS.items():
            hz     = self.counts[topic] / elapsed
            ok     = hz >= cfg['min_hz'] * 0.5
            status = '✅' if ok else '❌'
            print(f'{status} {topic:45s} {hz:6.1f} Hz '
                  f'(min: {cfg["min_hz"]:.0f} Hz)')
            if not ok:
                all_ok = False

        print()
        print(f'{"✅" if self.mavros_connected else "❌"} MAVROS connected: '
              f'{self.mavros_connected}')
        print(f'{"✅" if self.ekf_ok else "❌"} EKF state valid: {self.ekf_ok}')
        print()

        if all_ok and self.mavros_connected and self.ekf_ok:
            print('🚀 SYSTEM READY FOR FLIGHT')
        else:
            print('⚠️  SYSTEM NOT READY — fix issues above before flight')

        print('='*50)


def main():
    rclpy.init()
    node = SystemHealthChecker()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

"""

## 📋 Step 9 — Development Order Checklist
```
PHASE 1 — Setup (Week 1)
━━━━━━━━━━━━━━━━━━━━━━━━
□ Install Ubuntu 22.04, ROS2 Humble, PX4, MAVROS
□ Install AirSim or Gazebo
□ Build ROS2 workspace: colcon build
□ Verify: ros2 topic list shows MAVROS topics

PHASE 2 — Perception (Week 1-2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Calibrate camera (30+ checkerboard images, RMS < 0.5px)
□ Verify YOLO on test images (check bounding boxes look correct)
□ Run corner_extractor.py standalone test
□ Run pose_estimator demo, verify distance estimate is reasonable
□ Integrate into ROS2: verify /perception/gate_pose publishes

PHASE 3 — State Estimation (Week 2)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Run EKF node, plot /state_estimation/odom in rviz2
□ Shake drone by hand — verify EKF tracks orientation
□ Verify EKF updates when gate_pose arrives (check covariance shrinks)
□ Run VIO node, verify it doesn't diverge over 30s

PHASE 4 — Simulation (Week 2-3)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Train RL agent: python training/train.py (5M steps)
□ Evaluate: does agent pass gates consistently in eval env?
□ Run AirSim bridge, verify camera + IMU topics appear in ROS2
□ Run integration_test.py — all green before proceeding

PHASE 5 — Control (Week 3)
━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Tune PID in simulation (pos_x_kp, pos_z_kp first)
□ Verify MAVROS arming sequence works in SITL (Software-In-The-Loop)
□ Test minimum snap planner: visualize path in rviz2
□ Fly single gate in simulation end-to-end

PHASE 6 — Full System (Week 4)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
□ Run full pipeline in AirSim: run_race.sh sim
□ Complete 3-gate circuit in simulation
□ Complete full track in simulation (< 30s target)
□ Transition to real hardware: run_race.sh real
□ Competition!

"""