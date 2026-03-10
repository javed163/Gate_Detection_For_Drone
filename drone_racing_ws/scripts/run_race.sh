#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_race.sh — Launch full autonomous drone racing system
# ─────────────────────────────────────────────────────────────────────────────
set -e

WORKSPACE="$HOME/drone_racing_ws"
source "$WORKSPACE/install/setup.bash"
source /opt/ros/humble/setup.bash

MODE="${1:-sim}"   # sim | real

echo "=============================================="
echo "  Autonomous Drone Racing System"
echo "  Mode: $MODE"
echo "=============================================="

if [ "$MODE" = "sim" ]; then
    echo "[1/4] Launching AirSim bridge..."
    ros2 launch simulation airsim_sim.launch.py &
    sleep 3

elif [ "$MODE" = "real" ]; then
    echo "[1/4] Launching MAVROS for real drone..."
    ros2 launch mavros px4.launch fcu_url:=/dev/ttyUSB0:921600 &
    sleep 2
fi

echo "[2/4] Launching perception pipeline..."
ros2 launch perception perception.launch.py &
sleep 2

echo "[3/4] Launching state estimation (EKF + VIO)..."
ros2 launch state_estimation state_estimation.launch.py &
sleep 2

echo "[4/4] Launching planning + control..."
ros2 launch control full_system.launch.py &

echo ""
echo "✅ All nodes launched. Monitor with:"
echo "   ros2 topic list"
echo "   rqt_graph"
echo "   rviz2"
echo ""
echo "Kill all: Ctrl+C, then: pkill -f ros2"

wait