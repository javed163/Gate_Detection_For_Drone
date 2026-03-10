#!/bin/bash

# Root folder
mkdir -p drone_racing_ws/src

# -----------------------------
# Drone Racing Meta Package
# -----------------------------
mkdir -p drone_racing_ws/src/drone_racing
touch drone_racing_ws/src/drone_racing/package.xml

# -----------------------------
# Perception Package
# -----------------------------
mkdir -p drone_racing_ws/src/perception/perception
mkdir -p drone_racing_ws/src/perception/config
mkdir -p drone_racing_ws/src/perception/models
mkdir -p drone_racing_ws/src/perception/launch

touch drone_racing_ws/src/perception/perception/__init__.py
touch drone_racing_ws/src/perception/perception/gate_detector.py
touch drone_racing_ws/src/perception/perception/corner_extractor.py
touch drone_racing_ws/src/perception/perception/pose_estimator.py
touch drone_racing_ws/src/perception/perception/camera_calibration.py

touch drone_racing_ws/src/perception/config/camera_params.yaml
touch drone_racing_ws/src/perception/config/gate_geometry.yaml

touch drone_racing_ws/src/perception/models/gate_detector.pt

touch drone_racing_ws/src/perception/launch/perception.launch.py
touch drone_racing_ws/src/perception/package.xml
touch drone_racing_ws/src/perception/setup.py

# -----------------------------
# State Estimation Package
# -----------------------------
mkdir -p drone_racing_ws/src/state_estimation/state_estimation
mkdir -p drone_racing_ws/src/state_estimation/config
mkdir -p drone_racing_ws/src/state_estimation/launch

touch drone_racing_ws/src/state_estimation/state_estimation/__init__.py
touch drone_racing_ws/src/state_estimation/state_estimation/ekf_node.py
touch drone_racing_ws/src/state_estimation/state_estimation/imu_preintegration.py
touch drone_racing_ws/src/state_estimation/state_estimation/vio_node.py

touch drone_racing_ws/src/state_estimation/config/ekf_params.yaml

touch drone_racing_ws/src/state_estimation/launch/state_estimation.launch.py
touch drone_racing_ws/src/state_estimation/package.xml
touch drone_racing_ws/src/state_estimation/setup.py

# -----------------------------
# Planning Package
# -----------------------------
mkdir -p drone_racing_ws/src/planning/planning
mkdir -p drone_racing_ws/src/planning/config
mkdir -p drone_racing_ws/src/planning/launch

touch drone_racing_ws/src/planning/planning/__init__.py
touch drone_racing_ws/src/planning/planning/minimum_snap.py
touch drone_racing_ws/src/planning/planning/mpc_planner.py
touch drone_racing_ws/src/planning/planning/gate_sequencer.py
touch drone_racing_ws/src/planning/planning/trajectory_server.py

touch drone_racing_ws/src/planning/config/planner_params.yaml

touch drone_racing_ws/src/planning/launch/planning.launch.py
touch drone_racing_ws/src/planning/package.xml
touch drone_racing_ws/src/planning/setup.py

# -----------------------------
# Control Package
# -----------------------------
mkdir -p drone_racing_ws/src/control/control
mkdir -p drone_racing_ws/src/control/config
mkdir -p drone_racing_ws/src/control/models
mkdir -p drone_racing_ws/src/control/launch

touch drone_racing_ws/src/control/control/__init__.py
touch drone_racing_ws/src/control/control/rl_controller.py
touch drone_racing_ws/src/control/control/pid_controller.py
touch drone_racing_ws/src/control/control/mavros_bridge.py
touch drone_racing_ws/src/control/control/mixer.py

touch drone_racing_ws/src/control/config/pid_params.yaml
touch drone_racing_ws/src/control/config/rl_params.yaml

touch drone_racing_ws/src/control/models/rl_policy.zip

touch drone_racing_ws/src/control/launch/control.launch.py
touch drone_racing_ws/src/control/package.xml
touch drone_racing_ws/src/control/setup.py

# -----------------------------
# Simulation Package
# -----------------------------
mkdir -p drone_racing_ws/src/simulation/simulation
mkdir -p drone_racing_ws/src/simulation/worlds
mkdir -p drone_racing_ws/src/simulation/launch

touch drone_racing_ws/src/simulation/simulation/__init__.py
touch drone_racing_ws/src/simulation/simulation/airsim_bridge.py
touch drone_racing_ws/src/simulation/simulation/gazebo_bridge.py
touch drone_racing_ws/src/simulation/simulation/race_environment.py

touch drone_racing_ws/src/simulation/worlds/race_track.world

touch drone_racing_ws/src/simulation/launch/airsim_sim.launch.py
touch drone_racing_ws/src/simulation/launch/gazebo_sim.launch.py

touch drone_racing_ws/src/simulation/package.xml
touch drone_racing_ws/src/simulation/setup.py

# -----------------------------
# Drone Racing Messages Package
# -----------------------------
mkdir -p drone_racing_ws/src/drone_racing_msgs/msg
mkdir -p drone_racing_ws/src/drone_racing_msgs/srv
mkdir -p drone_racing_ws/src/drone_racing_msgs/action

touch drone_racing_ws/src/drone_racing_msgs/msg/GateDetection.msg
touch drone_racing_ws/src/drone_racing_msgs/msg/GatePose.msg
touch drone_racing_ws/src/drone_racing_msgs/msg/DroneState.msg
touch drone_racing_ws/src/drone_racing_msgs/msg/RaceTrajectory.msg

touch drone_racing_ws/src/drone_racing_msgs/srv/PlanTrajectory.srv
touch drone_racing_ws/src/drone_racing_msgs/action/ExecuteRace.action

touch drone_racing_ws/src/drone_racing_msgs/package.xml
touch drone_racing_ws/src/drone_racing_msgs/CMakeLists.txt

# -----------------------------
# Scripts
# -----------------------------
mkdir -p drone_racing_ws/scripts

touch drone_racing_ws/scripts/setup_environment.sh
touch drone_racing_ws/scripts/calibrate_camera.sh
touch drone_racing_ws/scripts/train_rl_agent.py
touch drone_racing_ws/scripts/run_race.sh

# -----------------------------
# Training
# -----------------------------
mkdir -p drone_racing_ws/training/envs
mkdir -p drone_racing_ws/training/configs

touch drone_racing_ws/training/envs/drone_racing_env.py
touch drone_racing_ws/training/envs/reward_functions.py

touch drone_racing_ws/training/configs/ppo_config.yaml

touch drone_racing_ws/training/train.py
touch drone_racing_ws/training/evaluate.py

# -----------------------------
# Calibration
# -----------------------------
mkdir -p drone_racing_ws/calibration/checkerboard_images

touch drone_racing_ws/calibration/calibration_results.yaml
touch drone_racing_ws/calibration/run_calibration.py

# -----------------------------
# Docs
# -----------------------------
mkdir -p drone_racing_ws/docs

touch drone_racing_ws/docs/architecture.md
touch drone_racing_ws/docs/ros2_topics.md
touch drone_racing_ws/docs/tuning_guide.md

# -----------------------------
# Docker
# -----------------------------
mkdir -p drone_racing_ws/docker

touch drone_racing_ws/docker/Dockerfile
touch drone_racing_ws/docker/docker-compose.yml

echo "Drone Racing Workspace Structure Created Successfully!"