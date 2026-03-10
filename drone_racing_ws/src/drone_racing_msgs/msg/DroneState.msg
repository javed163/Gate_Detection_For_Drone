std_msgs/Header header
geometry_msgs/Point position        # x, y, z in world frame
geometry_msgs/Vector3 velocity      # vx, vy, vz
geometry_msgs/Quaternion orientation
geometry_msgs/Vector3 angular_velocity
float32[36] covariance              # Full 6x6 state covariance