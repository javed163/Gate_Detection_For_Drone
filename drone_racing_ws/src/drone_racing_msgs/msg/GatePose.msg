std_msgs/Header header
uint32 gate_id
geometry_msgs/PoseStamped pose    # 6DoF pose in camera frame
float32[9] covariance             # 3x3 position covariance
bool is_valid