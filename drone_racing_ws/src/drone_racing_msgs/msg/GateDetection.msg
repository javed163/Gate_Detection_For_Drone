std_msgs/Header header
uint32 gate_id
float32[4] bbox          # [x_center, y_center, width, height] normalized
float32 confidence
float32[8] corners_2d    # 4 corner points [u1,v1, u2,v2, u3,v3, u4,v4]