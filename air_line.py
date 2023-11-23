import numpy as np

def generate(start_point, target_point, total_time, max_speed, camera_pos):
    # 相机的位置
    camera_x, camera_y, camera_z = camera_pos
    camera_z1 =camera_z + 100

    start_x, start_y, start_z = start_point[0], start_point[1], start_point[2]
    target_x, target_y, target_z = target_point[0], target_point[1], target_point[2]

    # 计算飞行方向和距离
    direction1 = np.array([camera_x - start_x, camera_y - start_y, camera_z1 - start_z], dtype=float)
    direction2 = np.array([target_x - camera_x, target_y - camera_y, target_z - camera_z1], dtype=float)
    distance1 = np.linalg.norm(direction1, ord=2).astype(float)
    distance2 = np.linalg.norm(direction2, ord=2).astype(float)

    # 计算速度和加速度
    speed1 = max_speed  # 前半段时间
    speed2 = max_speed  # 后半段时间

    # 归一化方向向量
    direction1 /= distance1
    direction2 /= distance2

    # 生成时间序列
    time = np.linspace(0, total_time, num=int(total_time * 10))
    # 使用 round 函数将时间点数组中的每个元素保留一位小数
    time = np.round(time, 1)

    # 打印起始点和目标点
    # print(f"Start Point: {start_point}")
    # print(f"Target Point: {target_point}")
    # print(f"{distance1, distance2}")

    # 计算每个时间点的位置
    positions = []
    for t in time:
        if t <= total_time / 2:
            # 飞机向目标点飞行（前半段时间）
            displacement = speed1 * t
            displacement = min(displacement, distance1)  # 限制位移不超过总距离
            displacement *= direction1

            # 计算当前位置
            x = start_x + displacement[0]
            x = round(x, 2)

            y = start_y + displacement[1]
            y = round(y, 2)
            z = start_z + displacement[2]
            z = round(z, 2)

            positions.append((x, y, z))
        else:
            # 计算当前时间在后半段的百分比
            t_back = t - total_time / 2

            # 飞机向目标点飞行（后半段时间）
            displacement = speed2 * t_back
            displacement = min(displacement, distance2)  # 限制位移不超过总距离
            displacement *= direction2

            # 计算当前位置
            x = camera_x + displacement[0]
            x = round(x, 2)

            y = camera_y + displacement[1]
            y = round(y, 2)

            z = camera_z1 + displacement[2]
            z = round(z, 2)

            positions.append((x, y, z))

        # 打印当前坐标
        # print(f"Time {t}: Position ({x}, {y}, {z})")

    return positions

# 设置参数
#start_point = np.array([900, 1200, 200])
#target_point = np.array([-900, -1200, 200])
#camera_pos = np.array([0, 0, 0])
#total_time = 120
#max_speed = 25
#trajectory, target_point = generate_trajectory(start_point, target_point, total_time, max_speed, camera_pos)
#x = np.array([point[0] for point in trajectory])
#y = np.array([point[1] for point in trajectory])
#z = np.array([point[2] for point in trajectory])
#max_frames = len(x)
#print(f'{trajectory}')
#print(f'{target_point, max_frames}')