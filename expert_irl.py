from env import CameraControlEnv  # CameraControlEnv是你的自定义环境
import numpy as np
import random
import pickle

# 创建相机控制环境
env = CameraControlEnv()

max_episodes = 10

camera_pos = np.array([0, 0, 0])

# 存储专家数据的列表
expert_data = []

# 循环与代理交互并手动输入经验
for episode in range(max_episodes):
    state = env.reset()
    episode_data = []  # 存储当前 episode 的专家数据
    done = False

    while not done:
        azimuth, elevation, focal_length, gud_a, gud_e, x, y, z, current_t = state
        # 计算目标相对于相机的位置矢量
        target_vector = np.array([x, y, z]) - camera_pos

        # 计算俯角和仰角
        distance = np.linalg.norm(target_vector)
        elevation_rad = np.arcsin(target_vector[2] / distance)
        azimuth_rad = np.arctan2(target_vector[1], target_vector[0])

        elevation_deg = np.rad2deg(elevation_rad)
        elevation_deg = round(elevation_deg, 2)

        azimuth_deg = np.rad2deg(azimuth_rad)
        azimuth_deg = round(azimuth_deg, 2)

        a1 = 1 * random.uniform(0, 1)
        e1 = 1 * random.uniform(0, 1)

        azimuth1 = azimuth_deg - azimuth
        elevation1 = elevation_deg - elevation

        act1 = azimuth1 + a1
        act2 = elevation1 + e1

        if current_t <= 3600:
            focal = 0.006
        else:
            focal = -0.006       
        action = np.array([act1, act2, focal])

        # 保存专家数据到当前 episode 的数据列表
        episode_data.append({'state': state, 'action': action})

        next_state, reward, done, _ = env.step(action)

        # 更新当前状态
        state = next_state
    
    # 将当前 episode 的专家数据添加到专家数据列表
    expert_data.extend(episode_data)

    print(f"Episode: {episode + 1}")

# 保存专家数据到文件
with open('expert_data.pkl', 'wb') as f:
    pickle.dump(expert_data, f)

# 关闭环境
env.close()