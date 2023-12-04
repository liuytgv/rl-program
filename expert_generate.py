from SAC import SAC  # SAC类是你提供的SAC算法实现
from env import CameraControlEnv  # CameraControlEnv是你的自定义环境
import numpy as np
import random
import os

# 创建相机控制环境
env = CameraControlEnv()

# SAC算法的参数
state_dim = env.observation_space.shape[0]  # 观察空间为5维
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low  # 添加min_action参数

sac_args = {
    'gamma': 0.99,  # 折扣率，控制未来奖励的重要性0.99
    'tau': 0.005,  # 软更新参数，控制目标网络的平滑更新
    'alpha': 0.4,  # 熵调节参数，控制策略的探索程度0.2
    'policy': 'Deterministic',  # 策略类型，可以是Gaussian或Deterministic策略
    'target_update_interval': 5,  # 目标网络更新的频率
    'automatic_entropy_tuning': True,  # 是否自动调节熵参数
    'cuda': True,
    'hidden_size': 1024,  # Critic和Actor网络的隐藏层大小
    'lr': 0.001,  # 学习率0.0001
    'buffer_capacity': 10000000,  # 添加buffer_capacity属性
    'seed': 256  # 添加seed属性
}

# 在这里初始化SAC代理
agent = SAC(state_dim, env.action_space, sac_args)

buffer_save_path = 'checkpoints/sac_buffer_camera_control_.pkl'

# 尝试加载 Replay Buffer
if os.path.exists(buffer_save_path):
    print(f"Loading Replay Buffer from {buffer_save_path}...")
    agent.buffer.load_buffer(buffer_save_path)
else:
    print("No saved Replay Buffer found. Starting with an empty buffer.")

camera_pos = np.array([0, 0, 0])
max_episodes = 100

for episode in range(max_episodes):
    # 循环与代理交互并手动输入经验
    state = env.reset()
    total_reward = 0
    done = False

    while not done:
        azimuth, elevation, focal_length, gud_a, gud_e, x, y, z, current_t = state
        observation = state[:5]
        #print(observation)

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

        a1 = 0.1 * random.uniform(0, 1)
        e1 = 0.1 * random.uniform(0, 1)

        azimuth1 = azimuth_deg - azimuth
        elevation1 = elevation_deg - elevation

        act1 = azimuth1 + a1
        if act1 >= 6:
            act1 = 6
        else:
            if act1 <= -6:
                act1 = -6
        act2 = elevation1 + e1
        if act2 >= 6:
            act2 = 6
        else:
            if act2 <= -6:
                act2 = -6

        if current_t <= 801:
            focal = 0.006
        else:
            focal = -0.006       
        action = np.array([act1, act2, focal])
        #print(action)

        next_state, reward, done, _ = env.step(action)
        # 打印获得的奖励
        # print(f"Reward: {reward}")
        next_observation = next_state[:5]
        agent.buffer.push(observation, action, reward, next_observation, done)
        # print(f"{len(agent.buffer)}")
        # 保存 Replay Buffer
        state = next_state
        total_reward += reward
    
    print(f"Total Reward: {total_reward}")

# 关闭环境
env.close()

agent.buffer.save_buffer('camera_control')