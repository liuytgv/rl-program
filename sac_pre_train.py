import torch
import os
import numpy as np
from SAC import SAC  # SAC类是你提供的SAC算法实现
from env import CameraControlEnv  # CameraControlEnv是你的自定义环境

# 检查是否可用GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建相机控制环境
env = CameraControlEnv()

# SAC算法的参数
state_dim = env.observation_space.shape[0]  # 观察空间为5维
action_dim = env.action_space.shape[0]
max_action = env.action_space.high
min_action = env.action_space.low  # 添加min_action参数

sac_args = {
    'gamma': 0.999,  # 折扣率，控制未来奖励的重要性0.99
    'tau': 0.005,  # 软更新参数，控制目标网络的平滑更新
    'alpha': 0.4,  # 熵调节参数，控制策略的探索程度0.2
    'policy': 'Gaussian',  # 策略类型，可以是Gaussian或Deterministic策略
    'target_update_interval': 1,  # 目标网络更新的频率
    'automatic_entropy_tuning': True,  # 是否自动调节熵参数
    'cuda': True,
    'hidden_size': 1024,  # Critic和Actor网络的隐藏层大小
    'lr': 0.001,  # 学习率0.0001
    'buffer_capacity': 10000000,  # 添加buffer_capacity属性
    'seed': 256  # 添加seed属性
}

# 在这里初始化SAC代理
agent = SAC(state_dim, env.action_space, sac_args)

# 检查是否有预训练的模型参数
#pretrained_model_directory = 'sac_model'
#pretrained_model_episode = 1000  # 将这个 episode 修改为你想加载的模型参数的 episode

#if os.path.exists(pretrained_model_directory):
    #print(f"Loading pretrained model from episode {pretrained_model_episode}...")
    #agent.load_model(pretrained_model_directory, 'sac', pretrained_model_episode)
#else:
    #print("No pretrained model found. Starting from scratch.")

# 保存 Replay Buffer 的位置
buffer_save_path = 'checkpoints/sac_buffer_camera_control_.pkl'

# 尝试加载 Replay Buffer
if os.path.exists(buffer_save_path):
    print(f"Loading Replay Buffer from {buffer_save_path}...")
    agent.buffer.load_buffer(buffer_save_path)
else:
    print("No saved Replay Buffer found. Starting with an empty buffer.")

print(f"Buffer size: {len(agent.buffer)}")

# 存储每轮训练的奖励
episode_rewards = []

# 训练主循环
max_episodes = 200 # 最大训练轮数

updates = 0

for episode in range(max_episodes):
    # SAC算法中的训练步骤
    qf1_loss, qf2_loss, policy_loss, alpha_loss, alpha = agent.update_parameters(agent.buffer, 512, updates)
    updates += 1  # 每次更新后递增更新次数

print(f"{updates}")

# 保存训练后的模型参数
save_dir = 'sac_model'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

agent.save_model(save_dir, 'sac', max_episodes)