import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
import pickle
from env import CameraControlEnv

# 加载专家数据
def load_expert_data(file_path):
    with open(file_path, 'rb') as f:
        expert_data = pickle.load(f)
    return expert_data


class PolicyNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, action_dim)

    def forward(self, state):
        x = F.relu(self.fc1(state.view(1, -1)))  # Reshape state to (1, -1) to match weight matrix dimensions
        logits = self.fc2(x)
        return F.softmax(logits, dim=1) ###

class DiscriminatorNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_size):
        super(DiscriminatorNetwork, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        return torch.sigmoid(self.fc2(x))

def airl(env, expert_data_path, num_episodes=1000, learning_rate=1e-3, hidden_size=32):
    state_dim = env.state_space.shape[0]
    action_dim = env.action_space.shape[0]
    
    policy_network = PolicyNetwork(state_dim, action_dim, hidden_size)
    discriminator_network = DiscriminatorNetwork(state_dim + action_dim, 1, hidden_size)
    
    optimizer_policy = optim.Adam(policy_network.parameters(), lr=learning_rate)
    optimizer_discriminator = optim.Adam(discriminator_network.parameters(), lr=learning_rate)

    # 加载专家数据
    expert_dataset = load_expert_data(expert_data_path)
    expert_states, expert_actions = generate_expert_data(expert_dataset)

    for episode in range(num_episodes):
        states = []
        actions = []

        state = env.reset()
        done = False

        while not done:
            action_probs = policy_network(torch.tensor(state, dtype=torch.float32))###
            action_dist = Categorical(action_probs)###
            action = action_dist.sample().numpy()###

            next_state, _, done, _ = env.step(action)

            states.append(state)
            actions.append(action)

            state = next_state

        # 更新策略###
        optimizer_policy.zero_grad()
        action_probs = policy_network(torch.tensor(states, dtype=torch.float32))
        action_dist = Categorical(action_probs)
        loss_policy = -torch.mean(action_dist.log_prob(torch.tensor(actions)) * torch.ones(len(states)))
        loss_policy.backward()
        optimizer_policy.step()

        # 更新鉴别器
        optimizer_discriminator.zero_grad()
        logits_expert = discriminator_network(torch.tensor(np.concatenate([expert_states, expert_actions], axis=1), dtype=torch.float32))
        logits_policy = discriminator_network(torch.tensor(np.concatenate([states, actions], axis=1), dtype=torch.float32))
        loss_discriminator = -torch.mean(torch.log(logits_expert) + torch.log(1 - logits_policy))
        loss_discriminator.backward()
        optimizer_discriminator.step()

        if episode % 10 == 0:
            print(f"Episode {episode}, Loss Policy: {loss_policy.item()}, Loss Discriminator: {loss_discriminator.item()}")

    # 返回训练好的奖励函数逼近网络的权重
    return discriminator_network.state_dict()

def generate_expert_data(expert_dataset):
    states = np.array([sample['state'] for sample in expert_dataset])
    actions = np.array([sample['action'] for sample in expert_dataset])
    return states, actions


# 专家数据文件路径
expert_data_path = 'expert_data.pkl'
env = CameraControlEnv()

# 训练 AIRL 算法并获取训练好的奖励函数逼近网络的权重
trained_weights = airl(env, expert_data_path)
