# src/rl_agents.py

import numpy as np
import random
import torch
import torch.nn as nn
from torch.distributions import Categorical


# --- 低层RL Agent: UCB/Thompson 用于客户端选择 ---
class LocalExecutorUCB:
    """
    农场内部的低层RL Agent，使用UCB1算法选择客户端。
    """

    def __init__(self, num_clients, exploration_factor):
        self.num_clients = num_clients
        self.exploration_factor = exploration_factor
        self.counts = np.zeros(num_clients)
        self.values = np.zeros(num_clients)
        self.total_rounds = 0

    def select_clients(self, num_to_select):
        """选择一小组客户端参与训练。"""
        self.total_rounds += 1
        # 对从未被选择过的客户端给予最高优先级
        unexplored = np.where(self.counts == 0)[0]
        if len(unexplored) > 0:
            return np.random.choice(unexplored, size=min(num_to_select, len(unexplored)), replace=False)

        # UCB1分数计算
        ucb_scores = self.values + self.exploration_factor * np.sqrt(
            np.log(self.total_rounds) / self.counts
        )
        # 选择分数最高的客户端
        selected_indices = np.argsort(ucb_scores)[-num_to_select:]
        return selected_indices

    def update(self, client_indices, rewards):
        """根据客户端的奖励更新其统计值。"""
        for idx, reward in zip(client_indices, rewards):
            self.counts[idx] += 1
            # 增量式更新平均值
            self.values[idx] = ((self.counts[idx] - 1) * self.values[idx] + reward) / self.counts[idx]


# --- 高层RL Agent: Actor-Critic (A2C) ---
class Actor(nn.Module):
    """策略网络"""

    def __init__(self, state_dim, action_dim, hidden_dim=64):
        super(Actor, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.actor = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        # 输出每个动作的logits
        action_logits = self.actor(x)
        return torch.softmax(action_logits, dim=-1)


class Critic(nn.Module):
    """值函数网络"""

    def __init__(self, state_dim, hidden_dim=64):
        super(Critic, self).__init__()
        self.layer1 = nn.Linear(state_dim, hidden_dim)
        self.layer2 = nn.Linear(hidden_dim, hidden_dim)
        self.critic = nn.Linear(hidden_dim, 1)

    def forward(self, state):
        x = torch.relu(self.layer1(state))
        x = torch.relu(self.layer2(x))
        state_value = self.critic(x)
        return state_value


class MetaControllerA2C:
    """
    高层RL Agent (A2C)，负责为每个农场发布策略指令。
    这是一个简化的实现，用于演示。
    """

    def __init__(self, num_farms, state_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, device='cpu'):
        self.num_farms = num_farms
        # 动作空间：0=EXPLOIT, 1-N = TRANSFER_IN from farm 1-N
        self.action_dim = num_farms + 1
        self.gamma = gamma
        self.device = device

        self.actor = Actor(state_dim, self.action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)

        self.log_probs = []
        self.state_values = []
        self.rewards = []

    def select_action(self, state):
        """为单个农场选择一个指令。"""
        state = torch.FloatTensor(state).to(self.device)
        action_probs = self.actor(state)
        dist = Categorical(action_probs)
        action = dist.sample()

        self.log_probs.append(dist.log_prob(action))
        self.state_values.append(self.critic(state))

        return action.item()

    def update(self):
        """在一轮结束后，用收集到的数据更新网络。"""
        if not self.rewards:
            return

        # 计算累积折扣奖励
        returns = []
        discounted_reward = 0
        for reward in reversed(self.rewards):
            discounted_reward = reward + self.gamma * discounted_reward
            returns.insert(0, discounted_reward)

        returns = torch.tensor(returns).to(self.device)
        # 对returns进行标准化
        returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        log_probs = torch.stack(self.log_probs)
        state_values = torch.stack(self.state_values).squeeze()

        # 计算优势
        advantage = returns - state_values.detach()

        # 计算损失
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = nn.MSELoss()(state_values, returns)

        # 更新网络
        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        actor_loss.backward()
        critic_loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()

        # 清空缓冲区
        self.log_probs, self.state_values, self.rewards = [], [], []