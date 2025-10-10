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
    支持个性化状态和动作掩码。
    """
    def __init__(self, num_farms, state_dim, lr_actor=0.001, lr_critic=0.001, gamma=0.99, entropy_coeff=0.01, device='cpu'):
        self.num_farms = num_farms
        # 动作空间：0=EXPLOIT, 1..N = TRANSFER_IN from farm 0..N-1
        self.action_dim = num_farms + 1 
        self.gamma = gamma
        self.entropy_coeff = entropy_coeff
        self.device = device
        
        self.actor = Actor(state_dim, self.action_dim).to(device)
        self.critic = Critic(state_dim).to(device)
        self.optimizer_actor = torch.optim.Adam(self.actor.parameters(), lr=lr_actor)
        self.optimizer_critic = torch.optim.Adam(self.critic.parameters(), lr=lr_critic)
        
        self.mse_loss = nn.MSELoss()

        # 使用一个缓冲区来存储一个完整轮次的数据
        self.buffer = []

    def select_action(self, state, farm_idx_to_decide, available_teachers_indices):
        """
        为单个农场选择一个指令，并应用动作掩码。
        """
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        
        with torch.no_grad():
            action_probs = self.actor(state)

        # --- 动作掩码 ---
        mask = torch.ones_like(action_probs)
        # 1. 不能向自己学习
        mask[farm_idx_to_decide + 1] = 0
        # 2. 不能向没有模型的老师学习
        all_teacher_indices = set(range(self.num_farms))
        unavailable_teachers = all_teacher_indices - set(available_teachers_indices)
        for unavailable_idx in unavailable_teachers:
            mask[unavailable_idx + 1] = 0
        
        # 应用掩码
        masked_action_probs = action_probs * mask
        # 重新归一化概率
        if torch.sum(masked_action_probs) > 0:
            masked_action_probs /= torch.sum(masked_action_probs)
        else:
            # 如果所有TRANSFER动作都被屏蔽，则只能EXPLOIT
            masked_action_probs = torch.zeros_like(action_probs)
            masked_action_probs[0] = 1.0

        dist = Categorical(masked_action_probs)
        action = dist.sample()
        
        return action.item(), dist.log_prob(action)

    def store_transition(self, state, action_log_prob):
        """将一次决策的数据存入缓冲区。"""
        self.buffer.append((state, action_log_prob))

    def update(self, rewards: list):
        if not self.buffer:
            return
        
        # 个性化奖励已经传入，直接使用
        returns = torch.tensor(rewards, dtype=torch.float32).to(self.device)
        if len(returns) > 1:
            returns = (returns - returns.mean()) / (returns.std() + 1e-5)

        states = torch.tensor([item[0] for item in self.buffer], dtype=torch.float32).to(self.device)
        old_log_probs = torch.stack([item[1] for item in self.buffer]).to(self.device)

        state_values = self.critic(states).squeeze()
        advantage = returns - state_values.detach()
        
        # 计算熵
        dist_entropy = Categorical(self.actor(states)).entropy().mean()

        actor_loss = -(old_log_probs * advantage).mean()
        critic_loss = self.mse_loss(state_values, returns)

        loss = actor_loss + 0.5 * critic_loss - self.entropy_coeff * dist_entropy

        self.optimizer_actor.zero_grad()
        self.optimizer_critic.zero_grad()
        loss.backward()
        self.optimizer_actor.step()
        self.optimizer_critic.step()
        
        self.buffer = []