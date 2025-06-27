# src/rl_aggregator.py

import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np

# --- 1. Actor-Critic 模型 ---
# PPO Agent由两个网络组成：Actor（决定动作）和Critic（评估状态价值）
class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ActorCritic, self).__init__()

        # Actor Network: 状态 -> 动作概率分布的均值
        self.actor = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, action_dim) # 输出每个动作的logit
        )

        # Critic Network: 状态 -> 状态价值
        self.critic = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.Tanh(),
            nn.Linear(128, 64),
            nn.Tanh(),
            nn.Linear(64, 1)
        )
        # 动作的协方差矩阵对角线元素，可训练，确保为正
        self.action_log_std = nn.Parameter(torch.zeros(action_dim))

    def forward(self):
        raise NotImplementedError

    def act(self, state):
        # 从actor网络获取动作均值
        action_mean = self.actor(state)
        
        # 创建协方差矩阵
        cov_mat = torch.diag(torch.exp(self.action_log_std))
        
        # 创建多维正态分布
        dist = MultivariateNormal(action_mean, cov_mat)
        
        # 从分布中采样一个动作
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        # 使用softmax将动作转换为权重
        weights = F.softmax(action, dim=-1)
        
        return action.detach(), action_logprob.detach(), weights.detach()

    def evaluate(self, state, action):
        action_mean = self.actor(state)
        
        action_var = torch.exp(self.action_log_std * 2)
        cov_mat = torch.diag(action_var)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy

# --- 2. PPO Agent ---
class PPOAgent:
    def __init__(self, state_dim, action_dim, lr_actor, lr_critic, gamma, K_epochs, eps_clip, device):
        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        self.device = device
        
        # PPO需要一个策略网络和一个旧策略网络用于计算比率
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.policy.actor.parameters(), 'lr': lr_actor},
            {'params': self.policy.critic.parameters(), 'lr': lr_critic}
        ])

        self.policy_old = ActorCritic(state_dim, action_dim).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        
        # 用于存储一个回合（在我们的例子中是一次聚合）的数据
        self.buffer = []

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action, action_logprob, weights = self.policy_old.act(state)
        
        # 存储转换 (state, action, logprob)
        self.buffer.append((state, action, action_logprob))
        
        return weights.cpu().numpy().flatten()

    def update(self, reward):
        # 从buffer中获取数据，并添加奖励
        # PPO通常处理(s, a, logp, r, s')，但我们的s'是下一个FL round的状态，
        # 且奖励是即时的，所以简化处理
        states, actions, old_logprobs = zip(*self.buffer)
        states = torch.stack(states).to(self.device).detach()
        actions = torch.stack(actions).to(self.device).detach()
        old_logprobs = torch.stack(old_logprobs).to(self.device).detach()
        
        # 奖励转换为tensor
        rewards = torch.tensor([reward] * len(states), dtype=torch.float32).to(self.device).view(-1, 1)

        # 核心PPO更新循环
        for _ in range(self.K_epochs):
            # 评估旧动作
            logprobs, state_values, dist_entropy = self.policy.evaluate(states, actions)
            
            # 计算优势 A(s,a) = R - V(s)
            advantages = rewards - state_values.detach()
            
            # 计算比率 r = P_new(a|s) / P_old(a|s)
            ratios = torch.exp(logprobs - old_logprobs.detach())
            
            # 计算 Surrogate Loss (PPO-Clip 目标)
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages
            
            # 最终损失 = Actor损失 + Critic损失 - 熵奖励
            loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values, rewards) - 0.01 * dist_entropy
            
            # 梯度下降
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # 更新旧策略网络
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        # 清空buffer
        self.buffer = []

    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=self.device))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=self.device))