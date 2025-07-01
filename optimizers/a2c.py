# optimizers/a2c.py
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal
import torch.nn.functional as F
import numpy as np

# A2C可以复用和PPO完全相同的ActorCritic网络结构
from .ppo import ActorCritic 

class A2COptimizer:
    def __init__(self, state_dim, action_dim, config, device):
        self.gamma = config.A2C_GAMMA
        self.device = device
        
        self.policy = ActorCritic(state_dim, action_dim).to(device)
        self.optimizer = torch.optim.Adam(self.policy.parameters(), lr=config.A2C_LR)
        self.MseLoss = nn.MSELoss()

    def select_action(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(self.device)
            action_mean = self.policy.actor(state)
            weights = F.softmax(action_mean, dim=-1) # A2C可以直接用确定性动作或随机动作，这里简化
        return weights.cpu().numpy().flatten()

    def update(self, state, reward, done, next_state):
        state = torch.FloatTensor(state).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        
        # A2C 核心更新逻辑
        # 评估当前状态和动作
        action_logprobs, state_values, dist_entropy = self.policy.evaluate(state, self.policy.actor(state))
        
        # 计算优势
        # 由于我们是单步决策，next_state_value 可以简化
        advantage = reward - state_values

        # 计算 Actor 和 Critic 损失
        actor_loss = -(action_logprobs * advantage.detach()).mean()
        critic_loss = self.MseLoss(state_values, reward.unsqueeze(0))
        
        # 总损失
        loss = actor_loss + 0.5 * critic_loss - 0.01 * dist_entropy.mean()
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, evaluate_weights_fn, state_vector, epochs, **kwargs):
        best_acc = 0.0
        best_weights = None
        for epoch in range(epochs):
            weights = self.select_action(state_vector)
            current_acc, reward = evaluate_weights_fn(weights)
            
            # 在我们的单步场景中，done=True, next_state=state
            self.update(state_vector, reward, True, state_vector)

            print(f"Epoch {epoch+1}/{epochs} [A2C] | Weights: {np.round(weights,2)} -> Acc: {current_acc:.2f}% | Reward: {reward:.4f}")
            if current_acc > best_acc:
                best_acc = current_acc
                best_weights = weights
        return best_acc, best_weights