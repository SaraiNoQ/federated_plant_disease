# src/routing_agent.py
import numpy as np
import random

class RoutingAgent:
    """
    服务器端的慢尺度RL Agent，负责知识路由。
    它为每个客户端选择一个最合适的导师模型。
    """
    def __init__(self, farm_ids: list, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0, exploration_decay=0.995):
        self.farm_ids = list(farm_ids)
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01
        
        # Q-table: Q(state, action)
        # State: 当前客户端的ID (我们用其索引作为状态)
        # Action: 从知识库中选择的导师模型的ID (我们用其索引作为动作)
        # Q-table 存储了在某个状态下，采取某个动作的预期未来奖励。
        self.num_states = len(farm_ids)
        self.num_actions = len(farm_ids) # 假设知识库和客户端列表是对应的
        self.q_table = np.zeros((self.num_states, self.num_actions))

        self.farm_to_idx = {fid: i for i, fid in enumerate(self.farm_ids)}

    def choose_action(self, current_farm_id: str):
        """
        根据epsilon-greedy策略为当前农场选择一个导师农场。
        """
        state_idx = self.farm_to_idx[current_farm_id]
        
        # 探索
        if random.uniform(0, 1) < self.epsilon:
            action_idx = random.randint(0, self.num_actions - 1)
        # 利用
        else:
            action_idx = np.argmax(self.q_table[state_idx])
            
        teacher_farm_id = self.farm_ids[action_idx]
        return teacher_farm_id

    def update(self, current_farm_id: str, teacher_farm_id: str, reward: float):
        """
        根据获得的奖励更新Q-table。
        """
        state_idx = self.farm_to_idx[current_farm_id]
        action_idx = self.farm_to_idx[teacher_farm_id]
        
        old_value = self.q_table[state_idx, action_idx]
        
        # Q-learning 公式: Q_new = Q_old + lr * (reward + gamma * max(Q_next) - Q_old)
        # 在我们的单步场景中，没有 "next_state"，所以公式简化
        new_value = old_value + self.lr * (reward - old_value)
        
        self.q_table[state_idx, action_idx] = new_value
        
        # 更新探索率
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def print_q_table(self):
        print("\n--- Q-Table (Routing Policy) ---")
        header = "      | " + " | ".join([f"T_{fid[-1]}" for fid in self.farm_ids])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(self.q_table):
            client_id = self.farm_ids[i]
            row_str = f"C_{client_id[-1]}   | " + " | ".join([f"{v:5.2f}" for v in row])
            print(row_str)