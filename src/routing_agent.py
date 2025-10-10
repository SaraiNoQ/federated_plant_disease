# src/routing_agent.py
import numpy as np
import random

class RoutingAgent:
    """
    服务器端的慢尺度RL Agent，负责知识路由。
    支持动态添加新农场。
    """

    def __init__(self, initial_farm_ids: list, learning_rate=0.1, discount_factor=0.9, exploration_rate=1.0,
                 exploration_decay=0.995):
        self.farm_ids = list(initial_farm_ids)  # 初始农场列表
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = exploration_rate
        self.epsilon_decay = exploration_decay
        self.epsilon_min = 0.01

        self.farm_to_idx = {fid: i for i, fid in enumerate(self.farm_ids)}

        # Q-table: Q(state, action)
        # State: 学生农场的索引
        # Action: 教师农场的索引
        num_states = len(self.farm_ids)
        num_actions = len(self.farm_ids)
        self.q_table = np.zeros((num_states, num_actions))

    def choose_action(self, current_farm_id: str):
        """
        根据epsilon-greedy策略为当前农场选择一个导师农场。
        """
        if current_farm_id not in self.farm_to_idx:
            print(f"警告: 农场 {current_farm_id} 不在路由代理的已知列表中。将随机选择一个导师。")
            # 为一个完全未知的农场随机选择一个动作
            action_idx = random.randint(0, self.q_table.shape[1] - 1)
        else:
            state_idx = self.farm_to_idx[current_farm_id]
            # 探索
            if random.uniform(0, 1) < self.epsilon:
                action_idx = random.randint(0, self.q_table.shape[1] - 1)
            # 利用
            else:
                action_idx = np.argmax(self.q_table[state_idx])

        # 动作索引对应的是教师农场
        teacher_farm_id = self.farm_ids[action_idx]
        return teacher_farm_id

    def update(self, current_farm_id: str, teacher_farm_id: str, reward: float):
        """
        根据获得的奖励更新Q-table。
        """
        if current_farm_id not in self.farm_to_idx or teacher_farm_id not in self.farm_to_idx:
            # 如果有新农场加入，可能出现这种情况，此时我们忽略更新，等待下一次add_farms
            return

        state_idx = self.farm_to_idx[current_farm_id]
        action_idx = self.farm_to_idx[teacher_farm_id]

        old_value = self.q_table[state_idx, action_idx]
        new_value = old_value + self.lr * (reward - old_value)
        self.q_table[state_idx, action_idx] = new_value

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def add_farms(self, new_farm_ids: list):
        """
        动态地将新农场添加到Agent中，并扩展Q-table。
        """
        if not new_farm_ids:
            return

        print(f"--- RoutingAgent: 正在添加新农场: {new_farm_ids} ---")
        num_old_farms = len(self.farm_ids)
        num_new_farms = len(new_farm_ids)

        # 1. 更新内部ID列表和映射
        self.farm_ids.extend(new_farm_ids)
        self.farm_to_idx = {fid: i for i, fid in enumerate(self.farm_ids)}

        # 2. 扩展Q-table
        # Q-table的维度是 (num_students, num_teachers)
        # 教师的数量就是旧农场的数量，因为新农场还没有模型在知识库里
        # 学生的数量是新旧农场的总和
        new_q_table = np.zeros((num_old_farms + num_new_farms, num_old_farms))

        # 复制旧的Q-table部分
        new_q_table[:num_old_farms, :num_old_farms] = self.q_table

        # 为新农场（作为学生）初始化Q值
        # 可以用0初始化，也可以用现有Q值的平均值来加速学习
        avg_q_values = np.mean(self.q_table, axis=0)
        for i in range(num_new_farms):
            new_q_table[num_old_farms + i, :] = avg_q_values

        self.q_table = new_q_table
        print("Q-table已扩展以容纳新农场。新维度:", self.q_table.shape)

    def update_action_space(self):
        """
        当知识库更新后（例如，新农场贡献了模型），更新Q-table的动作空间。
        """
        num_students, num_old_teachers = self.q_table.shape
        num_total_farms = len(self.farm_ids)

        if num_total_farms > num_old_teachers:
            num_new_teachers = num_total_farms - num_old_teachers
            # 创建一个更大的Q-table
            new_q_table = np.zeros((num_students, num_total_farms))
            # 复制旧的部分
            new_q_table[:, :num_old_teachers] = self.q_table
            # 为新的教师动作初始化Q值（可以取该学生所有旧动作的平均Q值）
            for i in range(num_students):
                avg_q_for_student = np.mean(self.q_table[i, :])
                new_q_table[i, num_old_teachers:] = avg_q_for_student

            self.q_table = new_q_table
            print(f"--- RoutingAgent: 动作空间已扩展。新维度: {self.q_table.shape} ---")

    def print_q_table(self):
        print("\n--- Q-Table (Routing Policy) ---")
        # 教师（列标题）应该是所有在知识库中有模型的农场
        teacher_ids = self.farm_ids[:self.q_table.shape[1]]
        student_ids = self.farm_ids[:self.q_table.shape[0]]

        header = "      | " + " | ".join([f"T_{fid[-1]}" for fid in teacher_ids])
        print(header)
        print("-" * len(header))
        for i, row in enumerate(self.q_table):
            client_id = student_ids[i]
            row_str = f"S_{client_id[-1]}   | " + " | ".join([f"{v:5.2f}" for v in row])
            print(row_str)