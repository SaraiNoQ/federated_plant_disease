# src/rl_selector.py

import numpy as np

class UCB1Selector:
    """
    使用UCB1算法的农场/客户端选择器。
    每个农场被视为一个“臂”（arm）。
    """
    def __init__(self, num_farms, farm_ids, exploration_factor=2.0):
        """
        初始化选择器。
        Args:
            num_farms (int): 可选农场的总数。
            farm_ids (list): 农场的ID列表，顺序必须固定。
            exploration_factor (float): UCB1算法中的探索常数C。
        """
        self.num_farms = num_farms
        self.farm_ids = farm_ids
        self.farm_id_to_idx = {farm_id: i for i, farm_id in enumerate(farm_ids)}
        self.exploration_factor = exploration_factor
        
        # 初始化统计数据
        self.pull_counts = np.zeros(num_farms, dtype=int)
        self.avg_rewards = np.zeros(num_farms, dtype=float)
        self.total_pulls = 0

    def select_farms(self, num_to_select):
        """
        根据UCB1策略选择最优的农场子集。
        Args:
            num_to_select (int): 要选择的农场数量。
        Returns:
            list: 被选中的农场ID列表。
        """
        # 优先选择从未被选过的农场
        unpulled_indices = np.where(self.pull_counts == 0)[0]
        if len(unpulled_indices) > 0:
            # 如果未选择的农场数量大于等于需要选择的数量，从中随机选
            if len(unpulled_indices) >= num_to_select:
                selected_indices = np.random.choice(unpulled_indices, num_to_select, replace=False)
            # 否则，选择所有未选择的，再从已选择的里面补齐
            else:
                selected_indices = list(unpulled_indices)
                remaining_needed = num_to_select - len(selected_indices)
                pulled_indices = np.where(self.pull_counts > 0)[0]
                selected_indices.extend(np.random.choice(pulled_indices, remaining_needed, replace=False))
            
            selected_farm_ids = [self.farm_ids[i] for i in selected_indices]
            print(f"[RL Selector] 优先选择未探索的农场: {selected_farm_ids}")
            return selected_farm_ids

        # 如果所有农场都至少被选择过一次，计算UCB1分数
        ucb1_scores = self.avg_rewards + self.exploration_factor * np.sqrt(
            np.log(self.total_pulls) / self.pull_counts
        )
        
        # 选择分数最高的N个农场
        # np.argsort返回的是从小到大的索引，所以我们取最后N个
        selected_indices = np.argsort(ucb1_scores)[-num_to_select:]
        
        selected_farm_ids = [self.farm_ids[i] for i in selected_indices]
        print(f"[RL Selector] 根据UCB1分数 {ucb1_scores.round(3)} 选择农场: {selected_farm_ids}")
        return selected_farm_ids

    def update(self, selected_farm_ids, reward):
        """
        根据本轮聚合的奖励来更新被选中的农场。
        Args:
            selected_farm_ids (list): 被选中的农场ID。
            reward (float): 本轮聚合后获得的奖励（例如，全局模型准确率的提升）。
        """
        self.total_pulls += 1
        for farm_id in selected_farm_ids:
            idx = self.farm_id_to_idx[farm_id]
            self.pull_counts[idx] += 1
            # 更新平均奖励 (增量式更新)
            n = self.pull_counts[idx]
            old_avg = self.avg_rewards[idx]
            self.avg_rewards[idx] = old_avg + (reward - old_avg) / n
        
        print(f"[RL Selector] 已更新农场 {selected_farm_ids} 的统计数据，奖励为: {reward:.4f}")
        print(f"  - 新的平均奖励: {self.avg_rewards.round(3)}")
        print(f"  - 新的选择计数: {self.pull_counts}")