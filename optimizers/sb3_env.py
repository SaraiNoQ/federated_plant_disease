# optimizers/sb3_env.py

import gymnasium as gym
from gymnasium import spaces
import numpy as np

class FLWeightEnv(gym.Env):
    """
    一个用于联邦学习权重优化的自定义Gym环境，兼容Stable-Baselines3。
    """
    metadata = {'render_modes': []}

    def __init__(self, state_dim, action_dim, evaluate_weights_fn):
        super(FLWeightEnv, self).__init__()
        
        # 外部传入的核心评估函数
        self.evaluate_weights_fn = evaluate_weights_fn
        self._state_dim = state_dim
        
        # 动作空间：连续的，每个权重在[-1, 1]之间。SB3的算法会自动处理这个范围。
        # 我们将在step函数中对动作进行softmax归一化。
        self.action_space = spaces.Box(low=-1, high=1, shape=(action_dim,), dtype=np.float32)
        
        # 状态空间：连续的，描述了联邦学习系统的状态。
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(state_dim,), dtype=np.float32)

    def _get_obs(self):
        # 在我们的场景中，状态是固定的，由外部传入
        # 但为了符合gym规范，我们从一个成员变量中获取
        return self._current_state

    def _get_info(self):
        # 可以返回一些调试信息
        return {}

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        
        # `options` 字典可以用来在重置时传入固定的状态
        if options and 'initial_state' in options:
            self._current_state = options['initial_state']
        else:
            # 如果没有提供，则使用一个零向量作为默认状态
            self._current_state = np.zeros(self._state_dim, dtype=np.float32)

        observation = self._get_obs()
        info = self._get_info()
        return observation, info

    def step(self, action):
        # 1. 将SB3输出的动作（在[-1, 1]范围内）转换为和为1的权重
        # 使用softmax可以确保权重非负且和为1
        weights = np.exp(action) / np.sum(np.exp(action))
        
        # 2. 调用我们的核心评估函数
        accuracy, reward = self.evaluate_weights_fn(weights)
        
        # 3. 在我们的单步决策问题中，每次step后环境就结束了
        terminated = True
        
        # 4. 下一个状态与当前状态相同，因为我们的系统状态只在每轮FL开始时更新
        observation = self._get_obs()
        info = self._get_info()
        info['accuracy'] = accuracy # 可以将准确率放入info中方便监控

        return observation, reward, terminated, False, info # (obs, reward, terminated, truncated, info)