# optimizers/sb3_wrapper.py

from .sb3_env import FLWeightEnv
from stable_baselines3 import A2C, SAC, PPO # PPO也一并使用SB3的实现，保持一致性
from stable_baselines3.common.callbacks import BaseCallback
import numpy as np # <-- 确保导入numpy

class SB3ProgressCallback(BaseCallback):
    """一个简单的回调函数，用于打印训练进度。"""
    def __init__(self, total_timesteps, algorithm_name, verbose=0):
        super(SB3ProgressCallback, self).__init__(verbose)
        self.total_timesteps = total_timesteps
        self.algorithm_name = algorithm_name
        self.best_acc = -1

    def _on_step(self) -> bool:
        # infos是环境step返回的info字典的列表
        if len(self.model.ep_info_buffer) > 0:
            last_info = self.model.ep_info_buffer[-1]
            accuracy = last_info.get('accuracy', -1)
            reward = last_info.get('r', -1) # 'r' 是SB3记录的奖励
            
            if accuracy > self.best_acc:
                self.best_acc = accuracy

            print(f"\rStep {self.n_calls}/{self.total_timesteps} [{self.algorithm_name}] | Last Acc: {accuracy:.2f}% | Best Acc: {self.best_acc:.2f}% | Last Reward: {reward:.4f}", end="")
        return True

class SB3Optimizer:
    def __init__(self, algorithm_name, state_dim, action_dim, config, device):
        self.algorithm_name = algorithm_name.upper()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        
        # 从SB3库中获取算法类
        self.ALGO_MAP = {'A2C': A2C, 'SAC': SAC, 'PPO': PPO}
        if self.algorithm_name not in self.ALGO_MAP:
            raise ValueError(f"Unknown SB3 algorithm: {self.algorithm_name}")
        
        # 不再这里初始化 self.model
        self.model = None 
        self.best_acc = -1
        self.best_weights = None

    def run(self, evaluate_weights_fn, state_vector, epochs, **kwargs):
        # 1. 创建我们的自定义环境
        env = FLWeightEnv(self.state_dim, self.action_dim, evaluate_weights_fn)
        
        # 2. **在这里初始化模型**，因为现在有env了
        AlgoClass = self.ALGO_MAP[self.algorithm_name]
        self.model = AlgoClass("MlpPolicy", env, verbose=0, device=self.device)
        
        # 3. 定义一个回调函数，在每次step后评估并保存最佳权重
        class CustomCallback(BaseCallback):
            def __init__(self, optimizer_instance, verbose=0):
                super(CustomCallback, self).__init__(verbose)
                self.optimizer = optimizer_instance

            def _on_step(self) -> bool:
                # 检查ep_info_buffer是否为空且其最后一个元素不为空
                if len(self.model.ep_info_buffer) > 0 and len(self.model.ep_info_buffer[-1]) > 0:
                    last_info = self.model.ep_info_buffer[-1]
                    accuracy = last_info.get('accuracy', -1)
                    if accuracy > self.optimizer.best_acc:
                        self.optimizer.best_acc = accuracy
                        # 从上一步的动作中恢复权重
                        # self.locals['actions'] 包含上一步的动作
                        if 'actions' in self.locals and self.locals['actions'] is not None:
                            last_action = self.locals['actions'][0]
                            weights = np.exp(last_action) / np.sum(np.exp(last_action))
                            self.optimizer.best_weights = weights
                return True
        
        # 4. 创建进度条和自定义回调
        progress_callback = SB3ProgressCallback(epochs, self.algorithm_name)
        custom_callback = CustomCallback(self)
        
        # 5. 开始训练
        # SB3的learn方法会在内部循环调用env.step()
        # 我们需要通过reset的options传入固定的state_vector
        # 注意：SB3的learn方法会自动调用reset，我们不需要手动调用
        
        print(f"--- 开始使用 Stable-Baselines3 训练 {self.algorithm_name} ---")
        self.model.learn(
            total_timesteps=epochs, 
            callback=[progress_callback, custom_callback],
            reset_num_timesteps=False,
            log_interval=-1,
            # 通过 reset_kwargs 将 options 传递给 env.reset()
            tb_log_name=f"{self.algorithm_name}_run",
            progress_bar=False, # 我们有自己的进度条
            # ** 核心修复 **
            # learn 方法在开始时会自动调用reset，我们可以把初始状态传进去
            # 但更稳妥的做法是在learn之前手动调用一次reset，确保环境状态正确
        )
        # 在 learn 之前手动重置环境并设置初始状态
        # 这是为了确保第一次 learn 调用时，环境处于我们期望的初始状态
        # 尽管 learn 会自动 reset，但明确指定可以避免潜在问题
        env.reset(options={'initial_state': np.array(state_vector, dtype=np.float32)})
        
        print("\n--- 训练完成 ---")
        
        # 如果训练结束时没有找到任何最佳权重（例如epochs=1），则使用最后一个动作
        if self.best_weights is None and self.model._last_obs is not None:
             # _last_action 在 SB3 v2.x 中可用，如果是更早版本可能需要其他方式获取
            with torch.no_grad():
                last_action, _ = self.model.predict(self.model._last_obs, deterministic=True)
            self.best_weights = np.exp(last_action[0]) / np.sum(np.exp(last_action[0]))
            self.best_acc, _ = evaluate_weights_fn(self.best_weights)

        return self.best_acc, self.best_weights