# optimizers/bayesian_opt.py
import numpy as np
from skopt import gp_minimize
from skopt.space import Real
from skopt.utils import use_named_args

class BayesianOptimizer:
    def __init__(self, action_dim, config, device):
        self.action_dim = action_dim
        # 定义搜索空间，每个权重都在 [0, 1] 之间
        self.space = [Real(0, 1, name=f'w_{i}') for i in range(action_dim)]
        self.config = config

    def run(self, evaluate_weights_fn, epochs, **kwargs):
        # 由于evaluate_weights_fn需要归一化的权重，而gp_minimize的输入是独立的
        # 我们需要一个包装函数
        
        # gp_minimize 的目标是最小化，所以我们要对准确率取反
        @use_named_args(self.space)
        def objective(**params):
            # 将字典形式的参数转换为numpy数组
            raw_weights = np.array(list(params.values()))
            # 归一化，使其和为1
            if np.sum(raw_weights) == 0:
                weights = np.ones(self.action_dim) / self.action_dim
            else:
                weights = raw_weights / np.sum(raw_weights)
            
            # 调用外部评估函数
            acc, _ = evaluate_weights_fn(weights)
            
            # 返回负准确率
            return -acc

        print("--- 开始贝叶斯优化 ---")
        # n_calls 是总的评估次数
        # n_initial_points 是初始的随机探索点数
        res_gp = gp_minimize(
            func=objective,
            dimensions=self.space,
            n_calls=epochs,
            n_initial_points=self.config.BO_INITIAL_POINTS,
            random_state=42,
            verbose=True
        )

        best_acc = -res_gp.fun
        best_raw_weights = np.array(res_gp.x)
        best_weights = best_raw_weights / np.sum(best_raw_weights)

        print(f"\n贝叶斯优化完成。最佳准确率: {best_acc:.2f}%")
        return best_acc, best_weights