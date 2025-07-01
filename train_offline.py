# train_offline.py

import torch
import os
import numpy as np
import argparse
import copy
from functools import partial

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import aggregate_models_rl
from src.utils import evaluate_model

# 动态导入优化器
from optimizers.sb3_wrapper import SB3Optimizer # <-- 导入新的SB3包装器
from optimizers.bayesian_opt import BayesianOptimizer
from optimizers.evolutionary import EvolutionaryOptimizer
from optimizers.bayesian_opt import BayesianOptimizer
from optimizers.evolutionary import EvolutionaryOptimizer

OPTIMIZER_MAP = {
    'ppo_sb3': lambda **kwargs: SB3Optimizer(algorithm_name='PPO', **kwargs),
    'a2c_sb3': lambda **kwargs: SB3Optimizer(algorithm_name='A2C', **kwargs),
    'sac_sb3': lambda **kwargs: SB3Optimizer(algorithm_name='SAC', **kwargs),
    'bayes': BayesianOptimizer,
    'evo': EvolutionaryOptimizer,
}

def load_round_data(round_dir: str, device: torch.device):
    """加载指定轮次的所有模型和相关数据"""
    print(f"--- 正在从 {round_dir} 加载数据 ---")
    
    # 加载基线服务器模型
    server_model_path = os.path.join(round_dir, 'server_model_before_agg.pth')
    if not os.path.exists(server_model_path):
        raise FileNotFoundError(f"基线服务器模型未找到: {server_model_path}")
    
    baseline_server_model = build_model(num_classes=config.NUM_CLASSES_PLANTVILLAGE).to(device)
    baseline_server_model.load_state_dict(torch.load(server_model_path, map_location=device))
    print("基线服务器模型加载成功。")

    # 加载所有农场的局部模型
    local_models = {}
    farm_ids = sorted(list(config.FARM_CLASS_ALLOCATION.keys())) # 确保顺序固定

    for farm_id in farm_ids:
        farm_model_path = os.path.join(round_dir, f'local_model_{farm_id}.pth')
        if os.path.exists(farm_model_path):
            num_classes = len(config.FARM_CLASS_ALLOCATION[farm_id])
            model = build_model(num_classes=num_classes).to(device)
            model.load_state_dict(torch.load(farm_model_path, map_location=device))
            local_models[farm_id] = model
            print(f"  - 加载 {farm_id} 的局部模型成功 ({num_classes} 类)。")
        else:
            print(f"  - 警告: 未找到 {farm_id} 的模型，跳过。")
            
    if not local_models:
        raise ValueError("未能加载任何局部模型。")
        
    return baseline_server_model, local_models

def main(args):
    """主执行函数：调度不同的离线优化算法"""
    print(f"--- 离线优化器调度程序 ---")
    print(f"选择的算法: {args.optimizer}")
    print(f"目标训练轮次目录: {args.round_dir}")
    print(f"训练周期/评估次数: {args.epochs}")
    print(f"使用设备: {config.DEVICE}")

    # 1. 加载所有需要的模型
    try:
        baseline_model, local_models = load_round_data(args.round_dir, config.DEVICE)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    # 2. 准备数据加载器 (仅验证集)
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=0, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, create_global_val_set=True,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )

    # 3. 计算基线和固定的模型/数据
    print("\n--- 准备评估环境 ---")
    baseline_metrics = evaluate_model(baseline_model, global_val_loader, config.DEVICE)
    baseline_acc = baseline_metrics['accuracy']
    print(f"基线服务器模型全局准确率: {baseline_acc:.2f}%")

    farm_ids_participating = sorted(list(local_models.keys()))
    local_model_backbones = [{k: v for k, v in local_models[fid].state_dict().items() if 'fc' not in k} for fid in farm_ids_participating]

    # 4. 定义通用的评估函数 (所有优化器共用)
    def evaluate_weights_fn(weights: np.ndarray):
        """
        接收一组权重，返回 (准确率, 奖励值)。
        这是所有优化算法的核心目标函数。
        """
        agg_backbone = aggregate_models_rl(local_model_backbones, weights)
        if agg_backbone is None:
            return 0.0, -baseline_acc # 返回一个很差的奖励

        temp_eval_model = copy.deepcopy(baseline_model)
        temp_eval_model.load_state_dict(agg_backbone, strict=False)
        current_metrics = evaluate_model(temp_eval_model, global_val_loader, config.DEVICE)
        current_acc = current_metrics['accuracy']
        reward = current_acc - baseline_acc
        return current_acc, reward

    # 5. 准备状态向量 (仅RL算法需要)
    state_vector = []
    if args.optimizer in ['ppo', 'a2c', 'sac']:
        print("\n--- 正在为RL算法构建状态向量 ---")
        state_vector.append(baseline_acc / 100.0)
        for farm_id in farm_ids_participating:
            val_loader = all_farms_data_loaders[farm_id]['val_loader']
            metrics = evaluate_model(local_models[farm_id], val_loader, config.DEVICE)
            state_vector.extend([
                metrics['accuracy'] / 100.0,
                min(metrics['loss'], 5.0),
                len(val_loader.dataset) / 1000.0
            ])
        print("状态向量构建完成。")
        
    # 6. 初始化并运行选择的优化器
    action_dim = len(farm_ids_participating)
    
    OptimizerFactory = OPTIMIZER_MAP.get(args.optimizer)
    if not OptimizerFactory:
        raise ValueError(f"未知的优化器: {args.optimizer}. 可选项: {list(OPTIMIZER_MAP.keys())}")
    
    # 统一调用接口
    if args.optimizer in ['ppo_sb3', 'a2c_sb3', 'sac_sb3']:
        state_dim = len(state_vector)
        optimizer = OptimizerFactory(state_dim=state_dim, action_dim=action_dim, config=config, device=config.DEVICE)
        best_acc, best_weights = optimizer.run(evaluate_weights_fn, state_vector, args.epochs)
    else: # Bayesian or Evolutionary
        optimizer = OptimizerFactory(action_dim=action_dim, config=config, device=config.DEVICE)
        best_weights = optimizer.run(evaluate_weights_fn, args.epochs)

    # 7. 结束
    print("\n--- 优化流程完成 ---")
    print(f"算法: {args.optimizer.upper()}")
    print(f"最佳准确率: {best_acc:.4f}%")
    print(f"最佳权重: {np.round(best_weights, 4)}")
    
    # 保存结果
    results = {
        'optimizer': args.optimizer,
        'best_accuracy': best_acc,
        'best_weights': best_weights.tolist(),
        'farm_ids': farm_ids_participating
    }
    save_path = os.path.join(args.round_dir, f'optimization_results_{args.optimizer}.json')
    import json
    with open(save_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"优化结果已保存到: {save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline Aggregation Weight Optimization")
    parser.add_argument(
        '--optimizer',
        type=str,
        required=True,
        choices=list(OPTIMIZER_MAP.keys()), # choices会自动更新
        help="选择要使用的优化算法。"
    )
    parser.add_argument(
        '--round_dir',
        type=str,
        required=True,
        help="包含该轮次所有局部模型的目录 (例如: ./outputs/round_1)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help="优化算法的训练周期数或总评估次数。"
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)