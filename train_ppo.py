# train_ppo_offline.py

import torch
import os
import numpy as np
import argparse
import copy

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import aggregate_models_rl
from src.utils import evaluate_model
from src.rl_aggregator import PPOAgent

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
    """主执行函数：离线训练PPO聚合器"""
    print("--- PPO聚合器离线训练程序 ---")
    print(f"使用设备: {config.DEVICE}")
    print(f"目标训练轮次目录: {args.round_dir}")
    print(f"训练周期数: {args.epochs}")

    # 1. 加载所有需要的模型
    try:
        baseline_model, local_models = load_round_data(args.round_dir, config.DEVICE)
    except (FileNotFoundError, ValueError) as e:
        print(f"错误: {e}")
        return

    # 2. 准备数据加载器，只需要全局验证集和每个农场的验证集
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=0, # 不需要客户端训练数据
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=True,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )

    if not global_val_loader:
        print("错误: 全局验证集加载失败，无法进行训练。")
        return

    # 3. 初始化PPO Agent
    farm_ids_participating = sorted(list(local_models.keys()))
    num_farms = len(farm_ids_participating)
    
    STATE_DIM = 1 + num_farms * 3  # last_global_acc + N * (acc, loss, samples)
    ACTION_DIM = num_farms

    ppo_agent = PPOAgent(
        state_dim=STATE_DIM, action_dim=ACTION_DIM,
        lr_actor=config.PPO_LR_ACTOR, lr_critic=config.PPO_LR_CRITIC,
        gamma=config.PPO_GAMMA, K_epochs=config.PPO_K_EPOCHS,
        eps_clip=config.PPO_EPS_CLIP, device=config.DEVICE
    )
    print("PPO Agent 初始化完成。")

    # 4. 构建固定的状态向量
    print("\n--- 正在构建状态向量 ---")
    # a. 获取基线服务器模型在全局验证集上的性能
    baseline_metrics = evaluate_model(baseline_model, global_val_loader, config.DEVICE)
    baseline_acc = baseline_metrics['accuracy']
    print(f"基线服务器模型全局准确率: {baseline_acc:.2f}%")

    # b. 获取每个局部模型在自己验证集上的性能
    state_vector = [baseline_acc / 100.0]
    local_model_backbones = []

    for farm_id in farm_ids_participating:
        model = local_models[farm_id]
        val_loader = all_farms_data_loaders[farm_id]['val_loader']
        metrics = evaluate_model(model, val_loader, config.DEVICE)
        num_samples = len(val_loader.dataset)
        
        state_vector.extend([
            metrics['accuracy'] / 100.0,
            min(metrics['loss'], 5.0),
            num_samples / 1000.0
        ])
        print(f"  - {farm_id} 局部性能: Acc={metrics['accuracy']:.2f}%, Loss={metrics['loss']:.4f}")
        
        # 提取backbone以备后续聚合使用
        backbone = {k: v for k, v in model.state_dict().items() if 'fc' not in k}
        local_model_backbones.append(backbone)
    
    # 5. 离线训练循环
    print("\n--- 开始PPO离线训练 ---")
    best_acc = 0.0
    
    for epoch in range(args.epochs):
        # a. Agent决策，获取聚合权重
        weights = ppo_agent.select_action(state_vector)
        
        # b. 模拟聚合
        agg_backbone = aggregate_models_rl(local_model_backbones, weights)
        if agg_backbone is None:
            ppo_agent.buffer.pop() # 移除无效转换
            continue

        # c. 将聚合后的backbone加载到临时模型中并评估
        temp_eval_model = copy.deepcopy(baseline_model)
        temp_eval_model.load_state_dict(agg_backbone, strict=False)
        current_metrics = evaluate_model(temp_eval_model, global_val_loader, config.DEVICE)
        current_acc = current_metrics['accuracy']

        # d. 计算奖励并更新PPO Agent
        reward = current_acc - baseline_acc
        ppo_agent.update(reward)

        print(f"Epoch {epoch+1}/{args.epochs} | Weights: {np.round(weights,2)} -> Global Acc: {current_acc:.2f}% | Reward: {reward:.4f}")

        # 保存表现最好的策略网络
        if current_acc > best_acc:
            best_acc = current_acc
            save_path = os.path.join(args.round_dir, 'ppo_policy_best.pth')
            ppo_agent.save(save_path)
            print(f"  ** 新的最佳准确率! 模型已保存到 {save_path} **")
            
    print("\n--- PPO离线训练完成 ---")
    print(f"最佳聚合准确率: {best_acc:.2f}%")
    final_save_path = os.path.join(args.round_dir, 'ppo_policy_final.pth')
    ppo_agent.save(final_save_path)
    print(f"最终的PPO策略已保存到: {final_save_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Offline PPO Aggregator Training")
    parser.add_argument(
        '--round_dir',
        type=str,
        required=True,
        help="包含该轮次所有局部模型的目录 (例如: ./outputs/round_1)"
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=500,
        help="PPO离线训练的总周期数"
    )
    
    parsed_args = parser.parse_args()
    main(parsed_args)