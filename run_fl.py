# run_fl.py

import torch
import os
import numpy as np
import copy

# 导入我们自己的模块
import config
from src.data_loader import get_dataloaders
from src.models import build_model
from src.federated import client_update, server_aggregate
from src.utils import evaluate_model, plot_history

def main():
    """主执行函数"""
    # 打印配置信息
    print("--- 联邦学习项目启动 ---")
    print(f"设备: {config.DEVICE}")
    print(f"客户端总数: {config.NUM_CLIENTS}, 每轮参与数: {config.CLIENTS_PER_ROUND}")
    print(f"通信轮数: {config.NUM_ROUNDS}, 客户端Epochs: {config.EPOCHS_PER_CLIENT}")
    
    # 准备数据
    print("\n--- 正在准备数据集 ---")
    client_loaders, val_loader = get_dataloaders(
        data_dir=config.DATA_DIR,
        num_clients=config.NUM_CLIENTS,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )
    for i, loader in enumerate(client_loaders):
        print(f"客户端 {i} 的数据量: {len(loader.dataset)}")
    print(f"全局验证集数据量: {len(val_loader.dataset)}")

    # 初始化全局模型
    print("\n--- 正在初始化全局模型 ---")
    if config.USE_CUSTOM_PRETRAINED:
        print("模式: 继续训练 (使用自定义预训练模型)")
        model_path_to_use = config.CUSTOM_PRETRAINED_MODEL_PATH
    else:
        print("模式: 从头开始 (使用ImageNet预训练模型)")
        model_path_to_use = config.IMAGENET_PRETRAINED_PATH
        
    global_model = build_model(
        num_classes=config.NUM_CLASSES, 
        pretrained_path=model_path_to_use
    ).to(config.DEVICE)

    # 开始联邦学习主循环
    print("\n--- 开始联邦学习训练循环 ---")
    history = {'round': [], 'loss': [], 'accuracy': []}

    for round_idx in range(config.NUM_ROUNDS):
        print(f"\n====== 全局轮次: {round_idx + 1}/{config.NUM_ROUNDS} ======")
        
        # 1. 随机选择客户端
        selected_clients_indices = np.random.choice(
            range(config.NUM_CLIENTS), config.CLIENTS_PER_ROUND, replace=False
        )
        print(f"本轮选中的客户端: {selected_clients_indices}")
        
        client_updates = []
        
        # 2. 分发模型并在客户端上训练
        for client_idx in selected_clients_indices:
            local_model = copy.deepcopy(global_model).to(config.DEVICE)
            
            # 本地训练
            local_weights = client_update(
                model=local_model,
                train_loader=client_loaders[client_idx],
                epochs=config.EPOCHS_PER_CLIENT,
                lr=config.LEARNING_RATE,
                device=config.DEVICE
            )
            client_updates.append(local_weights)
            print(f"  - 客户端 {client_idx} 本地训练完成。")
            
        # 3. 聚合客户端权重
        aggregated_weights = server_aggregate(client_updates)
        global_model.load_state_dict(aggregated_weights)
        print("服务器聚合完成，全局模型已更新。")
        
        # 4. 评估更新后的全局模型
        val_loss, val_accuracy = evaluate_model(global_model, val_loader, config.DEVICE)
        print(f"--- 全局轮次 {round_idx + 1} 结束 ---")
        print(f"全局模型在验证集上的性能: Loss = {val_loss:.4f}, Accuracy = {val_accuracy:.2f}%")

        # 记录历史
        history['round'].append(round_idx + 1)
        history['loss'].append(val_loss)
        history['accuracy'].append(val_accuracy)

    # 训练结束，保存和可视化
    print("\n--- 联邦学习训练完成 ---")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    final_model_path = os.path.join(config.OUTPUT_DIR, 'final_federated_model.pth')
    torch.save(global_model.state_dict(), final_model_path)
    print(f"最终模型已保存至: {final_model_path}")
    
    plot_history(history, config.OUTPUT_DIR)

if __name__ == '__main__':
    main()