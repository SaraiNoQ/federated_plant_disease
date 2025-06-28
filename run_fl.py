# run_fl.py (V5 - Data Generation Mode)

import torch
import os
import numpy as np
import copy
import datetime

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import farm_unit_update_fedprox, aggregate_models
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history

def main():
    """主执行函数 - 现在只负责生成和保存每轮的局部模型"""
    print("--- 多农场联邦学习模型生成程序 (V5: Data Generation) ---")
    print(f"使用设备: {config.DEVICE}")

    # --- 1. 设置日志 ---
    log_dir = os.path.join(config.OUTPUT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'farm_fl_accuracy_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write(f"Log start time: {datetime.datetime.now()}\n")
        f.write("Server_Round,Farm_ID,Farm_FL_Round,Accuracy_on_Farm_Val_Set,Loss_on_Farm_Val_Set\n")

    def append_to_log(message):
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')

    # 2. 加载数据
    print("\n--- 步骤1: 准备数据集 ---")
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=config.CLIENT_UNITS_PER_FARM,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=True, # 依然需要全局验证集来评估
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )

    if not all_farms_data_loaders:
        print("错误: 未能加载任何农场数据，程序终止。")
        return

    # 3. 初始化服务器模型
    print("\n--- 步骤2: 初始化服务器模型 ---")
    server_global_model = build_model(
        num_classes=config.NUM_CLASSES_PLANTVILLAGE,
        pretrained_path=config.INITIAL_SOURCE_MODEL_PATH
    ).to(config.DEVICE)
    
    # 4. 服务器聚合轮次循环
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")

        # --- **核心修改1**: 所有农场都参与 ---
        participating_farms = list(all_farms_data_loaders.keys())
        print(f"本轮所有农场都将参与训练: {participating_farms}")
        
        farm_final_models = {}
        
        # --- 遍历所有农场 ---
        for farm_id in participating_farms:
            farm_data = all_farms_data_loaders[farm_id]
            print(f"\n  --- Processing Farm: {farm_id} ---")
            
            farm_global_model = build_model(
                num_classes=farm_data["num_classes"], pretrained_path=None
            ).to(config.DEVICE)
            # 从服务器同步最新的 backbone
            server_state_dict = server_global_model.state_dict()
            backbone_state_dict = {k: v for k, v in server_state_dict.items() if k in farm_global_model.state_dict() and 'fc' not in k}
            farm_global_model.load_state_dict(backbone_state_dict, strict=False)
            
            # --- Intra-Farm FL Loop ---
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if len(loader.dataset) > 0]
                if not active_unit_indices: break
                
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0: break
                
                # 简单随机选择客户端
                selected_unit_indices = np.random.choice(active_unit_indices, num_units_to_select, replace=False)

                farm_round_start_model = copy.deepcopy(farm_global_model).to(config.DEVICE)
                
                unit_model_updates = [
                    farm_unit_update_fedprox(
                        model=copy.deepcopy(farm_round_start_model),
                        global_model=farm_round_start_model,
                        train_loader=farm_data["unit_loaders"][unit_idx],
                        epochs=config.EPOCHS_PER_UNIT, lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE, farm_id=farm_id, unit_id=unit_idx,
                        mu=config.FEDPROX_MU
                    ) for unit_idx in selected_unit_indices
                ]
                
                if unit_model_updates:
                    agg_weights = aggregate_models(unit_model_updates, f"Farm {farm_id} internal")
                    if agg_weights: farm_global_model.load_state_dict(agg_weights)
            
            # 训练完成后，评估并记录最终的局部模型性能
            final_metrics = evaluate_model(farm_global_model, farm_data["val_loader"], config.DEVICE)
            print(f"  Farm {farm_id} final model local val Acc: {final_metrics['accuracy']:.2f}%")
            append_to_log(f"{server_round_idx+1},{farm_id},FINAL,{final_metrics['accuracy']:.4f},{final_metrics['loss']:.4f}")

            farm_final_models[farm_id] = copy.deepcopy(farm_global_model)

        # --- **核心修改2**: 保存所有农场的最终模型 ---
        round_output_dir = os.path.join(config.OUTPUT_DIR, f'round_{server_round_idx + 1}')
        os.makedirs(round_output_dir, exist_ok=True)
        print(f"\n--- 保存第 {server_round_idx + 1} 轮的所有农场模型到: {round_output_dir} ---")

        for farm_id, model in farm_final_models.items():
            model_save_path = os.path.join(round_output_dir, f'local_model_{farm_id}.pth')
            torch.save(model.state_dict(), model_save_path)
            print(f"  - 已保存 {farm_id} 的模型.")
        
        # 保存本轮开始前的服务器模型，作为下一轮训练的起点和离线训练的基线模型
        prev_server_model_path = os.path.join(round_output_dir, 'server_model_before_agg.pth')
        torch.save(server_global_model.state_dict(), prev_server_model_path)
        print(f"  - 已保存本轮的基线服务器模型.")

        # --- 聚合更新服务器模型，为下一轮做准备 ---
        print(f"\n--- (For Next Round) Aggregating Backbones from all farms using FedAvg ---")
        backbone_weights_list = [
            {k: v for k, v in model.state_dict().items() if 'fc' not in k}
            for model in farm_final_models.values()
        ]

        if backbone_weights_list:
            # 在这里，我们仍然使用简单的FedAvg来更新全局模型，以便下一轮FL有一个更好的起点
            aggregated_weights = aggregate_models(backbone_weights_list, "Server Backbone")
            if aggregated_weights:
                server_model_state_dict = server_global_model.state_dict()
                server_model_state_dict.update(aggregated_weights)
                server_global_model.load_state_dict(server_model_state_dict)
                print("  Server global model's backbone has been updated for the next round.")

    print("\n\n--- 模型生成流程完成 ---")
    final_server_model_path = os.path.join(config.OUTPUT_DIR, 'server_final_aggregated_model.pth')
    torch.save(server_global_model.state_dict(), final_server_model_path)
    print(f"\n最终服务器模型已保存至: {final_server_model_path}")
    print(f"所有轮次的局部模型已保存到 '{config.OUTPUT_DIR}/round_n/' 目录中。")

if __name__ == '__main__':
    main()