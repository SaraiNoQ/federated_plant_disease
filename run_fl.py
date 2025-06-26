# run_fl.py

import torch
import os
import numpy as np
import copy
import datetime

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import farm_unit_update, aggregate_models, farm_unit_update_fedprox
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history
from src.rl_selector import UCB1Selector # 导入RL选择器

def main():
    """主执行函数"""
    print("--- 多农场联邦迁移学习与模型蒸馏项目启动 (V3: Two-Layer RL & Distillation) ---")
    print(f"使用设备: {config.DEVICE}")

    # --- 1. 设置日志 ---
    log_dir = os.path.join(config.OUTPUT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'farm_fl_accuracy_log.txt')
    
    with open(log_file_path, 'w') as f:
        f.write(f"Log start time: {datetime.datetime.now()}\n")
        f.write("Server_Round,Farm_ID,Farm_FL_Round,Accuracy_on_Farm_Val_Set\n")

    def append_to_log(message):
        with open(log_file_path, 'a') as f:
            f.write(message + '\n')

    # 2. 加载数据，并创建全局验证集
    print("\n--- 步骤1: 准备数据集与全局验证集 ---")
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=config.CLIENT_UNITS_PER_FARM,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=config.CREATE_GLOBAL_VALIDATION_SET,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )

    if not all_farms_data_loaders:
        print("错误: 未能加载任何农场数据，程序终止。")
        return

    # 3. 初始化服务器模型和RL选择器
    print("\n--- 步骤2: 初始化服务器模型与RL选择器 ---")
    server_global_model = build_model(
        num_classes=config.NUM_CLASSES_PLANTVILLAGE,
        pretrained_path=config.INITIAL_SOURCE_MODEL_PATH
    ).to(config.DEVICE)

    # Layer 1: Farm Selector
    server_rl_selector = None
    if config.USE_RL_FARM_SELECTION:
        farm_ids = list(all_farms_data_loaders.keys())
        server_rl_selector = UCB1Selector(
            num_farms=len(farm_ids),
            farm_ids=farm_ids,
            exploration_factor=config.RL_EXPLORATION_FACTOR
        )
        print("Layer 1 RL (Farm Selection) is ENABLED.")
    
    # Layer 2: Client Selectors (one per farm)
    farm_rl_selectors = {}
    if config.USE_RL_CLIENT_SELECTION:
        print("Layer 2 RL (Client Selection) is ENABLED for each farm.")
        for farm_id, farm_data in all_farms_data_loaders.items():
            num_units = len(farm_data["unit_loaders"])
            if num_units > 0:
                farm_rl_selectors[farm_id] = UCB1Selector(
                    num_farms=num_units,
                    farm_ids=list(range(num_units)),
                    exploration_factor=config.RL_EXPLORATION_FACTOR
                )

    # State tracking for rewards
    server_history = {'round': [], 'loss': [], 'accuracy': [], 'f1_score': []}
    last_server_accuracy = 0.0
    last_farm_accuracies = {farm_id: 0.0 for farm_id in all_farms_data_loaders.keys()}

    # --- 4. 服务器聚合轮次循环 ---
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")

        # --- Layer 1 RL: Select Farms ---
        participating_farms = list(all_farms_data_loaders.keys())
        if config.USE_RL_FARM_SELECTION and server_rl_selector:
            participating_farms = server_rl_selector.select_farms(config.FARMS_PER_SERVER_ROUND)
            print(f"[Server RL] Selected farms for this round: {participating_farms}")
        
        farm_final_models = {}
        
        # --- 遍历所有选中农场 ---
        for farm_id in participating_farms:
            farm_data = all_farms_data_loaders[farm_id]
            print(f"\n  --- Processing Farm: {farm_id} ---")
            
            farm_global_model = build_model(
                num_classes=farm_data["num_classes"], pretrained_path=None
            ).to(config.DEVICE)
            server_state_dict = server_global_model.state_dict()
            backbone_state_dict = {k: v for k, v in server_state_dict.items() if k in farm_global_model.state_dict() and 'fc' not in k}
            farm_global_model.load_state_dict(backbone_state_dict, strict=False)
            
            # --- Intra-Farm FL Loop ---
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                
                # --- Layer 2 RL: Select Clients ---
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if len(loader.dataset) > 0]
                if not active_unit_indices: break
                
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0: break

                selected_unit_indices = []
                if config.USE_RL_CLIENT_SELECTION and farm_id in farm_rl_selectors:
                    selected_unit_indices = farm_rl_selectors[farm_id].select_farms(num_units_to_select)
                    # print(f"    [Farm {farm_id} RL] Round {farm_fl_round+1}, Selected clients: {selected_unit_indices}")
                else:
                    selected_unit_indices = np.random.choice(active_unit_indices, num_units_to_select, replace=False)

                farm_round_start_model = copy.deepcopy(farm_global_model).to(config.DEVICE)
                # Train selected clients
                unit_model_updates = [
                    farm_unit_update_fedprox(
                        model=copy.deepcopy(farm_round_start_model),  # Send a fresh copy
                        global_model=farm_round_start_model,  # Send reference for prox term
                        train_loader=farm_data["unit_loaders"][unit_idx],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx,
                        mu=config.FEDPROX_MU  # Use the new mu parameter
                    ) for unit_idx in selected_unit_indices
                ]
                
                # Aggregate and evaluate
                if unit_model_updates:
                    agg_weights = aggregate_models(unit_model_updates, f"Farm {farm_id} internal")
                    if agg_weights: farm_global_model.load_state_dict(agg_weights)
                
                metrics = evaluate_model(farm_global_model, farm_data["val_loader"], config.DEVICE)
                current_acc = metrics['accuracy']
                
                print(f"    FL Round {farm_fl_round+1}/{config.FARM_FL_ROUNDS} | Clients: {selected_unit_indices} -> Acc: {current_acc:.2f}%")
                append_to_log(f"{server_round_idx+1},{farm_id},{farm_fl_round+1},{current_acc:.4f}")

                # Update Layer 2 RL
                if config.USE_RL_CLIENT_SELECTION and farm_id in farm_rl_selectors:
                    reward = current_acc - last_farm_accuracies[farm_id]
                    farm_rl_selectors[farm_id].update(selected_unit_indices, reward)
                    last_farm_accuracies[farm_id] = current_acc
            
            farm_final_models[farm_id] = copy.deepcopy(farm_global_model)

            # --- Model Distillation Step ---
            print(f"  --- Starting Model Distillation for Farm: {farm_id} ---")
            teacher_model = farm_final_models[farm_id]
            distill_loader = farm_data.get('val_loader')
            if distill_loader and len(distill_loader.dataset) > 0:
                distill_model(
                    teacher_model=teacher_model,
                    farm_id=farm_id,
                    num_farm_classes=farm_data['num_classes'],
                    distill_train_loader=distill_loader,
                    distill_val_loader=distill_loader,
                    epochs=config.DISTILLATION_EPOCHS,
                    lr=config.LEARNING_RATE_DISTILL,
                    temperature=config.TEMPERATURE, alpha=config.ALPHA_DISTILLATION,
                    device=config.DEVICE, output_dir=config.OUTPUT_DIR
                )
            else:
                print(f"  Skipping distillation for Farm {farm_id}: No data available.")
            
        # 5. Server Aggregation & Layer 1 RL Update
        print(f"\n--- Server Aggregating Backbones from {list(farm_final_models.keys())} ---")
        
        backbone_weights_list = [
            {k: v for k, v in model.state_dict().items() if 'fc' not in k}
            for model in farm_final_models.values()
        ]

        if backbone_weights_list:
            aggregated_weights = aggregate_models(backbone_weights_list, "Server Backbone")
            if aggregated_weights:
                server_model_state_dict = server_global_model.state_dict()
                server_model_state_dict.update(aggregated_weights)
                server_global_model.load_state_dict(server_model_state_dict)
                print("  Server global model's backbone has been updated.")
                
                if global_val_loader:
                    metrics = evaluate_model(server_global_model, global_val_loader, config.DEVICE, "Server Post-Aggregation")
                    print(f"  Server Model on Global Val Set: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1_score']:.4f}")
                    server_history['round'].append(server_round_idx + 1)
                    server_history['loss'].append(metrics['loss'])
                    server_history['accuracy'].append(metrics['accuracy'])
                    server_history['f1_score'].append(metrics['f1_score'])

                    # Update Layer 1 RL
                    if config.USE_RL_FARM_SELECTION and server_rl_selector:
                        reward = metrics['accuracy'] - last_server_accuracy
                        server_rl_selector.update(participating_farms, reward)
                        last_server_accuracy = metrics['accuracy']

    print("\n\n--- 顶级联邦学习流程完成 ---")
    plot_server_fl_history(server_history, config.OUTPUT_DIR)


    # 7. 保存最终的服务器模型
    final_server_model_path = os.path.join(config.OUTPUT_DIR, 'server_final_aggregated_model.pth')
    torch.save(server_global_model.state_dict(), final_server_model_path)
    print(f"\n最终服务器模型已保存至: {final_server_model_path}")
    print(f"农场内联邦学习准确率日志已保存至: {log_file_path}")

if __name__ == '__main__':
    main()