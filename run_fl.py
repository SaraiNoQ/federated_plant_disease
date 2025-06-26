# run_fl.py

import torch
import os
import numpy as np
import copy
import collections
import datetime

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import farm_unit_update, aggregate_models
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history
from src.rl_selector import UCB1Selector # 导入RL选择器

def main():
    """主执行函数"""
    print("--- 多农场联邦迁移学习与模型蒸馏项目启动 (V3: 内部客户端RL选择与日志记录) ---")
    print(f"使用设备: {config.DEVICE}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # --- 1. 设置日志 ---
    log_dir = os.path.join(config.OUTPUT_DIR, 'log')
    os.makedirs(log_dir, exist_ok=True)
    log_file_path = os.path.join(log_dir, 'farm_fl_accuracy_log.txt')

    # 写入日志文件头
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
    
    farm_rl_selectors = {}
    last_farm_accuracies = {}
    if config.USE_RL_CLIENT_SELECTION:
        print("为每个农场启用强化学习(UCB1)客户端选择器。")
        for farm_id, farm_data in all_farms_data_loaders.items():
            num_units = len(farm_data["unit_loaders"])
            if num_units > 0:
                farm_rl_selectors[farm_id] = UCB1Selector(
                    num_farms=num_units,
                    farm_ids=list(range(num_units)), # 臂是客户端索引
                    exploration_factor=config.RL_EXPLORATION_FACTOR
                )
                last_farm_accuracies[farm_id] = 0.0 # 初始化上次准确率

    server_history = {'round': [], 'loss': [], 'accuracy': [], 'f1_score': []}

    # --- 4. 服务器聚合轮次循环 ---
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")

        farm_final_models = {} # 存储本轮各农场训练好的模型对象
        
       # --- 遍历所有农场 ---
        for farm_id, farm_data in all_farms_data_loaders.items():
            print(f"\n\n  J{'='*8} 开始处理农场: {farm_id} J{'='*8}")
            
            # 为当前农场加载服务器骨干网络
            farm_global_model = build_model(
                num_classes=farm_data["num_classes"], pretrained_path=None
            ).to(config.DEVICE)
            server_state_dict = server_global_model.state_dict()
            backbone_state_dict = {k: v for k, v in server_state_dict.items() if k in farm_global_model.state_dict() and 'fc' not in k}
            farm_global_model.load_state_dict(backbone_state_dict, strict=False)
            
            # --- 农场内联邦学习循环 ---
            print(f"  --- 开始农场 {farm_id} 内部联邦学习 (FTL) ---")
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                
                # --- vvvvvv 核心修改：RL选择客户端 vvvvvv ---
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if len(loader.dataset) > 0]
                if not active_unit_indices:
                    print(f"    农场 {farm_id} 没有活跃的计算单元，跳过本轮农场FL。")
                    break
                
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0:
                     break

                selected_unit_indices = []
                if config.USE_RL_CLIENT_SELECTION and farm_id in farm_rl_selectors:
                    # 使用RL选择器
                    selected_unit_indices = farm_rl_selectors[farm_id].select_farms(num_units_to_select)
                else:
                    # 默认随机选择
                    selected_unit_indices = np.random.choice(active_unit_indices, num_units_to_select, replace=False)
                
                print(f"    回合 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} (农场 {farm_id}) | 选中单元: {selected_unit_indices}")

                unit_model_updates = []
                for unit_idx in selected_unit_indices:
                    unit_model_copy = copy.deepcopy(farm_global_model).to(config.DEVICE)
                    unit_weights = farm_unit_update(
                        model=unit_model_copy,
                        train_loader=farm_data["unit_loaders"][unit_idx],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx
                    )
                    unit_model_updates.append(unit_weights)
                
                # 聚合农场内单元的模型
                if unit_model_updates:
                    aggregated_farm_weights = aggregate_models(unit_model_updates, f"农场{farm_id}内部")
                    if aggregated_farm_weights:
                        farm_global_model.load_state_dict(aggregated_farm_weights)
                
                # --- vvvvvv 核心修改：评估、记录日志、更新RL vvvvvv ---
                if farm_data["val_loader"] and len(farm_data["val_loader"].dataset) > 0:
                    metrics = evaluate_model(farm_global_model, farm_data["val_loader"], config.DEVICE, context=f"农场{farm_id} FTL评估")
                    current_acc = metrics['accuracy']
                    
                    # 打印到控制台
                    print(f"      聚合后准确率: {current_acc:.2f}%")
                    
                    # 写入日志文件
                    log_message = f"{server_round_idx + 1},{farm_id},{farm_fl_round + 1},{current_acc:.4f}"
                    append_to_log(log_message)

                    # 如果使用RL，计算奖励并更新选择器
                    if config.USE_RL_CLIENT_SELECTION and farm_id in farm_rl_selectors:
                        reward = current_acc - last_farm_accuracies[farm_id]
                        farm_rl_selectors[farm_id].update(selected_unit_indices, reward)
                        last_farm_accuracies[farm_id] = current_acc # 更新该农场的最新准确率
            
            # 存储本轮服务器训练最终的农场模型
            farm_final_models[farm_id] = copy.deepcopy(farm_global_model)
            final_farm_model_path = os.path.join(config.OUTPUT_DIR, f'farm_{farm_id}_final_ftl_model_round_{server_round_idx+1}.pth')
            torch.save(farm_global_model.state_dict(), final_farm_model_path)
            
        # 5. 服务器聚合 (只聚合骨干网络)
        print(f"\n--- 服务器全局轮次 {server_round_idx + 1}: 聚合各农场骨干网络 ---")
        
        backbone_weights_list = [
            {k: v for k, v in model.state_dict().items() if 'fc' not in k}
            for model in farm_final_models.values()
        ]

        if backbone_weights_list:
            aggregated_backbone_weights = aggregate_models(backbone_weights_list, "服务器骨干网络")
            if aggregated_backbone_weights:
                server_model_state_dict = server_global_model.state_dict()
                server_model_state_dict.update(aggregated_backbone_weights)
                server_global_model.load_state_dict(server_model_state_dict)
                print("  服务器全局模型的骨干网络已更新。")
                
                if global_val_loader:
                    metrics = evaluate_model(server_global_model, global_val_loader, config.DEVICE, "服务器聚合后评估")
                    print(f"  服务器模型在全局验证集上: Acc={metrics['accuracy']:.2f}%, F1={metrics['f1_score']:.4f}")
                    server_history['round'].append(server_round_idx + 1)
                    server_history['loss'].append(metrics['loss'])
                    server_history['accuracy'].append(metrics['accuracy'])
                    server_history['f1_score'].append(metrics['f1_score'])

    print("\n\n--- 顶级联邦学习流程完成 ---")
    plot_server_fl_history(server_history, config.OUTPUT_DIR)
    
    # 6. (可选) 对所有训练好的农场模型进行蒸馏
    # print("\n--- 注意: 本次运行执行蒸馏步骤---")
    # for farm_id, teacher_model in farm_final_models.items():
    #     print(f"\n--- 开始为农场 {farm_id} 进行模型蒸馏 ---")
    #     farm_data = all_farms_data_loaders[farm_id]
    #     distill_train_loader_for_farm = farm_data['val_loader'] # 使用验证集进行蒸馏
    #     if distill_train_loader_for_farm and len(distill_train_loader_for_farm.dataset) > 0:
    #         distill_model(
    #             teacher_model=teacher_model,
    #             farm_id=farm_id,
    #             num_farm_classes=farm_data['num_classes'],
    #             distill_train_loader=distill_train_loader_for_farm,
    #             distill_val_loader=farm_data['val_loader'],
    #             epochs=config.DISTILLATION_EPOCHS,
    #             lr=config.LEARNING_RATE_DISTILL,
    #             temperature=config.TEMPERATURE,
    #             alpha=config.ALPHA_DISTILLATION,
    #             device=config.DEVICE,
    #             output_dir=config.OUTPUT_DIR
    #         )
    #     else:
    #         print(f"农场 {farm_id} 无数据可用于蒸馏，跳过。")


    # 7. 保存最终的服务器模型
    final_server_model_path = os.path.join(config.OUTPUT_DIR, 'server_final_aggregated_model.pth')
    torch.save(server_global_model.state_dict(), final_server_model_path)
    print(f"\n最终服务器模型已保存至: {final_server_model_path}")
    print(f"农场内联邦学习准确率日志已保存至: {log_file_path}")


if __name__ == '__main__':
    main()