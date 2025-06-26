# run_fl.py

import torch
import os
import numpy as np
import copy
import collections

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model
from src.federated import farm_unit_update, aggregate_models, calculate_wasserstein_distance
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history
from src.rl_selector import UCB1Selector # 导入RL选择器

def main():
    """主执行函数"""
    print("--- 多农场联邦迁移学习与模型蒸馏项目启动 (V2: 包含RL与高级评估) ---")
    print(f"使用设备: {config.DEVICE}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. 加载数据，并创建全局验证集
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

    # 2. 初始化服务器模型和RL选择器
    print("\n--- 步骤2: 初始化服务器模型与RL选择器 ---")
    server_global_model = build_model(
        num_classes=config.NUM_CLASSES_PLANTVILLAGE,
        pretrained_path=config.INITIAL_SOURCE_MODEL_PATH
    ).to(config.DEVICE)
    
    rl_selector = None
    if config.USE_RL_FARM_SELECTION:
        farm_ids = list(all_farms_data_loaders.keys())
        rl_selector = UCB1Selector(
            num_farms=len(farm_ids),
            farm_ids=farm_ids,
            exploration_factor=config.RL_EXPLORATION_FACTOR
        )
        print("强化学习(UCB1)农场选择器已启用。")

    server_history = {'round': [], 'loss': [], 'accuracy': [], 'f1_score': []}
    last_round_acc = 0.0

    # --- 服务器聚合轮次循环 ---
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")
        
        participating_farms = list(all_farms_data_loaders.keys())

        # 2.1 (可选) 使用RL选择参与本轮的农场
        if config.USE_RL_FARM_SELECTION and rl_selector:
            participating_farms = rl_selector.select_farms(config.FARMS_PER_SERVER_ROUND)
        
        farm_final_models = {} # 存储本轮各农场训练好的模型对象

        # --- 遍历选中的农场 ---
        for farm_id in participating_farms:
            farm_data = all_farms_data_loaders.get(farm_id)
            if not farm_data:
                print(f"警告: 找不到农场 {farm_id} 的数据，跳过。")
                continue
                
            print(f"\n\n  J{'='*8} 开始处理农场: {farm_id} J{'='*8}")
            farm_num_classes = farm_data["num_classes"]
            
            # 3. 为当前农场构建模型，并从服务器模型加载骨干网络权重
            farm_global_model = build_model(
                num_classes=farm_num_classes,
                pretrained_path=None # 我们将手动加载权重
            ).to(config.DEVICE)
            
            # 从服务器模型state_dict中拷贝权重，但排除FC层以避免维度不匹配
            server_state_dict = server_global_model.state_dict()
            farm_state_dict = farm_global_model.state_dict()
            
            # 创建一个新的字典，只包含服务器模型中与农场模型骨干网络匹配的权重
            backbone_state_dict = {
                k: v for k, v in server_state_dict.items()
                if k in farm_state_dict and 'fc' not in k
            }
            
            # 加载骨干网络权重
            farm_global_model.load_state_dict(backbone_state_dict, strict=False)
            print(f"  农场 {farm_id} 模型已从当前服务器模型加载骨干网络权重。FC层保持随机初始化。")
            
            # --- 农场内联邦学习循环 ---
            print(f"  --- 开始农场 {farm_id} 内部联邦学习 (FTL) ---")
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                print(f"    回合 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} (农场 {farm_id})")
                
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if len(loader.dataset) > 0]
                if not active_unit_indices:
                    print(f"    农场 {farm_id} 没有活跃的计算单元，跳过本轮农场FL。")
                    break
                
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0:
                     print(f"    农场 {farm_id} 可选单元不足，跳过本轮农场FL。")
                     break

                selected_unit_indices_local = np.random.choice(
                    active_unit_indices, num_units_to_select, replace=False
                )

                unit_model_updates = []
                for unit_idx_local in selected_unit_indices_local:
                    unit_model_copy = copy.deepcopy(farm_global_model).to(config.DEVICE)
                    unit_weights = farm_unit_update(
                        model=unit_model_copy,
                        train_loader=farm_data["unit_loaders"][unit_idx_local],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx_local
                    )
                    unit_model_updates.append(unit_weights)
                
                if unit_model_updates:
                    aggregated_farm_weights = aggregate_models(unit_model_updates, farm_id_for_log=f"农场{farm_id}内部")
                    if aggregated_farm_weights:
                        farm_global_model.load_state_dict(aggregated_farm_weights)
            
            # 存储并保存最终的农场模型
            print(f"  --- 农场 {farm_id} 内部联邦学习完成 ---")
            farm_final_models[farm_id] = copy.deepcopy(farm_global_model)
            final_farm_model_path = os.path.join(config.OUTPUT_DIR, f'farm_{farm_id}_final_ftl_model.pth')
            torch.save(farm_global_model.state_dict(), final_farm_model_path)
            print(f"  农场 {farm_id} 微调后的教师模型已保存至: {final_farm_model_path}")
            
            # 计算Wasserstein距离
            initial_farm_model_fc_state = build_model(num_classes=farm_num_classes).state_dict()
            dist = calculate_wasserstein_distance(farm_global_model.state_dict(), initial_farm_model_fc_state, layer_prefix='fc')
            if dist != -1:
                print(f"  [分析] 农场 {farm_id} 训练后FC层与初始随机FC层的Wasserstein距离: {dist:.4f}")

        # 3. 服务器聚合 (只聚合骨干网络)
        print(f"\n--- 服务器全局轮次 {server_round_idx + 1}: 准备聚合来自 {list(farm_final_models.keys())} 的模型 ---")
        
        backbone_weights_list = []
        for farm_model in farm_final_models.values():
            backbone_state_dict = {k: v for k, v in farm_model.state_dict().items() if 'fc' not in k}
            backbone_weights_list.append(backbone_state_dict)

        if backbone_weights_list:
            aggregated_backbone_weights = aggregate_models(backbone_weights_list, "服务器骨干网络")
            if aggregated_backbone_weights:
                server_model_state_dict = server_global_model.state_dict()
                server_model_state_dict.update(aggregated_backbone_weights)
                server_global_model.load_state_dict(server_model_state_dict)
                print("服务器全局模型的骨干网络已更新。")
                
                # 评估更新后的服务器模型
                if global_val_loader:
                    metrics = evaluate_model(server_global_model, global_val_loader, config.DEVICE, context="服务器聚合后评估")
                    print(f"  服务器模型在全局验证集上: Loss={metrics['loss']:.4f}, Acc={metrics['accuracy']:.2f}%, F1={metrics['f1_score']:.4f}")
                    server_history['round'].append(server_round_idx + 1)
                    server_history['loss'].append(metrics['loss'])
                    server_history['accuracy'].append(metrics['accuracy'])
                    server_history['f1_score'].append(metrics['f1_score'])

                    # 为RL选择器计算奖励并更新
                    if config.USE_RL_FARM_SELECTION and rl_selector:
                        # 奖励 = 本轮准确率提升值
                        reward = metrics['accuracy'] - last_round_acc
                        rl_selector.update(list(participating_farms), reward)
                        last_round_acc = metrics['accuracy']
            else:
                print("服务器聚合失败，模型未更新。")
        else:
            print("没有农场模型可供聚合。")

    print("\n\n--- 顶级联邦学习流程完成 ---")
    plot_server_fl_history(server_history, config.OUTPUT_DIR)
    
    # 4. (可选) 对所有训练好的农场模型进行蒸馏
    print("\n--- 注意: 本次运行执行蒸馏步骤---")
    for farm_id, teacher_model in farm_final_models.items():
        print(f"\n--- 开始为农场 {farm_id} 进行模型蒸馏 ---")
        farm_data = all_farms_data_loaders[farm_id]
        distill_train_loader_for_farm = farm_data['val_loader'] # 使用验证集进行蒸馏
        if distill_train_loader_for_farm and len(distill_train_loader_for_farm.dataset) > 0:
            distill_model(
                teacher_model=teacher_model,
                farm_id=farm_id,
                num_farm_classes=farm_data['num_classes'],
                distill_train_loader=distill_train_loader_for_farm,
                distill_val_loader=farm_data['val_loader'],
                epochs=config.DISTILLATION_EPOCHS,
                lr=config.LEARNING_RATE_DISTILL,
                temperature=config.TEMPERATURE,
                alpha=config.ALPHA_DISTILLATION,
                device=config.DEVICE,
                output_dir=config.OUTPUT_DIR
            )
        else:
            print(f"农场 {farm_id} 无数据可用于蒸馏，跳过。")


    # 5. 保存最终的服务器模型
    final_server_model_path = os.path.join(config.OUTPUT_DIR, 'server_final_aggregated_model.pth')
    torch.save(server_global_model.state_dict(), final_server_model_path)
    print(f"\n最终服务器模型已保存至: {final_server_model_path}")


if __name__ == '__main__':
    main()