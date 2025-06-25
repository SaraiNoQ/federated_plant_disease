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
from src.federated import farm_unit_update, aggregate_models
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history

def main():
    """主执行函数"""
    print("--- 多农场联邦迁移学习与模型蒸馏项目启动 ---")
    print(f"使用设备: {config.DEVICE}")
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)

    # 1. 加载并划分数据给各个农场和农场内的单元
    print("\n--- 步骤1: 准备各农场数据集 ---")
    all_farms_data_loaders = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=config.CLIENT_UNITS_PER_FARM,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS
    )

    if not all_farms_data_loaders:
        print("错误: 未能加载任何农场数据，程序终止。")
        return

    # 2. 初始化一个全局源模型 (例如，基于ImageNet预训练)
    # 这个模型将被分发给每个农场作为其内部FTL的起点
    print(f"\n--- 步骤2: 初始化全局初始源模型 (来自: {config.INITIAL_SOURCE_MODEL_PATH}) ---")
    # 注意：这个初始模型的类别数是针对PlantVillage总类别数，还是一个通用特征提取器？
    # 如果每个农场FTL时模型fc层会变成农场特定类别数，那么初始模型的fc层可以暂时是全局类别数，
    # 或者在加载后被替换。我们的build_model会处理这个。
    # 为了让农场能基于此进行微调并输出农场特定类别，我们需要在农场层面构建模型时传入农场的类别数。
    # 所以，这里的 `initial_global_source_model` 只是一个权重源。
    # 或者，我们也可以在这里就构建一个num_classes=PLANTVILLAGE_NUM_CLASSES的模型作为起点。
    # 为简单起见，我们先不实例化一个全局模型变量，而是在每个农场需要时基于路径构建。
    # 如果要进行服务器聚合，则需要一个服务器端的全局模型实例。

    # 服务器端的“源模型”，用于聚合农场模型。
    # 它的类别数应该是多少？如果农场模型是特定类别的，直接FedAvg会失败。
    # 策略：服务器聚合后的模型，其目标是成为一个更好的“通用特征提取器”或一个能处理所有已知农场类别的模型。
    # 为简化，如果SERVER_ROUNDS > 0，我们假设服务器聚合的是可以预测全局38类的模型。
    # 这意味着农场在上传模型给服务器前，需要将其模型转换回能输出38类。这是一个复杂步骤。

    # **简化策略调整：**
    # - 每个农场基于 INITIAL_SOURCE_MODEL_PATH (ImageNet预训练) 开始，微调得到一个针对其特定类别的最优农场模型。
    # - 然后对这个农场最优模型进行蒸馏。
    # - 服务器聚合 (SERVER_ROUNDS) 暂时简化为：
    #   如果要做，它应该聚合那些已经被转换回能够预测全局类别（或至少是一个共同超集类别）的农场模型。
    #   或者，服务器聚合的只是农场的骨干网络。
    #   **当前最简化的实现：如果 SERVER_ROUNDS > 1，我们将跳过服务器聚合部分，或只做象征性的（假设模型结构已对齐）。**
    #   **当前代码将主要关注农场内的FTL和后续的蒸馏。**
    #   **如果 SERVER_ROUNDS = 1， 意味着没有服务器级的模型迭代更新，每个农场独立进化然后蒸馏。**

    current_server_source_model_path = config.INITIAL_SOURCE_MODEL_PATH
    server_history = {'round': [], 'loss': [], 'accuracy': []} # 用于记录服务器聚合模型的性能

    # --- 服务器聚合轮次循环 (如果需要多轮服务器与农场间的交互) ---
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")
        
        farm_models_for_server_aggregation = [] # 存储本轮各农场上传的模型权重

        # --- 遍历每个农场 ---
        for farm_id, farm_data in all_farms_data_loaders.items():
            print(f"\n\n  J{'='*8} 开始处理农场: {farm_id} J{'='*8}")
            farm_unit_loaders = farm_data["unit_loaders"]
            farm_val_loader = farm_data["val_loader"]
            farm_num_classes = farm_data["num_classes"]
            # farm_class_names = farm_data["class_names"]

            if not farm_unit_loaders or not any(len(loader.dataset) > 0 for loader in farm_unit_loaders):
                print(f"  农场 {farm_id} 没有计算单元数据，跳过农场内联邦学习。")
                # 即使没有单元数据，如果farm_val_loader有数据，也可以尝试用初始模型直接蒸馏（如果适用）
                # 但通常FTL是蒸馏的前提
                continue
            
            # 3. 为当前农场构建/加载其联邦学习的初始模型
            #    基于 current_server_source_model_path (可能是ImageNet或上一轮服务器聚合的模型)
            #    并调整FC层以适应本农场的类别数。
            print(f"  正在为农场 {farm_id} 构建模型 (类别数: {farm_num_classes}), 基于源: {current_server_source_model_path}")
            farm_global_model = build_model(
                num_classes=farm_num_classes,
                pretrained_path=current_server_source_model_path,
                farm_specific_load=False # 因为源模型是通用的，不是农场特定的
            ).to(config.DEVICE)
            
            # --- 农场内联邦学习循环 ---
            print(f"  --- 开始农场 {farm_id} 内部联邦学习 (FTL) ---")
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                print(f"    回合 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} (农场 {farm_id})")
                
                # 随机选择农场内的计算单元
                # 确保选择的单元有数据
                active_unit_indices = [i for i, loader in enumerate(farm_unit_loaders) if len(loader.dataset) > 0]
                if not active_unit_indices:
                    print(f"    农场 {farm_id} 没有活跃的计算单元，跳过本轮农场FL。")
                    break
                
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0 :
                     print(f"    农场 {farm_id} 可选单元不足，跳过本轮农场FL。")
                     break

                selected_unit_indices_local = np.random.choice(
                    active_unit_indices, num_units_to_select, replace=False
                )
                print(f"    本轮农场 {farm_id} 选中的计算单元: {selected_unit_indices_local}")

                unit_model_updates = []
                for unit_idx_local in selected_unit_indices_local:
                    unit_model_copy = copy.deepcopy(farm_global_model).to(config.DEVICE)
                    
                    # 单元本地训练 (FTL的微调部分)
                    unit_weights = farm_unit_update(
                        model=unit_model_copy,
                        train_loader=farm_unit_loaders[unit_idx_local],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx_local
                    )
                    unit_model_updates.append(unit_weights)
                
                # 聚合农场内单元的模型
                if unit_model_updates:
                    aggregated_farm_weights = aggregate_models(unit_model_updates, farm_id_for_log=f"农场{farm_id}内部")
                    if aggregated_farm_weights:
                        farm_global_model.load_state_dict(aggregated_farm_weights)
                        print(f"    农场 {farm_id} 内部模型聚合完成。")
                    else:
                        print(f"    警告: 农场 {farm_id} 内部聚合失败，模型未更新。")

                # 评估农场聚合模型（可选）
                if farm_val_loader and len(farm_val_loader.dataset) > 0 :
                    farm_loss, farm_accuracy = evaluate_model(farm_global_model, farm_val_loader, config.DEVICE, context=f"农场{farm_id} FTL评估")
                    print(f"    农场 {farm_id} FTL模型在本地验证集: Loss={farm_loss:.4f}, Acc={farm_accuracy:.2f}%")
            
            print(f"  --- 农场 {farm_id} 内部联邦学习完成 ---")
            farm_final_model_path = os.path.join(config.OUTPUT_DIR, f'farm_{farm_id}_final_ftl_model.pth')
            torch.save(farm_global_model.state_dict(), farm_final_model_path)
            print(f"  农场 {farm_id} 微调后的教师模型已保存至: {farm_final_model_path}")

            # (可选) 将此农场模型添加到服务器聚合列表
            # **重要：** 如果要进行服务器FedAvg，farm_global_model的FC层是针对farm_num_classes的。
            # 这与全局38类的初始模型结构不同，直接聚合会失败。
            # 解决方案：
            # 1. 服务器只聚合骨干网络 (忽略FC层)。
            # 2. 农场在上传前，将其模型转换回一个能预测全局类别（或一个标准化的超集类别）的模型。
            # 3. 使用更高级的联邦学习算法，如FedMA，它可以处理异构模型。
            # 为简化，如果 SERVER_ROUNDS > 0, 我们需要一个策略。
            # 暂时假设：如果进行服务器聚合，我们期望农场模型与初始源模型结构一致（即38类）。
            # 这意味着农场的FTL实际上是在微调一个38类的模型，但在计算损失时只关注其拥有的类别。
            # 这个实现会更复杂。

            # **当前简化：** 我们只取农场模型的state_dict，如果服务器聚合那边能处理（例如，只取匹配的层），
            # 或者我们跳过服务器聚合，如果SERVER_ROUNDS <=1。
            if config.SERVER_ROUNDS > 0 : # 仅当需要服务器聚合时才添加
                # farm_models_for_server_aggregation.append(copy.deepcopy(farm_global_model.state_dict()))
                # ^^^ 上面这行会导致聚合问题，因为FC层维度不同。

                # **折中方案/待办事项：** 如果要做服务器聚合，需要一个转换步骤
                # farm_model_for_server = convert_farm_model_to_global_classes(farm_global_model, config.NUM_CLASSES_PLANTVILLAGE)
                # farm_models_for_server_aggregation.append(farm_model_for_server.state_dict())
                print(f"  注意: 农场 {farm_id} 的模型具有 {farm_num_classes} 个输出类别。若要进行服务器FedAvg，需特殊处理。")
                # 暂时不加入，除非我们解决了维度问题。


            # 4. 模型蒸馏 (对每个农场微调好的教师模型进行)
            print(f"  --- 开始为农场 {farm_id} 进行模型蒸馏 ---")
            # 用于蒸馏的数据加载器可以是农场的训练集或验证集的一部分
            # 为简单，我们这里使用农场的验证集作为蒸馏的“无标签”数据源（硬标签也用上）
            # 或者，如果农场训练数据很多，可以用一部分训练数据。
            # 注意：distill_train_loader 的标签应该是农场本地的 [0, num_farm_classes-1]
            # farm_val_loader 的标签已经是本地的了 (由FarmSubsetWrapper处理)
            distill_train_loader_for_farm = farm_val_loader # 使用验证集进行蒸馏
            if not distill_train_loader_for_farm or len(distill_train_loader_for_farm.dataset) == 0:
                # 如果验证集为空，尝试使用该农场第一个单元的训练数据加载器作为蒸馏数据源
                if farm_unit_loaders and len(farm_unit_loaders[0].dataset) > 0:
                    distill_train_loader_for_farm = farm_unit_loaders[0]
                    print(f"  农场 {farm_id} 蒸馏：使用单元0的训练数据进行蒸馏。")
                else:
                    print(f"  警告: 农场 {farm_id} 没有可用于蒸馏的数据。跳过蒸馏。")
                    continue # 跳过此农场的蒸馏

            _ = distill_model(
                teacher_model=farm_global_model, # 使用该农场FTL后的模型作为教师
                farm_id=farm_id,
                num_farm_classes=farm_num_classes,
                distill_train_loader=distill_train_loader_for_farm,
                distill_val_loader=farm_val_loader, # 也用它来评估学生模型
                epochs=config.DISTILLATION_EPOCHS,
                lr=config.LEARNING_RATE_DISTILL,
                temperature=config.TEMPERATURE,
                alpha=config.ALPHA_DISTILLATION,
                device=config.DEVICE,
                output_dir=config.OUTPUT_DIR
            )
            print(f"  --- 农场 {farm_id} 模型蒸馏完成 ---")


        # --- 服务器聚合步骤 (如果 SERVER_ROUNDS > 0 且有模型可聚合) ---
        if config.SERVER_ROUNDS > 0 and server_round_idx < config.SERVER_ROUNDS -1 : #最后一轮不需要再聚合服务器模型了
            if farm_models_for_server_aggregation:
                print(f"\n--- 服务器全局轮次 {server_round_idx + 1}: 准备聚合各农场模型 ---")
                # **这里的聚合需要非常小心，因为农场模型的FC层维度不同！**
                # 一个简单的FedAvg会失败。
                # 你需要实现一个只聚合共同层（骨干网络）的策略，或者农场上传时转换模型。
                # **为使代码能运行，我们暂时跳过这一步，或者只在所有农场类别数相同时才聚合。**
                print("警告: 服务器聚合不同类别数的农场模型功能复杂，当前版本未完全实现此聚合。")
                print("       服务器源模型将不会通过聚合农场模型来更新。")
                # aggregated_server_weights = aggregate_models(farm_models_for_server_aggregation, "服务器全局")
                # if aggregated_server_weights:
                #     # 这里需要一个服务器端的全局模型实例来加载权重
                #     # server_global_model = build_model(config.NUM_CLASSES_PLANTVILLAGE, ...).to(config.DEVICE)
                #     # server_global_model.load_state_dict(aggregated_server_weights)
                #     # current_server_source_model_path = os.path.join(config.OUTPUT_DIR, f'server_aggregated_model_round_{server_round_idx+1}.pth')
                #     # torch.save(server_global_model.state_dict(), current_server_source_model_path)
                #     # print(f"服务器聚合模型已保存并作为下一轮源: {current_server_source_model_path}")

                #     # (可选) 评估服务器聚合模型 - 需要一个统一的全局验证集
                #     # global_val_loss, global_val_acc = evaluate_model(server_global_model, global_unified_val_loader, config.DEVICE)
                #     # server_history['round'].append(server_round_idx+1)
                #     # server_history['loss'].append(global_val_loss)
                #     # server_history['accuracy'].append(global_val_acc)
                # else:
                #     print("服务器聚合失败。")
            else:
                print(f"服务器全局轮次 {server_round_idx + 1}: 没有农场模型可供聚合。")
        elif config.SERVER_ROUNDS > 0 :
             print(f"\n服务器全局轮次 {server_round_idx + 1}: 这是最后一轮服务器轮次，不再进行服务器模型聚合。")


    print("\n\n--- 所有农场处理和蒸馏流程完成 ---")
    if config.SERVER_ROUNDS > 1: # 只有多轮服务器交互时绘图才有意义
        # plot_server_fl_history(server_history, config.OUTPUT_DIR)
        print("服务器聚合历史绘图功能需要实现统一的全局验证集和正确的服务器聚合逻辑。")

if __name__ == '__main__':
    main()