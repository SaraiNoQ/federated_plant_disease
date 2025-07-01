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
from src.knowledge_base import KnowledgeBase
from src.routing_agent import RoutingAgent
from src.fusion import feddf_fusion # 导入FedDF

def main():
    """主执行函数 - 现在只负责生成和保存每轮的局部模型"""
    print("--- 多农场联邦学习模型生成程序 (V6: 联邦知识路由与融合框架启动) ---")
    print(f"使用设备: {config.DEVICE}")

    # 1. 初始化
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR, farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=config.CLIENT_UNITS_PER_FARM, batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS, create_global_val_set=True,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )
    farm_ids = list(all_farms_data_loaders.keys())
    
    knowledge_base = KnowledgeBase(config.OUTPUT_DIR)
    routing_agent = RoutingAgent(farm_ids)
    
    # 全局服务器模型，用于FedDF或作为初始模型
    # 注意：这个模型的输出头必须是全局的类别数
    server_global_model = build_model(
        num_classes=config.NUM_CLASSES_PLANTVILLAGE,
        pretrained_path=config.INITIAL_SOURCE_MODEL_PATH
    ).to(config.DEVICE)
    
    # 记录每个农场上一轮的性能，用于计算奖励
    last_farm_accuracies = {fid: 0.0 for fid in farm_ids}

    # 2. 主循环
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 全局知识路由轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")

        teacher_assignments = {}
        # a. 服务器为每个农场做路由决策
        if len(knowledge_base) > 0: # 如果知识库非空
            print("--- 慢尺度RL: 正在进行知识路由决策 ---")
            for farm_id in farm_ids:
                teacher_farm_id = routing_agent.choose_action(farm_id)
                teacher_assignments[farm_id] = teacher_farm_id
                print(f"  - 为 {farm_id} 分配的导师是: {teacher_farm_id}")
        else:
            print("--- 知识库为空，所有农场从初始全局模型开始 ---")

        newly_trained_models = {}
        # b. 客户端接收导师模型并进行本地训练
        for farm_id in farm_ids:
            farm_data = all_farms_data_loaders[farm_id]
            print(f"\n  --- 开始处理农场: {farm_id} ---")

            # 创建本地模型
            local_model = build_model(num_classes=farm_data["num_classes"]).to(config.DEVICE)

            # 加载导师模型的backbone
            teacher_id = teacher_assignments.get(farm_id)
            if teacher_id:
                teacher_state_dict = knowledge_base.get_model(teacher_id)
            else: # 第一轮
                teacher_state_dict = server_global_model.state_dict()
            
            backbone_state_dict = {k: v for k, v in teacher_state_dict.items() if k in local_model.state_dict() and 'fc' not in k}
            local_model.load_state_dict(backbone_state_dict, strict=False)
            print(f"    已从导师 '{teacher_id or 'Initial'}' 加载backbone。")

            # --- Intra-Farm FL Loop (这部分逻辑不变) ---
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if len(loader.dataset) > 0]
                if not active_unit_indices: break
                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                if num_units_to_select == 0: break
                selected_unit_indices = np.random.choice(active_unit_indices, num_units_to_select, replace=False)
                farm_round_start_model = copy.deepcopy(local_model).to(config.DEVICE)
                unit_model_updates = [
                    farm_unit_update_fedprox(
                        model=copy.deepcopy(farm_round_start_model),
                        global_model=farm_round_start_model,
                        train_loader=farm_data["unit_loaders"][unit_idx],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx,
                        mu=config.FEDPROX_MU
                    ) for unit_idx in selected_unit_indices
                ]
                # ...
            # 训练完成后，得到最终的本地模型
            newly_trained_models[farm_id] = local_model

        # c. 服务器收集新模型，计算奖励，并更新知识库和RL Agent
        print("\n--- 服务器收集结果并更新策略 ---")
        for farm_id, model in newly_trained_models.items():
            # 评估性能
            metrics = evaluate_model(model, all_farms_data_loaders[farm_id]['val_loader'], config.DEVICE)
            current_acc = metrics['accuracy']
            print(f"  - {farm_id} 本轮训练后本地准确率: {current_acc:.2f}%")
            
            # 计算奖励
            reward = current_acc - last_farm_accuracies[farm_id]
            
            # 更新RL Agent
            teacher_id = teacher_assignments.get(farm_id)
            if teacher_id: # 只有在有决策发生时才更新
                routing_agent.update(farm_id, teacher_id, reward)
                print(f"    - RoutingAgent更新: C={farm_id[-1]}, T={teacher_id[-1]}, R={reward:.2f}")

            # 更新知识库
            knowledge_base.update(farm_id, model.state_dict())
            
            # 更新上一轮性能记录
            last_farm_accuracies[farm_id] = current_acc

        # 打印Q-table，观察策略学习情况
        routing_agent.print_q_table()

        # d. (可选但推荐) 使用FedDF融合知识库中的模型，以评估全局性能
        if global_val_loader:
            print("\n--- 正在使用FedDF融合知识，以评估全局性能 ---")
            
            # FedDF需要完整的模型对象，而不仅仅是state_dict
            teacher_model_objects = {}
            for fid in knowledge_base.get_all_farm_ids():
                num_local_classes = len(all_farms_data_loaders[fid]['class_names'])
                teacher = build_model(num_classes=num_local_classes).to(config.DEVICE)
                teacher.load_state_dict(knowledge_base.get_model(fid))
                
                # 为了让所有教师模型都能对全局数据进行推理，我们需要一个hack
                # 将它们的本地化fc层替换为能输出全局类别数的fc层
                # 这样做是为了在FedDF中，所有教师能对同一批数据输出维度一致的logits
                num_backbone_features = teacher.fc.in_features
                teacher.fc = nn.Linear(num_backbone_features, config.NUM_CLASSES_PLANTVILLAGE).to(config.DEVICE)
                teacher_model_objects[fid] = teacher
                
            if teacher_model_objects:
                # 使用全局验证集作为公共代理数据集来驱动蒸馏
                fused_server_model = feddf_fusion(
                    teacher_models=teacher_model_objects,
                    server_model=server_global_model,
                    public_dataloader=global_val_loader,
                    device=config.DEVICE,
                    epochs=5 # FedDF的蒸馏轮数
                )
                
                # 评估融合后的全局模型
                global_metrics = evaluate_model(fused_server_model, global_val_loader, config.DEVICE)
                print(f"  ** 融合后的全局模型在全局验证集上表现: Acc = {global_metrics['accuracy']:.2f}% **")
                
                # 更新下一轮的 server_global_model，使其成为一个更强的起点
                server_global_model = fused_server_model
            else:
                print("  知识库中没有足够的模型来进行FedDF融合。")

    print("\n--- 联邦知识路由流程完成 ---")
    # 保存最终的、经过多轮蒸馏融合的全局模型
    final_model_path = os.path.join(config.OUTPUT_DIR, 'fused_global_model_final.pth')
    torch.save(server_global_model.state_dict(), final_model_path)
    print(f"最终的融合全局模型已保存至: {final_model_path}")

if __name__ == '__main__':
    main()