# run_fl.py (版本：时延感知RL + 联邦蒸馏)

import torch
import os
import numpy as np
import copy
import time

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model, build_student_model
from src.federated import farm_unit_update_fedprox, aggregate_models, distill_unit_update
from src.utils import evaluate_model
from src.knowledge_base import KnowledgeBase
from src.routing_agent import RoutingAgent


# ### 新增：一个辅助函数用于模型冻结 ###
def freeze_layers(model, freeze_level: float):
    """
    冻结模型的一部分层。

    Args:
        model (nn.Module): The model to be modified.
        freeze_level (float): 0.0 to 1.0. 0.0 means no layers are frozen.
                              1.0 means all layers except the final classifier are frozen.
    """
    if freeze_level <= 0:
        print("    冻结级别为0，所有层都可训练。")
        return model

    # 将模型的所有参数转换为一个列表
    parameters = list(model.parameters())

    # 确定要冻结的参数数量
    # 我们总是保留最后的分类头可训练
    # ResNet的分类头是'fc'，2个参数(weight, bias)
    # MobileNet/EfficientNet是'classifier'里的最后一个线性层，也是2个参数
    num_trainable_classifier = 2
    total_params = len(parameters)
    num_to_freeze = int((total_params - num_trainable_classifier) * freeze_level)

    print(f"    模型总参数层数: {total_params}。冻结级别: {freeze_level}。")
    print(f"    将冻结前 {num_to_freeze} 层参数。")

    # 冻结指定数量的层
    for i, param in enumerate(parameters):
        if i < num_to_freeze:
            param.requires_grad = False
        else:
            param.requires_grad = True

    # 验证冻结结果
    # trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f"    冻结后，可训练参数数量: {trainable_params}")

    return model

def main():
    """
    主执行函数，集成了：
    1. 时延感知的强化学习知识路由（农场间）。
    2. 农场内部的联邦学习，训练专家模型。
    3. 农场内部的联邦蒸馏，产出轻量级模型。
    """
    print("--- 层级冻结FL + 蒸馏 + 量化框架 ---")
    print(f"使用设备: {config.DEVICE}")

    # 1. 初始化
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        client_units_per_farm=config.CLIENT_UNITS_PER_FARM,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=config.CREATE_GLOBAL_VALIDATION_SET,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )
    farm_ids = list(all_farms_data_loaders.keys())

    # 知识库，用于存储每个农场的专家模型
    knowledge_base = KnowledgeBase(config.OUTPUT_DIR)
    # 强化学习路由代理
    routing_agent = RoutingAgent(farm_ids)

    # 准备一个初始的、在ImageNet上预训练的模型，仅用于第一轮的冷启动
    initial_source_model = build_model(
        num_classes=config.NUM_CLASSES_PLANTVILLAGE,  # 类别数在这里不重要，因为我们只用backbone
        use_pretrained_weights=True
    ).to(config.DEVICE)
    initial_source_model_state = initial_source_model.state_dict()

    # 记录每个农场上一轮的性能，用于计算奖励
    last_farm_performance = {
        fid: {"accuracy": 0.0, "latency": 0.0} for fid in farm_ids
    }

    # 定义复合奖励函数的权重
    W_ACCURACY = 1.0  # 对每1%的准确率提升，奖励1.0分
    W_LATENCY = 0.05  # 对每1秒的训练时延，惩罚0.05分

    # 2. 主循环：多轮知识路由与本地优化
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'=' * 10} 全局知识路由轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'=' * 10}")

        teacher_assignments = {}
        # a. 服务器为每个农场做路由决策 (如果知识库非空)
        if len(knowledge_base) > 0:
            print("--- 慢尺度RL: 正在进行知识路由决策 ---")
            for farm_id in farm_ids:
                teacher_farm_id = routing_agent.choose_action(farm_id)
                teacher_assignments[farm_id] = teacher_farm_id
                print(f"  - 为农场 {farm_id} 分配的导师来自: {teacher_farm_id}")
        else:
            print("--- 知识库为空，所有农场从初始ImageNet模型开始 ---")

        newly_trained_results = {}

        # b. 各个农场并行进行内部联邦学习和蒸馏
        for farm_id in farm_ids:
            farm_data = all_farms_data_loaders[farm_id]
            print(f"\n  --- 开始处理农场: {farm_id} ---")

            # 记录开始时间
            farm_start_time = time.time()

            # 1. 创建本地模型结构 (分类头匹配该农场的类别数)
            local_model = build_model(
                num_classes=farm_data["num_classes"],
                use_pretrained_weights=False
            ).to(config.DEVICE)

            # ### 新增：层级冻结 ###
            # 假设我们在config中定义了FREEZE_LEVEL

            freeze_level = getattr(config, 'FREEZE_LEVEL', 0.0)  # 如果未定义则不冻结
            local_model = freeze_layers(local_model, freeze_level)

            # 2. 加载“导师模型”的骨干网络 (backbone) 权重
            teacher_id = teacher_assignments.get(farm_id)
            if teacher_id:
                teacher_state_dict = knowledge_base.get_model(teacher_id)
                source_log = f"导师 '{teacher_id}'"
            else:  # 仅在第一轮发生
                teacher_state_dict = initial_source_model_state
                source_log = "初始ImageNet模型"

            backbone_state_dict = {
                k: v for k, v in teacher_state_dict.items()
                if 'fc' not in k and 'classifier' not in k
            }
            local_model.load_state_dict(backbone_state_dict, strict=False)
            print(f"    已从 {source_log} 加载骨干网络。")

            # --- 3. 农场内部的联邦学习循环，训练专家模型 ---
            print(f"    开始进行 {config.FARM_FL_ROUNDS} 轮内部联邦学习 (训练专家模型)...")
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                active_unit_indices = [i for i, loader in enumerate(farm_data["unit_loaders"]) if
                                       len(loader.dataset) > 0]
                if not active_unit_indices:
                    print("    农场内没有可用的客户端，跳过内部FL。")
                    break

                num_units_to_select = min(config.UNITS_PER_FARM_ROUND, len(active_unit_indices))
                selected_unit_indices = np.random.choice(active_unit_indices, num_units_to_select, replace=False)

                unit_model_updates = [
                    farm_unit_update_fedprox(
                        model=copy.deepcopy(local_model),
                        global_model=local_model,
                        train_loader=farm_data["unit_loaders"][unit_idx],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_FTL,
                        device=config.DEVICE,
                        farm_id=farm_id,
                        unit_id=unit_idx,
                        mu=config.FEDPROX_MU
                    ) for unit_idx in selected_unit_indices
                ]

                if unit_model_updates:
                    aggregated_weights = aggregate_models(unit_model_updates, f"Farm {farm_id} internal")
                    if aggregated_weights:
                        local_model.load_state_dict(aggregated_weights)

                if (farm_fl_round + 1) % 5 == 0 or farm_fl_round == config.FARM_FL_ROUNDS - 1:
                    temp_metrics = evaluate_model(local_model, farm_data["val_loader"], config.DEVICE)
                    print(
                        f"    - 内部FL轮次 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} | 专家模型精度: {temp_metrics['accuracy']:.2f}%")

            # --- 4. 本地专家模型训练结束，评估最终模型 ---
            final_expert_metrics = evaluate_model(local_model, farm_data["val_loader"], config.DEVICE)
            print(f"    >>> 专家模型训练完成! 最终本地精度: {final_expert_metrics['accuracy']:.2f}%")

            # --- 5. 联邦蒸馏 (Federated Distillation) ---
            # 这里的触发条件可以根据你的需求调整，例如达到某个精度阈值
            if final_expert_metrics['accuracy'] >= 80.0:  # 假设专家模型精度超过95%就进行蒸馏
                print(f"    --- 专家模型性能达标，开始在农场 {farm_id} 内部进行联邦蒸馏 ---")

                # a. 初始化共享的轻量级学生模型
                shared_student_model = build_student_model(
                    num_classes=farm_data["num_classes"],
                    architecture=config.DISTILL_MODEL_ARCH  # 或其他轻量级模型
                ).to(config.DEVICE)

                # b. 进行多轮联邦蒸馏
                num_fd_rounds = config.DISTILL_ROUNDS  # 协同蒸馏的轮数
                for fd_round in range(num_fd_rounds):
                    distill_clients = [i for i, loader in enumerate(farm_data["unit_loaders"]) if
                                       len(loader.dataset) > 0]
                    student_updates = []

                    for unit_idx in distill_clients:
                        # 教师是刚刚训练好的、整个农场的专家模型 (local_model)
                        # 学生是共享的轻量级模型 (shared_student_model)
                        client_distill_loader = farm_data["unit_loaders"][unit_idx]

                        updated_student_state = distill_unit_update(
                            teacher_model=local_model,
                            student_model=copy.deepcopy(shared_student_model),
                            train_loader=client_distill_loader,
                            epochs=config.EPOCHS_PER_UNIT,
                            lr=config.LEARNING_RATE_DISTILL,
                            temperature=config.TEMPERATURE,
                            alpha=config.ALPHA_DISTILLATION,
                            device=config.DEVICE
                        )
                        student_updates.append(updated_student_state)

                    if student_updates:
                        aggregated_student_weights = aggregate_models(student_updates, f"Farm {farm_id} student agg")
                        shared_student_model.load_state_dict(aggregated_student_weights)

                    student_metrics = evaluate_model(shared_student_model, farm_data["val_loader"], config.DEVICE)
                    print(
                        f"      FD Round {fd_round + 1}/{num_fd_rounds} | 学生模型精度: {student_metrics['accuracy']:.2f}%")

                # c. 保存最终的、经过联邦蒸馏的轻量级学生模型
                student_model_save_path = os.path.join(config.OUTPUT_DIR,
                                                       f'distilled_student_model_farm_{farm_id}_round_{server_round_idx + 1}.pth')
                torch.save(shared_student_model.state_dict(), student_model_save_path)

                student_fp32_metrics_before_quant = evaluate_model(
                    shared_student_model,
                    farm_data["val_loader"],
                    config.DEVICE
                )
                print(f"    --- 联邦蒸馏完成，FP32学生模型已保存至: {student_model_save_path}，模型基准精度: {student_fp32_metrics_before_quant['accuracy']:.2f}% ---")

                # --- 阶段三：知识部署 (Post-Training Quantization) ---
                print(f"    --- 开始对学生模型进行训练后量化 (PTQ) ---")

                # 2. 准备用于量化的模型副本
                #    `student_to_quantize` 在这里被定义。
                #    它是 `shared_student_model` 的一个深拷贝，并被移动到CPU。
                student_to_quantize = copy.deepcopy(shared_student_model).to('cpu')
                student_to_quantize.eval()

                # 3. 准备用于校准和评估的CPU数据加载器
                #    `cpu_val_loader` 在这里被定义。
                #    它使用和GPU验证加载器相同的底层数据集，但确保数据在CPU上。
                cpu_val_loader = torch.utils.data.DataLoader(
                    farm_data["val_loader"].dataset,
                    batch_size=config.BATCH_SIZE,
                    shuffle=False,  # 验证时不需要打乱
                    num_workers=config.NUM_WORKERS
                )

                # 4. 配置量化器
                student_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')

                # 5. 准备量化
                torch.quantization.prepare(student_to_quantize, inplace=True)

                # 6. 校准
                print("      正在使用验证数据进行校准...")
                with torch.no_grad():
                    for data, _ in cpu_val_loader:  # 使用 cpu_val_loader
                        student_to_quantize(data)  # 使用 student_to_quantize
                        break

                # 7. 转换模型
                torch.quantization.convert(student_to_quantize, inplace=True)
                print("      量化完成！")

                # 8. 评估量化后的模型
                print("      正在评估INT8量化模型的性能...")
                quantized_metrics = evaluate_model(
                    student_to_quantize,  # 使用已量化的模型
                    cpu_val_loader,  # 使用CPU数据加载器
                    torch.device('cpu'),  # 在CPU上评估
                    context="INT8 Quantized Model"
                )

                # 9. 打印对比结果
                print("\n    ----------------------------------------------------")
                print(f"    [性能对比] 农场 {farm_id}:")
                print(f"    - FP32 学生模型精度: {student_fp32_metrics_before_quant['accuracy']:.2f}%")
                print(f"    - INT8 量化模型精度: {quantized_metrics['accuracy']:.2f}%")
                accuracy_drop = student_fp32_metrics_before_quant['accuracy'] - quantized_metrics['accuracy']
                print(f"    - 精度下降: {accuracy_drop:.2f}%")
                print("    ----------------------------------------------------")

                # 6. 保存最终的INT8模型
                quantized_model_path = os.path.join(config.OUTPUT_DIR,
                                                    f'quantized_student_model_farm_{farm_id}_round_{server_round_idx + 1}.pth')
                torch.save(student_to_quantize.state_dict(), quantized_model_path)
                print(f"    最终可部署的INT8学生模型已保存至: {quantized_model_path}")

            # 记录结束时间并计算时延 (包括了FL和FD的时间)
            farm_end_time = time.time()
            latency = farm_end_time - farm_start_time
            print(f"    农场 {farm_id} 本轮训练总耗时 (时延): {latency:.2f} 秒。")

            # 存储所有结果，注意这里用于RL奖励的是专家模型的性能
            newly_trained_results[farm_id] = {
                "expert_model": local_model,
                "metrics": final_expert_metrics,
                "latency": latency
            }

        # c. 服务器收集结果，计算复合奖励，并更新所有组件
        print("\n--- 服务器收集结果并更新时延感知路由策略 ---")
        knowledge_updated = False
        for farm_id, results in newly_trained_results.items():
            expert_model = results["expert_model"]
            current_metrics = results["metrics"]
            current_latency = results["latency"]

            current_acc = current_metrics['accuracy']
            last_acc = last_farm_performance[farm_id]['accuracy']

            print(f"  - 收到来自 {farm_id} 的新专家模型。 Acc: {current_acc:.2f}%, Latency: {current_latency:.2f}s")

            # 计算复合奖励
            accuracy_gain = current_acc - last_acc
            reward = (W_ACCURACY * accuracy_gain) - (W_LATENCY * current_latency)

            # 更新RL Agent的Q-table
            teacher_id = teacher_assignments.get(farm_id)
            if teacher_id:
                routing_agent.update(farm_id, teacher_id, reward)
                print(
                    f"    - RoutingAgent更新: C={farm_id[-1]}, T={teacher_id[-1]}, AccGain={accuracy_gain:.2f}, Latency={current_latency:.2f}s => Reward={reward:.2f}")

            # 将训练好的专家模型存入知识库
            knowledge_base.update(farm_id, expert_model.state_dict())
            print(f"    - {farm_id} 的专家模型已更新至知识库。")

            # 更新性能记录
            last_farm_performance[farm_id] = {
                "accuracy": current_acc,
                "latency": current_latency
            }

            knowledge_base.update(farm_id, expert_model.state_dict())
            knowledge_updated = True

        # 如果知识库被更新（有新农场贡献了模型），则扩展动作空间
        if knowledge_updated:
            routing_agent.update_action_space()

        routing_agent.print_q_table()

    print("\n--- 联邦知识路由与赋能流程完成 ---")
    print("知识库中包含了每个农场训练出的最新专家模型。")
    print(f"模型保存在: {os.path.join(config.OUTPUT_DIR, 'knowledge_base')}")
    print("每个农场的轻量级蒸馏模型已分别保存在输出目录中。")


if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    main()