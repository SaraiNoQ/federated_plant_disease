# run_fl.py (版本：时延感知RL + 联邦蒸馏)
import torch
import os
import numpy as np
import copy
import time
from torch.utils.data import DataLoader

# 导入我们自己的模块
import config
from src.data_loader import get_farm_dataloaders
from src.models import build_model, build_student_model
from src.federated import local_distill_update, aggregate_models
from src.utils import evaluate_model, print_metrics, quantize_and_evaluate_model
from src.knowledge_base import KnowledgeBase
from src.rl_agents import MetaControllerA2C, LocalExecutorUCB


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
    active_farm_ids = list(config.INITIAL_FARMS)

    # 知识库，用于存储每个农场的专家模型
    knowledge_base = KnowledgeBase(config.OUTPUT_DIR)

    # --- HRL 初始化 ---
    # 高层RL Agent
    # 状态维度可以简化，例如: [avg_acc, avg_f1, avg_loss, diversity_metric]
    state_dim = 4
    meta_controller = MetaControllerA2C(
        num_farms=config.NUM_FARMS,
        state_dim=state_dim,
        device=config.DEVICE
    )

    # 为每个农场创建低层RL Agent和本地教师模型
    farm_executors = {}
    local_teachers = {}
    for farm_id in config.FARM_CLASS_ALLOCATION.keys():
        num_clients = config.CLIENT_UNITS_PER_FARM
        farm_executors[farm_id] = LocalExecutorUCB(num_clients, config.LOCAL_RL_EXPLORATION)
        # 初始化每个客户端私有的教师模型
        farm_data = all_farms_data_loaders[farm_id]
        client_teachers = [
            build_model(num_classes=farm_data["num_classes"], use_pretrained_weights=True).to(config.DEVICE)
            for _ in range(num_clients)
        ]
        local_teachers[farm_id] = client_teachers

    # 全局状态追踪
    last_global_efficiency = 0.0
    last_global_diversity = 0.0

    # 2. 主循环：多轮知识路由与本地优化
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'=' * 10} 全局知识路由轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'=' * 10}")

        # --- 高层决策阶段 ---
        # 1. 构建高层状态
        # (这是一个简化的例子，实际可以更复杂)
        all_metrics = [m['metrics'] for m in knowledge_base.get_all_models()]
        if all_metrics:
            avg_acc = np.mean([m['accuracy'] for m in all_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
            avg_loss = np.mean([m['loss'] for m in all_metrics])
            # 简化多样性：模型参数的方差
            all_params = [torch.nn.utils.parameters_to_vector(m['state_dict'].values()) for m in
                          knowledge_base.get_all_models()]
            diversity = torch.stack(all_params).var().item() if len(all_params) > 1 else 0.0
        else:
            avg_acc, avg_f1, avg_loss, diversity = 0, 0, 0, 0

        current_state = [avg_acc / 100, avg_f1, avg_loss, diversity]

        # 2. 为每个活跃农场生成指令
        farm_directives = {}
        print("--- 高层Agent正在发布指令 ---")
        for farm_id in active_farm_ids:
            action = meta_controller.select_action(current_state)
            if action == 0 or len(knowledge_base) == 0:
                directive = {'role': 'EXPLOIT'}
                print(f"  - 指令 to {farm_id}: {directive}")
            else:
                # 选择一个非自身的教师
                teacher_options = [fid for fid in knowledge_base.get_all_farm_ids() if fid != farm_id]
                if not teacher_options:
                    directive = {'role': 'EXPLOIT'}
                else:
                    teacher_id = teacher_options[(action - 1) % len(teacher_options)]
                    directive = {'role': 'TRANSFER_IN', 'source': teacher_id, 'budget': config.TRANSFER_BUDGET}
                print(f"  - 指令 to {farm_id}: {directive}")
            farm_directives[farm_id] = directive

        # --- 低层执行阶段 ---
        round_latencies = []
        newly_trained_student_models = {}

        for farm_id in active_farm_ids:
            print(f"\n  --- 农场 {farm_id} 开始执行指令 (低层任务) ---")
            farm_start_time = time.time()
            farm_data = all_farms_data_loaders[farm_id]
            directive = farm_directives[farm_id]

            # 准备本轮的初始学生模型和教师模型
            farm_student_model = build_student_model(
                num_classes=farm_data["num_classes"],
                architecture=config.DISTILL_MODEL_ARCH
            ).to(config.DEVICE)

            # 如果是TRANSFER_IN，加载教师学生模型
            transfer_teacher_student_model = None
            if directive['role'] == 'TRANSFER_IN':
                teacher_info = knowledge_base.get_model(directive['source'])
                # 注意：知识库现在存的是学生模型
                transfer_teacher_student_model = build_student_model(
                    num_classes=all_farms_data_loaders[directive['source']]["num_classes"],
                    architecture=config.DISTILL_MODEL_ARCH
                ).to(config.DEVICE)
                transfer_teacher_student_model.load_state_dict(teacher_info['state_dict'])

            # 内部FedMD循环
            for farm_fl_round in range(config.FARM_FL_ROUNDS):
                # 低层RL选择客户端
                selected_client_indices = farm_executors[farm_id].select_clients(config.UNITS_PER_FARM_ROUND)

                student_updates = []
                client_rewards = []
                for client_idx in selected_client_indices:
                    # 获取该客户端私有的教师模型
                    private_teacher = local_teachers[farm_id][client_idx]

                    # 客户端本地蒸馏
                    updated_student_dict = local_distill_update(
                        local_teacher_model=private_teacher,
                        student_model_to_train=copy.deepcopy(farm_student_model),
                        global_student_model=farm_student_model,  # 将本轮开始时的学生模型传入
                        prox_mu=config.FEDPROX_MU,  # 使用config中的mu值
                        train_loader=farm_data["unit_loaders"][client_idx],
                        epochs=config.EPOCHS_PER_UNIT,
                        lr=config.LEARNING_RATE_DISTILL,
                        device=config.DEVICE,
                        # HRL参数
                        transfer_teacher_model=transfer_teacher_student_model,
                        transfer_budget=directive.get('budget', 0.0),
                        # 标准蒸馏参数
                        temperature=config.TEMPERATURE,
                        alpha=config.ALPHA_DISTILLATION,
                    )
                    student_updates.append(updated_student_dict)

                    # 计算该客户端的内在奖励 (简化版)
                    temp_student = copy.deepcopy(farm_student_model)
                    temp_student.load_state_dict(updated_student_dict)
                    metrics = evaluate_model(temp_student, farm_data["val_loader"], config.DEVICE)
                    client_rewards.append(metrics['accuracy'])  # 奖励=该更新带来的精度

                # 更新低层RL Agent
                farm_executors[farm_id].update(selected_client_indices, client_rewards)

                # 聚合学生模型
                if student_updates:
                    aggregated_weights = aggregate_models(student_updates, f"Farm {farm_id} Student Models")
                    farm_student_model.load_state_dict(aggregated_weights)

                final_metrics = evaluate_model(farm_student_model, farm_data["val_loader"], config.DEVICE)
                print_metrics(final_metrics, f"    内部FL轮次 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} | 学生模型")

            # 农场本轮任务结束
            farm_end_time = time.time()
            latency = farm_end_time - farm_start_time
            round_latencies.append(latency)

            final_farm_metrics = evaluate_model(farm_student_model, farm_data["val_loader"], config.DEVICE)
            print_metrics(final_farm_metrics, f">>> 农场 {farm_id} 最终学生模型")

            # --- 量化与评估 ---
            quantized_state_dict = quantize_and_evaluate_model(
                farm_student_model, farm_data["val_loader"], config.DEVICE
            )

            # 保存结果，准备更新高层RL
            newly_trained_student_models[farm_id] = {
                'state_dict': farm_student_model.state_dict(),
                'metrics': final_farm_metrics,
                'latency': latency
            }

        # --- 高层奖励计算与更新 ---
        # 1. 更新知识库
        for farm_id, model_info in newly_trained_student_models.items():
            knowledge_base.update(farm_id, model_info)

        # 2. 计算高层奖励
        current_metrics = [m['metrics'] for m in knowledge_base.get_all_models()]
        current_avg_acc = np.mean([m['accuracy'] for m in current_metrics]) if current_metrics else 0
        current_avg_lat = np.mean(round_latencies) if round_latencies else 0

        # 计算效率
        current_efficiency = config.W_ACC_GLOBAL * current_avg_acc - config.W_LAT_GLOBAL * current_avg_lat
        efficiency_gain = current_efficiency - last_global_efficiency

        # 计算多样性
        all_params = [torch.nn.utils.parameters_to_vector(m['state_dict'].values()) for m in
                      knowledge_base.get_all_models()]
        current_diversity = torch.stack(all_params).var().item() if len(all_params) > 1 else 0.0
        diversity_gain = current_diversity - last_global_diversity

        # 最终高层奖励
        meta_reward = config.W_EFFICIENCY_GLOBAL * efficiency_gain + config.W_DIVERSITY_GLOBAL * diversity_gain
        meta_controller.rewards.append(meta_reward)
        print(f"\n--- 高层奖励计算 ---")
        print(f"  效率提升: {efficiency_gain:.2f}, 多样性提升: {diversity_gain:.4f}")
        print(f"  本轮高层总奖励: {meta_reward:.4f}")

        # 3. 更新高层RL Agent
        meta_controller.update()

        # 更新状态
        last_global_efficiency = current_efficiency
        last_global_diversity = current_diversity

    print("\n--- HRL 框架所有轮次执行完毕 ---")


if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    main()