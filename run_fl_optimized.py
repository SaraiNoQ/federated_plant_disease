# run_fl.py (减少显存占用)
import torch
import os
import numpy as np
import copy
import time
from torch.utils.data import DataLoader

# 导入我们自己的模块
import config_optimized as config  # 使用优化后的配置
from src.data_loader import get_farm_dataloaders
from src.models import build_model, build_student_model
from src.federated import local_distill_update, aggregate_models, print_memory_usage, local_teacher_update
from src.utils import evaluate_model, print_metrics, quantize_and_evaluate_model
from src.knowledge_base import KnowledgeBase
from src.rl_agents import MetaControllerA2C, LocalExecutorUCB


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

    return model

def create_model_copy(model, num_classes, device):
    """
    创建模型的浅拷贝，避免深度复制占用过多显存
    """
    model_copy = type(model)(num_classes=num_classes)
    model_copy.load_state_dict(model.state_dict())
    model_copy.to(device)
    return model_copy

def main():
    """
    主执行函数，集成了显存优化措施
    """
    print("--- 优化版：层级冻结FL + 蒸馏 + 量化框架 ---")
    print(f"使用设备: {config.DEVICE}")
    print_memory_usage("程序启动时")

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

    # 农场ID到索引的映射
    farm_ids_list = sorted(list(config.FARM_CLASS_ALLOCATION.keys()))
    farm_to_idx = {fid: i for i, fid in enumerate(farm_ids_list)}

    # --- HRL 初始化 ---
    # 高层RL Agent(4全局 + 8农场特定) - 更新状态维度
    state_dim = 12
    meta_controller = MetaControllerA2C(
        num_farms=config.NUM_FARMS,
        state_dim=state_dim,
        device=config.DEVICE,
        entropy_coeff=0.01 # <--- 引入熵正则化
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

    print_memory_usage("初始化完成后")

    # 创建持久化学生模型字典
    farm_student_models = {}
    for farm_id in config.FARM_CLASS_ALLOCATION.keys():
        farm_data = all_farms_data_loaders[farm_id]
        farm_student_models[farm_id] = build_student_model(
            num_classes=farm_data["num_classes"],
            architecture=config.DISTILL_MODEL_ARCH
        ).to(config.DEVICE)
    print("持久化学生模型字典初始化完成")

    # 2. 主循环：多轮知识路由与本地优化
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'=' * 10} 全局知识路由轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'=' * 10}")
        print_memory_usage(f"服务器轮次 {server_round_idx + 1} 开始前")

        # --- 高层决策阶段 ---
        # 1. 构建高层状态
        # 1. 构建全局上下文
        all_model_info = knowledge_base.get_all_models()
        if all_model_info:
            all_metrics = [info['metrics'] for info in all_model_info]
            avg_acc = np.mean([m['accuracy'] for m in all_metrics])
            avg_f1 = np.mean([m['f1_score'] for m in all_metrics])
            avg_loss = np.mean([m['loss'] for m in all_metrics])
            all_params = [torch.nn.utils.parameters_to_vector(info['state_dict'].values()) for info in all_model_info]
            diversity = torch.stack(all_params).var().item() if len(all_params) > 1 else 0.0
        else:
            avg_acc, avg_f1, avg_loss, diversity = 0, 0, 0, 0
        global_context = [avg_acc/100.0, avg_f1, avg_loss, diversity]

        # 2. 为每个活跃农场生成指令
        farm_directives = {}
        print("--- 高层Agent正在发布指令 ---")

        # 获取当前可用的教师
        available_teachers_ids = knowledge_base.get_all_farm_ids()
        available_teachers_indices = [farm_to_idx[fid] for fid in available_teachers_ids]
        
        # 计算动态探索率：随着轮次增加而减少，但保持最小探索率
        exploration_rate = max(0.15, 0.4 * (1 - server_round_idx / config.SERVER_ROUNDS))
        
        print(f"  可用教师: {available_teachers_ids}")
        print(f"  探索率: {exploration_rate:.3f}")
        
        for farm_id in active_farm_ids:
            # --- 构建该农场的个性化状态 ---
            farm_idx = farm_to_idx[farm_id]
            farm_info = knowledge_base.get_model(farm_id)
            
            # 改进的状态构建：包含更多农场特定信息
            if farm_info:
                metrics = farm_info['metrics']
                # 添加农场类别多样性、历史性能趋势等特征
                farm_classes = config.FARM_CLASS_ALLOCATION[farm_id]
                class_diversity = len(farm_classes) / 10.0  # 归一化
                # 计算性能趋势（更精细）
                performance_trend = min(1.0, max(0.0, (metrics['accuracy'] - 30) / 70.0))  # 30-100映射到0-1
                # 添加类别互补性特征
                farm_class_set = set(farm_classes)
                complementary_score = 0.0
                for teacher_id in available_teachers_ids:
                    if teacher_id != farm_id:
                        teacher_classes = set(config.FARM_CLASS_ALLOCATION[teacher_id])
                        overlap = len(farm_class_set & teacher_classes) / len(farm_class_set | teacher_classes)
                        complementary_score += (1 - overlap)  # 互补性越高，分数越高
                complementary_score = min(1.0, complementary_score / len(available_teachers_ids))
                
                farm_specific_features = [
                    metrics['accuracy']/100, 
                    metrics['f1_score'], 
                    metrics['loss'], 
                    class_diversity,
                    performance_trend,
                    farm_idx / config.NUM_FARMS,  # 农场索引归一化
                    complementary_score,  # 类别互补性
                    len(farm_classes) / 15.0  # 类别数量归一化
                ]
            else: # 新农场
                farm_classes = config.FARM_CLASS_ALLOCATION[farm_id]
                class_diversity = len(farm_classes) / 10.0
                farm_specific_features = [
                    0.0, 0.0, 1.0, class_diversity, 0.0, farm_idx / config.NUM_FARMS, 0.5, len(farm_classes) / 15.0
                ]
            
            # 拼接成最终状态（现在状态维度为12）
            current_state = global_context + farm_specific_features

            # --- 获取决策 ---
            action, log_prob = meta_controller.select_action(
                current_state, 
                farm_idx, 
                available_teachers_indices,
                exploration_rate=exploration_rate
            )
            # --- 存储决策信息 ---
            meta_controller.store_transition(current_state, log_prob)

            # --- 解析动作为指令 ---
            if action == 0 or not available_teachers_ids:
                directive = {'role': 'EXPLOIT'}
            else:
                valid_teacher_ids = [fid for fid in available_teachers_ids if fid != farm_id]
                if not valid_teacher_ids:
                    directive = {'role': 'EXPLOIT'}
                else:
                    teacher_id = valid_teacher_ids[(action - 1) % len(valid_teacher_ids)]
                    directive = {'role': 'TRANSFER_IN', 'source': teacher_id, 'budget': config.TRANSFER_BUDGET}
            print(f"  - 指令 to {farm_id}: {directive} (动作: {action})")
            farm_directives[farm_id] = directive

        # --- 低层执行阶段 ---
        round_latencies = []
        newly_trained_student_models = {}

        for farm_id in active_farm_ids:
            print(f"\n  --- 农场 {farm_id} 开始执行指令 (低层任务) ---")
            farm_start_time = time.time()
            farm_data = all_farms_data_loaders[farm_id]
            directive = farm_directives[farm_id]

            # 从持久化字典中获取学生模型，继承上一轮的状态
            farm_student_model = farm_student_models[farm_id]
            print(f"  使用持久化学生模型作为起点 (农场 {farm_id})")

            # --- 阶段一：教师模型微调 ---
            print(f"  --- 阶段一：教师模型微调 (农场 {farm_id}) ---")
            for client_idx in range(config.CLIENT_UNITS_PER_FARM):
                print(f"    客户端 {client_idx} 教师模型微调开始...")
                teacher_model = local_teachers[farm_id][client_idx]
                
                # 微调教师模型
                updated_teacher_dict = local_teacher_update(
                    teacher_model=teacher_model,
                    train_loader=farm_data["unit_loaders"][client_idx],
                    epochs=config.EPOCHS_PER_UNIT,  # 使用与蒸馏相同的轮数
                    lr=config.LEARNING_RATE_DISTILL,
                    device=config.DEVICE,
                    freeze_level=0.5  # 冻结50%的层
                )
                
                # 更新教师模型状态
                teacher_model.load_state_dict(updated_teacher_dict)
                print(f"    客户端 {client_idx} 教师模型微调完成")
            
            print(f"  --- 阶段一完成：所有教师模型已微调 (农场 {farm_id}) ---")

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

                    # 使用浅拷贝而不是深度拷贝来减少显存占用
                    student_model_copy = create_model_copy(
                        farm_student_model, 
                        farm_data["num_classes"], 
                        config.DEVICE
                    )
                    
                    # 客户端本地蒸馏
                    updated_student_dict = local_distill_update(
                        local_teacher_model=private_teacher,
                        student_model_to_train=student_model_copy,
                        global_student_model=farm_student_model,
                        prox_mu=config.FEDPROX_MU,
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
                    temp_student = create_model_copy(
                        farm_student_model, 
                        farm_data["num_classes"], 
                        config.DEVICE
                    )
                    temp_student.load_state_dict(updated_student_dict)
                    metrics = evaluate_model(temp_student, farm_data["val_loader"], config.DEVICE)
                    client_rewards.append(metrics['accuracy'])

                    # 及时删除临时模型以释放显存
                    del student_model_copy, temp_student
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()

                # 更新低层RL Agent
                farm_executors[farm_id].update(selected_client_indices, client_rewards)

                # 聚合学生模型
                if student_updates:
                    aggregated_weights = aggregate_models(student_updates, f"Farm {farm_id} Student Models")
                    farm_student_model.load_state_dict(aggregated_weights)

                # 评估学生模型
                student_metrics = evaluate_model(farm_student_model, farm_data["val_loader"], config.DEVICE)
                
                # 评估教师模型（使用第一个客户端的教师模型作为代表）
                teacher_metrics = evaluate_model(local_teachers[farm_id][0], farm_data["val_loader"], config.DEVICE)
                
                print(f"    内部FL轮次 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} | 学生模型 (MobileNet_v2):")
                print_metrics(student_metrics, "      ")
                print(f"    内部FL轮次 {farm_fl_round + 1}/{config.FARM_FL_ROUNDS} | 教师模型 (ResNet34):")
                print_metrics(teacher_metrics, "      ")

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

            # 将训练好的学生模型状态保存回持久化字典
            farm_student_models[farm_id] = farm_student_model
            print(f"  已保存农场 {farm_id} 的学生模型状态到持久化字典")

            # 清理显存
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # --- 高层奖励计算与更新 ---
        # 1. 更新知识库
        for farm_id, model_info in newly_trained_student_models.items():
            knowledge_base.update(farm_id, model_info)

        # 2. 计算全局奖励分量
        current_metrics = [m['metrics'] for m in knowledge_base.get_all_models()]
        current_avg_acc = np.mean([m['accuracy'] for m in current_metrics]) if current_metrics else 0
        current_avg_lat = np.mean(round_latencies) if round_latencies else 0

        # 计算效率
        current_efficiency = config.W_ACC_GLOBAL * current_avg_acc - config.W_LAT_GLOBAL * current_avg_lat
        efficiency_gain = current_efficiency - last_global_efficiency

        # 计算多样性
        all_state_dicts = [m['state_dict'] for m in knowledge_base.get_all_models()]
        if len(all_state_dicts) > 1:
            all_params = [torch.nn.utils.parameters_to_vector(sd.values()) for sd in all_state_dicts]
            current_diversity = torch.stack(all_params).var().item()
        else:
            current_diversity = 0.0
        diversity_gain = current_diversity - last_global_diversity
        
        global_reward_component = config.W_EFFICIENCY_GLOBAL * efficiency_gain + config.W_DIVERSITY_GLOBAL * diversity_gain
        # 3. 计算并分配个性化奖励
        individual_rewards = []
        w_local_gain = 0.5  # 增加本地增益权重
        w_transfer_success = 0.2  # 知识转移成功奖励
        w_diversity_contribution = 0.1  # 多样性贡献奖励
        
        for farm_id in active_farm_ids:
            current_info = newly_trained_student_models.get(farm_id)
            last_info = knowledge_base.get_model(farm_id)
            local_acc_gain = 0.0
            transfer_success_bonus = 0.0
            diversity_contribution = 0.0
            
            if current_info:
                if last_info:
                    local_acc_gain = current_info['metrics']['accuracy'] - last_info['metrics']['accuracy']
                else: # 新农场
                    local_acc_gain = current_info['metrics']['accuracy']
                
                # 检查知识转移是否成功
                directive = farm_directives.get(farm_id, {})
                if directive.get('role') == 'TRANSFER_IN':
                    # 如果转移后性能提升，给予额外奖励
                    if local_acc_gain > 0:
                        transfer_success_bonus = 0.8  # 增加成功奖励
                    else:
                        transfer_success_bonus = -0.3  # 转移失败惩罚
                
                # 计算多样性贡献
                farm_classes = config.FARM_CLASS_ALLOCATION[farm_id]
                farm_class_set = set(farm_classes)
                unique_contribution = 0.0
                for other_farm_id in active_farm_ids:
                    if other_farm_id != farm_id:
                        other_classes = set(config.FARM_CLASS_ALLOCATION[other_farm_id])
                        overlap = len(farm_class_set & other_classes) / len(farm_class_set | other_classes)
                        unique_contribution += (1 - overlap)
                diversity_contribution = min(1.0, unique_contribution / (len(active_farm_ids) - 1))
            
            # 组合奖励：全局效率 + 本地增益 + 转移成功奖励 + 多样性贡献
            reward = (1 - w_local_gain - w_transfer_success - w_diversity_contribution) * global_reward_component + \
                    w_local_gain * (local_acc_gain / 10.0) + \
                    w_transfer_success * transfer_success_bonus + \
                    w_diversity_contribution * diversity_contribution
            
            individual_rewards.append(reward)

        print(f"\n--- 高层奖励计算 (个性化) ---")
        print(f"  全局奖励分量: {global_reward_component:.4f}")
        print(f"  本轮个体奖励 (前5个): {np.round(individual_rewards[:5], 4)}")
        print(f"  探索率: {exploration_rate:.3f}")

        # 3. 批量更新高层Agent，将本轮的全局奖励作为参数传入
        print("  正在批量更新高层Agent策略...")
        meta_controller.update(individual_rewards)

        # 更新状态
        last_global_efficiency = current_efficiency
        last_global_diversity = current_diversity

        print_memory_usage(f"服务器轮次 {server_round_idx + 1} 结束后")

    print("\n--- HRL 框架所有轮次执行完毕 ---")
    print_memory_usage("程序结束时")


if __name__ == '__main__':
    # 确保输出目录存在
    if not os.path.exists(config.OUTPUT_DIR):
        os.makedirs(config.OUTPUT_DIR)
    main()
