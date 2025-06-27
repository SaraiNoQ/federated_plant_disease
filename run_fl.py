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
from src.federated import farm_unit_update, aggregate_models, farm_unit_update_fedprox, aggregate_models_rl
from src.distillation import distill_model
from src.utils import evaluate_model, plot_server_fl_history
from src.rl_selector import UCB1Selector, ThompsonSamplingSelector
from src.rl_aggregator import PPOAgent # <<< 新增导入

def main():
    """主执行函数"""
    print("--- 多农场联邦迁移学习与模型蒸馏项目启动 (V4: RL-Powered Aggregation) ---")
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

    # 定义选择器类
    selector_class = None
    if config.RL_STRATEGY.lower() == 'thompson':
        selector_class = ThompsonSamplingSelector
        print("RL Strategy: Thompson Sampling")
    elif config.RL_STRATEGY.lower() == 'ucb':
        selector_class = UCB1Selector
        print("RL Strategy: UCB1")

    # Layer 1: Farm Selector
    server_rl_selector = None
    if config.USE_RL_FARM_SELECTION and selector_class:
        farm_ids = list(all_farms_data_loaders.keys())
        server_rl_selector = selector_class(
            num_arms=len(farm_ids),
            arm_ids=farm_ids,
            exploration_factor=config.RL_EXPLORATION_FACTOR # Thompson会忽略这个
        )
        print("Layer 1 RL (Farm Selection) is ENABLED.")
    
    # Layer 2: Client Selectors (one per farm)
    farm_rl_selectors = {}
    if config.USE_RL_CLIENT_SELECTION and selector_class:
        print("Layer 2 RL (Client Selection) is ENABLED for each farm.")
        for farm_id, farm_data in all_farms_data_loaders.items():
            num_units = len(farm_data["unit_loaders"])
            if num_units > 0:
                farm_rl_selectors[farm_id] = selector_class(
                    num_arms=num_units,
                    arm_ids=list(range(num_units)),
                    exploration_factor=config.RL_EXPLORATION_FACTOR
                )

    # === 新增：初始化PPO聚合器 ===
    # 假设我们最多选择FARMS_PER_SERVER_ROUND个农场
    ppo_agent = None
    if config.USE_RL_AGGREGATION:
        # 状态维度 = 1 (last_global_acc) + N_farms * 3 (acc, loss, num_samples)
        STATE_DIM = 1 + config.FARMS_PER_SERVER_ROUND * 3
        ACTION_DIM = config.FARMS_PER_SERVER_ROUND

        ppo_agent = PPOAgent(
            state_dim=STATE_DIM,
            action_dim=ACTION_DIM,
            lr_actor=config.PPO_LR_ACTOR,
            lr_critic=config.PPO_LR_CRITIC,
            gamma=config.PPO_GAMMA,
            K_epochs=config.PPO_K_EPOCHS,
            eps_clip=config.PPO_EPS_CLIP,
            device=config.DEVICE
        )
        print("PPO Aggregation Agent is [ENABLED].")
    else:
        print("PPO Aggregation Agent is [DISABLED]. Using standard FedAvg.")

    # State tracking for rewards
    server_history = {'round': [], 'loss': [], 'accuracy': [], 'f1_score': []}
    last_server_accuracy = 0.0
    last_farm_accuracies = {farm_id: 0.0 for farm_id in all_farms_data_loaders.keys()}

    # --- 4. 服务器聚合轮次循环 ---
    for server_round_idx in range(config.SERVER_ROUNDS):
        print(f"\n\n K{'='*10} 服务器全局轮次: {server_round_idx + 1}/{config.SERVER_ROUNDS} K{'='*10}")

        # 保存本轮开始前的全局模型状态，用于PPO调优和奖励计算
        prev_server_model_state = copy.deepcopy(server_global_model.state_dict())

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
        
        # 准备用于聚合的模型列表和用于构建状态的数据
        participating_farm_ids = list(farm_final_models.keys())
        if not participating_farm_ids:
            print("本轮没有农场参与聚合，跳过。")
            continue

        models_to_aggregate = [farm_final_models[fid] for fid in participating_farm_ids]

        backbone_weights_list = [
            {k: v for k, v in model.state_dict().items() if 'fc' not in k}
            for model in models_to_aggregate
        ]
        if not backbone_weights_list:
            print("未能提取任何backbone权重，跳过聚合。")
            continue

        # === 核心：RL聚合或标准FedAvg ===
        aggregated_weights = None
        state_vector_for_ppo = []
        local_metrics_for_ppo = {}

        if config.USE_RL_AGGREGATION and ppo_agent:
            print("  Using PPO Agent for aggregation...")
            # a. 评估每个局部模型，构建状态向量
            for farm_id in participating_farm_ids:
                metrics = evaluate_model(farm_final_models[farm_id], all_farms_data_loaders[farm_id]['val_loader'], config.DEVICE)
                num_samples = len(all_farms_data_loaders[farm_id]['val_loader'].dataset)
                local_metrics_for_ppo[farm_id] = {'acc': metrics['accuracy'], 'loss': metrics['loss'], 'samples': num_samples}

            # b. 构建状态向量 [last_global_acc, acc_A, loss_A, samples_A, acc_B, ...]
            state_vector_for_ppo = [last_server_accuracy / 100.0]
            for farm_id in participating_farm_ids:
                metrics = local_metrics_for_ppo[farm_id]
                state_vector_for_ppo.extend([
                    metrics['acc'] / 100.0,
                    min(metrics['loss'], 5.0), # 限制损失最大值，防止梯度爆炸
                    metrics['samples'] / 1000.0
                ])
            # 填充以满足PPO Agent的输入维度
            padding = [0] * (ppo_agent.policy.actor[0].in_features - len(state_vector_for_ppo))
            state_vector_for_ppo.extend(padding)

            # c. PPO Agent决策，获取权重
            aggregation_weights = ppo_agent.select_action(state_vector_for_ppo)
            
            # d. 使用RL权重聚合
            aggregated_weights = aggregate_models_rl(backbone_weights_list, aggregation_weights)

        else: # 标准FedAvg
            print("  Using standard FedAvg for aggregation...")
            aggregated_weights = aggregate_models(backbone_weights_list, "Server Backbone")

        # === 更新全局模型并评估 ===
        if aggregated_weights:
            server_model_state_dict = server_global_model.state_dict()
            server_model_state_dict.update(aggregated_weights)
            server_global_model.load_state_dict(server_model_state_dict)
            
            if global_val_loader:
                metrics = evaluate_model(server_global_model, global_val_loader, config.DEVICE, "Server Post-Aggregation")
                current_accuracy = metrics['accuracy']
                print(f"  Server Model on Global Val Set: Acc={current_accuracy:.2f}%, F1={metrics['f1_score']:.4f}")
                
                # 计算奖励
                reward = current_accuracy - last_server_accuracy
                
                # 更新PPO Agent (如果启用)
                if config.USE_RL_AGGREGATION and ppo_agent:
                    ppo_agent.update(reward)
                    print(f"  PPO Agent updated with online reward: {reward:.4f}")

                # 更新历史记录和基线
                server_history['round'].append(server_round_idx + 1)
                server_history['loss'].append(metrics['loss'])
                server_history['accuracy'].append(metrics['accuracy'])
                server_history['f1_score'].append(metrics['f1_score'])
                last_server_accuracy = current_accuracy

                if config.USE_RL_FARM_SELECTION and server_rl_selector:
                    # 使用相同的奖励值来更新农场选择器
                    # 告诉选择器，它这次选择的农场组合 (participating_farms) 带来了多大的收益
                    server_rl_selector.update(participating_farms, reward)
                    print(f"  Farm Selector RL (Layer 1) updated for farms {participating_farms} with reward: {reward:.4f}")
        
        # === PPO离线调优 ===
        if config.USE_RL_AGGREGATION and ppo_agent and config.TUNE_PPO_OFFLINE:
            # `baseline_acc` 应该是本轮聚合开始前的准确率，即 last_server_accuracy 在被更新前的旧值。
            # 我们在循环开始时保存了 prev_server_model_state，但没有保存旧的 acc。
            # 不过，我们可以在更新 last_server_accuracy 之前，把它的旧值传给调优器。
            # 为了代码清晰，我们用一个新变量保存一下。
            
            # reward的计算已经完成，last_server_accuracy也已更新为current_accuracy
            # 所以，本轮开始前的准确率是 current_accuracy - reward
            baseline_accuracy_for_tuning = last_server_accuracy - reward

            # 确保传递的参数与新函数定义完全匹配
            tune_ppo_aggregator(
                ppo_agent=ppo_agent,
                local_model_backbones=backbone_weights_list, # <<< 现在函数认识这个参数了
                prev_global_model_state=prev_server_model_state,
                global_val_loader=global_val_loader,
                state_vector=state_vector_for_ppo, # <<< 传递预先计算好的状态
                baseline_acc=baseline_accuracy_for_tuning, # <<< 传递正确的基线准确率
                num_tuning_epochs=config.PPO_TUNING_EPOCHS
            )

    print("\n\n--- 顶级联邦学习流程完成 ---")
    plot_server_fl_history(server_history, config.OUTPUT_DIR)


    # 7. 保存最终的服务器模型
    final_server_model_path = os.path.join(config.OUTPUT_DIR, 'server_final_aggregated_model.pth')
    torch.save(server_global_model.state_dict(), final_server_model_path)
    print(f"\n最终服务器模型已保存至: {final_server_model_path}")
    print(f"农场内联邦学习准确率日志已保存至: {log_file_path}")

def tune_ppo_aggregator(
    ppo_agent,
    local_model_backbones: list,
    prev_global_model_state: dict,
    global_val_loader: torch.utils.data.DataLoader,
    state_vector: list,
    baseline_acc: float,
    num_tuning_epochs: int = 10
):
    """
    利用已有的模型，对PPO聚合器进行高效的离线调优。
    此版本接受预先计算好的状态和基线准确率，以提高效率。

    Args:
        ppo_agent (PPOAgent): 要调优的PPO智能体。
        local_model_backbones (list): 参与聚合的农场模型的backbone state_dicts列表。
        prev_global_model_state (dict): 上一轮的全局模型state_dict，用于构建模拟聚合的模型。
        global_val_loader (DataLoader): 用于评估模拟聚合后模型的全局验证集。
        state_vector (list): 为本次聚合预先计算好的状态向量。
        baseline_acc (float): 上一轮全局模型的准确率，作为计算奖励的基线。
        num_tuning_epochs (int): 离线调优的迭代次数。
    """
    print(f"\n--- Starting Offline PPO Agent Tuning ({num_tuning_epochs} epochs) ---")
    
    device = ppo_agent.device
    
    # 状态向量在整个调优过程中是固定的
    fixed_state_tensor = torch.FloatTensor(state_vector).to(device)

    # 4. 训练循环
    for epoch in range(num_tuning_epochs):
        # a. Agent决策
        # 注意：PPO的select_action内部会处理state的tensor转换，但我们传入tensor更规范
        # 同时，select_action会缓存(state, action, logprob)到buffer
        weights = ppo_agent.select_action(state_vector)
        
        # b. 模拟聚合
        # 我们只聚合backbone
        agg_backbone = aggregate_models_rl(local_model_backbones, weights)
        
        # 如果聚合失败，则跳过本次调优
        if agg_backbone is None:
            print(f"  Tuning Epoch {epoch+1}/{num_tuning_epochs} | Aggregation failed. Skipping.")
            ppo_agent.buffer.pop() # 移除刚才添加的无效转换
            continue

        # c. 创建一个临时全局模型并评估
        # 从上一轮的全局模型状态开始，只更新其backbone
        tuned_global_model_state = copy.deepcopy(prev_global_model_state)
        tuned_global_model_state.update(agg_backbone)

        # 加载到模型中进行评估
        temp_eval_model = build_model(num_classes=config.NUM_CLASSES_PLANTVILLAGE).to(device)
        temp_eval_model.load_state_dict(tuned_global_model_state)
        
        current_metrics = evaluate_model(temp_eval_model, global_val_loader, device)
        current_acc = current_metrics['accuracy']
        
        # d. 计算奖励并更新PPO Agent
        # 奖励 = 本次模拟聚合后的准确率 - 上一轮真实聚合的准确率
        reward = current_acc - baseline_acc
        ppo_agent.update(reward) # update会使用并清空buffer中刚刚由select_action存入的记录
        
        print(f"  Tuning Epoch {epoch+1}/{num_tuning_epochs} | Weights: {np.round(weights,2)} -> SimAcc: {current_acc:.2f}% | Reward: {reward:.4f}")
        
    print("--- Offline PPO Agent Tuning Finished ---")

if __name__ == '__main__':
    main()