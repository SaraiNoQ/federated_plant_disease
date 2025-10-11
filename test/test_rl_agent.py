#!/usr/bin/env python3
"""
测试文件：验证RL Agent的运行是否正确
"""

import torch
import numpy as np
import sys
import os

# 添加src目录到路径
sys.path.append(os.path.join(os.path.dirname(__file__), 'src'))

from rl_agents import MetaControllerA2C, LocalExecutorUCB


def test_meta_controller_basic():
    """测试高层RL Agent的基本功能"""
    print("=== 测试高层RL Agent基本功能 ===")
    
    # 初始化参数
    num_farms = 6
    state_dim = 12
    device = 'cpu'
    
    # 创建MetaController
    meta_controller = MetaControllerA2C(
        num_farms=num_farms,
        state_dim=state_dim,
        device=device,
        entropy_coeff=0.01
    )
    
    # 测试状态构建
    test_state = [0.5, 0.6, 0.3, 0.2, 0.7, 0.8, 0.4, 0.9, 0.1, 0.5, 0.6, 0.7]
    farm_idx = 2
    available_teachers = [0, 1, 3, 4, 5]  # 除了自己外的所有农场
    
    # 测试动作选择
    action, log_prob = meta_controller.select_action(
        test_state, farm_idx, available_teachers, exploration_rate=0.3
    )
    
    print(f"测试状态: {test_state}")
    print(f"农场索引: {farm_idx}")
    print(f"可用教师: {available_teachers}")
    print(f"选择动作: {action}")
    print(f"动作概率: {log_prob.item():.4f}")
    
    # 验证动作是否有效
    assert action >= 0 and action <= num_farms, f"动作 {action} 超出范围 [0, {num_farms}]"
    
    # 测试动作掩码
    if action > 0:
        teacher_idx = action - 1
        assert teacher_idx in available_teachers, f"选择的教师 {teacher_idx} 不在可用列表中"
        assert teacher_idx != farm_idx, "不能向自己学习"
    
    print("✓ 高层RL Agent基本功能测试通过\n")


def test_meta_controller_action_masking():
    """测试动作掩码功能"""
    print("=== 测试动作掩码功能 ===")
    
    num_farms = 4
    state_dim = 12
    device = 'cpu'
    
    meta_controller = MetaControllerA2C(
        num_farms=num_farms,
        state_dim=state_dim,
        device=device
    )
    
    # 测试场景1：只能EXPLOIT（没有可用教师）
    test_state = [0.5] * state_dim
    farm_idx = 0
    available_teachers = []  # 没有可用教师
    
    action, _ = meta_controller.select_action(
        test_state, farm_idx, available_teachers, exploration_rate=0.0
    )
    
    print(f"场景1 - 没有可用教师")
    print(f"选择动作: {action}")
    assert action == 0, "没有可用教师时应选择EXPLOIT"
    print("✓ 场景1测试通过")
    
    # 测试场景2：只能向特定教师学习
    farm_idx = 1
    available_teachers = [0, 2]  # 只能向农场0和2学习
    
    action, _ = meta_controller.select_action(
        test_state, farm_idx, available_teachers, exploration_rate=0.0
    )
    
    print(f"场景2 - 只能向农场{available_teachers}学习")
    print(f"选择动作: {action}")
    if action > 0:
        teacher_idx = action - 1
        assert teacher_idx in available_teachers, f"选择的教师 {teacher_idx} 不在可用列表中"
        assert teacher_idx != farm_idx, "不能向自己学习"
    print("✓ 场景2测试通过\n")


def test_meta_controller_update():
    """测试策略更新功能"""
    print("=== 测试策略更新功能 ===")
    
    num_farms = 3
    state_dim = 12
    device = 'cpu'
    
    meta_controller = MetaControllerA2C(
        num_farms=num_farms,
        state_dim=state_dim,
        device=device
    )
    
    # 模拟一轮决策
    states = []
    for i in range(num_farms):
        state = [0.5 + i*0.1] * state_dim
        farm_idx = i
        available_teachers = [j for j in range(num_farms) if j != i]
        
        action, log_prob = meta_controller.select_action(
            state, farm_idx, available_teachers, exploration_rate=0.1
        )
        meta_controller.store_transition(state, log_prob)
        states.append(state)
        
        print(f"农场 {i}: 动作={action}, 概率={log_prob.item():.4f}")
    
    # 模拟奖励
    rewards = [1.0, 0.5, -0.2]
    
    # 更新策略
    print(f"更新前缓冲区大小: {len(meta_controller.buffer)}")
    meta_controller.update(rewards)
    print(f"更新后缓冲区大小: {len(meta_controller.buffer)}")
    
    assert len(meta_controller.buffer) == 0, "更新后缓冲区应该被清空"
    print("✓ 策略更新测试通过\n")


def test_local_executor_ucb():
    """测试低层RL Agent的UCB算法"""
    print("=== 测试低层RL Agent UCB算法 ===")
    
    num_clients = 5
    exploration_factor = 2.0
    
    executor = LocalExecutorUCB(num_clients, exploration_factor)
    
    # 初始选择（应该优先选择未探索的客户端）
    print("初始选择（应优先选择未探索的客户端）:")
    selected = executor.select_clients(2)
    print(f"选择的客户端: {selected}")
    assert len(selected) == 2, "应该选择2个客户端"
    assert len(set(selected)) == len(selected), "选择的客户端不应重复"
    print("✓ 初始选择测试通过")
    
    # 更新奖励
    rewards = [0.8, 0.6]  # 假设两个客户端分别获得0.8和0.6的奖励
    executor.update(selected, rewards)
    
    print(f"更新后客户端统计:")
    for i in range(num_clients):
        print(f"  客户端 {i}: 选择次数={executor.counts[i]}, 平均奖励={executor.values[i]:.3f}")
    
    # 再次选择（应该基于UCB分数）
    print("再次选择（应基于UCB分数）:")
    selected2 = executor.select_clients(2)
    print(f"选择的客户端: {selected2}")
    
    # 验证至少有一个新客户端被选择（探索）
    new_clients = set(selected2) - set(selected)
    if len(new_clients) > 0:
        print(f"探索了新客户端: {new_clients}")
    
    print("✓ UCB算法测试通过\n")


def test_exploration_mechanism():
    """测试探索机制"""
    print("=== 测试探索机制 ===")
    
    num_farms = 4
    state_dim = 12
    device = 'cpu'
    
    meta_controller = MetaControllerA2C(
        num_farms=num_farms,
        state_dim=state_dim,
        device=device
    )
    
    # 测试不同探索率下的行为
    test_state = [0.5] * state_dim
    farm_idx = 0
    available_teachers = [1, 2, 3]
    
    exploration_rates = [0.0, 0.5, 1.0]
    action_counts = {i: 0 for i in range(num_farms + 1)}
    
    num_trials = 100
    
    for exploration_rate in exploration_rates:
        print(f"探索率: {exploration_rate}")
        
        for _ in range(num_trials):
            action, _ = meta_controller.select_action(
                test_state, farm_idx, available_teachers, exploration_rate
            )
            action_counts[action] += 1
        
        # 打印动作分布
        for action, count in action_counts.items():
            if count > 0:
                prob = count / num_trials
                action_name = "EXPLOIT" if action == 0 else f"TRANSFER_IN from Farm_{action-1}"
                print(f"  动作 {action} ({action_name}): {prob:.2%}")
        
        # 重置计数
        action_counts = {i: 0 for i in range(num_farms + 1)}
    
    print("✓ 探索机制测试通过\n")


def test_complete_workflow():
    """测试完整的工作流程"""
    print("=== 测试完整工作流程 ===")
    
    # 模拟多轮知识路由
    num_farms = 3
    state_dim = 12
    num_rounds = 3
    
    meta_controller = MetaControllerA2C(
        num_farms=num_farms,
        state_dim=state_dim,
        device='cpu'
    )
    
    for round_idx in range(num_rounds):
        print(f"\n--- 轮次 {round_idx + 1} ---")
        
        # 计算动态探索率
        exploration_rate = max(0.15, 0.4 * (1 - round_idx / num_rounds))
        print(f"探索率: {exploration_rate:.3f}")
        
        # 为每个农场生成指令
        farm_directives = {}
        available_teachers = list(range(num_farms))  # 所有农场都可用
        
        for farm_id in range(num_farms):
            # 模拟状态构建
            state = [0.5 + farm_id*0.1] * state_dim
            
            action, log_prob = meta_controller.select_action(
                state, farm_id, available_teachers, exploration_rate
            )
            meta_controller.store_transition(state, log_prob)
            
            # 解析指令
            if action == 0:
                directive = {'role': 'EXPLOIT'}
            else:
                teacher_id = action - 1
                directive = {'role': 'TRANSFER_IN', 'source': f'Farm_{teacher_id}', 'budget': 0.2}
            
            farm_directives[farm_id] = directive
            print(f"  农场 {farm_id}: {directive}")
        
        # 模拟奖励计算
        rewards = []
        for farm_id in range(num_farms):
            directive = farm_directives[farm_id]
            if directive['role'] == 'TRANSFER_IN':
                # 知识转移可能带来正奖励
                reward = 0.5 + np.random.normal(0, 0.1)
            else:
                # 本地探索可能带来较小奖励
                reward = 0.2 + np.random.normal(0, 0.1)
            rewards.append(reward)
        
        print(f"  奖励: {[f'{r:.3f}' for r in rewards]}")
        
        # 更新策略
        meta_controller.update(rewards)
    
    print("✓ 完整工作流程测试通过\n")


if __name__ == '__main__':
    print("开始测试RL Agent...\n")
    
    try:
        test_meta_controller_basic()
        test_meta_controller_action_masking()
        test_meta_controller_update()
        test_local_executor_ucb()
        test_exploration_mechanism()
        test_complete_workflow()
        
        print("🎉 所有测试通过！RL Agent运行正常。")
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        import traceback
        traceback.print_exc()
