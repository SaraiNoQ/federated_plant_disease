# src/federated.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import copy
import numpy as np
import torch.nn.functional as F
import config_optimized as config


def farm_unit_update(model, train_loader, epochs, lr, device, farm_id, unit_id, global_model_state=None, mu=0.0):
    """
    Local training process for a farm's computational unit.
    Now includes the FedProx proximal term.

    Args:
        model (nn.Module): The local model to be trained.
        train_loader (DataLoader): The local data loader.
        epochs (int): Number of local epochs.
        lr (float): Learning rate.
        device (torch.device): The device to train on.
        farm_id (str): The ID of the farm.
        unit_id (int): The ID of the client unit.
        global_model_state (OrderedDict, optional): The state_dict of the global model from the
                                                     start of the round. Required for FedProx.
        mu (float): The mu parameter for the FedProx proximal term. If 0, it's standard FedAvg.
    """
    model.train()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # If FedProx is enabled, store the initial global model parameters
    global_params = None
    if mu > 0 and global_model_state is not None:
        global_params = [param.detach().clone() for param in torch.nn.utils.parameters_to_vector(
            [p for p in nn.Module().load_state_dict(global_model_state, strict=False) or model.parameters() if
             p.requires_grad]
        )]
        temp_global_model = copy.deepcopy(model).to(device)
        temp_global_model.load_state_dict(global_model_state)
        global_params = [param.detach().clone() for param in temp_global_model.parameters()]

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            # --- FedProx Implementation ---
            loss = criterion(output, target)

            if mu > 0 and global_params is not None:
                prox_term = 0.0
                local_params = model.parameters()
                for local_p, global_p in zip(local_params, global_params):
                    prox_term += (local_p - global_p).norm(2)

                loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

    return model.state_dict()


def farm_unit_update_fedprox(model, global_model, train_loader, epochs, lr, device, farm_id, unit_id, mu=0.0):
    """
    A cleaner implementation of the local training process with FedProx.
    Args:
        model (nn.Module): The local model to be trained (a copy).
        global_model (nn.Module): The global model from the start of the round (for reference).
        ... (other args are the same)
    """
    model.train()
    global_model.eval()  # Keep the global model in eval mode
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)

            loss = criterion(output, target)

            if mu > 0:
                prox_term = 0.0
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    prox_term += (local_param - global_param.detach()).norm(2)
                loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

    return model.state_dict()


def aggregate_models(model_weights_list: list, farm_id_for_log: str = "服务器"):
    """
    Generic model aggregation function (FedAvg).
    """
    if not model_weights_list:
        return None

    keys = model_weights_list[0].keys()
    aggregated_weights = OrderedDict()

    for key in keys:
        if all(key in weights for weights in model_weights_list):
            try:
                layer_weights = [weights[key].float() for weights in model_weights_list]
                if all(w.shape == layer_weights[0].shape for w in layer_weights):
                    aggregated_weights[key] = torch.stack(layer_weights).mean(0)
            except RuntimeError as e:
                print(f"!! Aggregation error on layer '{key}': {e}. Skipping this layer.")

    return aggregated_weights

def aggregate_models_rl(model_weights_list: list, aggregation_weights: np.ndarray, farm_id_for_log: str = "服务器 (RL)"):
    """
    Generic model aggregation function using specified weights.
    """
    if not model_weights_list or aggregation_weights is None:
        return None
    
    if len(model_weights_list) != len(aggregation_weights):
        print(f"!! Aggregation error: Mismatch between number of models ({len(model_weights_list)}) and weights ({len(aggregation_weights)})")
        return None

    keys = model_weights_list[0].keys()
    aggregated_weights = OrderedDict()

    print(f"[{farm_id_for_log}] Aggregating with RL weights: {np.round(aggregation_weights, 3)}")

    for key in keys:
        if all(key in weights for weights in model_weights_list):
            try:
                # 获取所有模型的当前层，并应用权重
                weighted_layers = [weights[key].float() * weight for weights, weight in zip(model_weights_list, aggregation_weights)]
                
                # 检查形状是否一致
                if all(w.shape == weighted_layers[0].shape for w in weighted_layers):
                    # 叠加求和
                    aggregated_weights[key] = torch.stack(weighted_layers).sum(0)
            except RuntimeError as e:
                print(f"!! Aggregation error on layer '{key}': {e}. Skipping this layer.")

    return aggregated_weights


def distill_unit_update(
        teacher_model: nn.Module,
        student_model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        temperature: float,
        alpha: float,
        device: torch.device
):
    """
    单个客户端上的蒸馏训练过程。
    """
    teacher_model.eval()
    student_model.train()

    optimizer = torch.optim.Adam(student_model.parameters(), lr=lr)

    for epoch in range(epochs):
        for inputs, hard_labels in train_loader:
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            optimizer.zero_grad()

            with torch.no_grad():
                teacher_outputs = teacher_model(inputs)

            student_outputs = student_model(inputs)

            loss_kd = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)

            loss_ce = nn.CrossEntropyLoss()(student_outputs, hard_labels)
            total_loss = alpha * loss_kd + (1 - alpha) * loss_ce

            total_loss.backward()
            optimizer.step()

    return student_model.state_dict()


def print_memory_usage(prefix=""):
    """打印GPU显存使用情况"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU显存使用: {allocated:.2f}GB / {cached:.2f}GB")

def local_distill_update(
        local_teacher_model: nn.Module,
        student_model_to_train: nn.Module,
        global_student_model: nn.Module, # 传入上一轮的全局学生模型
        prox_mu: float,                  # FedProx的mu，我们复用它来控制相似度
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        temperature: float,
        alpha: float,
        device: torch.device,
        # HRL 新增参数
        transfer_teacher_model: nn.Module = None,
        transfer_budget: float = 0.0
):
    """
    客户端本地蒸馏过程 (FedMD)。
    学生模型学习私有教师模型，并可选地学习一个外部迁移教师。
    添加了显存管理优化。
    """
    # 初始化显存管理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    local_teacher_model.eval()
    if transfer_teacher_model:
        transfer_teacher_model.eval()

    student_model_to_train.train()
    optimizer = torch.optim.Adam(student_model_to_train.parameters(), lr=lr)

    for epoch in range(epochs):
        for batch_idx, (inputs, hard_labels) in enumerate(train_loader):
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            optimizer.zero_grad(set_to_none=True)  # 使用set_to_none减少显存占用

            # 1. 从私有教师获取软标签
            with torch.no_grad():
                local_teacher_outputs = local_teacher_model(inputs)

            # 2. 学生模型推理
            student_outputs = student_model_to_train(inputs)

            # 3. 计算标准蒸馏损失
            loss_ce = nn.CrossEntropyLoss()(student_outputs, hard_labels)
            loss_kd_local = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(local_teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)

            loss_exploit = alpha * loss_kd_local + (1 - alpha) * loss_ce

            # 释放中间变量
            del local_teacher_outputs

            # 4. 计算知识迁移损失
            loss_transfer = 0.0
            if transfer_teacher_model and transfer_budget > 0:
                with torch.no_grad():
                    transfer_teacher_outputs = transfer_teacher_model(inputs)

                loss_kd_transfer = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(transfer_teacher_outputs / temperature, dim=1)
                ) * (temperature * temperature)
                loss_transfer = loss_kd_transfer

                # 释放中间变量
                del transfer_teacher_outputs

            # 5. 最终复合损失
            total_loss = (1 - transfer_budget) * loss_exploit + transfer_budget * loss_transfer

            if prox_mu > 0:
                prox_term = 0.0
                for local_p, global_p in zip(student_model_to_train.parameters(), global_student_model.parameters()):
                    prox_term += (local_p - global_p.detach()).norm(2)

                total_loss += (prox_mu / 2) * prox_term

            total_loss.backward()
            optimizer.step()

            # 定期清理显存和监控
            # if batch_idx % config.CLEAN_CACHE_INTERVAL == 0 and torch.cuda.is_available():
            #     torch.cuda.empty_cache()
            
            # if batch_idx % config.MEMORY_MONITOR_INTERVAL == 0 and torch.cuda.is_available():
            #     print_memory_usage(f"蒸馏轮次 {epoch+1}/{epochs} 批次 {batch_idx}")

    return student_model_to_train.state_dict()
