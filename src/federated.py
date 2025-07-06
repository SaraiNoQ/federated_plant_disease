# src/federated.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import copy
import numpy as np
import torch.nn.functional as F


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
             p.requires_grad]  # Re-create a temporary model to load state dict to get parameters
        )]
        # This is a bit tricky; a simpler way is to pass the global model object itself
        # For now, let's pass the state_dict and re-create parameters
        # A much cleaner way:
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

            # Add the proximal term if mu > 0
            if mu > 0 and global_params is not None:
                prox_term = 0.0
                local_params = model.parameters()
                for local_p, global_p in zip(local_params, global_params):
                    prox_term += (local_p - global_p).norm(2)

                loss += (mu / 2) * prox_term

            loss.backward()
            optimizer.step()

    return model.state_dict()


# --- A more robust farm_unit_update using a simpler FedProx implementation ---
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
                # Iterate over the parameters of the local and global models
                for local_param, global_param in zip(model.parameters(), global_model.parameters()):
                    # The .norm(2) computes the L2 norm (Euclidean distance)
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