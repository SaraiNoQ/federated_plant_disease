# src/federated.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict

def client_update(model, train_loader, epochs, lr, device):
    """
    客户端本地训练过程。

    Returns:
        OrderedDict: 训练后模型的权重。
    """
    model.train()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
    return model.state_dict()

def server_aggregate(client_weights: list):
    """
    服务器端聚合权重 (FedAvg)。

    Args:
        client_weights (list): 包含多个客户端模型权重的列表。

    Returns:
        OrderedDict: 聚合后的全局模型权重。
    """
    if not client_weights:
        return None
        
    # 获取第一个模型的键
    keys = client_weights[0].keys()
    aggregated_weights = OrderedDict()

    for key in keys:
        # 收集所有客户端在当前层的权重
        layer_weights = [weights[key].float() for weights in client_weights]
        # 计算平均值
        aggregated_weights[key] = torch.stack(layer_weights).mean(0)
        
    return aggregated_weights