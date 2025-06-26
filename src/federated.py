# src/federated.py

import torch
import torch.nn as nn
import torch.optim as optim
from collections import OrderedDict
import copy
import numpy as np

def farm_unit_update(model, train_loader, epochs, lr, device, farm_id, unit_id):
    """
    农场内部计算单元的本地训练过程。
    模型传入时应已配置为该农场的特定类别数。

    Returns:
        OrderedDict: 训练后模型的权重。
    """
    model.train()
    # 使用 filter(lambda p: p.requires_grad, model.parameters()) 确保只优化需要训练的参数
    # 这在迁移学习中很重要
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    criterion = nn.CrossEntropyLoss() # 标签已经是农场本地的 [0, num_farm_classes-1]

    # print(f"  [农场 {farm_id} - 单元 {unit_id}] 开始本地训练, Epochs: {epochs}...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = 0
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            num_batches +=1
        # if num_batches > 0:
        #      print(f"    Epoch {epoch+1}/{epochs}, 平均损失: {epoch_loss/num_batches:.4f}")
        # else:
        #     print(f"    Epoch {epoch+1}/{epochs}, 无数据进行训练。")


    return model.state_dict()

def aggregate_models(model_weights_list: list, farm_id_for_log: str = "服务器"):
    """
    通用的模型聚合函数 (FedAvg)。
    可用于聚合农场内部单元的模型，或聚合不同农场的模型。

    Args:
        model_weights_list (list): 包含多个模型state_dict的列表。
        farm_id_for_log (str): 用于日志记录的标识。

    Returns:
        OrderedDict: 聚合后的模型权重。
    """
    if not model_weights_list:
        print(f"警告 [{farm_id_for_log}]: 模型权重列表为空，无法聚合。")
        return None

    # 检查所有state_dict是否具有相同的键结构，如果不同，FedAvg会出问题
    # 为简单起见，我们假设它们是相同的 (例如，来自同一初始模型的副本)
    # 但在实际中，如果模型结构不同（例如，不同农场的模型有不同类别数），则不能直接平均
    # 我们的策略是：农场内FL聚合的模型具有相同的（农场特定）类别数
    # 服务器聚合农场模型时，如果农场模型类别数不同，这个简单的FedAvg不适用！
    # **重要：** 当前的服务器聚合假设所有农场模型都已转换回能预测全局38类的模型，
    # 或者，我们只聚合特征提取层，分类头分别处理。
    # 为简化本次修改，我们将假设：服务器聚合时，如果目标是生成新的通用源模型，
    # 那么农场上传的模型需要是针对全局38类的。
    # 或者，更简单：服务器只聚合那些在“通用任务”上训练的农场模型。
    # **当前代码的简化：农场上传的模型state_dict，如果用于服务器FedAvg，必须结构一致。**
    # **这意味着农场微调后的模型，其fc层需要与全局初始模型一致（38类）。**
    # **这是一个设计选择：农场内部微调时，模型输出农场特定类别。上传给服务器时，需要转换。**
    # **或者，服务器聚合的是一个“通用特征提取器”，然后每个农场再接自己的头。**

    # **当前的简化方案：** 我们假设在服务器聚合前，农场模型会被转换回输出全局类别。
    # 但在 `run_fl.py` 中，农场上传的是其微调后的模型，fc层是农场特定的。
    # 这意味着 `server_aggregate_farm_models` 需要更复杂的逻辑，或者我们改变策略。

    # **策略调整：农场内部FL聚合后，得到一个农场最优模型。这个模型用于蒸馏。**
    # **服务器聚合（如果要做）需要对这些不同输出维度的农场模型进行特殊处理，**
    # **例如，只聚合共享的骨干网络层，或者采用更高级的联邦聚合算法。**
    # **为使代码能运行，我们暂时让服务器聚合也只聚合结构完全相同的模型。**
    # **这意味着，如果服务器要聚合农场模型，这些农场模型在上传前，其FC层需要被标准化。**
    # **或者，我们只对来自同一“任务类型”（例如，都预测全局38类）的农场模型进行聚合。**

    # **当前最简化的实现：此函数用于农场内聚合，结构是相同的。**
    # **用于服务器聚合时，需要确保传入的农场模型state_dict结构一致。**

    # print(f"[{farm_id_for_log}] 正在聚合 {len(model_weights_list)} 个模型...")
    keys = model_weights_list[0].keys()
    aggregated_weights = OrderedDict()

    for key in keys:
        if all(key in weights for weights in model_weights_list):
            try:
                layer_weights = [weights[key].float() for weights in model_weights_list]
                if all(w.shape == layer_weights[0].shape for w in layer_weights):
                    aggregated_weights[key] = torch.stack(layer_weights).mean(0)
                # else:
                #     print(f"  - 跳过聚合层 '{key}'，因为形状不匹配。")
            except RuntimeError as e:
                print(f"!! 聚合错误在层 '{key}': {e}。跳过此层。")
        # else:
        #     print(f"  - 跳过聚合层 '{key}'，因为它并非在所有模型中都存在。")

    # print(f"[{farm_id_for_log}] 模型聚合完成。")
    return aggregated_weights