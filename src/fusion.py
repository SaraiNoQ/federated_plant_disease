# src/fusion.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy

from .models import build_model # 确保可以从你的模型工厂导入

def feddf_fusion(
    teacher_models: dict, # {'Farm_A': model_obj, ...}
    server_model: nn.Module,
    public_dataloader: torch.utils.data.DataLoader,
    device: torch.device,
    epochs=5,
    lr=0.01
):
    """
    使用知识蒸馏进行模型融合 (FedDF)。

    Args:
        teacher_models (dict): 包含所有客户端（教师）模型对象的字典。
        server_model (nn.Module): 要训练的服务器（学生）模型。
        public_dataloader (DataLoader): 公开的、无标签的代理数据集。
        device (torch.device): 计算设备。
        epochs (int): 蒸馏训练的轮数。
        lr (float): 学习率。
    
    Returns:
        nn.Module: 融合训练后的服务器模型。
    """
    print("\n--- Starting Federated Distillation Fusion (FedDF) ---")
    
    # 冻结所有教师模型
    for teacher in teacher_models.values():
        teacher.eval()

    student_model = copy.deepcopy(server_model).to(device)
    student_model.train()
    optimizer = optim.Adam(student_model.parameters(), lr=lr)

    for epoch in range(epochs):
        total_loss = 0.0
        for data in public_dataloader:
            # 我们只需要数据，不需要标签
            if isinstance(data, list):
                inputs = data[0].to(device)
            else:
                inputs = data.to(device)
            
            optimizer.zero_grad()

            # 1. 获取所有教师的平均logits
            with torch.no_grad():
                avg_logits = None
                num_teachers = len(teacher_models)
                for teacher in teacher_models.values():
                    # 注意：教师模型可能有不同的输出头，这里假设服务器模型有全局的输出头
                    # 为了简化，我们假设所有模型都有相同的输出维度，或者只使用backbone
                    # 更稳健的做法是让所有模型输出到统一维度的logits
                    # 此处假设模型结构兼容，可以输出相同维度的logits
                    logits = teacher(inputs)
                    if avg_logits is None:
                        avg_logits = torch.zeros_like(logits)
                    avg_logits += logits / num_teachers

            # 2. 学生模型进行推理
            student_logits = student_model(inputs)

            # 3. 计算KL散度损失
            # 学生向教师们的共识学习
            loss = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_logits, dim=1),
                F.softmax(avg_logits, dim=1)
            )

            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        print(f"  FedDF Epoch {epoch+1}/{epochs} | Average Loss: {total_loss / len(public_dataloader):.4f}")

    print("--- FedDF Fusion Finished ---")
    return student_model