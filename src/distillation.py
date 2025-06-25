# src/distillation.py

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR # 导入学习率调度器
from .models import build_student_model
from .utils import evaluate_model
import copy
import os # vvvvvvvvvv 导入 os 模块 vvvvvvvvvv

def distill_model(
    teacher_model: nn.Module,
    farm_id: str,
    num_farm_classes: int,
    distill_train_loader: torch.utils.data.DataLoader,
    distill_val_loader: torch.utils.data.DataLoader,
    epochs: int,
    lr: float,
    temperature: float,
    alpha: float,
    device: torch.device,
    output_dir: str
):
    """
    使用知识蒸馏训练学生模型。
    """
    print(f"\n--- 开始为农场 {farm_id} 进行模型蒸馏 ---")
    print(f"教师模型类别数 (预期): {num_farm_classes}")

    # 1. 构建学生模型
    student_model = build_student_model(num_classes=num_farm_classes).to(device)

    # 2. 定义损失函数、优化器和学习率调度器
    optimizer = optim.Adam(student_model.parameters(), lr=lr)
    # vvvvvvvvvv 添加学习率调度器 vvvvvvvvvv
    # 使用余弦退火调度器，它会在整个训练过程中平滑地降低学习率
    scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-6) # T_max是周期，eta_min是最小学习率
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    teacher_model.eval()
    student_model.train()

    best_student_accuracy = 0.0
    best_student_model_state = None

    for epoch in range(epochs):
        student_model.train() # 确保在每个epoch开始时模型处于训练模式
        running_loss_kd = 0.0
        running_loss_ce = 0.0
        running_loss_total = 0.0
        processed_samples = 0

        for inputs, hard_labels in distill_train_loader:
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

            running_loss_kd += loss_kd.item() * inputs.size(0)
            running_loss_ce += loss_ce.item() * inputs.size(0)
            running_loss_total += total_loss.item() * inputs.size(0)
            processed_samples += inputs.size(0)
        
        # vvvvvvvvvv 在每个epoch结束后更新学习率 vvvvvvvvvv
        current_lr = scheduler.get_last_lr()[0]
        scheduler.step()
        # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

        avg_loss_kd = running_loss_kd / processed_samples if processed_samples > 0 else 0
        avg_loss_ce = running_loss_ce / processed_samples if processed_samples > 0 else 0
        avg_loss_total = running_loss_total / processed_samples if processed_samples > 0 else 0

        print(f"  [农场 {farm_id} - 蒸馏 Epoch {epoch+1}/{epochs}] "
              f"LR: {current_lr:.6f}, " # 打印当前学习率
              f"Avg KD Loss: {avg_loss_kd:.4f}, Avg CE Loss: {avg_loss_ce:.4f}, Avg Total Loss: {avg_loss_total:.4f}")

        # 评估学生模型
        if distill_val_loader and len(distill_val_loader.dataset) > 0:
            # 评估时需要将模型设置为评估模式
            student_val_loss, student_val_accuracy = evaluate_model(student_model, distill_val_loader, device)
            print(f"    学生模型在农场验证集上: Loss = {student_val_loss:.4f}, Accuracy = {student_val_accuracy:.2f}%")
            if student_val_accuracy > best_student_accuracy:
                best_student_accuracy = student_val_accuracy
                best_student_model_state = copy.deepcopy(student_model.state_dict())
                print(f"    新最佳学生模型 (农场 {farm_id})! 准确率: {best_student_accuracy:.2f}%")
        else:
             best_student_model_state = copy.deepcopy(student_model.state_dict())

    if best_student_model_state:
        student_model.load_state_dict(best_student_model_state)
        print(f"农场 {farm_id} 最佳学生模型已加载，准确率: {best_student_accuracy:.2f}%")

        student_model_save_path = os.path.join(output_dir, f'student_model_farm_{farm_id}.pth')
        torch.save(student_model.state_dict(), student_model_save_path)
        print(f"农场 {farm_id} 学生模型已保存至: {student_model_save_path}")
    else:
        print(f"警告: 未能为农场 {farm_id} 生成有效的学生模型。")

    return student_model