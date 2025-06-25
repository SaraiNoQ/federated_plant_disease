# src/utils.py (evaluate_model保持不变, plot_history可以暂时不用于农场内FL)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def evaluate_model(model, dataloader, device, context="评估"):
    """评估模型在给定数据集上的性能。"""
    model.eval()
    criterion = nn.CrossEntropyLoss() # 假设是分类任务
    correct = 0
    total = 0
    running_loss = 0.0
    if not dataloader or len(dataloader.dataset) == 0:
        print(f"[{context}] 验证数据加载器为空或数据集为空，跳过评估。")
        return 0.0, 0.0

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)

    if total == 0:
        print(f"[{context}] 验证集中没有样本，无法评估。")
        return 0.0, 0.0

    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def plot_server_fl_history(history, output_dir):
    """绘制并保存服务器聚合的联邦学习历史曲线。"""
    if not history['round']: # 如果历史记录为空
        print("没有服务器聚合历史记录可供绘制。")
        return

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(history['round'], history['accuracy'], marker='o')
    plt.title("服务器聚合 - 全局模型准确率")
    plt.xlabel("服务器通信轮次 (Round)")
    plt.ylabel("在某个统一验证集上的准确率 (%)") # 需要定义一个统一的验证集
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(history['round'], history['loss'], marker='o', color='r')
    plt.title("服务器聚合 - 全局模型损失")
    plt.xlabel("服务器通信轮次 (Round)")
    plt.ylabel("在某个统一验证集上的损失")
    plt.grid(True)

    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, 'server_federated_learning_plot.png')
    plt.savefig(plot_save_path)
    print(f"服务器FL训练曲线图已保存至: {plot_save_path}")
    plt.close() # 关闭图像，避免在notebook中重复显示