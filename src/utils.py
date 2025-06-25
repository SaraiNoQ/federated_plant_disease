# src/utils.py

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def evaluate_model(model, dataloader, device):
    """评估模型在给定数据集上的性能。"""
    model.eval()
    criterion = nn.CrossEntropyLoss()
    correct = 0
    total = 0
    loss = 0.0
    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss += criterion(output, target).item() * data.size(0)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            total += target.size(0)
            
    avg_loss = loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy

def plot_history(history, output_dir):
    """绘制并保存训练历史曲线。"""
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history['round'], history['accuracy'], marker='o')
    plt.title("联邦学习全局模型准确率")
    plt.xlabel("通信轮次 (Round)")
    plt.ylabel("验证集准确率 (%)")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(history['round'], history['loss'], marker='o', color='r')
    plt.title("联邦学习全局模型损失")
    plt.xlabel("通信轮次 (Round)")
    plt.ylabel("验证集损失 (Loss)")
    plt.grid(True)
    
    plt.tight_layout()
    plot_save_path = os.path.join(output_dir, 'federated_learning_plot.png')
    plt.savefig(plot_save_path)
    print(f"训练曲线图已保存至: {plot_save_path}")
    plt.show()