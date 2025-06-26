# src/utils.py (evaluate_model保持不变, plot_history可以暂时不用于农场内FL)

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import numpy as np

def evaluate_model(model, dataloader, device, context="评估"):
    """
    评估模型在给定数据集上的性能，返回损失、准确率和F1分数。
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    if not dataloader or len(dataloader.dataset) == 0:
        print(f"[{context}] 数据加载器为空，跳过评估。")
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0}

    with torch.no_grad():
        for data, target in dataloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = criterion(output, target)
            running_loss += loss.item() * data.size(0)
            
            pred = output.argmax(dim=1)
            all_preds.extend(pred.cpu().numpy())
            all_targets.extend(target.cpu().numpy())

    total_samples = len(all_targets)
    if total_samples == 0:
        print(f"[{context}] 验证集中没有样本，无法评估。")
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0}

    avg_loss = running_loss / total_samples
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / total_samples
    # 使用 'macro' 平均，因为它对每个类别同等对待，适合类别不平衡的情况
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {"loss": avg_loss, "accuracy": accuracy, "f1_score": f1}


def plot_server_fl_history(history, output_dir):
    """绘制并保存服务器聚合的联邦学习历史曲线。"""
    if not history['round']:
        print("没有服务器聚合历史记录可供绘制。")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    ax1.plot(history['round'], history['accuracy'], marker='o', label='Accuracy')
    ax1.set_ylabel("准确率 (%)")
    ax1.set_title("服务器全局模型在统一验证集上的性能")
    ax1.grid(True)
    
    ax2.plot(history['round'], history['f1_score'], marker='o', color='g', label='F1-Score (Macro)')
    ax2.set_ylabel("F1 Score (Macro)")
    ax2.grid(True)

    ax3.plot(history['round'], history['loss'], marker='o', color='r', label='Loss')
    ax3.set_ylabel("损失")
    ax3.set_xlabel("服务器通信轮次 (Round)")
    ax3.grid(True)
    
    fig.tight_layout()
    plot_save_path = os.path.join(output_dir, 'server_federated_learning_plot.png')
    plt.savefig(plot_save_path)
    print(f"服务器FL训练曲线图已保存至: {plot_save_path}")
    plt.close()