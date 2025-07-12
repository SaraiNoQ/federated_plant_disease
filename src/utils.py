import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
from sklearn.metrics import f1_score
import numpy as np
import copy

def evaluate_model(model, dataloader, device, context="Evaluation"):
    """
    Evaluates the model's performance on a given dataset, returning loss, accuracy, and F1-score.
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_preds = []
    all_targets = []
    running_loss = 0.0
    
    if not dataloader or len(dataloader.dataset) == 0:
        # print(f"[{context}] DataLoader is empty, skipping evaluation.")
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
        # print(f"[{context}] No samples in validation set, cannot evaluate.")
        return {"loss": 0.0, "accuracy": 0.0, "f1_score": 0.0}

    avg_loss = running_loss / total_samples
    accuracy = 100. * np.sum(np.array(all_preds) == np.array(all_targets)) / total_samples
    f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)

    return {"loss": avg_loss, "accuracy": accuracy, "f1_score": f1}


def plot_server_fl_history(history, output_dir):
    """
    Plots and saves the federated learning history curves for the server model.
    """
    if not history['round']:
        print("No server aggregation history to plot.")
        return

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    # --- English Labels ---
    ax1.plot(history['round'], history['accuracy'], marker='o', label='Accuracy')
    ax1.set_ylabel("Accuracy (%)")
    ax1.set_title("Server Global Model Performance on Unified Validation Set")
    ax1.grid(True)
    
    ax2.plot(history['round'], history['f1_score'], marker='o', color='g', label='F1-Score (Macro)')
    ax2.set_ylabel("F1 Score (Macro)")
    ax2.grid(True)

    ax3.plot(history['round'], history['loss'], marker='o', color='r', label='Loss')
    ax3.set_ylabel("Loss")
    ax3.set_xlabel("Server Communication Round")
    ax3.grid(True)
    
    fig.tight_layout()
    plot_save_path = os.path.join(output_dir, 'server_federated_learning_plot.png')
    plt.savefig(plot_save_path)
    print(f"Server FL training plot saved to: {plot_save_path}")
    plt.close()

def fuse_model(model: nn.Module, is_qat: bool = False):
    """
    对模型进行融合，将Conv-BN或Conv-BN-ReLU融合成一个层。
    这对量化是至关重要的。
    
    Args:
        model (nn.Module): The model to be fused.
        is_qat (bool): 是否为量化感知训练。PTQ用False。
    """
    # 遍历所有模块
    for module_name, module in model.named_children():
        # 如果有子模块，递归进去
        if hasattr(module, "children") and len(list(module.children())) > 0:
            fuse_model(module, is_qat)

        # 检查融合模式
        # 对于 PTQ，我们融合 Conv-BN
        if isinstance(module, nn.Sequential):
            # 模式: (Conv, BN, ReLU), (Conv, BN)
            patterns = [
                ['0', '1', '2'], # Conv, BN, ReLU
                ['0', '1']       # Conv, BN
            ]
            for pattern in patterns:
                # 检查序列中的模块类型是否匹配
                is_match = True
                # 确保序列长度足够
                if len(module) != len(pattern):
                    continue
                
                types_to_check = []
                if len(pattern) == 3:
                    types_to_check = [nn.Conv2d, nn.BatchNorm2d, nn.ReLU]
                elif len(pattern) == 2:
                    types_to_check = [nn.Conv2d, nn.BatchNorm2d]
                
                for i, module_idx in enumerate(pattern):
                    if not isinstance(module[int(module_idx)], types_to_check[i]):
                        is_match = False
                        break
                
                if is_match:
                    try:
                        torch.quantization.fuse_modules(module, pattern, inplace=True)
                    except Exception as e:
                        # print(f"Could not fuse modules in sequence {module_name}: {e}")
                        pass
    return model


def print_metrics(metrics: dict, title: str):
    """一个统一的打印函数，用于显示评估结果。"""
    print(f"{title} | "
          f"Loss: {metrics['loss']:.4f}, "
          f"Accuracy: {metrics['accuracy']:.2f}%, "
          f"F1-Score: {metrics['f1_score']:.4f}")


def quantize_and_evaluate_model(
        model_fp32: nn.Module,
        val_loader: torch.utils.data.DataLoader,
        device: torch.device
):
    """
    对给定的FP32模型进行训练后量化(PTQ)，并评估其性能。
    """
    print(f"\n--- 开始训练后量化 (PTQ) ---")

    # 评估FP32模型的基准性能
    metrics_fp32 = evaluate_model(model_fp32, val_loader, device)
    print_metrics(metrics_fp32, "[基准] FP32 学生模型性能")

    # 1. 准备模型：深拷贝，移至CPU，设为eval模式
    model_to_quantize = copy.deepcopy(model_fp32).to('cpu')
    model_to_quantize.eval()

    # 2. 融合模块
    # vvvvvvvvvvvvvvvvvvvvvvvv 修改融合逻辑 vvvvvvvvvvvvvvvvvvvvvvv
    print("      正在融合模型模块...")
    # torchvision.models.quantization.* 中的模型自带 fuse_model() 方法
    if hasattr(model_to_quantize, 'fuse_model'):
        model_to_quantize.fuse_model()
    else:
        print("      警告: 模型没有 'fuse_model' 方法，跳过融合。这可能会影响量化性能。")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

    # 3. 配置量化器
    # 对于PTQ，静态量化通常效果更好
    model_to_quantize.qconfig = torch.quantization.get_default_qconfig('fbgemm')

    # 4. 准备量化 (插入观察者)
    torch.quantization.prepare(model_to_quantize, inplace=True)

    # 5. 校准
    print("      正在使用验证数据进行校准...")
    cpu_val_loader = torch.utils.data.DataLoader(val_loader.dataset, batch_size=val_loader.batch_size)
    with torch.no_grad():
        for data, _ in cpu_val_loader:
            model_to_quantize(data)
            break

    # 6. 转换模型
    torch.quantization.convert(model_to_quantize, inplace=True)
    print("      量化完成！")

    # 7. 评估量化后的模型
    print("      正在评估INT8量化模型的性能...")
    metrics_int8 = evaluate_model(model_to_quantize, cpu_val_loader, torch.device('cpu'), "INT8 Quantized Model")

    # 8. 打印对比结果
    print("\n    --- PTQ 性能对比 ---")
    print_metrics(metrics_fp32, "[对比] FP32 学生模型")
    print_metrics(metrics_int8, "[对比] INT8 量化模型")
    acc_drop = metrics_fp32['accuracy'] - metrics_int8['accuracy']
    f1_drop = metrics_fp32['f1_score'] - metrics_int8['f1_score']
    print(f"    [结果] 精度下降: {acc_drop:.2f}%, F1分数下降: {f1_drop:.4f}")
    print("    -----------------------\n")

    return model_to_quantize.state_dict()