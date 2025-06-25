# src/models.py

import torch
from torchvision import models
import torch.nn as nn

def build_model(num_classes: int, pretrained_path: str = None, farm_specific_load: bool = False):
    """
    构建并返回一个ResNet34模型。
    可以加载ImageNet预训练权重，或针对特定农场任务加载微调后的权重。

    Args:
        num_classes (int): 输出类别的数量。
        pretrained_path (str, optional): 预训练权重文件的本地路径。
        farm_specific_load (bool): 如果为True，意味着pretrained_path指向的是一个
                                   已经为特定农场类别数调整过的模型（例如，农场内FL聚合后的模型）。
                                   这种情况下，加载权重时假设fc层维度已匹配。
                                   如果为False，通常用于加载ImageNet权重，fc层会被替换。
    Returns:
        torch.nn.Module: 构建好的PyTorch模型。
    """
    print(f"正在初始化ResNet34模型结构，目标类别数: {num_classes}")
    model = models.resnet34(pretrained=False) # 始终先用False，手动加载

    num_ftrs_original = model.fc.in_features # ResNet34原始fc输入特征数

    # 替换最后的全连接层以匹配我们的任务
    model.fc = nn.Linear(num_ftrs_original, num_classes)
    print(f"模型分类头已修改为支持 {num_classes} 个类别。")

    if pretrained_path:
        try:
            print(f"正在从本地路径加载权重: {pretrained_path}")
            checkpoint_state_dict = torch.load(pretrained_path, map_location=torch.device('cpu'))
            current_model_dict = model.state_dict()

            if farm_specific_load:
                # 如果是加载农场特定的微调模型，我们期望fc层维度已经匹配
                # 或者至少键名是 'fc.weight' 和 'fc.bias'
                # 这种情况下，直接尝试加载，如果fc层不匹配，load_state_dict会报错
                # 或者，我们可以像下面一样更安全地加载
                print("加载农场特定模型，将尝试匹配所有层...")
                # 过滤不匹配或形状不符的层
                filtered_state_dict = {
                    k: v for k, v in checkpoint_state_dict.items()
                    if k in current_model_dict and current_model_dict[k].shape == v.shape
                }
                if not filtered_state_dict:
                    print(f"警告: 从 {pretrained_path} 加载时，没有找到任何匹配的层。模型将是随机初始化的。")

                current_model_dict.update(filtered_state_dict)
                model.load_state_dict(current_model_dict)
                print(f"农场特定权重加载完成。共加载了 {len(filtered_state_dict)}/{len(current_model_dict)} 个层。")
                if 'fc.weight' not in filtered_state_dict and 'fc.weight' in checkpoint_state_dict:
                     print("提示: 'fc.weight' 层可能因形状不匹配未加载。请检查num_classes是否正确。")

            else: # 加载通用预训练权重 (如ImageNet)，fc层通常不匹配
                print("加载通用预训练模型 (如ImageNet)，将忽略不匹配的分类头。")
                # 过滤掉预训练权重中的fc层（如果存在且形状不匹配）
                # 以及其他可能不匹配的层
                pretrained_dict_for_transfer = {}
                for k, v in checkpoint_state_dict.items():
                    if k in current_model_dict and current_model_dict[k].shape == v.shape:
                        pretrained_dict_for_transfer[k] = v
                    elif 'fc' in k:
                        print(f"  - 忽略预训练权重中的层: {k} (分类头不匹配)")
                    else:
                        print(f"  - 忽略预训练权重中的层: {k} (未知原因，可能模型结构已变)")


                current_model_dict.update(pretrained_dict_for_transfer)
                model.load_state_dict(current_model_dict, strict=False) # strict=False允许部分加载
                print(f"通用预训练权重加载完成。尝试加载了 {len(pretrained_dict_for_transfer)} 个层。")
                if 'fc.weight' not in pretrained_dict_for_transfer and 'fc.weight' in checkpoint_state_dict :
                     print("提示: 'fc.weight' 层被忽略，这是加载ImageNet权重到不同类别数模型时的预期行为。")


        except FileNotFoundError:
            print(f"警告: 权重文件未找到: {pretrained_path}. 模型将是随机或部分随机初始化的。")
        except Exception as e:
            print(f"加载权重时发生未知错误: {e}. 模型将是随机或部分随机初始化的。")
    else:
        print("未提供预训练权重路径。模型将从随机权重开始训练（除了fc层）。")

    return model

def build_student_model(num_classes: int, architecture='mobilenet_v2'):
    """构建轻量级学生模型。"""
    if architecture == 'mobilenet_v2':
        student_model = models.mobilenet_v2(pretrained=True) # 可以用ImageNet预训练的MobileNet
        student_model.classifier[1] = nn.Linear(student_model.last_channel, num_classes)
    elif architecture == 'shufflenet_v2': # 另一个选择
        student_model = models.shufflenet_v2_x1_0(pretrained=True)
        student_model.fc = nn.Linear(student_model.fc.in_features, num_classes)
    else: # 简单自定义CNN示例
        print(f"警告: 未知的学生模型架构 '{architecture}', 将使用一个非常简单的自定义CNN。")
        student_model = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(), nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(32 * 56 * 56, 128), nn.ReLU(), # 输入尺寸依赖于224x224输入
            nn.Linear(128, num_classes)
        )
    print(f"学生模型 ({architecture}) 已构建，支持 {num_classes} 个类别。")
    return student_model