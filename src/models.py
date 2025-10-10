# src/models.py

import torch
from torchvision import models
from torchvision.models import quantization as quant_models
import torch.nn as nn
import config # 导入config来获取模型选择
from torch.hub import load_state_dict_from_url

def _modify_classifier(model, model_name, num_classes):
    """一个辅助函数，用于修改不同模型的分类头。"""
    if 'resnet' in model_name:
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, num_classes)
    elif 'efficientnet' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif 'mobilenet_v2' in model_name:
        num_ftrs = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif 'mobilenet_v3' in model_name:
        num_ftrs = model.classifier[3].in_features
        model.classifier[3] = nn.Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"未知的模型架构 '{model_name}'，无法修改分类头。")
    
    print(f"模型 '{model_name}' 的分类头已修改为支持 {num_classes} 个类别。")
    return model

def get_classifier_in_features(model, model_name):
    """一个辅助函数，用于获取不同模型分类头的输入特征数。"""
    if 'resnet' in model_name:
        return model.fc.in_features
    elif 'efficientnet' in model_name:
        return model.classifier[1].in_features
    elif 'mobilenet_v2' in model_name:
        return model.classifier[1].in_features
    elif 'mobilenet_v3' in model_name:
        return model.classifier[3].in_features
    raise ValueError(f"未知的模型架构 '{model_name}'，无法获取分类头特征数。")

def build_model(num_classes: int, pretrained_path: str = None, use_pretrained_weights=True):
    """
    构建并返回一个指定架构的模型。
    采用更稳健的权重加载方式，以避免分类头冲突的警告。

    Args:use_pretrained_weights
        num_classes (int): 输出类别的数量。
        pretrained_path (str, optional): 自定义预训练权重文件的本地路径。
        use_pretrained_weights (bool): 是否使用PyTorch官方的ImageNet预训练权重。
    
    Returns:
        torch.nn.Module: 构建好的PyTorch模型。
    """
    model_name = config.MODEL_ARCHITECTURE.lower()
    print(f"\n正在初始化模型: {model_name}，目标类别数: {num_classes}")

    # --- 1. 创建模型骨架 ---
    if model_name == 'resnet18':
        model = models.resnet18(pretrained=False)
        model_urls = {'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth'}
    elif model_name == 'resnet34':
        model = models.resnet34(pretrained=False)
        model_urls = {'resnet34': 'https://download.pytorch.org/models/resnet34-b627a593.pth'}
    elif model_name == 'efficientnet_b0':
        model = models.efficientnet_b0(pretrained=False)
        model_urls = {'efficientnet_b0': 'https://download.pytorch.org/models/efficientnet_b0_rwightman-3dd342df.pth'}
    elif model_name == 'mobilenet_v3_small':
        model = models.mobilenet_v3_small(pretrained=False)
        model_urls = {'mobilenet_v3_small': 'https://download.pytorch.org/models/mobilenet_v3_small-047dcff4.pth'}
    elif model_name == 'mobilenet_v2':
        model = models.mobilenet_v2(pretrained=False)
        model_urls = {'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'}
    else:
        raise ValueError(f"不支持的模型架构: {model_name}")

    # --- 2. 修改分类头以匹配我们的任务 ---
    model = _modify_classifier(model, model_name, num_classes)

    # --- 3. 加载权重 ---
    if pretrained_path:
        # 优先加载用户提供的本地权重文件
        try:
            print(f"正在从本地路径加载权重: {pretrained_path}")
            checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
            
            if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            print(f"从 {pretrained_path} 加载权重完成 (strict=False)。")

        except FileNotFoundError:
            print(f"警告: 权重文件未找到: {pretrained_path}. 将尝试加载ImageNet权重或使用随机权重。")
            use_pretrained_weights = True # 如果本地文件未找到，回退到使用官方预训练权重
        except Exception as e:
            print(f"加载本地权重时发生错误: {e}. 将尝试加载ImageNet权重或使用随机权重。")
            use_pretrained_weights = True

    if not pretrained_path and use_pretrained_weights:
        # 如果没有提供本地路径，且要求使用预训练权重，则从URL加载
        try:
            print(f"正在从PyTorch Hub加载 '{model_name}' 的ImageNet预训练权重...")
            # 从URL加载官方预训练权重
            state_dict = load_state_dict_from_url(model_urls[model_name], progress=True)
            
            # 核心：从下载的state_dict中移除分类头的权重
            # ResNet & ShuffleNet
            state_dict.pop('fc.weight', None)
            state_dict.pop('fc.bias', None)
            # EfficientNet & MobileNet
            # 遍历并移除所有键名包含 'classifier' 的项
            classifier_keys = [k for k in state_dict if 'classifier' in k]
            for k in classifier_keys:
                state_dict.pop(k, None)
            
            # 使用 strict=False 加载，它会只加载键名匹配的层（即所有backbone层）
            model.load_state_dict(state_dict, strict=False)
            print("ImageNet backbone权重加载成功。")

        except Exception as e:
            print(f"从URL加载ImageNet权重时发生错误: {e}. 模型将从随机权重开始。")
            
    elif not use_pretrained_weights:
        print("未提供预训练路径，且 use_pretrained_weights=False。模型将从随机权重开始训练。")

    return model


def build_student_model(num_classes: int, architecture: str = 'mobilenet_v2'):
    """
    构建一个轻量级的、为量化准备好的学生模型。
    我们直接使用 torchvision.models.quantization 中的模型，
    它们内部已经包含了 QuantStub 和 DeQuantStub。
    """
    arch_lower = architecture.lower()
    print(f"\n正在构建可量化的学生模型 ({arch_lower})，支持 {num_classes} 个类别...")

    model_builder = None
    if arch_lower == 'mobilenet_v2':
        # 使用 torchvision.models.quantization 中的 mobilenet_v2
        model_builder = quant_models.mobilenet_v2
    elif arch_lower == 'shufflenet_v2_x0_5':
        # 使用 torchvision.models.quantization 中的 shufflenet_v2_x0_5
        model_builder = quant_models.shufflenet_v2_x0_5
    else:
        raise NotImplementedError(f"可量化的学生模型架构 '{architecture}' 暂不支持。")

    # 构建预训练的FP32模型，这个模型已经准备好被量化
    student_model = model_builder(pretrained=True, quantize=False)  # quantize=False表示先加载FP32权重

    # 替换分类头以匹配我们的任务
    if 'mobilenet' in arch_lower:
        num_ftrs = student_model.classifier[1].in_features
        student_model.classifier[1] = nn.Linear(num_ftrs, num_classes)
    elif 'shufflenet' in arch_lower:
        num_ftrs = student_model.fc.in_features
        student_model.fc = nn.Linear(num_ftrs, num_classes)

    print("可量化的学生模型构建完成。")
    return student_model