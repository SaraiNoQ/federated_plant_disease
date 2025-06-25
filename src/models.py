# src/models.py

import torch
from torchvision import models
import torch.nn as nn

def build_model(num_classes: int, pretrained_path: str = None):
    """
    构建并返回一个ResNet34模型，采用更健壮的方式加载预训练权重。
    
    该函数的核心逻辑是：
    1. 先初始化一个带有随机权重的模型架构。
    2. 立刻修改模型的最后一层（分类头）以匹配目标任务的类别数。
    3. 然后，尝试从本地路径加载预训练权重，并手动过滤掉不匹配的层。
    
    Args:
        num_classes (int): 输出类别的数量。
        pretrained_path (str, optional): 预训练权重文件的本地路径。

    Returns:
        torch.nn.Module: 构建好的PyTorch模型。
    """
    # 1. 初始化模型，始终不使用在线预训练权重
    print("正在初始化ResNet34模型结构...")
    model = models.resnet34(pretrained=False)

    # 2. 替换最后的全连接层以匹配我们的任务
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    print(f"模型分类头已修改为支持 {num_classes} 个类别。")

    # 3. 尝试从本地文件加载预训练权重（如果提供了路径）
    if pretrained_path:
        try:
            print(f"正在从本地路径加载预训练权重: {pretrained_path}")
            
            # 加载预训练的 state_dict
            checkpoint = torch.load(pretrained_path, map_location=torch.device('cpu'))
            
            # 获取当前模型的 state_dict
            model_dict = model.state_dict()
            
            # --- (!!! 这是解决问题的核心代码 !!!) ---
            # 1. 过滤掉不匹配的层 (特别是fc层)
            #    只保留在两个模型中都存在且形状相同的层
            pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict and model_dict[k].shape == v.shape}
            
            # 2. 更新当前模型的 state_dict
            model_dict.update(pretrained_dict)
            
            # 3. 加载我们手动筛选过的 state_dict
            model.load_state_dict(model_dict)
            # --- (核心代码结束) ---
            
            print(f"权重加载成功。总共加载了 {len(pretrained_dict)}/{len(model_dict)} 个匹配的层。")
            if 'fc.weight' not in pretrained_dict:
                print("提示: 'fc.weight' 层由于形状不匹配已被成功忽略，这是预期的行为。")

        except FileNotFoundError:
            print(f"警告: 预训练权重文件未找到: {pretrained_path}")
            print("模型将从随机权重开始训练。")
        except Exception as e:
            print(f"加载本地权重时发生未知错误: {e}")
            print("模型将从随机权重开始训练。")
    else:
        print("未提供预训练权重路径。模型将从随机权重开始训练。")
        
    return model