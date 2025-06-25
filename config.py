# config.py

import torch

# --- 1. 路径设置 ---
# 在Kaggle环境中, input目录是只读的
DATA_DIR = './data/color'
# Kaggle的工作目录是 /kaggle/working/，所有输出文件会在这里
OUTPUT_DIR = './outputs'

# --- 2. 模型预训练配置 ---
# 设置为 True，可加载下面 CUSTOM_PRETRAINED_MODEL_PATH 指定的、我们自己训练好的模型（例如，继续训练）
# 设置为 False，将使用标准的 ImageNet 预训练权重
USE_CUSTOM_PRETRAINED = False  # <--- 新增的开关

# 标准 ImageNet 预训练权重路径
IMAGENET_PRETRAINED_PATH = './data/models/resnet34-b627a593.pth'

# 我们自己训练好的模型路径 (例如，上次联邦学习的输出)
# 注意：第一次运行时，这个文件可能不存在。
CUSTOM_PRETRAINED_MODEL_PATH = './data/models/best_plant_disease_model.pth'

# --- 3. 联邦学习参数 ---
NUM_CLIENTS = 10         # 客户端总数
NUM_ROUNDS = 15          # 全局通信轮数
CLIENTS_PER_ROUND = 5    # 每轮选择的客户端数量
EPOCHS_PER_CLIENT = 5    # 每个客户端本地训练的epoch数

# --- 4. 模型与训练参数 ---
NUM_CLASSES = 38         # 数据集类别数
BATCH_SIZE = 32          # 批量大小
LEARNING_RATE = 0.001    # 学习率
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2          # Kaggle中建议的数据加载进程数