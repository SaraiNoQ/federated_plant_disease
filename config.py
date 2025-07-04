# config.py

import torch

# --- 1. 路径设置 ---
DATA_DIR = './data/color'  # PlantVillage数据集的根目录
OUTPUT_DIR = './outputs'

# --- 2. 模型预训练配置 ---
# 初始源模型路径 (例如，ImageNet预训练的ResNet34)
# INITIAL_SOURCE_MODEL_PATH = './data/models/resnet18-f37072fd.pth'
# INITIAL_SOURCE_MODEL_PATH = './data/models/resnet34-b627a593.pth'

# --- 3. 顶级联邦学习参数 (农场间) ---
NUM_FARMS = 6             # 总农场数量 (A, B, C, D, E, F)
# SERVER_ROUNDS: 控制服务器聚合农场模型的轮数，如果只有一轮微调和蒸馏，则设为1
SERVER_ROUNDS = 3         # 服务器聚合农场模型的全局轮数,增加轮数以观察RL效果

# --- 4. 农场内联邦学习参数 (模拟设备/计算单元) ---
CLIENT_UNITS_PER_FARM = 5 # 每个农场内部的计算单元数量 (客户端)
FARM_FL_ROUNDS = 10       # 每个农场内部联邦学习的通信轮数
UNITS_PER_FARM_ROUND = 3  # 每轮农场内FL选择的计算单元数量
EPOCHS_PER_UNIT = 3       # 每个计算单元本地训练的epoch数

# --- 5. 模型与训练参数 ---
NUM_CLASSES_PLANTVILLAGE = 38 # PlantVillage总类别数
BATCH_SIZE = 32
LEARNING_RATE_FTL = 0.001   # 联邦迁移学习的初始学习率
LEARNING_RATE_DISTILL = 0.001 # 模型蒸馏的学习率
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 2
# 选择要使用的模型架构。可选项: 'resnet18', 'resnet34', 'efficientnet_b0', 'mobilenet_v3_small', 'mobilenet_v2'
MODEL_ARCHITECTURE = 'mobilenet_v3_small'

# --- 6. 蒸馏参数 ---
DISTILLATION_EPOCHS = 20
TEMPERATURE = 3.0           # 蒸馏温度
ALPHA_DISTILLATION = 0.7    # 蒸馏损失中，软目标损失的权重

# --- 7. 数据集划分参数 ---
# 这个字典定义了每个农场拥有的作物类别 (基于PlantVillage的文件夹名)
# 确保所有38个类别都被分配，且不重叠
FARM_CLASS_ALLOCATION = {
    'Farm_A': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew'],
    'Farm_B': ['Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
               'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'],
    'Farm_C': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)'],
    'Farm_D': ['Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy', 'Potato___Early_blight'],
    'Farm_E': ['Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy',
               'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch'],
    'Farm_F': ['Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
}
# 校验类别分配 (确保总共38类)
assert sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values()) == NUM_CLASSES_PLANTVILLAGE, \
    f"类别分配错误，应有{NUM_CLASSES_PLANTVILLAGE}个类别，实际分配了{sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values())}个。"

# 8.1 强化学习客户端选择
RL_STRATEGY = 'Thompson'        # <<< 新增：选择 'Thompson' 或 'UCB'
USE_RL_FARM_SELECTION = True # 是否启用RL选择农场进行聚合
FARMS_PER_SERVER_ROUND = 3     # 如果启用RL，每轮服务器聚合选择多少个农场
RL_EXPLORATION_FACTOR = 2.0    # RL (UCB1) 探索因子 C
USE_RL_CLIENT_SELECTION = True  # 是否启用RL在农场内部选择客户端

# 8.2 评估
# 创建一个包含所有38个类别的全局验证集，用于评估服务器模型
CREATE_GLOBAL_VALIDATION_SET = True
GLOBAL_VALIDATION_SPLIT = 0.1 # 从所有数据中分出10%作为全局验证集

# --- 9. FedProx Regularization ---
FEDPROX_MU = 0.01                # FedProx proximal term strength (a good starting point)

# --- 10. RL aggregation ---
USE_RL_AGGREGATION = True       # 是否启用RL进行服务器聚合
TUNE_PPO_OFFLINE = True         # 是否在每轮后进行PPO离线调优
PPO_TUNING_EPOCHS = 50          # 离线调优的轮数

# --- 10. Offline Optimization Hyperparameters ---

# PPO (保留或调整)
PPO_LR_ACTOR = 0.0003
PPO_LR_CRITIC = 0.001
PPO_GAMMA = 0.99
PPO_K_EPOCHS = 40
PPO_EPS_CLIP = 0.2

# A2C
A2C_LR = 0.001
A2C_GAMMA = 0.99

# Bayesian Optimization
BO_INITIAL_POINTS = 10 # 初始随机探索点数

# Evolutionary Algorithm
EVO_POPULATION_SIZE = 50  # 种群大小
EVO_MUTATION_RATE = 0.1   # 变异率
EVO_CROSSOVER_RATE = 0.8  # 交叉率
EVO_ELITISM_COUNT = 2     # 精英保留数量