# config_optimized.py - 精简优化版本，降低显存占用

import torch

# --- 1. 路径设置 ---
DATA_DIR = './data/color'  # PlantVillage数据集的根目录
OUTPUT_DIR = './outputs_optimized'

# --- 2. 顶级联邦学习参数 (农场间) ---
NUM_FARMS = 6             # 总农场数量 (A, B, C, D, E, F)
SERVER_ROUNDS = 5         # 服务器聚合农场模型的全局轮数，从10降低到5

# --- 3. 农场内联邦学习参数 (模拟设备/计算单元) ---
CLIENT_UNITS_PER_FARM = 5 # 每个农场内部的计算单元数量 (客户端)，从5降低到3
FARM_FL_ROUNDS = 5       # 每个农场内部联邦学习的通信轮数，从10降低到5
UNITS_PER_FARM_ROUND = 3  # 每轮农场内FL选择的计算单元数量，从4降低到2
EPOCHS_PER_UNIT = 2       # 每个计算单元本地训练的epoch数，从3降低到2

# --- 4. 模型与训练参数 ---
NUM_CLASSES_PLANTVILLAGE = 38 # PlantVillage总类别数
BATCH_SIZE = 64              # 批量大小，从128降低到32
LEARNING_RATE_DISTILL = 0.001 # 模型蒸馏的学习率
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 8              # 从6降低到4
MODEL_ARCHITECTURE = 'resnet34'  # 从resnet34改为更轻量的resnet18
DISTILL_MODEL_ARCH = 'mobilenet_v2'

# --- 5. 蒸馏参数 ---
TEMPERATURE = 3.0           # 蒸馏温度
ALPHA_DISTILLATION = 0.7    # 蒸馏损失中，软目标损失的权重

# --- 6. 数据集划分参数 ---
DATA_PARTITION_STRATEGY = 'dirichlet'
DIRICHLET_ALPHA = 0.5

# 这个字典定义了每个农场拥有的作物类别 (基于PlantVillage的文件夹名)
FARM_CLASS_ALLOCATION = {
    'Farm_A': ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
               'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy'],
    'Farm_B': ['Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
                'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy'],
    'Farm_C': ['Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
               'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)'],
    'Farm_D': ['Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot',
               'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy'],
    'Farm_E': ['Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                'Strawberry___Leaf_scorch', 'Strawberry___healthy'],
    'Farm_F': ['Tomato___Bacterial_spot', 'Tomato___Early_blight',
               'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot',
               'Tomato___Spider_mites Two-spotted_spider_mite', 'Tomato___Target_Spot',
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']
}
# 校验类别分配 (确保总共38类)
assert sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values()) == NUM_CLASSES_PLANTVILLAGE, \
    f"类别分配错误，应有{NUM_CLASSES_PLANTVILLAGE}个类别，实际分配了{sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values())}个。"

# --- 7. 强化学习配置 ---
USE_RL_CLIENT_SELECTION = True  # 是否启用RL在农场内部选择客户端
LOCAL_RL_EXPLORATION = 2.0 # UCB的探索因子C

# --- 8. 评估配置 ---
CREATE_GLOBAL_VALIDATION_SET = True
GLOBAL_VALIDATION_SPLIT = 0.1 # 从所有数据中分出10%作为全局验证集

# --- 9. FedProx Regularization ---
FEDPROX_MU = 0.01                # FedProx proximal term strength

# --- 10. 动态农场加入配置 ---
INITIAL_FARMS = ['Farm_A', 'Farm_B', 'Farm_C', 'Farm_D', 'Farm_E', 'Farm_F']

# --- 11. 分层强化学习 (HRL) 配置 ---
# 高层RL奖励函数权重
W_EFFICIENCY_GLOBAL = 1.0  # 全局效率提升的权重
W_DIVERSITY_GLOBAL = 0.2   # 全局知识多样性的权重
W_ACC_GLOBAL = 1.0         # 平均准确率的权重
W_LAT_GLOBAL = 0.05        # 平均时延的权重

# HRL 耦合参数
TRANSFER_BUDGET = 0.2

# --- 12. 显存优化配置 ---
MEMORY_MONITOR_INTERVAL = 10        # 显存监控间隔
CLEAN_CACHE_INTERVAL = 5            # 清理显存缓存的间隔
