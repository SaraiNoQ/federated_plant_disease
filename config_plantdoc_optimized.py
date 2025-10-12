# config_plantdoc_optimized.py - 专门为PlantDoc小样本数据集优化的配置

import torch

# --- 1. 路径设置 ---
OUTPUT_DIR = './outputs_plantdoc_optimized'

# --- 2. 顶级联邦学习参数 (农场间) ---
NUM_FARMS = 6             # 总农场数量
SERVER_ROUNDS = 15        # 增加服务器聚合轮数，给小样本更多训练机会

# --- 3. 农场内联邦学习参数 (模拟设备/计算单元) ---
CLIENT_UNITS_PER_FARM = 3 # 减少每个农场的计算单元数量，避免数据过度分割
FARM_FL_ROUNDS = 8        # 增加农场内部联邦学习轮数
UNITS_PER_FARM_ROUND = 3  # 每轮选择较少的计算单元
EPOCHS_PER_UNIT = 2       # 增加每个单元的本地训练轮数
EPOCHS_PER_TEACHER_TRAIN = 3

# --- 4. 模型与训练参数 ---
BATCH_SIZE = 16           # 减小批量大小，适应小样本
LEARNING_RATE_DISTILL = 0.0005 # 降低学习率，防止过拟合
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
NUM_WORKERS = 4           # 减少工作进程数
MODEL_ARCHITECTURE = 'resnet18'  # 使用更轻量的模型
DISTILL_MODEL_ARCH = 'mobilenet_v2'

# --- 5. 蒸馏参数 ---
TEMPERATURE = 2.5         # 降低蒸馏温度，减少噪声
ALPHA_DISTILLATION = 0.7  # 增加软目标权重，更多依赖教师知识

# --- 6. 数据集划分参数 ---
DATA_PARTITION_STRATEGY = 'dirichlet'
DIRICHLET_ALPHA = 1.0     # 增加alpha值，使数据分布更均匀

# --- 7. 小样本学习专用配置 ---
USE_DATASET = 'plantdoc'
DATA_DIR = './data/plantdoc_merged'  # 需要您提供PlantDoc数据集路径

# PlantDoc数据集有27个类别
NUM_CLASSES_TOTAL = 27

# 为PlantDoc重新设计的农场类别分配（需要根据实际文件夹名调整）
FARM_CLASS_ALLOCATION = {
    'Farm_A': ['Apple rust leaf', 'Apple leaf', 'Apple Scab Leaf', 'Bell_pepper leaf spot', 'Bell_pepper leaf'],
    'Farm_B': ['Blueberry leaf', 'Cherry leaf', 'Corn Gray leaf spot', 'Corn rust leaf', 'Corn leaf blight'],
    'Farm_C': ['Peach leaf', 'Potato leaf early blight', 'Potato leaf late blight', 'Raspberry leaf', 'Soyabean leaf'],
    'Farm_D': ['Grape leaf blight', 'Grape leaf', 'Grape Esca (Black Measles)', 'Squash Powdery mildew leaf', 'Strawberry leaf'],
    'Farm_E': ['Tomato Early blight leaf', 'Tomato leaf', 'Tomato leaf bacterial spot', 'Tomato leaf mosaic virus'],
    'Farm_F': ['Tomato Septoria leaf spot', 'Tomato leaf yellow virus', 'Tomato two spotted spider mites leaf']
}

NUM_CLASSES_DATASET = NUM_CLASSES_TOTAL

# --- 8. 数据增强配置 ---
USE_ADVANCED_AUGMENTATION = True
AUGMENTATION_INTENSITY = 0.8  # 数据增强强度

# --- 9. 迁移学习配置 ---
USE_TRANSFER_LEARNING = True
FREEZE_BACKBONE_LEVEL = 0.7   # 冻结70%的backbone层

# --- 10. 小样本学习策略 ---
USE_FEW_SHOT_STRATEGIES = True
MIN_SAMPLES_PER_CLASS = 5     # 每个类别最少样本数阈值

# --- 11. 强化学习配置 ---
USE_RL_CLIENT_SELECTION = True
LOCAL_RL_EXPLORATION = 1.5    # 降低探索因子

# --- 12. 评估配置 ---
CREATE_GLOBAL_VALIDATION_SET = True
GLOBAL_VALIDATION_SPLIT = 0.15 # 增加验证集比例

# --- 13. FedProx Regularization ---
FEDPROX_MU = 0.02             # 增加正则化强度

# --- 14. 动态农场加入配置 ---
INITIAL_FARMS = ['Farm_A', 'Farm_B', 'Farm_C', 'Farm_D', 'Farm_E', 'Farm_F']

# --- 15. 分层强化学习 (HRL) 配置 ---
W_EFFICIENCY_GLOBAL = 1.0
W_DIVERSITY_GLOBAL = 0.3      # 增加多样性权重
W_ACC_GLOBAL = 1.0
W_LAT_GLOBAL = 0.03

# HRL 耦合参数
TRANSFER_BUDGET = 0.3         # 增加知识迁移预算

# --- 16. 显存优化配置 ---
MEMORY_MONITOR_INTERVAL = 20
CLEAN_CACHE_INTERVAL = 10

# --- 17. 早停配置 ---
USE_EARLY_STOPPING = True
EARLY_STOPPING_PATIENCE = 5
MIN_DELTA = 0.001

# --- 18. 学习率调度 ---
USE_LR_SCHEDULER = True
LR_SCHEDULER_STEP_SIZE = 5
LR_SCHEDULER_GAMMA = 0.5

# 校验类别分配
assert sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values()) == NUM_CLASSES_TOTAL, \
f"类别分配错误，应有{NUM_CLASSES_TOTAL}个类别，实际分配了{sum(len(classes) for classes in FARM_CLASS_ALLOCATION.values())}个。"
