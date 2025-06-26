import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import os

# --- 1. 数据预处理转换 ---
train_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- 2. 包装器，用于应用变换和标签映射 ---
class FarmSubsetWrapper(Dataset):
    """
    一个包装器，用于处理农场特定子集的变换和标签映射。
    它将全局标签转换为农场本地的 [0, N-1] 标签。
    """
    def __init__(self, subset, transform, global_classes, local_class_map):
        self.subset = subset
        self.transform = transform
        self.global_classes = global_classes
        self.local_class_map = local_class_map

    def __getitem__(self, index):
        # 从原始数据集中获取图像(PIL)和全局标签
        x, global_label_idx = self.subset[index]

        # 应用变换
        if self.transform:
            x = self.transform(x)

        # 将全局标签映射到农场本地标签
        global_class_name = self.global_classes[global_label_idx]
        if global_class_name in self.local_class_map:
            y_farm_local = self.local_class_map[global_class_name]
        else:
            # 对于全局验证集，本地映射就是全局映射
            y_farm_local = global_label_idx
            
        return x, y_farm_local

    def __len__(self):
        return len(self.subset)

# --- 3. 核心数据加载与划分函数 ---
def get_farm_dataloaders(
    data_dir: str,
    farm_class_allocation: dict,
    client_units_per_farm: int,
    batch_size: int,
    num_workers: int,
    create_global_val_set: bool = False,
    global_val_split: float = 0.1
):
    """
    为每个农场准备数据加载器，并可选择性地创建一个全局验证集。
    """
    all_farm_data = {}
    
    # --- vvvvvvvvvvvvvv 修改开始 vvvvvvvvvvvvvv ---
    # 步骤1: 加载原始数据集，不附加任何变换，以获取原始PIL图像
    full_plantvillage_dataset = datasets.ImageFolder(data_dir)
    # --- ^^^^^^^^^^^^^^ 修改结束 ^^^^^^^^^^^^^^ ---

    print(f"PlantVillage总类别数: {len(full_plantvillage_dataset.classes)}")
    print(f"PlantVillage总样本数: {len(full_plantvillage_dataset)}")

    global_val_loader = None
    main_dataset_indices = list(range(len(full_plantvillage_dataset)))

    if create_global_val_set:
        print(f"\n--- 正在创建全局验证集 (划分比例: {global_val_split}) ---")
        total_size = len(full_plantvillage_dataset)
        val_size = int(total_size * global_val_split)
        train_size = total_size - val_size
        
        main_dataset_indices, global_val_indices = random_split(
            range(total_size), [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )

        # 为全局验证集创建 Subset 和包装器，以应用 val_transform
        global_val_subset = Subset(full_plantvillage_dataset, global_val_indices)
        
        # 对于全局验证集，本地映射就是全局映射本身
        global_classes = full_plantvillage_dataset.classes
        global_class_to_idx = full_plantvillage_dataset.class_to_idx
        
        # 使用 FarmSubsetWrapper 来应用变换
        global_val_wrapped = FarmSubsetWrapper(global_val_subset, val_transform, global_classes, global_class_to_idx)
        global_val_loader = DataLoader(global_val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"全局验证集创建完成，样本数: {len(global_val_wrapped)}")
        print(f"用于农场划分的剩余样本数: {len(main_dataset_indices)}")

    global_classes = full_plantvillage_dataset.classes

    for farm_id, assigned_class_names in farm_class_allocation.items():
        print(f"\n--- 正在为农场 {farm_id} 准备数据 ---")
        
        # 从可用的主数据集索引中，筛选出属于该农场的样本索引
        target_class_indices_map = {name: i for i, name in enumerate(global_classes) if name in assigned_class_names}
        target_global_indices_set = set(target_class_indices_map.values())
        
        farm_specific_main_indices = []
        for idx in main_dataset_indices:
            _, target_idx = full_plantvillage_dataset.samples[idx]
            if target_idx in target_global_indices_set:
                farm_specific_main_indices.append(idx)
        
        if not farm_specific_main_indices:
            print(f"警告: 农场 {farm_id} 没有分配到任何数据样本。跳过此农场。")
            continue
            
        temp_class_to_idx = {name: i for i, name in enumerate(sorted(assigned_class_names))}
        
        # 分割农场数据为训练集和验证集 (80/20)
        farm_train_size = int(0.8 * len(farm_specific_main_indices))
        farm_val_size = len(farm_specific_main_indices) - farm_train_size
        farm_train_indices, farm_val_indices = random_split(
            farm_specific_main_indices, [farm_train_size, farm_val_size]
        )
        
        # 创建农场验证加载器
        farm_val_subset = Subset(full_plantvillage_dataset, farm_val_indices)
        farm_val_wrapped = FarmSubsetWrapper(farm_val_subset, val_transform, global_classes, temp_class_to_idx)
        farm_val_loader = DataLoader(farm_val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"农场 {farm_id}: 总样本数 {len(farm_specific_main_indices)}, 训练样本 {len(farm_train_indices)}, 验证样本 {len(farm_val_indices)}")
        
        # 将农场训练数据分割给内部的计算单元
        farm_unit_loaders = []
        if client_units_per_farm > 0 and len(farm_train_indices) > 0:
            shuffled_train_indices = list(farm_train_indices)
            np.random.shuffle(shuffled_train_indices)
            unit_data_size = len(shuffled_train_indices) // client_units_per_farm
            
            if unit_data_size == 0 and len(shuffled_train_indices) > 0:
                unit_splits = [shuffled_train_indices] + [[] for _ in range(client_units_per_farm - 1)]
            else:
                unit_splits = np.array_split(shuffled_train_indices, client_units_per_farm)

            for i, unit_indices in enumerate(unit_splits):
                if len(unit_indices) == 0:
                    # 创建一个空的加载器
                    farm_unit_loaders.append(DataLoader([], batch_size=batch_size))
                    print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 0 个训练样本。")
                    continue
                
                unit_subset = Subset(full_plantvillage_dataset, unit_indices)
                unit_dataset = FarmSubsetWrapper(unit_subset, train_transform, global_classes, temp_class_to_idx)
                loader = DataLoader(unit_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                farm_unit_loaders.append(loader)
                print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 {len(unit_dataset)} 个训练样本。")
        
        all_farm_data[farm_id] = {
            "unit_loaders": farm_unit_loaders,
            "val_loader": farm_val_loader,
            "num_classes": len(assigned_class_names),
            "class_names": assigned_class_names,
            "class_to_idx_local": temp_class_to_idx
        }

    return all_farm_data, global_val_loader