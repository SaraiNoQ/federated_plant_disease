# src/data_loader.py

import torch
from torch.utils.data import DataLoader, random_split, Subset
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

# --- 2. 辅助函数：根据类别过滤数据集 ---
def get_subset_by_class_names(dataset: datasets.ImageFolder, target_class_names: list):
    """从ImageFolder数据集中提取仅包含特定类别名称的子集。"""
    target_indices = []
    class_to_idx = dataset.class_to_idx
    target_class_indices = [class_to_idx[name] for name in target_class_names if name in class_to_idx]

    if not target_class_indices:
        print(f"警告: 在数据集中未找到任何目标类别: {target_class_names}")
        return Subset(dataset, [])

    for i, (path, target_idx) in enumerate(dataset.samples):
        if target_idx in target_class_indices:
            target_indices.append(i)
    return Subset(dataset, target_indices)

# --- vvvvvvvvvvvvvv 修改开始 vvvvvvvvvvvvvv ---

# 将 FarmSubsetWrapper 移到函数外部，并使其自包含
class FarmSubsetWrapper(torch.utils.data.Dataset):
    """
    一个包装器，用于处理农场特定子集的变换和标签映射。
    它将全局标签转换为农场本地的 [0, N-1] 标签。
    """
    def __init__(self, subset, transform, global_classes, local_class_map):
        self.subset = subset
        self.transform = transform
        # 存储必要的映射信息，而不是依赖外部作用域
        self.global_classes = global_classes
        self.local_class_map = local_class_map

    def __getitem__(self, index):
        # 从原始数据集中获取图像和全局标签
        x, global_label_idx = self.subset[index]

        # 应用变换
        if self.transform:
            x = self.transform(x)

        # 将全局标签映射到农场本地标签
        global_class_name = self.global_classes[global_label_idx]
        if global_class_name in self.local_class_map:
            y_farm_local = self.local_class_map[global_class_name]
        else:
            # 这个错误现在应该不会发生了，但保留作为健全性检查
            raise ValueError(
                f"标签映射错误: '{global_class_name}' 不在当前农场的本地类别映射中。"
            )
        return x, y_farm_local

    def __len__(self):
        return len(self.subset)

# --- ^^^^^^^^^^^^^^ 修改结束 ^^^^^^^^^^^^^^ ---


# --- 3. 核心数据加载与划分函数 ---
def get_farm_dataloaders(
    data_dir: str,
    farm_class_allocation: dict,
    client_units_per_farm: int,
    batch_size: int,
    num_workers: int
):
    """
    为每个农场准备数据加载器，并进行农场内计算单元的划分。
    """
    all_farm_data = {}
    full_plantvillage_dataset = datasets.ImageFolder(data_dir)
    print(f"PlantVillage总类别数: {len(full_plantvillage_dataset.classes)}")
    print(f"PlantVillage总样本数: {len(full_plantvillage_dataset)}")

    # 获取全局类别列表，以传递给包装器
    global_classes = full_plantvillage_dataset.classes

    for farm_id, assigned_class_names in farm_class_allocation.items():
        print(f"\n--- 正在为农场 {farm_id} 准备数据 ---")
        print(f"分配的类别: {assigned_class_names}")

        # 1. 提取农场特定数据子集
        farm_specific_dataset = get_subset_by_class_names(full_plantvillage_dataset, assigned_class_names)

        if len(farm_specific_dataset) == 0:
            print(f"警告: 农场 {farm_id} 没有分配到任何数据样本。跳过此农场。")
            continue

        # 为当前农场创建本地类别映射
        temp_class_to_idx = {name: i for i, name in enumerate(sorted(assigned_class_names))}

        # 2. 分割农场数据为训练集和验证集 (80/20)
        farm_train_size = int(0.8 * len(farm_specific_dataset))
        farm_val_size = len(farm_specific_dataset) - farm_train_size

        if farm_train_size == 0 or farm_val_size == 0:
            print(f"警告: 农场 {farm_id} 数据太少 ({len(farm_specific_dataset)}个样本)，无法有效分割。")
            if len(farm_specific_dataset) < client_units_per_farm:
                 print(f"农场 {farm_id} 数据过少，跳过。")
                 continue
            farm_train_indices, farm_val_indices = list(range(len(farm_specific_dataset))), []
        else:
            farm_train_indices, farm_val_indices = random_split(
                range(len(farm_specific_dataset)), [farm_train_size, farm_val_size]
            )

        # 创建验证集子集和加载器
        farm_val_subset = Subset(farm_specific_dataset, farm_val_indices)
        farm_val_wrapped = FarmSubsetWrapper(farm_val_subset, val_transform, global_classes, temp_class_to_idx)
        farm_val_loader = DataLoader(farm_val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(f"农场 {farm_id}: 总样本数 {len(farm_specific_dataset)}, 训练样本 {len(farm_train_indices)}, 验证样本 {len(farm_val_indices)}")

        # 3. 将农场训练数据分割给内部的计算单元 (IID方式)
        farm_unit_loaders = []
        if client_units_per_farm > 0 and len(farm_train_indices) > 0:
            shuffled_train_indices = list(farm_train_indices)
            np.random.shuffle(shuffled_train_indices)

            unit_data_size = len(shuffled_train_indices) // client_units_per_farm
            if unit_data_size == 0:
                print(f"警告: 农场 {farm_id} 的训练数据无法均匀分配给 {client_units_per_farm} 个单元。将所有数据给第一个单元。")
                unit_splits = [shuffled_train_indices] + [[] for _ in range(client_units_per_farm - 1)]
            else:
                unit_splits = []
                for i in range(client_units_per_farm):
                    start_idx = i * unit_data_size
                    end_idx = (i + 1) * unit_data_size if i < client_units_per_farm - 1 else len(shuffled_train_indices)
                    unit_splits.append(shuffled_train_indices[start_idx:end_idx])

            for i, unit_indices in enumerate(unit_splits):
                unit_subset = Subset(farm_specific_dataset, unit_indices)
                unit_dataset = FarmSubsetWrapper(unit_subset, train_transform, global_classes, temp_class_to_idx)

                loader = DataLoader(unit_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                farm_unit_loaders.append(loader)
                print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 {len(unit_dataset)} 个训练样本。")
        else:
             print(f"农场 {farm_id} 没有计算单元或没有训练数据。")

        all_farm_data[farm_id] = {
            "unit_loaders": farm_unit_loaders,
            "val_loader": farm_val_loader,
            "num_classes": len(assigned_class_names),
            "class_names": assigned_class_names,
            "class_to_idx_local": temp_class_to_idx
        }

    return all_farm_data