import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import config  # 导入config

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
        x, global_label_idx = self.subset[index]
        if self.transform:
            x = self.transform(x)

        # 对于狄利克雷划分或全局验证集，local_map就是全局map，因此映射不变
        # 对于手动划分，它会映射到更小的标签空间
        global_class_name = self.global_classes[global_label_idx]
        y_farm_local = self.local_class_map.get(global_class_name, global_label_idx)

        return x, y_farm_local

    def __len__(self):
        return len(self.subset)


# --- 3. 狄利克雷划分函数 ---
def dirichlet_split_noniid(dataset, num_farms, alpha, seed=42):
    """
    使用狄利克雷分布将数据集划分为Non-IID的子集。
    """
    np.random.seed(seed)

    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)

    idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}

    class_proportions = np.random.dirichlet([alpha] * num_classes, num_farms)

    farm_indices = [[] for _ in range(num_farms)]

    for c in range(num_classes):
        class_c_indices = idx_by_class[c]
        np.random.shuffle(class_c_indices)

        proportions = class_proportions[:, c]
        proportions = proportions / proportions.sum()
        proportions = (np.cumsum(proportions) * len(class_c_indices)).astype(int)[:-1]

        farm_splits = np.split(class_c_indices, proportions)

        for i in range(num_farms):
            farm_indices[i].extend(farm_splits[i].tolist())

    farm_data_indices = {f"Farm_{chr(65 + i)}": indices for i, indices in enumerate(farm_indices)}
    return farm_data_indices


# --- 4. 核心数据加载与划分函数 ---
def get_farm_dataloaders(
        data_dir: str,
        client_units_per_farm: int,
        batch_size: int,
        num_workers: int,
        create_global_val_set: bool = False,
        global_val_split: float = 0.1
):
    """
    为每个农场准备数据加载器，支持手动和狄利克雷划分。
    """
    all_farm_data = {}

    full_plantvillage_dataset = datasets.ImageFolder(data_dir)
    print(f"PlantVillage总类别数: {len(full_plantvillage_dataset.classes)}")
    print(f"PlantVillage总样本数: {len(full_plantvillage_dataset)}")

    global_val_loader = None
    main_dataset_indices = list(range(len(full_plantvillage_dataset)))

    if create_global_val_set:
        print(f"\n--- 正在创建全局验证集 (划分比例: {global_val_split}) ---")
        total_size = len(full_plantvillage_dataset)
        val_size = int(total_size * global_val_split)
        train_size = total_size - val_size

        main_dataset_indices_split, global_val_indices = random_split(
            range(total_size), [train_size, val_size], generator=torch.Generator().manual_seed(42)
        )
        main_dataset_indices = list(main_dataset_indices_split)  # 转换为list

        global_val_subset = Subset(full_plantvillage_dataset, global_val_indices)
        global_val_wrapped = FarmSubsetWrapper(global_val_subset, val_transform, full_plantvillage_dataset.classes,
                                               full_plantvillage_dataset.class_to_idx)
        global_val_loader = DataLoader(global_val_wrapped, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)

        print(f"全局验证集创建完成，样本数: {len(global_val_wrapped)}")
        print(f"用于农场划分的剩余样本数: {len(main_dataset_indices)}")

    # ---------------- 数据划分策略选择 ----------------
    farm_indices_to_process = {}
    global_classes = full_plantvillage_dataset.classes
    global_class_to_idx = full_plantvillage_dataset.class_to_idx

    if config.DATA_PARTITION_STRATEGY == 'dirichlet':
        print(f"\n--- 正在使用狄利克雷分布 (alpha={config.DIRICHLET_ALPHA}) 划分数据 ---")
        # 1. 获取主训练数据集对应的标签列表
        # full_plantvillage_dataset.targets 是一个包含所有样本标签的列表
        main_labels = np.array(full_plantvillage_dataset.targets)[main_dataset_indices]

        # 2. 准备 dirichlet_split 函数需要的数据
        # 我们需要一个临时的 "dataset-like" 对象，或者直接传递所需参数
        num_classes = len(global_classes)

        # 3. 按类别对索引进行分组，但只在 main_dataset_indices 范围内分组
        # key是类别号，value是该类别在 main_dataset_indices 中的索引位置
        idx_by_class = {i: np.where(main_labels == i)[0] for i in range(num_classes)}

        # 4. 调用狄利克雷分布，生成每个农场应该获取的各类别的样本数量
        class_proportions = np.random.dirichlet([config.DIRICHLET_ALPHA] * num_classes, config.NUM_FARMS)

        # 5. 分配索引
        farm_relative_indices_map = {i: [] for i in range(config.NUM_FARMS)}
        for c in range(num_classes):
            class_c_indices = idx_by_class[c]  # 这是类别c在main_labels中的索引
            np.random.shuffle(class_c_indices)

            # 计算每个农场从这个类别中分得的样本数
            proportions_c = class_proportions[:, c]
            num_samples_c = len(class_c_indices)

            # (num_samples_for_farm_c_0, num_samples_for_farm_c_1, ...)
            num_samples_per_farm_c = (proportions_c * num_samples_c).astype(int)

            # 修正由于取整可能造成的总数不匹配问题
            remainder = num_samples_c - num_samples_per_farm_c.sum()
            add_ons = np.random.choice(config.NUM_FARMS, remainder)
            for farm_idx in add_ons:
                num_samples_per_farm_c[farm_idx] += 1

            # 切分索引并分配
            start = 0
            for i in range(config.NUM_FARMS):
                end = start + num_samples_per_farm_c[i]
                farm_relative_indices_map[i].extend(class_c_indices[start:end])
                start = end

        # 6. 将相对索引转换回原始数据集的绝对索引
        for i in range(config.NUM_FARMS):
            farm_id = f"Farm_{chr(65 + i)}"
            relative_indices = farm_relative_indices_map[i]
            # main_dataset_indices[j] 可以将相对索引 j 映射回绝对索引
            absolute_indices = [main_dataset_indices[j] for j in relative_indices]
            farm_indices_to_process[farm_id] = absolute_indices

    elif config.DATA_PARTITION_STRATEGY == 'manual':
        print("\n--- 正在使用手动方式划分数据 ---")
        for farm_id, assigned_class_names in config.FARM_CLASS_ALLOCATION.items():
            target_global_indices_set = {global_class_to_idx[name] for name in assigned_class_names}
            farm_specific_indices = [
                idx for idx in main_dataset_indices
                if full_plantvillage_dataset.samples[idx][1] in target_global_indices_set
            ]
            farm_indices_to_process[farm_id] = farm_specific_indices
    else:
        raise ValueError(f"未知的数据划分策略: {config.DATA_PARTITION_STRATEGY}")

    for farm_id, farm_specific_main_indices in farm_indices_to_process.items():
        if not farm_specific_main_indices:
            print(f"警告: 农场 {farm_id} 没有分配到任何数据样本。跳过此农场。")
            continue

        if config.DATA_PARTITION_STRATEGY == 'dirichlet':
            num_farm_classes = len(global_classes)
            farm_class_names = global_classes
            farm_class_to_idx_local = global_class_to_idx
        else:
            farm_class_names = sorted(config.FARM_CLASS_ALLOCATION[farm_id])
            num_farm_classes = len(farm_class_names)
            farm_class_to_idx_local = {name: i for i, name in enumerate(farm_class_names)}

        print(f"\n--- 正在为农场 {farm_id} 准备数据 (类别数: {num_farm_classes}) ---")

        farm_train_size = int(0.8 * len(farm_specific_main_indices))
        farm_val_size = len(farm_specific_main_indices) - farm_train_size
        if farm_train_size == 0 or farm_val_size == 0:
            print(f"  警告: 农场 {farm_id} 样本数过少({len(farm_specific_main_indices)})，无法划分为训练/验证集。")
            continue

        farm_train_indices, farm_val_indices = random_split(
            farm_specific_main_indices, [farm_train_size, farm_val_size]
        )

        farm_val_subset = Subset(full_plantvillage_dataset, farm_val_indices)
        farm_val_wrapped = FarmSubsetWrapper(farm_val_subset, val_transform, global_classes, farm_class_to_idx_local)
        farm_val_loader = DataLoader(farm_val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(
            f"农场 {farm_id}: 总样本数 {len(farm_specific_main_indices)}, 训练样本 {len(farm_train_indices)}, 验证样本 {len(farm_val_indices)}")

        farm_unit_loaders = []
        if client_units_per_farm > 0 and len(farm_train_indices) > 0:
            unit_splits = np.array_split(list(farm_train_indices), client_units_per_farm)
            for i, unit_indices in enumerate(unit_splits):
                if len(unit_indices) == 0:
                    farm_unit_loaders.append(DataLoader([], batch_size=batch_size))
                    print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 0 个训练样本。")
                    continue

                unit_subset = Subset(full_plantvillage_dataset, unit_indices)
                unit_dataset = FarmSubsetWrapper(unit_subset, train_transform, global_classes, farm_class_to_idx_local)
                loader = DataLoader(unit_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                farm_unit_loaders.append(loader)
                print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 {len(unit_dataset)} 个训练样本。")

        all_farm_data[farm_id] = {
            "unit_loaders": farm_unit_loaders, "val_loader": farm_val_loader,
            "num_classes": num_farm_classes, "class_names": farm_class_names,
            "class_to_idx_local": farm_class_to_idx_local
        }

    return all_farm_data, global_val_loader