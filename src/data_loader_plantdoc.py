# src/data_loader_plantdoc.py - 专门为PlantDoc小样本数据集优化的数据加载器

import torch
from torch.utils.data import DataLoader, random_split, Subset, Dataset
from torchvision import datasets, transforms
import numpy as np
import os
import config_plantdoc_optimized as config
from PIL import Image, ImageEnhance, ImageFilter
import random

# --- 1. 高级数据预处理转换 ---
def get_advanced_transforms():
    """为小样本数据集提供更强大的数据增强"""
    
    # 基础训练变换
    base_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 高级训练变换（包含更多增强）
    advanced_train_transform = transforms.Compose([
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # 验证变换
    val_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    return base_train_transform, advanced_train_transform, val_transform

# --- 2. 自定义数据增强类 ---
class AdvancedAugmentation:
    """自定义高级数据增强方法"""
    
    def __init__(self, intensity=0.8):
        self.intensity = intensity
        
    def __call__(self, img):
        if random.random() < self.intensity:
            # 随机选择一种增强方法
            augment_method = random.choice([
                self._color_enhancement,
                self._sharpness_enhancement,
                self._gaussian_blur,
                self._random_erasing_simulated,
                self._perspective_transform
            ])
            img = augment_method(img)
        return img
    
    def _color_enhancement(self, img):
        """颜色增强"""
        enhancer = ImageEnhance.Color(img)
        factor = random.uniform(0.8, 1.2)
        return enhancer.enhance(factor)
    
    def _sharpness_enhancement(self, img):
        """锐度增强"""
        enhancer = ImageEnhance.Sharpness(img)
        factor = random.uniform(0.8, 1.5)
        return enhancer.enhance(factor)
    
    def _gaussian_blur(self, img):
        """高斯模糊"""
        radius = random.uniform(0.1, 1.0)
        return img.filter(ImageFilter.GaussianBlur(radius=radius))
    
    def _random_erasing_simulated(self, img):
        """模拟随机擦除"""
        img_array = np.array(img)
        h, w = img_array.shape[:2]
        
        # 随机选择一个区域进行遮挡
        erase_h = int(h * random.uniform(0.1, 0.3))
        erase_w = int(w * random.uniform(0.1, 0.3))
        erase_x = random.randint(0, w - erase_w)
        erase_y = random.randint(0, h - erase_h)
        
        # 用随机颜色填充
        fill_color = [random.randint(0, 255) for _ in range(3)]
        img_array[erase_y:erase_y+erase_h, erase_x:erase_x+erase_w] = fill_color
        
        return Image.fromarray(img_array)
    
    def _perspective_transform(self, img):
        """透视变换模拟"""
        # 简单的仿射变换
        angle = random.uniform(-10, 10)
        return img.rotate(angle, resample=Image.BILINEAR, expand=False)

# --- 3. 包装器，用于应用变换和标签映射 ---
class FarmSubsetWrapper(Dataset):
    """
    一个包装器，用于处理农场特定子集的变换和标签映射。
    它将全局标签转换为农场本地的 [0, N-1] 标签。
    """

    def __init__(self, subset, transform, global_classes, local_class_map, use_advanced_aug=False):
        self.subset = subset
        self.transform = transform
        self.global_classes = global_classes
        self.local_class_map = local_class_map
        self.use_advanced_aug = use_advanced_aug
        if use_advanced_aug:
            self.advanced_aug = AdvancedAugmentation(config.AUGMENTATION_INTENSITY)

    def __getitem__(self, index):
        x, global_label_idx = self.subset[index]
        
        # 应用基础变换
        if self.transform:
            x = self.transform(x)
        
        # 应用高级增强（如果需要）
        if self.use_advanced_aug:
            # 将tensor转换回PIL图像进行高级增强
            if isinstance(x, torch.Tensor):
                # 反标准化并转换回PIL
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                x_denorm = x * std + mean
                x_denorm = torch.clamp(x_denorm, 0, 1)
                x_pil = transforms.ToPILImage()(x_denorm)
                
                # 应用高级增强
                x_pil = self.advanced_aug(x_pil)
                
                # 转换回tensor并重新标准化
                x = transforms.ToTensor()(x_pil)
                x = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])(x)

        # 对于狄利克雷划分或全局验证集，local_map就是全局map，因此映射不变
        # 对于手动划分，它会映射到更小的标签空间
        global_class_name = self.global_classes[global_label_idx]
        y_farm_local = self.local_class_map.get(global_class_name, global_label_idx)

        return x, y_farm_local

    def __len__(self):
        return len(self.subset)

# --- 4. 狄利克雷划分函数（优化版本） ---
def dirichlet_split_noniid_balanced(dataset, num_farms, alpha, seed=42, min_samples_per_class=5):
    """
    使用狄利克雷分布将数据集划分为Non-IID的子集，但保证每个类别在每个农场都有最小样本数。
    """
    np.random.seed(seed)

    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)

    idx_by_class = {i: np.where(labels == i)[0] for i in range(num_classes)}
    
    # 检查每个类别的样本数是否足够
    for class_id, indices in idx_by_class.items():
        if len(indices) < min_samples_per_class * num_farms:
            print(f"警告: 类别 {class_id} 只有 {len(indices)} 个样本，少于推荐的最小值 {min_samples_per_class * num_farms}")

    class_proportions = np.random.dirichlet([alpha] * num_classes, num_farms)

    farm_indices = [[] for _ in range(num_farms)]

    for c in range(num_classes):
        class_c_indices = idx_by_class[c]
        np.random.shuffle(class_c_indices)

        # 确保每个农场至少获得min_samples_per_class个样本
        min_samples = min(min_samples_per_class, len(class_c_indices) // num_farms)
        
        # 为每个农场分配最小样本
        for i in range(num_farms):
            if len(class_c_indices) > 0:
                farm_indices[i].extend(class_c_indices[:min_samples].tolist())
                class_c_indices = class_c_indices[min_samples:]
        
        # 剩余样本按比例分配
        if len(class_c_indices) > 0:
            proportions = class_proportions[:, c]
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(class_c_indices)).astype(int)[:-1]

            farm_splits = np.split(class_c_indices, proportions)

            for i in range(num_farms):
                if i < len(farm_splits):
                    farm_indices[i].extend(farm_splits[i].tolist())

    farm_data_indices = {f"Farm_{chr(65 + i)}": indices for i, indices in enumerate(farm_indices)}
    return farm_data_indices

# --- 5. 核心数据加载与划分函数（优化版本） ---
def get_farm_dataloaders_plantdoc(
        data_dir: str,
        client_units_per_farm: int,
        batch_size: int,
        num_workers: int,
        create_global_val_set: bool = False,
        global_val_split: float = 0.15
):
    """
    为每个农场准备数据加载器，专门为PlantDoc小样本数据集优化。
    """
    all_farm_data = {}

    # 获取数据变换（移到函数开头）
    base_train_transform, advanced_train_transform, val_transform = get_advanced_transforms()

    # 检查数据集是否存在
    if not os.path.exists(data_dir):
        print(f"错误: 数据集路径不存在: {data_dir}")
        print("请确保PlantDoc数据集已正确放置在指定路径")
        return None, None

    try:
        full_dataset = datasets.ImageFolder(data_dir)
        print(f"PlantDoc总类别数: {len(full_dataset.classes)}")
        print(f"PlantDoc总样本数: {len(full_dataset)}")
        
        # 打印每个类别的样本数
        print("\n各类别样本统计:")
        for class_name in full_dataset.classes:
            class_indices = [i for i, (_, label) in enumerate(full_dataset.samples) 
                           if full_dataset.classes[label] == class_name]
            print(f"  {class_name}: {len(class_indices)} 个样本")
            
    except Exception as e:
        print(f"加载数据集时出错: {e}")
        return None, None

    global_val_loader = None
    main_dataset_indices = list(range(len(full_dataset)))

    if create_global_val_set:
        print(f"\n--- 正在创建全局验证集 (划分比例: {global_val_split}) ---")
        total_size = len(full_dataset)
        val_size = int(total_size * global_val_split)
        train_size = total_size - val_size

        # 使用分层抽样确保验证集中每个类别都有代表性
        from sklearn.model_selection import train_test_split
        labels = [full_dataset.targets[i] for i in main_dataset_indices]
        main_dataset_indices, global_val_indices = train_test_split(
            main_dataset_indices, 
            test_size=global_val_split, 
            stratify=labels,
            random_state=42
        )

        global_val_subset = Subset(full_dataset, global_val_indices)
        global_val_wrapped = FarmSubsetWrapper(global_val_subset, val_transform, full_dataset.classes,
                                               full_dataset.class_to_idx)
        global_val_loader = DataLoader(global_val_wrapped, batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)

        print(f"全局验证集创建完成，样本数: {len(global_val_wrapped)}")
        print(f"用于农场划分的剩余样本数: {len(main_dataset_indices)}")

    # ---------------- 数据划分策略选择 ----------------
    farm_indices_to_process = {}
    global_classes = full_dataset.classes
    global_class_to_idx = full_dataset.class_to_idx

    if config.DATA_PARTITION_STRATEGY == 'dirichlet':
        print(f"\n--- 正在使用平衡狄利克雷分布 (alpha={config.DIRICHLET_ALPHA}) 划分数据 ---")
        farm_indices_to_process = dirichlet_split_noniid_balanced(
            full_dataset, config.NUM_FARMS, config.DIRICHLET_ALPHA,
            min_samples_per_class=config.MIN_SAMPLES_PER_CLASS
        )

    elif config.DATA_PARTITION_STRATEGY == 'manual':
        print("\n--- 正在使用手动方式划分数据 ---")
        for farm_id, assigned_class_names in config.FARM_CLASS_ALLOCATION.items():
            target_global_indices_set = {global_class_to_idx[name] for name in assigned_class_names}
            farm_specific_indices = [
                idx for idx in main_dataset_indices
                if full_dataset.samples[idx][1] in target_global_indices_set
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

        # 检查是否可以进行分层抽样
        farm_labels = [full_dataset.targets[i] for i in farm_specific_main_indices]
        
        # 检查每个类别的样本数是否足够进行分层抽样
        from collections import Counter
        label_counts = Counter(farm_labels)
        min_samples_per_class = min(label_counts.values()) if label_counts else 0
        
        # 如果某个类别的样本数少于2，无法进行分层抽样，使用随机划分
        if min_samples_per_class < 2:
            print(f"  警告: 农场 {farm_id} 有类别样本数少于2，使用随机划分")
            total_size = len(farm_specific_main_indices)
            val_size = int(total_size * 0.2)
            train_size = total_size - val_size
            farm_train_indices, farm_val_indices = random_split(
                farm_specific_main_indices, [train_size, val_size], 
                generator=torch.Generator().manual_seed(42)
            )
        else:
            # 使用分层抽样划分训练/验证集
            farm_train_indices, farm_val_indices = train_test_split(
                farm_specific_main_indices,
                test_size=0.2,
                stratify=farm_labels,
                random_state=42
            )

        farm_val_subset = Subset(full_dataset, farm_val_indices)
        farm_val_wrapped = FarmSubsetWrapper(farm_val_subset, val_transform, global_classes, farm_class_to_idx_local)
        farm_val_loader = DataLoader(farm_val_wrapped, batch_size=batch_size, shuffle=False, num_workers=num_workers)

        print(
            f"农场 {farm_id}: 总样本数 {len(farm_specific_main_indices)}, 训练样本 {len(farm_train_indices)}, 验证样本 {len(farm_val_indices)}")

        farm_unit_loaders = []
        if client_units_per_farm > 0 and len(farm_train_indices) > 0:
            unit_splits = []
            unit_labels = [full_dataset.targets[i] for i in farm_train_indices]
            
            # 检查是否可以使用分层抽样划分计算单元
            from collections import Counter
            label_counts = Counter(unit_labels)
            min_samples_per_class = min(label_counts.values()) if label_counts else 0
            
            if min_samples_per_class < client_units_per_farm:
                print(f"  警告: 农场 {farm_id} 样本数不足，使用随机划分计算单元")
                # 使用随机划分
                unit_splits = np.array_split(farm_train_indices, client_units_per_farm)
            else:
                # 使用分层抽样划分计算单元
                try:
                    from sklearn.model_selection import StratifiedKFold
                    skf = StratifiedKFold(n_splits=client_units_per_farm, shuffle=True, random_state=42)
                    
                    for _, unit_indices in skf.split(farm_train_indices, unit_labels):
                        unit_splits.append([farm_train_indices[i] for i in unit_indices])
                except Exception as e:
                    print(f"  警告: 分层抽样失败 ({e})，使用随机划分计算单元")
                    unit_splits = np.array_split(farm_train_indices, client_units_per_farm)

            for i, unit_indices in enumerate(unit_splits):
                if len(unit_indices) == 0:
                    farm_unit_loaders.append(DataLoader([], batch_size=batch_size))
                    print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 0 个训练样本。")
                    continue

                unit_subset = Subset(full_dataset, unit_indices)
                # 使用高级数据增强
                unit_dataset = FarmSubsetWrapper(
                    unit_subset, 
                    advanced_train_transform if config.USE_ADVANCED_AUGMENTATION else base_train_transform, 
                    global_classes, 
                    farm_class_to_idx_local,
                    use_advanced_aug=config.USE_ADVANCED_AUGMENTATION
                )
                loader = DataLoader(unit_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
                farm_unit_loaders.append(loader)
                print(f"  - 农场 {farm_id} - 单元 {i}: 分配了 {len(unit_dataset)} 个训练样本。")

        all_farm_data[farm_id] = {
            "unit_loaders": farm_unit_loaders, "val_loader": farm_val_loader,
            "num_classes": num_farm_classes, "class_names": farm_class_names,
            "class_to_idx_local": farm_class_to_idx_local
        }

    return all_farm_data, global_val_loader
