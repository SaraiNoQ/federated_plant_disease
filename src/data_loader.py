# src/data_loader.py

from torch.utils.data import DataLoader, random_split, Subset
from torchvision import datasets, transforms
import numpy as np

def get_dataloaders(data_dir: str, num_clients: int, batch_size: int, num_workers: int):
    """
    加载数据、进行预处理，并将其分割为联邦学习客户端和全局验证集。

    Args:
        data_dir (str): 数据集根目录。
        num_clients (int): 客户端数量。
        batch_size (int): 批量大小。
        num_workers (int): DataLoader的工作进程数。

    Returns:
        tuple: (客户端加载器列表, 全局验证加载器)
    """
    # 定义预处理
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

    # 加载并分割训练/验证集
    full_dataset = datasets.ImageFolder(data_dir, transform=train_transform)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])
    
    # 为验证集应用正确的变换
    val_dataset.dataset.transform = val_transform
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # IID（独立同分布）方式分割训练数据给客户端
    client_data_size = len(train_dataset) // num_clients
    indices = list(range(len(train_dataset)))
    np.random.shuffle(indices)

    client_loaders = []
    for i in range(num_clients):
        start_idx = i * client_data_size
        end_idx = start_idx + client_data_size
        subset_indices = indices[start_idx:end_idx]
        client_subset = Subset(train_dataset, subset_indices)
        loader = DataLoader(client_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        client_loaders.append(loader)
    
    return client_loaders, val_loader