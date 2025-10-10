# 联邦学习项目显存溢出分析与解决方案

## 问题分析

根据错误信息和代码分析，显存溢出的主要原因如下：

### 1. 模型数量过多
- 每个农场有 `CLIENT_UNITS_PER_FARM = 5` 个客户端
- 每个客户端都有自己的私有教师模型
- 同时还有学生模型和迁移教师模型
- 在训练过程中同时存在多个模型副本

### 2. 批量大小过大
- `BATCH_SIZE = 128` 对于ResNet34和MobileNetV2来说过大
- 特别是在蒸馏过程中，需要同时处理多个模型的输出

### 3. 模型深度复制
- 代码中频繁使用 `copy.deepcopy()` 创建模型副本
- 这会占用大量显存，特别是对于大型模型

### 4. 梯度累积
- 在蒸馏过程中同时计算多个损失函数
- 没有及时清理中间变量

## 解决方案

### 方案1：立即降低显存占用的配置修改

```python
# 在config.py中添加以下配置优化

# 降低批量大小
BATCH_SIZE = 32  # 从128降低到32

# 减少客户端数量
CLIENT_UNITS_PER_FARM = 3  # 从5降低到3

# 减少内部联邦学习轮次
FARM_FL_ROUNDS = 5  # 从10降低到5

# 减少每轮选择的客户端数量
UNITS_PER_FARM_ROUND = 2  # 从4降低到2

# 减少蒸馏轮次
DISTILLATION_EPOCHS = 10  # 从20降低到10

# 使用更轻量的模型
MODEL_ARCHITECTURE = 'resnet18'  # 从resnet34改为resnet18
DISTILL_MODEL_ARCH = 'mobilenet_v2'  # 保持轻量级学生模型

# 启用梯度检查点（如果支持）
USE_GRADIENT_CHECKPOINTING = True
```

### 方案2：代码层面的显存优化

在 `src/federated.py` 的 `local_distill_update` 函数中添加显存管理：

```python
def local_distill_update(
        local_teacher_model: nn.Module,
        student_model_to_train: nn.Module,
        global_student_model: nn.Module,
        prox_mu: float,
        train_loader: torch.utils.data.DataLoader,
        epochs: int,
        lr: float,
        temperature: float,
        alpha: float,
        device: torch.device,
        transfer_teacher_model: nn.Module = None,
        transfer_budget: float = 0.0
):
    # 添加显存管理
    torch.cuda.empty_cache() if torch.cuda.is_available() else None
    
    local_teacher_model.eval()
    if transfer_teacher_model:
        transfer_teacher_model.eval()
    
    student_model_to_train.train()
    optimizer = torch.optim.Adam(student_model_to_train.parameters(), lr=lr)
    
    for epoch in range(epochs):
        for inputs, hard_labels in train_loader:
            inputs, hard_labels = inputs.to(device), hard_labels.to(device)
            optimizer.zero_grad()
            
            # 使用torch.no_grad()减少显存占用
            with torch.no_grad():
                local_teacher_outputs = local_teacher_model(inputs)
            
            student_outputs = student_model_to_train(inputs)
            
            # 及时释放不需要的变量
            loss_ce = nn.CrossEntropyLoss()(student_outputs, hard_labels)
            loss_kd_local = nn.KLDivLoss(reduction='batchmean')(
                F.log_softmax(student_outputs / temperature, dim=1),
                F.softmax(local_teacher_outputs / temperature, dim=1)
            ) * (temperature * temperature)
            
            loss_exploit = alpha * loss_kd_local + (1 - alpha) * loss_ce
            
            # 释放中间变量
            del local_teacher_outputs
            
            loss_transfer = 0.0
            if transfer_teacher_model and transfer_budget > 0:
                with torch.no_grad():
                    transfer_teacher_outputs = transfer_teacher_model(inputs)
                
                loss_kd_transfer = nn.KLDivLoss(reduction='batchmean')(
                    F.log_softmax(student_outputs / temperature, dim=1),
                    F.softmax(transfer_teacher_outputs / temperature, dim=1)
                ) * (temperature * temperature)
                loss_transfer = loss_kd_transfer
                
                # 释放中间变量
                del transfer_teacher_outputs
            
            total_loss = (1 - transfer_budget) * loss_exploit + transfer_budget * loss_transfer
            
            if prox_mu > 0:
                prox_term = 0.0
                for local_p, global_p in zip(student_model_to_train.parameters(), global_student_model.parameters()):
                    prox_term += (local_p - global_p.detach()).norm(2)
                total_loss += (prox_mu / 2) * prox_term
            
            total_loss.backward()
            optimizer.step()
            
            # 清理梯度
            optimizer.zero_grad(set_to_none=True)
            
            # 定期清理显存
            if batch_idx % 10 == 0 and torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    return student_model_to_train.state_dict()
```

### 方案3：训练脚本优化

在 `run_fl.py` 中减少模型副本：

```python
# 在客户端选择循环中，避免不必要的深度复制
for client_idx in selected_client_indices:
    # 使用浅拷贝而不是深度拷贝
    student_model_copy = type(farm_student_model)(num_classes=farm_data["num_classes"])
    student_model_copy.load_state_dict(farm_student_model.state_dict())
    student_model_copy.to(config.DEVICE)
    
    updated_student_dict = local_distill_update(
        local_teacher_model=local_teachers[farm_id][client_idx],
        student_model_to_train=student_model_copy,  # 使用浅拷贝
        # ... 其他参数
    )
    student_updates.append(updated_student_dict)
    
    # 及时删除模型副本
    del student_model_copy
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

## 推荐的配置修改

创建一个新的优化配置文件：

```python
# config_optimized.py
import torch

# 路径设置
DATA_DIR = './data/color'
OUTPUT_DIR = './outputs_optimized'

# 模型配置 - 使用更轻量的模型
MODEL_ARCHITECTURE = 'resnet18'  # 从resnet34改为resnet18
DISTILL_MODEL_ARCH = 'mobilenet_v2'

# 训练参数优化
BATCH_SIZE = 32  # 显著降低批量大小
CLIENT_UNITS_PER_FARM = 3  # 减少客户端数量
FARM_FL_ROUNDS = 5  # 减少内部轮次
UNITS_PER_FARM_ROUND = 2  # 减少每轮选择的客户端
EPOCHS_PER_UNIT = 2  # 减少本地训练轮次

# 蒸馏参数优化
DISTILLATION_EPOCHS = 10  # 减少蒸馏轮次
SERVER_ROUNDS = 5  # 减少服务器轮次

# 其他参数保持不变...
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
```

## 检测显存使用情况

添加显存监控代码：

```python
# 在训练循环中添加显存监控
def print_memory_usage(prefix=""):
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        cached = torch.cuda.memory_reserved() / 1024**3
        print(f"{prefix} GPU显存使用: {allocated:.2f}GB / {cached:.2f}GB")

# 在关键位置调用
print_memory_usage("训练开始前")
```

## 总结

通过以上优化措施，预计可以显著降低显存占用：
1. 批量大小从128降低到32（减少75%）
2. 客户端数量从5降低到3（减少40%）
3. 使用更轻量的ResNet18代替ResNet34
4. 优化代码中的显存管理

这些修改应该能够解决当前的显存溢出问题。
