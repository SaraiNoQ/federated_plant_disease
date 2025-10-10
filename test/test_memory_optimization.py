# test_memory_optimization.py - 测试显存优化效果

import torch
import sys
import os

def test_memory_usage():
    """测试显存使用情况"""
    print("=== 显存优化测试 ===")
    
    # 检查CUDA可用性
    if not torch.cuda.is_available():
        print("警告: CUDA不可用，无法测试GPU显存优化")
        return
    
    # 获取GPU信息
    gpu_count = torch.cuda.device_count()
    print(f"检测到 {gpu_count} 个GPU设备")
    
    for i in range(gpu_count):
        props = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {props.name}")
        print(f"  总显存: {props.total_memory / 1024**3:.2f} GB")
        print(f"  CUDA计算能力: {props.major}.{props.minor}")
    
    # 测试显存分配和释放
    print("\n=== 显存分配测试 ===")
    
    # 初始显存状态
    initial_allocated = torch.cuda.memory_allocated() / 1024**3
    initial_cached = torch.cuda.memory_reserved() / 1024**3
    print(f"初始显存使用: {initial_allocated:.2f}GB / {initial_cached:.2f}GB")
    
    # 分配一些显存
    print("\n分配1GB显存...")
    tensor1 = torch.randn(256, 1024, 1024).cuda()  # 约1GB
    allocated_after = torch.cuda.memory_allocated() / 1024**3
    print(f"分配后显存: {allocated_after:.2f}GB")
    
    # 释放显存
    print("\n释放显存...")
    del tensor1
    torch.cuda.empty_cache()
    final_allocated = torch.cuda.memory_allocated() / 1024**3
    print(f"释放后显存: {final_allocated:.2f}GB")
    
    print("\n=== 模型显存占用测试 ===")
    
    # 测试不同模型的显存占用
    from torchvision import models
    
    model_sizes = {
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'mobilenet_v2': models.mobilenet_v2,
    }
    
    batch_size = 32
    input_size = (3, 224, 224)
    
    for model_name, model_builder in model_sizes.items():
        print(f"\n测试 {model_name}:")
        
        # 创建模型
        model = model_builder(num_classes=38)
        model = model.cuda()
        
        # 计算模型参数数量
        total_params = sum(p.numel() for p in model.parameters())
        print(f"  参数数量: {total_params:,}")
        
        # 测试前向传播显存占用
        torch.cuda.empty_cache()
        initial_mem = torch.cuda.memory_allocated() / 1024**3
        
        # 模拟前向传播
        with torch.no_grad():
            dummy_input = torch.randn(batch_size, *input_size).cuda()
            output = model(dummy_input)
        
        after_forward = torch.cuda.memory_allocated() / 1024**3
        forward_memory = after_forward - initial_mem
        print(f"  前向传播显存占用: {forward_memory:.2f}GB (批量大小: {batch_size})")
        
        # 清理
        del model, dummy_input, output
        torch.cuda.empty_cache()

def compare_configurations():
    """比较原始配置和优化配置"""
    print("\n=== 配置对比 ===")
    
    # 原始配置
    original_config = {
        'BATCH_SIZE': 128,
        'CLIENT_UNITS_PER_FARM': 5,
        'FARM_FL_ROUNDS': 10,
        'UNITS_PER_FARM_ROUND': 4,
        'EPOCHS_PER_UNIT': 3,
        'MODEL_ARCHITECTURE': 'resnet34',
        'DISTILLATION_EPOCHS': 20,
        'SERVER_ROUNDS': 10,
    }
    
    # 优化配置
    optimized_config = {
        'BATCH_SIZE': 32,
        'CLIENT_UNITS_PER_FARM': 3,
        'FARM_FL_ROUNDS': 5,
        'UNITS_PER_FARM_ROUND': 2,
        'EPOCHS_PER_UNIT': 2,
        'MODEL_ARCHITECTURE': 'resnet18',
        'DISTILLATION_EPOCHS': 10,
        'SERVER_ROUNDS': 5,
    }
    
    print("原始配置 vs 优化配置:")
    for key in original_config.keys():
        original_val = original_config[key]
        optimized_val = optimized_config[key]
        
        # 对于数值类型的配置项计算减少百分比
        if isinstance(original_val, (int, float)) and isinstance(optimized_val, (int, float)):
            reduction = (original_val - optimized_val) / original_val * 100
            print(f"  {key}: {original_val} -> {optimized_val} (减少 {reduction:.1f}%)")
        else:
            # 对于字符串类型的配置项，只显示变化
            print(f"  {key}: {original_val} -> {optimized_val}")

def check_requirements():
    """检查依赖包版本"""
    print("\n=== 依赖包检查 ===")
    
    try:
        import torch
        import torchvision
        import numpy
        
        print(f"PyTorch版本: {torch.__version__}")
        print(f"TorchVision版本: {torchvision.__version__}")
        print(f"NumPy版本: {numpy.__version__}")
        
        # 检查CUDA版本
        if torch.cuda.is_available():
            print(f"CUDA版本: {torch.version.cuda}")
        else:
            print("CUDA: 不可用")
            
    except ImportError as e:
        print(f"导入错误: {e}")

def main():
    """主测试函数"""
    print("联邦学习项目显存优化测试")
    print("=" * 50)
    
    # 检查依赖
    check_requirements()
    
    # 比较配置
    compare_configurations()
    
    # 测试显存使用
    test_memory_usage()
    
    print("\n=== 优化建议 ===")
    print("1. 使用优化后的配置文件: config_optimized.py")
    print("2. 运行优化后的训练脚本: python run_fl_optimized.py")
    print("3. 监控显存使用情况，根据需要进一步调整配置")
    print("4. 如果仍然遇到显存问题，可以:")
    print("   - 进一步降低批量大小")
    print("   - 减少客户端数量")
    print("   - 使用更轻量的模型")
    print("   - 启用梯度检查点")

if __name__ == "__main__":
    main()
