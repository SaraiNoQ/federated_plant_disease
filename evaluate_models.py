# evaluate_models.py

import torch
import os
import pandas as pd
import re

# 确保能导入项目模块
import config
from src.models import build_model, build_student_model
from src.utils import evaluate_model
from src.data_loader import get_farm_dataloaders

def get_dataloaders_for_eval():
    """复用data_loader逻辑，为评估获取所有需要的数据加载器。"""
    print("正在为评估准备数据集...")
    all_farms_data, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        farm_class_allocation=config.FARM_CLASS_ALLOCATION,
        client_units_per_farm=0, # 不需要单元数据
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=True,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )
    
    farm_val_loaders = {farm_id: data['val_loader'] for farm_id, data in all_farms_data.items()}
    farm_num_classes_map = {farm_id: data['num_classes'] for farm_id, data in all_farms_data.items()}
    return farm_val_loaders, global_val_loader, farm_num_classes_map

def main():
    print("--- 开始对所有已保存的模型进行全面评估 ---")
    
    farm_val_loaders, global_val_loader, farm_num_classes_map = get_dataloaders_for_eval()
    
    if not global_val_loader:
        print("错误: 无法创建全局验证集，无法进行全面评估。")
        return

    results = []
    output_dir = config.OUTPUT_DIR
    model_files = [f for f in os.listdir(output_dir) if f.endswith('.pth')]

    for model_file in model_files:
        model_path = os.path.join(output_dir, model_file)
        print(f"\n>>> 正在评估模型: {model_file}")

        num_classes = 0
        is_student = 'student' in model_file
        
        # --- vvvvvvvvvvvvvv 修改开始 vvvvvvvvvvvvvv ---
        # 从文件名解析模型信息（修复后的正则表达式）
        farm_id_match = re.search(r'farm_(Farm_[A-F])', model_file, re.IGNORECASE)
        farm_id_key = None
        eval_loader_specific = None
        
        if farm_id_match:
            # 确保找到的farm_id格式与config中的key一致 (例如 'Farm_C')
            found_id = farm_id_match.group(1)
            # 遍历config中的keys，找到大小写不敏感的匹配项
            for key in config.FARM_CLASS_ALLOCATION.keys():
                if key.lower() == found_id.lower():
                    farm_id_key = key
                    break
        
        if farm_id_key:
            num_classes = farm_num_classes_map.get(farm_id_key)
            eval_loader_specific = farm_val_loaders.get(farm_id_key)
            print(f"  检测到农场模型: {farm_id_key}, 类别数: {num_classes}")
        else: # 服务器模型
            num_classes = config.NUM_CLASSES_PLANTVILLAGE
            print(f"  检测到服务器/全局模型, 类别数: {num_classes}")
        # --- ^^^^^^^^^^^^^^ 修改结束 ^^^^^^^^^^^^^^ ---

        if not num_classes or num_classes == 0:
            print("  无法确定类别数，跳过此模型。")
            continue

        # 构建模型
        try:
            if is_student:
                model = build_student_model(num_classes=num_classes)
            else:
                model = build_model(num_classes=num_classes)
            
            model.load_state_dict(torch.load(model_path, map_location=config.DEVICE))
            model.to(config.DEVICE)
        except RuntimeError as e:
            print(f"  加载模型权重时出错: {e}")
            print("  这通常意味着文件名解析出的类别数与实际文件中存储的模型不符。跳过此模型。")
            continue
        
        # 在对应数据集上评估
        metrics_specific = {'accuracy': 'N/A', 'f1_score': 'N/A'}
        if eval_loader_specific:
            metrics_specific = evaluate_model(model, eval_loader_specific, config.DEVICE, context=f"对应验证集 ({farm_id_key})")
            print(f"  在 {farm_id_key} 验证集上: Acc={metrics_specific['accuracy']:.2f}%, F1={metrics_specific['f1_score']:.4f}")
        else:
            print("  无对应的农场验证集 (可能是服务器模型)。")
            
        # 在全局数据集上评估
        metrics_global = {'accuracy': 'N/A', 'f1_score': 'N/A'}
        if num_classes == config.NUM_CLASSES_PLANTVILLAGE:
            metrics_global = evaluate_model(model, global_val_loader, config.DEVICE, context="全局验证集")
            print(f"  在全局验证集上: Acc={metrics_global['accuracy']:.2f}%, F1={metrics_global['f1_score']:.4f}")
        else:
            print(f"  模型类别数({num_classes})与全局({config.NUM_CLASSES_PLANTVILLAGE})不符，跳过全局评估。")

        results.append({
            "model_file": model_file,
            "type": "Student" if is_student else "Teacher/Server",
            "farm_id": farm_id_key if farm_id_key else "Server",
            "acc_specific": metrics_specific['accuracy'],
            "f1_specific": metrics_specific['f1_score'],
            "acc_global": metrics_global['accuracy'],
            "f1_global": metrics_global['f1_score']
        })

    # 打印最终结果表格
    if results:
        df = pd.DataFrame(results)
        print("\n\n--- 最终评估结果汇总 ---")
        print(df.to_string())
        
        csv_path = os.path.join(output_dir, 'evaluation_summary.csv')
        df.to_csv(csv_path, index=False)
        print(f"\n评估结果已保存至: {csv_path}")

if __name__ == '__main__':
    main()