import torch
import torch.nn as nn
from torchvision import models
import os
import re
import pandas as pd

# 导入你项目中的模块
import config
from src.data_loader import get_farm_dataloaders, FarmSubsetWrapper  # 明确导入FarmSubsetWrapper
from src.utils import evaluate_model


# 这个构建函数基本正确，我们保留它
def build_quantizable_shufflenet(num_classes: int):
    """
    构建一个可量化的ShuffleNetV2 x0.5模型。
    """
    # 1. 加载一个标准的ShuffleNet模型
    model = models.shufflenet_v2_x0_5(weights=None)  # 使用 weights=None 避免警告

    # 2. 修改分类头
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)

    # 3. 创建一个包装类以支持量化
    class QuantizableShuffleNet(nn.Module):
        def __init__(self, shufflenet_model):
            super(QuantizableShuffleNet, self).__init__()
            self.quant = torch.quantization.QuantStub()
            self.conv1 = shufflenet_model.conv1
            self.maxpool = shufflenet_model.maxpool
            self.stage2 = shufflenet_model.stage2
            self.stage3 = shufflenet_model.stage3
            self.stage4 = shufflenet_model.stage4
            self.conv5 = shufflenet_model.conv5
            self.fc = shufflenet_model.fc
            self.dequant = torch.quantization.DeQuantStub()

        def _forward_impl(self, x):
            # This is the original forward implementation
            x = self.conv1(x)
            x = self.maxpool(x)
            x = self.stage2(x)
            x = self.stage3(x)
            x = self.stage4(x)
            x = self.conv5(x)
            x = x.mean([2, 3])  # globalpool
            x = self.fc(x)
            return x

        def forward(self, x):
            x = self.quant(x)
            x = self._forward_impl(x)
            x = self.dequant(x)
            return x

    quantizable_model = QuantizableShuffleNet(model)
    return quantizable_model


def main():
    print("--- 开始评估量化后的学生模型 ---")

    # --- 1. 准备数据集 ---
    print("正在加载数据集...")
    all_farms_data_loaders, global_val_loader = get_farm_dataloaders(
        data_dir=config.DATA_DIR,
        client_units_per_farm=0,
        batch_size=config.BATCH_SIZE,
        num_workers=config.NUM_WORKERS,
        create_global_val_set=True,
        global_val_split=config.GLOBAL_VALIDATION_SPLIT
    )
    if not global_val_loader:
        print("错误：全局验证集未创建，无法进行评估。请检查config.py配置。")
        return
    print("数据集加载完成。")

    # --- 2. 发现并解析模型文件 ---
    model_dir = config.OUTPUT_DIR
    model_files_to_evaluate = []
    pattern = re.compile(r'quantized_student_model_farm_(Farm_[A-Z])_round_(\d+)\.pth')

    for filename in os.listdir(model_dir):
        match = pattern.match(filename)
        if match:
            farm_id = match.group(1)
            round_num = int(match.group(2))
            model_path = os.path.join(model_dir, filename)
            model_files_to_evaluate.append({
                "farm_id": farm_id,
                "round": round_num,
                "path": model_path
            })

    if not model_files_to_evaluate:
        print(f"在目录 '{model_dir}' 中没有找到符合格式 'quantized_student_model...' 的文件。")
        return

    model_files_to_evaluate.sort(key=lambda x: (x['farm_id'], x['round']))
    print(f"\n发现 {len(model_files_to_evaluate)} 个待评估的模型。")

    # --- 3. 循环评估 ---
    results = []
    device = torch.device('cpu')
    print(f"评估将在设备 '{device}' 上进行。")

    for model_info in model_files_to_evaluate:
        farm_id = model_info['farm_id']
        model_path = model_info['path']

        print(f"\n{'=' * 20}")
        print(f"正在评估模型: {os.path.basename(model_path)}")

        if farm_id not in all_farms_data_loaders:
            print(f"警告：找不到农场 '{farm_id}' 的数据加载器，跳过此模型。")
            continue

        # --- 3a. 构建并加载模型 (修改后的逻辑) ---
        farm_data = all_farms_data_loaders[farm_id]
        num_local_classes = farm_data['num_classes']

        # 1. 构建一个FP32的模型骨架
        model_to_test_fp32 = build_quantizable_shufflenet(num_classes=num_local_classes)
        model_to_test_fp32.eval()  # 设置为评估模式

        # 2. 为模型附加量化配置
        # 确保这里的 backend 与你训练时使用的 backend 一致
        model_to_test_fp32.qconfig = torch.quantization.get_default_qconfig('fbgemm')

        # 3. (关键步骤) 直接转换模型，创建量化层结构，但此时的scale/zero_point是默认值
        # 我们不是要训练它，只是为了让模型结构能够匹配state_dict
        model_quantized_prepared = torch.quantization.prepare(model_to_test_fp32, inplace=False)
        model_quantized = torch.quantization.convert(model_quantized_prepared, inplace=False)

        # 4. 现在，加载文件中保存的、已经校准好的权重和量化参数
        try:
            state_dict = torch.load(model_path, map_location=device)
            model_quantized.load_state_dict(state_dict)
        except Exception as e:
            print(f"加载模型权重失败: {e}")
            continue

        model_to_test = model_quantized.to(device)

        # --- 3b. 在其专属的本地验证集上评估 ---
        local_val_loader = farm_data['val_loader']
        print(f"  -> 在 '{farm_id}' 本地验证集上进行评估...")
        local_metrics = evaluate_model(model_to_test, local_val_loader, device, context=f"{farm_id} Local Val")
        print(f"     本地准确度: {local_metrics['accuracy']:.2f}% | 本地F1 Score: {local_metrics['f1_score']:.4f}")

        # --- 3c. 在过滤后的全局验证集上评估 ---
        # (这部分逻辑保持不变)
        global_class_to_idx = global_val_loader.dataset.subset.dataset.class_to_idx
        local_class_names = farm_data['class_names']
        known_global_indices = {global_class_to_idx[name] for name in local_class_names}
        original_global_dataset = global_val_loader.dataset
        global_val_indices = original_global_dataset.subset.indices

        filtered_indices = [idx for idx in global_val_indices if
                            original_global_dataset.subset.dataset.samples[idx][1] in known_global_indices]

        if filtered_indices:
            filtered_subset = torch.utils.data.Subset(original_global_dataset.subset.dataset, filtered_indices)
            local_class_map = farm_data['class_to_idx_local']
            global_classes_list = original_global_dataset.subset.dataset.classes

            # 重新检查 local_class_map 的创建逻辑
            # 它应该是一个从全局类别名到本地[0,N-1]索引的映射
            name_to_local_idx = {name: i for i, name in enumerate(sorted(local_class_names))}

            filtered_wrapped = FarmSubsetWrapper(
                filtered_subset,
                transform=local_val_loader.dataset.transform,
                global_classes=global_classes_list,
                local_class_map=name_to_local_idx
            )
            filtered_global_loader = torch.utils.data.DataLoader(filtered_wrapped, batch_size=config.BATCH_SIZE)

            print(f"  -> 在全局验证集 (已过滤，含{len(local_class_names)}个已知类别) 上评估...")
            global_metrics = evaluate_model(model_to_test, filtered_global_loader, device,
                                            context=f"{farm_id} Global Val (Filtered)")
            print(
                f"     全局准确度: {global_metrics['accuracy']:.2f}% | 全局F1 Score: {global_metrics['f1_score']:.4f}")
        else:
            print("  -> 全局验证集中不包含此农场已知的任何类别，跳过全局评估。")
            global_metrics = {"accuracy": 0.0, "f1_score": 0.0}

        results.append({
            "model_file": os.path.basename(model_path),
            "farm_id": farm_id,
            "round": model_info['round'],
            "local_accuracy": local_metrics['accuracy'],
            "local_f1_score": local_metrics['f1_score'],
            "global_accuracy_filtered": global_metrics['accuracy'],
            "global_f1_score_filtered": global_metrics['f1_score'],
        })

    # --- 4. 打印总结报告 ---
    # (这部分逻辑保持不变)
    if results:
        print("\n\n" + "=" * 30)
        print("          评估总结报告")
        print("=" * 30)
        df = pd.DataFrame(results)
        df = df.round(2)
        print(df.to_string())
        summary_path = os.path.join(model_dir, "quantized_models_evaluation_summary.csv")
        df.to_csv(summary_path, index=False)
        print(f"\n总结报告已保存到: {summary_path}")


if __name__ == '__main__':
    main()