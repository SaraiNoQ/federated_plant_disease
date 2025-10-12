# 示例 bash 命令 (请根据你的实际路径修改)
mkdir -p ./data/plantdoc_merged
DATA_SOURCE=./PlantDoc/PlantDoc-Dataset-windows-compatible-master # 你的原始PlantDoc路径
DATA_DEST=./data/plantdoc_merged

# 遍历所有子集 (train, test, val)
for subset in train test; do
  # 遍历每个类别文件夹
  for class_dir in "$DATA_SOURCE/$subset"/*; do
    class_name=$(basename "$class_dir")
    # 在目标路径创建类别文件夹
    mkdir -p "$DATA_DEST/$class_name"
    # 复制所有图片
    cp -r "$class_dir"/* "$DATA_DEST/$class_name/"
  done
done
