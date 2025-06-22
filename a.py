import os

# 设置标签目录路径
train_label_dir = "/home/a10/slh/yolo/datasets/TinyPerson/labels/train"
val_label_dir = "/home/a10/slh/yolo/datasets/TinyPerson/labels/val"


def fix_labels(label_dir):
    for filename in os.listdir(label_dir):
        if filename.endswith(".txt"):
            path = os.path.join(label_dir, filename)
            with open(path, 'r') as f:
                lines = f.readlines()

            corrected = []
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 5:  # 确保是有效标签行
                    if int(parts[0]) != 0:  # 检查类别ID是否为0
                        parts[0] = '0'  # 修正为类别0
                    corrected.append(" ".join(parts) + '\n')

            with open(path, 'w') as f:
                f.writelines(corrected)


fix_labels(train_label_dir)
fix_labels(val_label_dir)
print("标签修正完成！训练集和验证集所有非0类别已转换为0")