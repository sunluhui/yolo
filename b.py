import os
from pathlib import Path


def remove_duplicate_labels(label_dir):
    """删除标注文件中完全重复的行"""
    label_paths = list(Path(label_dir).glob('*.txt'))

    for label_path in label_paths:
        with open(label_path, 'r') as f:
            lines = f.readlines()

        # 使用集合去重（保留顺序）
        unique_lines = []
        seen = set()
        for line in lines:
            if line not in seen:
                seen.add(line)
                unique_lines.append(line)

        # 只有当有重复时才重写文件
        if len(unique_lines) < len(lines):
            with open(label_path, 'w') as f:
                f.writelines(unique_lines)
            print(f"Cleaned {label_path}: {len(lines) - len(unique_lines)} duplicates removed")


# 使用示例 - 清理训练集和验证集
remove_duplicate_labels('/home/a10/ZH/v8/ultralytics-main/datasets/VisDrone2019/VisDrone2019-DET-train/images/labels')
