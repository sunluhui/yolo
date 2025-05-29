import os
from collections import defaultdict

label_dir = '/home/a10/slh/yolo/datasets/UAVDT/train/labels'
class_counts = defaultdict(int)
class_names = set()

for lbl_file in os.listdir(label_dir):
    with open(os.path.join(label_dir, lbl_file)) as f:
        for line in f:
            class_id = int(line.strip().split()[0])
            class_names.add(class_id)
            class_counts[class_id] += 1

print(f"类别总数: {len(class_names)}")
print(f"类别分布: {dict(class_counts)}")