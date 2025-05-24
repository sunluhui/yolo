import os
import shutil
from tqdm import tqdm  # 可视化进度条


def clean_dataset(src_root, dst_root):
    for split in ['train', 'val', 'test']:
        # 定义路径
        img_src = os.path.join(src_root, 'images', split)
        label_src = os.path.join(src_root, 'labels', split)
        img_dst = os.path.join(dst_root, 'images', split)
        label_dst = os.path.join(dst_root, 'labels', split)

        # 创建目标目录
        os.makedirs(img_dst, exist_ok=True)
        os.makedirs(label_dst, exist_ok=True)

        # 获取有效图片列表
        valid_images = set(f.split('.')[0] for f in os.listdir(label_src) if f.endswith('.txt'))

        # 复制有效数据
        for img_file in tqdm(os.listdir(img_src), desc=f'Processing {split}'):
            if img_file.split('.')[0] in valid_images:
                # 复制图片
                shutil.copy(
                    src=os.path.join(img_src, img_file),
                    dst=os.path.join(img_dst, img_file)
                )
                # 复制标签
                label_file = img_file.rsplit('.', 1)[0] + '.txt'
                shutil.copy(
                    src=os.path.join(label_src, label_file),
                    dst=os.path.join(label_dst, label_file)
                )


# 使用示例
clean_dataset(
    src_root="/path/to/raw_dataset",
    dst_root="/path/to/cleaned_dataset"
)