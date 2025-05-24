import os
import shutil

src = '/home/a10/ZH/ultralytics-main/datasets/UAVDT'
dst = '/home/a10/ZH/ultralytics-main/datasets/UAVDT_cleaned'

for split in ['train', 'test']:  # UAVDT官方用test作val
    # 处理images
    os.makedirs(f'{dst}/images/{split}', exist_ok=True)
    os.makedirs(f'{dst}/labels/{split}', exist_ok=True)

    # 构建有效图片列表
    labels = {f.split('.')[0] for f in os.listdir(f'{src}/labels/{split}')}

    # 复制有效数据
    for img in os.listdir(f'{src}/images/{split}'):
        if img.split('.')[0] in labels:
            shutil.copy(
                src=f'{src}/images/{split}/{img}',
                dst=f'{dst}/images/{split}/{img}'
            )
            shutil.copy(
                src=f'{src}/labels/{split}/{img.replace(os.path.splitext(img)[1], ".txt")}',
                dst=f'{dst}/labels/{split}/{img.replace(os.path.splitext(img)[1], ".txt")}'
            )
