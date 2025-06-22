import os
import shutil

# 配置路径
image_dirs = [
    "/home/a10/slh/yolo/datasets/TinyPerson/images/train",
    "/home/a10/slh/yolo/datasets/TinyPerson/images/val"
]

label_dirs = [
    "/home/a10/slh/yolo/datasets/TinyPerson/labels/train",
    "/home/a10/slh/yolo/datasets/TinyPerson/labels/val"
]

backup_dir = "/home/a10/slh/yolo/datasets/TinyPerson/backup_no_labels"

# 创建备份目录
os.makedirs(backup_dir, exist_ok=True)


def clean_images(image_dir, label_dir):
    deleted_count = 0
    kept_count = 0

    # 获取所有标签文件名（不含扩展名）
    label_files = {os.path.splitext(f)[0] for f in os.listdir(label_dir) if f.endswith('.txt')}

    print(f"正在处理: {image_dir}")
    for img_file in os.listdir(image_dir):
        if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name = os.path.splitext(img_file)[0]

            if img_name not in label_files:
                # 移动无标签图片到备份目录
                src = os.path.join(image_dir, img_file)
                dst = os.path.join(backup_dir, img_file)
                shutil.move(src, dst)
                print(f"已移动: {img_file} -> backup_no_labels")
                deleted_count += 1
            else:
                kept_count += 1

    print(f"处理完成: 保留 {kept_count} 张图片, 移除 {deleted_count} 张无标签图片")
    return deleted_count


# 执行清理
total_deleted = 0
for img_dir, lbl_dir in zip(image_dirs, label_dirs):
    total_deleted += clean_images(img_dir, lbl_dir)

print(f"\n总移除图片数: {total_deleted}")
print(f"无标签图片已备份至: {backup_dir}")