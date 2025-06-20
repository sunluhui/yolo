import os
import shutil
from pathlib import Path


def delete_val_files_from_train(dataset_root):
    """
    从训练集中删除与验证集同名的图片和标签文件

    参数:
        dataset_root (str): 数据集根目录路径
    """
    # 创建备份目录
    backup_dir = Path(dataset_root) / f"backup_{Path(dataset_root).name}_{os.getpid()}"
    backup_dir.mkdir(parents=True, exist_ok=True)

    print(f"数据集根目录: {dataset_root}")
    print(f"备份目录: {backup_dir}\n")

    # 定义目录路径
    dirs = {
        "img_train": Path(dataset_root) / "/home/a10/slh/yolo/datasets/DOTAv1.5/images/train",
        "img_val": Path(dataset_root) / "/home/a10/slh/yolo/datasets/DOTAv1.5/images/val",
        "label_train": Path(dataset_root) / "/home/a10/slh/yolo/datasets/DOTAv1.5/labels/train",
        "label_val": Path(dataset_root) / "/home/a10/slh/yolo/datasets/DOTAv1.5/labels/val"
    }

    # 验证目录是否存在
    for name, path in dirs.items():
        if not path.exists():
            print(f"错误: 目录不存在 - {path}")
            return

    # 获取验证集文件名（不带扩展名）
    val_img_files = {f.stem for f in dirs["img_val"].glob("*") if f.is_file()}
    val_label_files = {f.stem for f in dirs["label_val"].glob("*") if f.is_file()}
    val_files = val_img_files | val_label_files

    print(f"验证集文件数量: {len(val_files)}")

    # 删除训练集中匹配的文件
    deleted_count = 0

    # 处理训练集图片
    for img_file in dirs["img_train"].glob("*"):
        if img_file.is_file() and img_file.stem in val_files:
            # 创建备份
            backup_path = backup_dir / "images/train" / img_file.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(img_file), str(backup_path))
            print(f"已备份并删除图片: {img_file.name}")
            deleted_count += 1

    # 处理训练集标签
    for label_file in dirs["label_train"].glob("*"):
        if label_file.is_file() and label_file.stem in val_files:
            # 创建备份
            backup_path = backup_dir / "labels/train" / label_file.name
            backup_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(label_file), str(backup_path))
            print(f"已备份并删除标签: {label_file.name}")
            deleted_count += 1

    # 输出结果
    print(f"\n操作完成! 共删除 {deleted_count} 个文件")
    print(f"所有删除的文件已备份到: {backup_dir}")

    # 统计剩余文件
    remaining_img = len(list(dirs["img_train"].glob("*")))
    remaining_label = len(list(dirs["label_train"].glob("*")))
    print(f"\n训练集剩余: {remaining_img} 张图片, {remaining_label} 个标签")


if __name__ == "__main__":
    # 配置数据集路径 - 修改为您的实际路径
    dataset_path = "/home/a10/slh/yolo/datasets/DOTAv1.5"

    # 执行删除操作
    delete_val_files_from_train(dataset_path)

    # 添加清理缓存文件的建议
    print("\n建议操作: 删除旧缓存文件后重启训练")
    print("rm -f /home/a10/slh/yolo/datasets/DOTAv1.5/labels/*.cache")
    print("cd /home/a10/slh/yolo && python train.py")