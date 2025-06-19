import os
from pathlib import Path


def find_extra_images(dataset_main, dataset_compare):
    """
    找出dataset_main中比dataset_compare多出的图片文件

    :param dataset_main: 主数据集路径
    :param dataset_compare: 对比数据集路径
    :return: 多出的图片文件列表
    """
    # 获取主数据集的所有图片文件名（不含扩展名）
    main_dir = Path(dataset_main) / "/home/a10/slh/yolo/DOTAv1.5/images/train"
    main_images = {f.stem for f in main_dir.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']}

    # 获取对比数据集的所有图片文件名（不含扩展名）
    compare_dir = Path(dataset_compare) / "/home/a10/slh/yolo/datasets/DOTAv1.5/images/train"
    compare_images = {f.stem for f in compare_dir.glob("*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']}

    # 找出主数据集中有而对比数据集中没有的图片
    extra_images = main_images - compare_images

    # 获取完整的文件名（带扩展名）
    extra_files = []
    for img_stem in extra_images:
        # 查找原始扩展名
        img_file = next((f for f in main_dir.glob(f"{img_stem}.*") if f.suffix.lower() in ['.jpg', '.png', '.jpeg']),
                        None)
        if img_file:
            extra_files.append(img_file.name)

    return sorted(extra_files)


def find_extra_labels(extra_images, dataset_main):
    """
    找出多出图片对应的标签文件

    :param extra_images: 多出的图片文件名列表
    :param dataset_main: 主数据集路径
    :return: 多出的标签文件列表
    """
    labels_dir = Path(dataset_main) / "labels/train"
    extra_labels = []

    for img_file in extra_images:
        # 去除扩展名
        img_stem = Path(img_file).stem
        label_file = f"{img_stem}.txt"

        # 检查标签文件是否存在
        if (labels_dir / label_file).exists():
            extra_labels.append(label_file)
        else:
            print(f"警告: 图片 {img_file} 没有对应的标签文件")

    return extra_labels


if __name__ == "__main__":
    # 配置路径 - 修改为你实际的数据集路径
    dataset_A = " /home/a10/slh/yolo/DOTAv1.5/images/train "  # 主数据集
    dataset_B = " /home/a10/slh/yolo/datasets/DOTAv1.5/images/train"  # 对比数据集

    # 找出多出的图片文件
    extra_images = find_extra_images(dataset_A, dataset_B)

    # 找出对应的标签文件
    extra_labels = find_extra_labels(extra_images, dataset_A)

    # 打印结果
    print(f"在 {dataset_A} 中多出的图片文件 ({len(extra_images)} 张):")
    for img in extra_images:
        print(f"  - {img}")

    print(f"\n对应的标签文件 ({len(extra_labels)} 个):")
    for label in extra_labels:
        print(f"  - {label}")

    # 保存结果到文件
    with open("extra_files_report.txt", "w") as f:
        f.write("多出的图片文件:\n")
        f.write("\n".join([f"- {img}" for img in extra_images]))
        f.write("\n\n多出的标签文件:\n")
        f.write("\n".join([f"- {label}" for label in extra_labels]))

    print("\n报告已保存到 extra_files_report.txt")