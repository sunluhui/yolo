import os
from pathlib import Path


def find_missing_files(dataset_A, dataset_B):
    # 获取数据集B的所有图片文件名（不含扩展名）
    b_img_dir = Path(dataset_B) / "/home/a10/slh/yolo/DOTAv1.5/images/train"
    b_images = {f.stem for f in b_img_dir.glob("*") if f.suffix in ['.jpg', '.png', '.jpeg']}

    # 获取数据集A的所有图片文件名（不含扩展名）
    a_img_dir = Path(dataset_A) / "/home/a10/slh/yolo/datasets/DOTAv1.5/images/train"
    a_images = {f.stem for f in a_img_dir.glob("*") if f.suffix in ['.jpg', '.png', '.jpeg']}

    # 计算差异
    missing_images = b_images - a_images

    # 检查对应的标注文件
    missing_files = []
    for img_stem in missing_images:
        img_ext = next((f.suffix for f in b_img_dir.glob(f"{img_stem}.*")), "")
        label_path = Path(dataset_B) / "labels/train" / f"{img_stem}.txt"

        if label_path.exists():
            missing_files.append({
                "image": f"{img_stem}{img_ext}",
                "label": f"{img_stem}.txt",
                "status": "完整"
            })
        else:
            missing_files.append({
                "image": f"{img_stem}{img_ext}",
                "label": f"{img_stem}.txt",
                "status": "缺失标注"
            })

    return missing_files


if __name__ == "__main__":
    # 配置路径
    dataset_A = "/home/a10/slh/yolo/datasets/DOTAv1.5/images/train"
    dataset_B = "/home/a10/slh/yolo/DOTAv1.5/images/train"

    # 查找缺失文件
    missing_files = find_missing_files(dataset_A, dataset_B)

    # 打印报告
    print(f"{'图片文件':<20} {'标注文件':<20} {'状态':<10}")
    print("-" * 50)
    for item in missing_files:
        print(f"{item['image']:<20} {item['label']:<20} {item['status']:<10}")

    # 保存结果
    with open("missing_files_report.csv", "w") as f:
        f.write("image,label,status\n")
        for item in missing_files:
            f.write(f"{item['image']},{item['label']},{item['status']}\n")

    print(f"\n找到 {len(missing_files)} 个缺失文件，报告已保存到 missing_files_report.csv")