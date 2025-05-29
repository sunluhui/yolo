import os
import random
import xml.etree.ElementTree as ET
from tqdm import tqdm
import cv2
import shutil


def create_voc_structure(root_dir):
    """创建VOC标准目录结构"""
    os.makedirs(os.path.join(root_dir, 'Annotations'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'JPEGImages'), exist_ok=True)
    os.makedirs(os.path.join(root_dir, 'ImageSets/Main'), exist_ok=True)


def convert_annotation(src_img_path, src_txt_path, dst_xml_path, class_mapping):
    """单文件转换核心函数"""
    # 读取图像信息
    img = cv2.imread(src_img_path)
    height, width, depth = img.shape

    # 创建XML根节点
    annotation = ET.Element('annotation')
    ET.SubElement(annotation, 'folder').text = 'JPEGImages'
    ET.SubElement(annotation, 'filename').text = os.path.basename(src_img_path)
    ET.SubElement(annotation, 'path').text = src_img_path

    # 添加图像尺寸信息
    size = ET.SubElement(annotation, 'size')
    ET.SubElement(size, 'width').text = str(width)
    ET.SubElement(size, 'height').text = str(height)
    ET.SubElement(size, 'depth').text = str(depth)

    # 处理空白图片
    if os.path.getsize(src_txt_path) == 0:
        ET.SubElement(annotation, 'segmented').text = '0'
        tree = ET.ElementTree(annotation)
        tree.write(dst_xml_path)
        return

    # 解析原始标注
    with open(src_txt_path, 'r') as f:
        lines = f.readlines()

    # 创建目标对象
    for line in lines:
        class_id, x_center, y_center, w, h = map(float, line.strip().split())
        class_name = class_mapping[int(class_id)]

        # 计算边界框坐标
        x_center *= width
        y_center *= height
        w *= width
        h *= height
        xmin = int(x_center - w / 2)
        ymin = int(y_center - h / 2)
        xmax = int(x_center + w / 2)
        ymax = int(y_center + h / 2)

        # 创建object节点
        obj = ET.SubElement(annotation, 'object')
        ET.SubElement(obj, 'name').text = class_name
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        bndbox = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bndbox, 'xmin').text = str(xmin)
        ET.SubElement(bndbox, 'ymin').text = str(ymin)
        ET.SubElement(bndbox, 'xmax').text = str(xmax)
        ET.SubElement(bndbox, 'ymax').text = str(ymax)

    # 保存XML文件
    tree = ET.ElementTree(annotation)
    tree.write(dst_xml_path, encoding='utf-8', xml_declaration=True)


def split_dataset(image_list, split_ratio=(0.8, 0.1, 0.1)):
    """数据集划分"""
    random.shuffle(image_list)
    total = len(image_list)
    train_end = int(total * split_ratio[0])
    val_end = train_end + int(total * split_ratio[1])

    with open('ImageSets/Main/train.txt', 'w') as f:
        f.writelines([f"{img}\n" for img in image_list[:train_end]])

    with open('ImageSets/Main/val.txt', 'w') as f:
        f.writelines([f"{img}\n" for img in image_list[train_end:val_end]])

    with open('ImageSets/Main/test.txt', 'w') as f:
        f.writelines([f"{img}\n" for img in image_list[val_end:]])


def main():
    # 配置参数
    raw_data_dir = '/home/a10/slh/yolo/datasets/UAVDT'  # 原始数据路径
    voc_root = '/home/a10/slh/yolo/datasets/UAVDT_VOC'  # 输出VOC路径
    class_mapping = {0: 'car', 1: 'truck', 2: 'bus'}  # 类别映射表

    # 创建目录结构
    create_voc_structure(voc_root)

    # 文件列表
    image_files = [f for f in os.listdir(raw_data_dir) if f.endswith('.jpg')]
    txt_files = [f.replace('.jpg', '.txt') for f in image_files]

    # 转换标注
    for img_name, txt_name in tqdm(zip(image_files, txt_files)):
        src_img = os.path.join(raw_data_dir, img_name)
        src_txt = os.path.join(raw_data_dir, txt_name)
        dst_xml = os.path.join(voc_root, 'Annotations', img_name.replace('.jpg', '.xml'))

        # 执行转换
        convert_annotation(src_img, src_txt, dst_xml, class_mapping)

    # 复制图像文件
    shutil.copytree(os.path.join(raw_data_dir, 'images'), os.path.join(voc_root, 'JPEGImages'))

    # 划分数据集
    with open(os.path.join(voc_root, 'ImageSets/Main/trainval.txt'), 'w') as f:
        f.writelines([f"{img}\n" for img in image_files])

    split_dataset(image_files)


if __name__ == '__main__':
    main()
