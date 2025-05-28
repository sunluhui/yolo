import os
import xml.etree.ElementTree as ET
from tqdm import tqdm

# 定义类别映射（根据UAVDT实际标注修改）
CLASS_MAP = {"car": 0, "truck": 1, "bus": 2}


def parse_xml(xml_path):
    """解析UAVDT XML文件，提取目标信息"""
    tree = ET.parse(xml_path)
    root = tree.getroot()

    # 获取图像尺寸
    size = root.find("size")
    width = int(size.find("width").text)
    height = int(size.find("height").text)

    objects = []
    for obj in root.findall("object"):
        # 提取类别名称
        cls = obj.find("name").text
        if cls not in CLASS_MAP:
            continue  # 跳过未定义的类别

        # 提取边界框坐标
        bbox = obj.find("bndbox")
        xmin = int(bbox.find("xmin").text)
        ymin = int(bbox.find("ymin").text)
        xmax = int(bbox.find("xmax").text)
        ymax = int(bbox.find("ymax").text)

        # 提取遮挡等级（UAVDT特有属性）
        occlusion = int(obj.find("occlusion").text) if obj.find("occlusion") is not None else 0

        objects.append((cls, xmin, ymin, xmax, ymax, occlusion))

    return width, height, objects


def convert_to_yolo(width, height, xmin, ymin, xmax, ymax):
    """将绝对坐标转换为YOLO归一化格式"""
    x_center = (xmin + xmax) / 2.0 / width
    y_center = (ymin + ymax) / 2.0 / height
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return x_center, y_center, w, h


def xml_to_yolo_txt(xml_dir, output_dir, class_map):
    """批量转换XML到YOLO格式TXT"""
    os.makedirs(output_dir, exist_ok=True)

    for xml_file in tqdm(os.listdir(xml_dir)):
        if not xml_file.endswith(".xml"):
            continue

        xml_path = os.path.join(xml_dir, xml_file)
        width, height, objects = parse_xml(xml_path)

        # 生成TXT文件名（与XML同名）
        txt_file = os.path.splitext(xml_file)[0] + ".txt"
        txt_path = os.path.join(output_dir, txt_file)

        with open(txt_path, "w") as f:
            for obj in objects:
                cls, xmin, ymin, xmax, ymax, occlusion = obj
                class_id = class_map[cls]

                # 计算YOLO格式坐标
                x_center, y_center, w, h = convert_to_yolo(width, height, xmin, ymin, xmax, ymax)

                # 写入TXT文件（格式：class_id x_center y_center width height [occlusion]）
                line = f"{class_id} {x_center:.6f} {y_center:.6f} {w:.6f} {h:.6f}"
                if occlusion > 0:
                    line += f" {occlusion}"  # 可选：保留遮挡等级
                f.write(line + "\n")


# 执行转换（指定输入输出路径）
xml_to_yolo_txt(
    xml_dir="/home/a10/slh/yolo/datasets/UAVDT/train/annotations",
    output_dir="/home/a10/slh/yolo/datasets/UAVDT/train",  # 根据划分结果调整路径
    class_map=CLASS_MAP
)