import xml.etree.ElementTree as ET
import os


def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)
    objects = []
    for obj in root.findall('object'):
        cls = obj.find('name').text  # UAVDT类别名（如"car"）
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        objects.append((cls, xmin, ymin, xmax, ymax))
    return width, height, objects


def convert_to_yolo(width, height, xmin, ymin, xmax, ymax):
    x_center = (xmin + xmax) / (2 * width)
    y_center = (ymin + ymax) / (2 * height)
    w = (xmax - xmin) / width
    h = (ymax - ymin) / height
    return x_center, y_center, w, h


def xml_to_txt(xml_dir, txt_dir, class_map):
    os.makedirs(txt_dir, exist_ok=True)
    for xml_file in os.listdir(xml_dir):
        if not xml_file.endswith('.xml'):
            continue
        xml_path = os.path.join(xml_dir, xml_file)
        width, height, objects = parse_xml(xml_path)
        txt_file = xml_file.replace('.xml', '.txt')
        txt_path = os.path.join(txt_dir, txt_file)
        with open(txt_path, 'w') as f:
            for obj in objects:
                cls, xmin, ymin, xmax, ymax = obj
                class_id = class_map[cls]  # 映射为数字ID（如0代表"car"）
                x, y, w, h = convert_to_yolo(width, height, xmin, ymin, xmax, ymax)
                f.write(f"{class_id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")


# 示例类别映射（需与UAVDT实际类别一致）
CLASS_MAP = {"car": 0, "truck": 1, "bus": 2}
xml_to_txt("Annotations", "labels", CLASS_MAP)