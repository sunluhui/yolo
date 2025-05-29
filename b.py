import os

# 文件夹的路径
folder_path = "/home/a10/slh/yolo/datasets/OpenDataLab___UAVDT/raw/UAV-benchmark-M"
folder_path1 = os.path.join(folder_path, 'gt')  # 存放分割后的txt文件的路径
folder_path2 = os.path.join(folder_path, 'images')  # 照片的路径，主要用于统计有多少张照片，方便创建txt文件
folder_path3 = os.path.join(folder_path, 'gt', 'labels')  # 存放标签的txt文件

photo_count = 0
# 判断图片数量
for file_name in os.listdir(folder_path2):
    # 使用os.path.splitext()函数获取文件扩展名
    _, extension = os.path.splitext(file_name)
    # 如果文件是照片文件（例如.jpg或.png），则增加计数器
    if extension.lower() in ['.jpg', '.jpeg', '.png', '.gif', '.bmp']:
        photo_count += 1
print("Total photos:", photo_count)

if not os.path.exists(folder_path):
    os.makedirs(folder_path)

with open(folder_path3, 'r') as file:
    lines = file.readlines()
    for i in range(1, photo_count + 1):
        file_path1 = os.path.join(folder_path1, str(i) + '.txt')
        with open(file_path1, 'w') as target_file:
            for line in lines:
                data = line.split(',')
                if data[0] == str(i):
                    target_file.write(line)
