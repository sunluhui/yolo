import pandas as pd
df = pd.read_csv('/home/a10/slh/yolo/datasets/UAVDT/train/annotations.txt')
print(df.isnull().sum())