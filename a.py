import pandas as pd
df = pd.read_csv('annotations.txt')
print(df.isnull().sum())