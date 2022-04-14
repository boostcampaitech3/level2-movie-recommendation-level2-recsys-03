import pandas as pd

df1 = pd.read_csv('/opt/ml/input/data/train/directors.tsv',sep='\t')
df2 = pd.read_csv('/opt/ml/input/data/train/writers.tsv',sep='\t')

df1 = df1.groupby('item').agg({'director':' '.join}).reset_index()
df2 = df2.groupby('item').agg({'writer':' '.join}).reset_index()

df1 = df1.merge(df2)

print(df1)
df1.to_csv('/opt/ml/input/data/train/directors.csv',index=False)