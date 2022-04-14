import pandas as pd

df1 = pd.read_csv('/opt/ml/input/data/train/genres.tsv',sep='\t')
df2 = pd.read_csv('/opt/ml/input/data/train/titles.tsv',sep='\t')
df3 = pd.read_csv('/opt/ml/input/data/train/years.tsv',sep='\t')
df4 = pd.read_csv('/opt/ml/input/data/train/writers.tsv',sep='\t')
df5 = pd.read_csv('/opt/ml/input/data/train/directors.tsv',sep='\t')

df1 = df1.groupby('item').agg({'genre':' '.join}).reset_index()

# print(df2.head())
df2 = df2.replace(to_replace='\s\(\d\d\d\d\)', value='', regex=True)
# print(df2.head())

df4 = df4.groupby('item').agg({'writer':' '.join}).reset_index()
df5 = df5.groupby('item').agg({'director':' '.join}).reset_index()

df1 = df1.merge(df2)

df1 = df1.merge(df3)

df1 = df1.merge(df4)
df1 = df1.merge(df5)

print(df1.head())

df1.to_csv('/opt/ml/input/data/train/all_items.csv',index=False)