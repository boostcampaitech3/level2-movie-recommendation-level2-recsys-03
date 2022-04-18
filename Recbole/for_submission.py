import pandas as pd
import torch

user = pd.read_csv('/opt/ml/input/data/train/train_ratings.csv')
user = user['user']
user = user.drop_duplicates()
user.to_csv('/opt/ml/Recbole/submission_index.csv',index=False)

