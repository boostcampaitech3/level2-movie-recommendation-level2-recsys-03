import pandas as pd
import torch
import numpy as np
import os

item1 = pd.read_csv('/opt/ml/Recbole/ensemble/output(4).csv')
item2 = pd.read_csv('/opt/ml/Recbole/ensemble/output(6).csv')
item3 = pd.read_csv('/opt/ml/Recbole/ensemble/output(7).csv')

item1 = item1['item']
item2 = item2['item']
item3 = item3['item']

item_list = [[]for _ in range(31360)]

for i in range(31360):
    for j in range(3):
        item_list[i].append(item3[i*10+j])
    for j in range(10):
        if item2[i*10+j] not in item_list[i]:
            item_list[i].append(item2[i*10+j])
        if len(item_list[i]) == 6:
            break
    
    for j in range(10):
        if item1[i*10+j] not in item_list[i]:
            item_list[i].append(item1[i*10+j])
        if len(item_list[i]) == 10:
            break


item_list = np.array(item_list)

item_list = item_list.flatten()



user_index = pd.read_csv('/opt/ml/Recbole/submission_index.csv')

user_index = np.array(user_index['user'],dtype=str)

user_index = np.repeat(user_index,10)


data = {'user' : user_index, 'item' :item_list}

df = pd.DataFrame(data)

print('- submission.csv -')
print(df)
print()


path = '/opt/ml/Recbole/output'

os.makedirs(path, exist_ok=True)

df.to_csv('/opt/ml/Recbole/output/Ensamble_Ranking_submission.csv',index=False)

print()
print('csv 파일 생성 완료')