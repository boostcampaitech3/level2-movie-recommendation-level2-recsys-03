import pandas as pd
import torch
import numpy as np
import os

item1 = pd.read_csv('/opt/ml/Recbole/ensemble/submission_EASE_all_user_top30.csv')
item2 = pd.read_csv('/opt/ml/Recbole/ensemble/RecVAE_30_submission.csv')


item1 = item1['item']
item2 = item2['item']

item_list = [[]for _ in range(31360)]


cnt_10 = 0
cnt_20 = 0
emp = 0
emp_arr = [0]*10

for i in range(31360):
    for j in range(30):
        for k in range(30):
            if item1[30*i+j] == item2[30*i+k]:
                item_list[i].append(item1[30*i+j])
                break
            
        if len(item_list[i]) == 10:
            if j == 9:
                cnt_10 += 1
            if j < 20:
                cnt_20 += 1
            break
    if len(item_list[i]) != 10:
        for j in range(30):
            if item1[30*i+j] not in item_list[i]:
                item_list[i].append(item1[30*i+j])
            if len(item_list[i]) == 10:
                break


# print()
# print('EASE의 TOP 10이 전부 RecVAE TOP 30 안에 있을 확률 : ',end=' ')
# print(round(cnt_10/31600*100,3),'%')
# print()

# print('EASE의 TOP 20 안에서 10개의 숫자가 RecVAE TOP 30 안에 있을 확률 : ',end=' ')
# print(round(cnt_20/31600*100,3),'%')
# print()

# print('EASE의 TOP 30 안에서 10개의 숫자가 RecVAE TOP 30 안에 있을 확률 : ',end=' ')
# print(round((31600-emp)/31600*100,3),'%')
# print()

# print('EASE의 TOP 30 과 RecVAE TOP 30의 중 10개도 겹치지 않을 확률 : ',end=' ')
# print(round(emp/31600*100,3),'%', ', 경우의 수 : ',emp)
# print()

# print('10개 미만 중복 아이템 갯수 : ')
# for i in range(10):
#     print(str(i)+' : '+str(emp_arr[i]), end='   ')

# print()

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

df.to_csv('/opt/ml/Recbole/output/Ensamble_Top_30_submission.csv',index=False)

print()
print('csv 파일 생성 완료')