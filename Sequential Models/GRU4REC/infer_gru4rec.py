# @Time   : 2022/4/8
# @Author : kysuk05 (https://github.com/kysuk05)
# @File   : inference.py
# UPDATE:
# @Time   : 2022/4/9
# @Author : Suyeon Hong (https://github.com/sparklingade)


import pandas as pd
from torch import tensor
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import numpy as np
import os

import torch
torch.cuda.empty_cache()

# submission_index.csv 파일이 생성되기 전이라면 for_submission.py 파일을 실행해주세요.
user_index = pd.read_csv('/opt/ml/RecBole/srcs/submission_index.csv')

user_index = np.array(user_index['user'],dtype=str)

# 불러올 모델의 주소를 넣어주세요.
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='/opt/ml/RecBole/srcs/saved/GRU4Rec-Apr-13-2022_05-09-18.pth'
)

##### START INFERENCE #####
num_u_group = 10
u_unit = len(user_index) // num_u_group
# print(f"u_unit: {u_unit}") 
# print(dataset.user_num)
for i in range(num_u_group):
    user_index_sub = user_index[i*u_unit:(i+1)*u_unit]
    # print(i*u_unit, (i+1)*u_unit, user_index_sub)
    uid_series = dataset.token2id(dataset.uid_field, user_index_sub)
    print(user_index_sub, uid_series)

    topk_items = []
    topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])

    print('- topk_score -')
    print(topk_score)  # scores of top 10 items
    print()

    print('- topk_list -')
    print(topk_iid_list)  # external tokens of top 10 items
    print(f"lenth : {len(topk_iid_list)}")
    print()

    user_index_sub = np.repeat(user_index_sub,10)

    external_item_list = external_item_list.flatten()

    data = {'user' : user_index_sub, 'item' :external_item_list}

    if i == 0:
        df = pd.DataFrame(data) 
    else:
        df = df + pd.DataFrame(data)

    print('- submission.csv -')
    print(df)
    print(f"len_data = len(df)")
    print()

#############################

path = '/opt/ml/Recbole/srcs/res'

os.makedirs(path, exist_ok=True)

df.to_csv('res/submission.csv',index=False)

print()
print('csv 파일 생성 완료')