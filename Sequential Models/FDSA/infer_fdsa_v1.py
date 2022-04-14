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
from tqdm import tqdm

import torch
torch.cuda.empty_cache()

# submission_index.csv 파일이 생성되기 전이라면 for_submission.py 파일을 실행해주세요.
user_index = pd.read_csv('/opt/ml/RecBole/srcs/submission_index.csv')

user_index = np.array(user_index['user'],dtype=str)

# 불러올 모델의 주소를 넣어주세요.
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='/opt/ml/RecBole/srcs/saved/FDSA-Apr-13-2022_14-43-01.pth' # /opt/ml/RecBole/srcs/saved/FDSA-Apr-10-2022_00-22-18.pth'       # 
)

print("check_0")
from recbole.utils.case_study import full_sort_topk
external_user_ids = dataset.id2token(
    dataset.uid_field, list(range(dataset.user_num)))[1:]

topk_items = []
users = []
for internal_user_id in tqdm(list(range(dataset.user_num))[1:]):  
    _, topk_iid_list = full_sort_topk([internal_user_id], model, test_data, k=10, device=config['device'])
    last_topk_iid_list = topk_iid_list[-1]
    external_item_list = dataset.id2token(dataset.iid_field, last_topk_iid_list.cpu()).tolist()
    topk_items.append(external_item_list)
print(len(topk_items))

user_index = np.repeat(user_index,10)
print(len(user_index))
topk_items = np.array(topk_items).flatten()
data = {'user' : user_index, 'item' :topk_items} # external_item_list}

df = pd.DataFrame(data)

print('- submission.csv -')
print(df)
print()


# path = '/opt/ml/Recbole/srcs/output'

# os.makedirs(path, exist_ok=True)

df.to_csv('/opt/ml/RecBole/srcs/output/submission_SINE_235845.csv',index=False)

print()
print('csv 파일 생성 완료')