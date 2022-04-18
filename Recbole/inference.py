import pandas as pd
from torch import tensor
from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk
import numpy as np
import os


# submission_index.csv 파일이 생성되기 전이라면 for_submission.py 파일을 실행해주세요.
user_index = pd.read_csv('/opt/ml/Recbole/submission_index.csv')

user_index = np.array(user_index['user'],dtype=str)


# 불러올 모델의 주소를 넣어주세요.
config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='/opt/ml/Recbole/saved/MacridVAE-Apr-06-2022_07-57-01.pth'
)

uid_series = dataset.token2id(dataset.uid_field, user_index)

topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])

print('- topk_score -')
print(topk_score)  # scores of top 10 items
print()

external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())

print('- topk_list -')
print(external_item_list)  # external tokens of top 10 items
print()

user_index = np.repeat(user_index,10)


external_item_list = external_item_list.flatten()


data = {'user' : user_index, 'item' :external_item_list}

df = pd.DataFrame(data)

print('- submission.csv -')
print(df)
print()


path = '/opt/ml/Recbole/output'

os.makedirs(path, exist_ok=True)

df.to_csv('/opt/ml/Recbole/output/submission.csv',index=False)

print()
print('csv 파일 생성 완료')