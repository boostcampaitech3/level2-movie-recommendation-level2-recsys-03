from recbole.quick_start.quick_start import load_data_and_model
from recbole.utils.case_study import full_sort_topk

config, model, dataset, train_data, valid_data, test_data = load_data_and_model(
    model_file='/opt/ml/duorec/log/CL4SRec/recbox_data/bs256-lmd0.1-sem0.1-us_x-Apr-05-2022_05-30-25-lr0.001-l20-tau1-cos-DPh0.5-DPa0.5/model.pth',
)

uid_series = dataset.token2id(dataset.uid_field)


topk_score, topk_iid_list = full_sort_topk(uid_series, model, test_data, k=10, device=config['device'])
print(topk_score)  # scores of top 10 items
print(topk_iid_list)  # internal id of top 10 items
external_item_list = dataset.id2token(dataset.iid_field, topk_iid_list.cpu())
print(external_item_list)  # external tokens of top 10 items