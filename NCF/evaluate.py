'''
/**
 * Original Code
 * https://github.com/pyy0715/Neural-Collaborative-Filtering
 * modified by Ye-ji Kim
 */
 '''

from ast import arg
import os
import argparse
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data

from model import NeuMF
from util import seed_everything
from dataset import NCF_Data
from trainer import Trainer

def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu_id',
        type=int,
        default=0 if torch.cuda.is_available() else -1,
        help="gpu id")
    parser.add_argument("--seed", 
        type=int, 
        default=42, 
        help="Seed")
    parser.add_argument("--lr", 
        type=float, 
        default=0.001, 
        help="learning rate")
    parser.add_argument("--dropout", 
        type=float,
        default=0.2,  
        help="dropout rate")
    parser.add_argument("--batch_size", 
        type=int, 
        default=256, 
        help="batch size for training")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=32, 
        help="predictive factors numbers in the model")
    parser.add_argument("--layers",
        nargs='+', 
        default=[64,32,16,8],
        help="MLP layers. Note that the first layer is the concatenation of user and item embeddings. So layers[0]/2 is the embedding size.")
    parser.add_argument("--num_ng", 
        type=int,
        default=4, 
        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test", 
        type=int,
        default=100, 
        help="Number of negative samples for test set")
        
    parser.add_argument("--DATA_PATH", 
        default='/opt/ml/input/data/train/train_ratings.csv',
        help="Data path")
    parser.add_argument("--MODEL_PATH", 
        default='/opt/ml/NCF/models/',
        help="Model path")
    parser.add_argument("--OUT_PATH", 
        default='/opt/ml/NCF/out_csv/',
        help="output csv file path")
    parser.add_argument("--OUT_CSV_FILE", 
        default='output',
        help="output csv file path")
    parser.add_argument("--MODEL", 
        default='PRE_NCF1',
        help="Model name")

    # set device and parameters
    config = parser.parse_args()

    return config


def main(config) :
    device = torch.device('cpu') if config.gpu_id < 0 else torch.device('cuda:%d' % config.gpu_id)   

    # seed for Reproducibility
    seed_everything(config.seed)

    # load data
    train_df = pd.read_csv(config.DATA_PATH)
    train_df.columns = ['user_id', 'item_id', 'timestamp']

    # set the num_users, items
    num_users = train_df['user_id'].nunique()
    num_items = train_df['item_id'].nunique()

    user_list = list(train_df['user_id'].drop_duplicates())
    user2id = {w: i for i, w in enumerate(user_list)}
    id2user = {i: w for i, w in enumerate(user_list)}

    item_list = list(train_df['item_id'].drop_duplicates())
    item2id = {w: i for i, w in enumerate(item_list)}
    id2item = {i: w for i, w in enumerate(item_list)}

    train_df['user_id'] = train_df['user_id'].apply(lambda x: user2id[x])
    train_df['item_id'] = train_df['item_id'].apply(lambda x: item2id[x])
    train_df['rating'] = train_df['timestamp'].apply(lambda x: float(x > 0))  

    result = np.zeros((2, config.top_k * num_users), dtype=np.int64) 
    
    interact_status = (
        train_df.groupby('user_id')['item_id']
        .apply(list)
        .reset_index()
        .rename(columns={'item_id': 'interacted_items'}))

    eval_user = torch.ones(num_items, dtype=torch.long)
    eval_items = torch.arange(0, num_items, dtype=torch.long)

    # set model
    input_path = os.path.join(config.MODEL_PATH, config.MODEL)
    model = torch.load(input_path).to(device)

    #model = NeuMF(config, num_users, num_items)
    #model.load_state_dict(torch.load('{}{}.pth'.format(config.MODEL_PATH, config.MODEL)))
    #model = model.to(device)
    
    model.eval()
    print('*'*40,'Evaluate','*'*40)

    with torch.no_grad():

        for user in tqdm(range(num_users)):
            users = (user * eval_user).to(device)
            items = eval_items.to(device)

            prediction = model(users, items)
            sorted_items = prediction.argsort(descending=True)
            positive_samples = interact_status.iloc[user]['interacted_items']

            rec_items = np.setdiff1d(np.array(sorted_items.to('cpu')), positive_samples)

            result[0, 10*user:10*(user+1)] = user
            result[1, 10*user:10*(user+1)] = rec_items[:config.top_k]
    
    test_df = pd.DataFrame()
    test_df['user_id'] = result[0].T
    test_df['item_id'] = result[1].T

    test_df['user'] = test_df['user_id'].apply(lambda x: id2user[x])
    test_df['item'] = test_df['item_id'].apply(lambda x: id2item[x])

    del test_df['item_id'], test_df['user_id'] 

    test_df.to_csv('{}{}.csv'.format(config.OUT_PATH, config.OUT_CSV_FILE), index=False)



if __name__ == '__main__':
    config = define_argparser()
    main(config)