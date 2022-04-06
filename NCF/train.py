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
import copy

import pandas as pd
import numpy as np

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
        default=128, 
        help="batch size for training")
    parser.add_argument("--n_epochs", 
        type=int,
        default=10,  
        help="training epochs")
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
        default=100, 
        help="Number of negative samples for training set")
    parser.add_argument("--num_ng_test", 
        type=int,
        default=100, 
        help="Number of negative samples for test set")
    parser.add_argument("--out", 
        default=True,
        help="save model or not")
    parser.add_argument("--DATA_PATH", 
        default='/opt/ml/input/data/train/train_ratings.csv',
        help="Data path")

    parser.add_argument("--MODEL_PATH", 
        default='/opt/ml/NCF/models/',
        help="Model path")
    parser.add_argument("--MODEL", 
        default='PRE_NCF1',
        help="Model name")

    parser.add_argument("--finetuning", 
        default=False,
        help="Load pretrained model or not")    
    parser.add_argument("--PRETRAINED_MODEL", 
        default='ml-1m_Neu_MF',
        help="Pretrained model name")

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

    # construct the train and test datasets
    data = NCF_Data(config, train_df)
    train_loader =data.get_train_instance()
    valid_loader =data.get_test_instance()

    # set model
    if config.finetuning: 
        model = torch.load('{}{}.pt'.format(config.MODEL_PATH, config.PRETRAINED_MODEL)).to(device)
    else:
        model = NeuMF(config, num_users, num_items).to(device)

    loss_function = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=config.lr)

    trainer = Trainer(model, optimizer, loss_function, config)

    trainer.train(train_loader, valid_loader)


if __name__ == '__main__':
    config = define_argparser()
    main(config)