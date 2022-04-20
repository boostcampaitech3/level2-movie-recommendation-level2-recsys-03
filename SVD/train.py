import os
import argparse

import pandas as pd

from surprise.dataset import DatasetAutoFolds
from surprise import Reader 
from surprise import SVD

from util import get_unseen_movieid, recommend_movie
from preprocessing import SVD_data


def define_argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--random_seed", 
        type=int, 
        default=0, 
        help="Set random seed")
    parser.add_argument("--n_epochs", 
        type=int,
        #default=200,  
        default=1,
        help="training epochs")
    parser.add_argument("--top_k", 
        type=int, 
        default=10, 
        help="compute metrics@top_k")
    parser.add_argument("--factor_num", 
        type=int,
        default=50, 
        help="predictive factors numbers in the model")   

    parser.add_argument("--DATA_PATH", 
        #default='/opt/ml/input/data/train',
        default='/home/yeji/data/train',
        help="Data path")
    parser.add_argument("--DATA_FILE",
        default='train_ratings.csv', 
        help="Data name")
    parser.add_argument("--TRAIN_DATA",
        default='SVD_input.csv', 
        help="Data name")

    parser.add_argument("--OUT_PATH", 
        #default='/opt/ml/SVD/out_csv/',
        default='/home/yeji/.',
        help="output csv file path")
    parser.add_argument("--OUT_CSV_FILE", 
        default='SVD_epoch200_fn50',
        help="output csv file path")

    # set parameters
    config = parser.parse_args()

    return config


def main(config):
    # load data
    train_df = pd.read_csv(os.path.join(config.DATA_PATH, config.DATA_FILE))
    dataset = SVD_data(config, train_df)
    data = dataset.data
    data.to_csv(os.path.join(config.DATA_PATH, config.TRAIN_DATA), index=False, header=False)

    reader = Reader(line_format='user item rating', sep=',', rating_scale=(0, 1))
    data_folds = DatasetAutoFolds(ratings_file=os.path.join(config.DATA_PATH, config.TRAIN_DATA), reader=reader)
    trainset = data_folds.build_full_trainset()
    print('Get data!')

    # modeling
    algo = SVD(n_epochs=config.n_epochs, n_factors=config.factor_num, random_state=config.random_seed)
    algo.fit(trainset)
    print('Training is over!')

    # Recommend movies
    unseen_movies = get_unseen_movieid(dataset.reindex_data)
    result = recommend_movie(algo, dataset.id2user, dataset.id2item, unseen_movies, config.top_k)
    result.to_csv('{}{}.csv'.format(config.OUT_PATH, config.OUT_CSV_FILE), index=False)
    print('Generate output file!')


if __name__ == '__main__':
    config = define_argparser()
    main(config)