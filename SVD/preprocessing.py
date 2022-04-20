import os
import pandas as pd
import random
import sklearn


class SVD_data(object):
    def __init__(self, args, ratings):
        self.args = args
        self.ratings = ratings
        self.reindex_data, self.id2user, self.id2item = self.reindex(self.ratings)
        self.negatives = self.negative_sampling(self.reindex_data)
        self.data = self.merge_negative(self.reindex_data, self.negatives)
        

    def reindex(self, ratings):
        user_list = list(ratings['user'].drop_duplicates())
        user2id = {w: i for i, w in enumerate(user_list)}
        id2user = {i: w for i, w in enumerate(user_list)}

        item_list = list(ratings['item'].drop_duplicates())
        item2id = {w: i for i, w in enumerate(item_list)}
        id2item = {i: w for i, w in enumerate(item_list)}

        ratings['user_id'] = ratings['user'].apply(lambda x: user2id[x])
        ratings['item_id'] = ratings['item'].apply(lambda x: item2id[x])
        ratings['rating'] = (ratings['time'] > 0).astype('float')
    
        return ratings[['user_id', 'item_id', 'rating']], id2user, id2item


    def negative_sampling(self, ratings):
        item_pool = set(ratings['item_id'].unique())
        num_item = len(item_pool)

        interact_status = (
            ratings.groupby('user_id')['item_id']
            .apply(set)
            .reset_index()
            .rename(columns={'item_id': 'interacted_items'}))
    
        interact_status['negative_items'] = interact_status['interacted_items'].apply(lambda x: item_pool - x)
        interact_status['negative_samples'] = interact_status['negative_items'].apply(lambda x: random.sample(x, (num_item - len(x))))

        return interact_status[['user_id', 'negative_items', 'negative_samples']]


    def merge_negative(self, ratings, negatives, shuffle=True):
        negatives = negatives.explode('negative_samples')
        del negatives['negative_items']
        negatives['rating'] = float(0)
        negatives = negatives.rename(columns = {'negative_samples':'item_id'})
        merge_df = pd.concat([ratings, negatives])
        
        if shuffle :
            merge_df = sklearn.utils.shuffle(merge_df)
        
        return merge_df

