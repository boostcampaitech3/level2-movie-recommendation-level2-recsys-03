import pandas as pd


def get_unseen_movieid(ratings):
    item_pool = set(ratings['item_id'].unique())

    interact_status = (
        ratings.groupby('user_id')['item_id']
        .apply(set)
        .reset_index()
        .rename(columns={'item_id': 'positive_items'}))
   
    interact_status['negative_items'] = interact_status['positive_items'].apply(lambda x: item_pool - x)

    return interact_status[['user_id', 'negative_items']]


def recommend_movie(algo, id2user, id2item, data, top_k):
    result = pd.DataFrame(columns=['user', 'item'])

    for user in range(len(data)):
        predictions = [algo.predict(str(user), str(movie)) for movie in data.iloc[user]['negative_items']]

        def sortkey_est(pred):
            return pred.est

        predictions.sort(key=sortkey_est, reverse=True)
        top_predictions = predictions[:top_k]

        top_movie_ids = [id2item[int(pred.iid)] for pred in top_predictions]
        
        tmp = pd.DataFrame({'user': [id2user[user]], 'item': [top_movie_ids]})
        tmp = tmp.explode('item')

        result = pd.concat([result, tmp])
     
    return result

