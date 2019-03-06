import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn import metrics
from tqdm import tqdm
import xgboost as xgb
import os
import lightgbm as lgb

class NewPreProcessing:
    def __init__(self, dataPercent):
        self.fraction = dataPercent

    def preprocessing(self):
        data_path = './data/'
        train = pd.read_csv(data_path + 'train.csv')
        train = train.sample(frac=self.fraction)
        test = pd.read_csv(data_path + 'test.csv')
        songs = pd.read_csv(data_path + 'songs.csv')
        members = pd.read_csv(data_path + 'members.csv')

        song_cols = ['song_id', 'artist_name', 'genre_ids', 'song_length', 'language']
        train = train.merge(songs[song_cols], on='song_id', how='left')
        test = test.merge(songs[song_cols], on='song_id', how='left')

        members['registration_year'] = members['registration_init_time'].apply(lambda x: int(str(x)[0:4]))
        members['registration_month'] = members['registration_init_time'].apply(lambda x: int(str(x)[4:6]))
        members['registration_date'] = members['registration_init_time'].apply(lambda x: int(str(x)[6:8]))

        members['expiration_year'] = members['expiration_date'].apply(lambda x: int(str(x)[0:4]))
        members['expiration_month'] = members['expiration_date'].apply(lambda x: int(str(x)[4:6]))
        members['expiration_date'] = members['expiration_date'].apply(lambda x: int(str(x)[6:8]))
        member = members.drop(['registration_year', 'expiration_year'], axis=1)

        members_cols = members.columns
        train = train.merge(members[members_cols], on='msno', how='left')
        test = test.merge(members[members_cols], on='msno', how='left')

        train = train.fillna(-1)
        test = test.fillna(-1)

        # Preprocess dataset
        cols = list(train.columns)
        cols.remove('target')

        for col in tqdm(cols):
            if train[col].dtype == 'object':
                train[col] = train[col].apply(str)
                test[col] = test[col].apply(str)

                le = LabelEncoder()
                train_vals = list(train[col].unique())
                test_vals = list(test[col].unique())
                le.fit(train_vals + test_vals)
                train[col] = le.transform(train[col])
                test[col] = le.transform(test[col])

            #print(col + ': ' + str(len(train_vals)) + ', ' + str(len(test_vals)))

        X = np.array(train.drop(['target'], axis=1))
        y = train['target'].values

        predictBase = np.array(test.copy().pop('id'))
        real_test_data = np.array(test.drop(['id'], axis=1))
        #ids = test['id'].values
        return X, y, real_test_data,predictBase
