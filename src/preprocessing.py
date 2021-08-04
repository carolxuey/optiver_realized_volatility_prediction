import pandas as pd
from sklearn.preprocessing import LabelEncoder

import path_utils


class PreprocessingPipeline:

    def __init__(self, df_train, df_test, split_type, n_splits, shuffle, random_state):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.split_type = split_type
        self.n_splits = n_splits
        self.shuffle = shuffle
        self.random_state = random_state

    def _label_encode(self):

        le = LabelEncoder()
        self.df_train['stock_id_encoded'] = le.fit_transform(self.df_train['stock_id'].values)
        self.df_test['stock_id_encoded'] = le.transform(self.df_test['stock_id'].values)

    def _get_folds(self):

        df_folds = pd.read_csv(f'{path_utils.DATA_PATH}/folds.csv')
        self.df_train['fold'] = df_folds[f'fold_{self.split_type}']

    def transform(self):

        self._get_folds()
        self._label_encode()

        return self.df_train, self.df_test
