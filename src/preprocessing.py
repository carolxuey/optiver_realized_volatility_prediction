from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder

import path_utils


class PreprocessingPipeline:

    def __init__(self, df_train, df_test, create_features, split_type):

        self.df_train = df_train.copy(deep=True)
        self.df_test = df_test.copy(deep=True)
        self.create_features = create_features
        self.split_type = split_type

    def _label_encode(self):

        le = LabelEncoder()
        self.df_train['stock_id_encoded'] = le.fit_transform(self.df_train['stock_id'].values)
        self.df_test['stock_id_encoded'] = le.transform(self.df_test['stock_id'].values)

    def _get_folds(self):

        df_folds = pd.read_csv(f'{path_utils.DATA_PATH}/folds.csv')
        self.df_train['fold'] = df_folds[f'fold_{self.split_type}']

    def _get_book_features(self):

        for dataset, df in zip(['train', 'test'], [self.df_train, self.df_test]):

            for stock_id in tqdm(df['stock_id'].unique()):

                df_book = pd.read_parquet(f'{path_utils.DATA_PATH}/book_{dataset}.parquet/stock_id={stock_id}')

                # Bid/Ask price competitiveness
                df_book['bid_price_competitiveness'] = np.abs(df_book['bid_price1'] - df_book['bid_price2'])
                df_book['ask_price_competitiveness'] = np.abs(df_book['ask_price1'] - df_book['ask_price2'])

                # Bid/Ask price distance
                df_book['bid_ask_price1_distance'] = np.abs(df_book['bid_price1'] - df_book['ask_price1'])
                df_book['bid_ask_price2_distance'] = np.abs(df_book['bid_price2'] - df_book['ask_price2'])

                # Volume
                df['bid_volume'] = df['bid_size1'] + df['bid_size2']
                df['ask_volume'] = df['ask_size1'] + df['ask_size2']
                df['total_volume'] = df['ask_size1'] + df['ask_size2'] + df['bid_size1'] + df['bid_size2']

                # Weighted average prices
                df_book['wap1'] = (df_book['bid_price1'] * df_book['ask_size1'] + df_book['ask_price1'] * df_book['bid_size1']) / \
                                  (df_book['bid_size1'] + df_book['ask_size1'])
                df_book['wap2'] = (df_book['bid_price2'] * df_book['ask_size2'] + df_book['ask_price2'] * df_book['bid_size2']) / \
                                  (df_book['bid_size2'] + df_book['ask_size2'])
                df_book['wap3'] = ((df_book['bid_price1'] * df_book['ask_size1'] + df_book['ask_price1'] * df_book['bid_size1']) +
                                   (df_book['bid_price2'] * df_book['ask_size2'] + df_book['ask_price2'] * df_book['bid_size2'])) / \
                                  (df_book['bid_size1'] + df_book['ask_size1'] + df_book['bid_size2'] + df_book['ask_size2'])

                # Squared log returns of sequences
                log_returns_sequences = [
                    'bid_price1', 'bid_price2', 'ask_price1', 'ask_price2',
                    'bid_size1', 'bid_size2', 'ask_size1', 'ask_size2',
                    'wap1', 'wap2', 'wap3'
                ]

                for sequence in log_returns_sequences:
                    df_book[f'{sequence}_squared_log_returns'] = np.log(df_book[sequence] / df_book.groupby('time_id')[sequence].shift(1)) ** 2

                # Aggregations on entire sequences
                feature_aggregations = {
                    'seconds_in_bucket': ['count'],
                    'bid_price1': [],
                    'bid_price2': [],
                    'ask_price1': [],
                    'ask_price2': [],
                    'bid_size1': ['std'],
                    'bid_size2': ['std'],
                    'ask_size1': ['std'],
                    'ask_size2': ['std'],
                    'bid_price_competitiveness': ['mean', 'max', 'distance'],
                    'ask_price_competitiveness': ['mean', 'max', 'distance'],
                    'bid_ask_price1_distance': ['mean', 'std', 'max'],
                    'bid_ask_price2_distance': ['mean', 'std', 'max'],
                    'bid_size': ['mean', 'std', 'max', 'min', 'distance'],
                    'ask_size': ['mean', 'std', 'max', 'min', 'distance'],
                    'total_size': ['mean', 'std', 'max', 'min', 'distance'],
                    'bid_price1_squared_log_returns': ['mean', 'std'],
                    'bid_price2_squared_log_returns': ['mean', 'std'],
                    'ask_price1_squared_log_returns': ['mean', 'std'],
                    'ask_price2_squared_log_returns': ['mean', 'std'],
                    'bid_size1_squared_log_returns': ['mean', 'std'],
                    'bid_size2_squared_log_returns': ['mean', 'std'],
                    'ask_size1_squared_log_returns': ['mean', 'std'],
                    'ask_size2_squared_log_returns': ['mean', 'std'],
                    'wap1': ['std'],
                    'wap2': ['std'],
                    'wap3': ['std'],
                    'wap1_squared_log_returns': ['mean', 'std', 'realized_volatility'],
                    'wap2_squared_log_returns': ['mean', 'std', 'realized_volatility'],
                    'wap3_squared_log_returns': ['mean', 'std', 'realized_volatility'],
                }

                for feature, aggregations in feature_aggregations.items():
                    if len(aggregations) > 0:
                        for aggregation in aggregations:
                            if aggregation == 'realized_volatility':
                                feature_aggregation = np.sqrt(df_book.groupby('time_id')[feature].sum())
                            elif aggregation == 'distance':
                                feature_aggregation = df_book.groupby('time_id')[feature].max() - df_book.groupby('time_id')[feature].min()
                            else:
                                feature_aggregation = df_book.groupby('time_id')[feature].agg(aggregation)

                            df.loc[df['stock_id'] == stock_id, f'book_{feature}_{aggregation}'] = df[df['stock_id'] == stock_id]['time_id'].map(feature_aggregation)

                # Aggregations on equally split sequences
                for n_splits in [2, 4, 10]:
                    timesteps = np.append(np.arange(0, 600, 600 // n_splits), [600])
                    for split, (t1, t2) in enumerate(zip(timesteps, timesteps[1:]), 1):
                        # Aggregating only on the last split
                        if t2 != timesteps[-1]:
                            continue

                        split_wap1_squared_log_returns_realized_volatility_aggregation = np.sqrt(df_book.loc[(df_book['seconds_in_bucket'] >= t1) & (df_book['seconds_in_bucket'] < t2), :].groupby('time_id')['wap1_squared_log_returns'].sum())
                        df.loc[df['stock_id'] == stock_id, f'book_wap1_squared_log_returns_{t1}-{t2}_realized_volatility'] = df[df['stock_id'] == stock_id]['time_id'].map(split_wap1_squared_log_returns_realized_volatility_aggregation)

                        split_seconds_in_bucket_count_aggregation = df_book.loc[(df_book['seconds_in_bucket'] >= t1) & (df_book['seconds_in_bucket'] < t2), :].groupby('time_id')['seconds_in_bucket'].count()
                        df.loc[df['stock_id'] == stock_id, f'book_seconds_in_bucket_{t1}-{t2}_count'] = df[df['stock_id'] == stock_id]['time_id'].map(split_seconds_in_bucket_count_aggregation)

    def _get_trade_features(self):

        for dataset, df in zip(['train', 'test'], [self.df_train, self.df_test]):

            for stock_id in tqdm(df['stock_id'].unique()):

                df_trade = pd.read_parquet(f'{path_utils.DATA_PATH}/trade_{dataset}.parquet/stock_id={stock_id}')

                df_trade['order_average_size'] = df_trade['size'] / df_trade['order_count']

                # Squared log returns of sequences
                log_returns_sequences = ['price']

                for sequence in log_returns_sequences:
                    df_trade[f'{sequence}_squared_log_returns'] = np.log(df_trade[sequence] / df_trade.groupby('time_id')[sequence].shift(1)) ** 2

                feature_aggregations = {
                    'seconds_in_bucket': ['count'],
                    'price': ['std'],
                    'size': ['std'],
                    'order_count': ['mean', 'sum'],
                    'order_average_size': ['mean', 'std'],
                    'price_squared_log_returns': ['mean', 'std', 'realized_volatility']
                }

                for feature, aggregations in feature_aggregations.items():
                    if len(aggregations) > 0:
                        for aggregation in aggregations:
                            if aggregation == 'realized_volatility':
                                feature_aggregation = np.sqrt(df_trade.groupby('time_id')[feature].sum())
                            else:
                                feature_aggregation = df_trade.groupby('time_id')[feature].agg(aggregation)

                            df.loc[df['stock_id'] == stock_id, f'trade_{feature}_{aggregation}'] = df[df['stock_id'] == stock_id]['time_id'].map(feature_aggregation)

                # Aggregations on equally split sequences
                for n_splits in [2, 4, 10]:
                    timesteps = np.append(np.arange(0, 600, 600 // n_splits), [600])
                    for split, (t1, t2) in enumerate(zip(timesteps, timesteps[1:]), 1):
                        # Aggregating only on the last split
                        if t2 != timesteps[-1]:
                            continue

                        split_price_squared_log_returns_realized_volatility_aggregation = np.sqrt(df_trade.loc[(df_trade['seconds_in_bucket'] >= t1) & (df_trade['seconds_in_bucket'] < t2), :].groupby('time_id')['price_squared_log_returns'].sum())
                        df.loc[df['stock_id'] == stock_id, f'trade_price_squared_log_returns_{t1}-{t2}_realized_volatility'] = df[df['stock_id'] == stock_id]['time_id'].map(split_price_squared_log_returns_realized_volatility_aggregation)

            # Fill missing trade data with zeros
            self.df_train = self.df_train.fillna(0)
            self.df_test = self.df_test.fillna(0)

    def transform(self):

        self._get_folds()
        self._label_encode()

        if self.create_features:

            self._get_book_features()
            self._get_trade_features()

            self.df_train.to_csv(f'{path_utils.DATA_PATH}/train_features.csv', index=False)
            self.df_test.to_csv(f'{path_utils.DATA_PATH}/test_features.csv', index=False)

        else:

            self.df_train = pd.read_csv(f'{path_utils.DATA_PATH}/train_features.csv')
            self.df_test = pd.read_csv(f'{path_utils.DATA_PATH}/test_features.csv')

        return self.df_train, self.df_test
