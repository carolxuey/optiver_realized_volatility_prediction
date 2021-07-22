import sys
import os
from tqdm import tqdm
import numpy as np
import pandas as pd

sys.path.append('..')
from src import preprocessing_utils


def write_book_npy_files(df, dataset):

    """
    Write individual time buckets from book data as npy files

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 2)]: Training or test set with stock_id column
    dataset (str): Name of the dataset (train or test)
    """

    book_features = [
        'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2',
        'bid_size1', 'ask_size1', 'bid_size2', 'ask_size2',
    ]

    root_dir = f'./book_{dataset}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_book = preprocessing_utils.read_book_data(dataset, stock_id, sort=True, forward_fill=True)
        stock_dir = os.path.join(root_dir, f'stock_{stock_id}')
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)

        for time_id in sorted(df_book['time_id'].unique()):
            sequences = df_book.loc[df_book['time_id'] == time_id, book_features].values
            filename = os.path.join(stock_dir, f'time_{time_id}.npy')
            np.save(filename, sequences)


def write_trade_npy_files(df, dataset):

    """
    Write individual time buckets from trade data as npy files

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 2)]: Training or test set with stock_id column
    dataset (str): Name of the dataset (train or test)
    """

    trade_features = ['price', 'size', 'order_count']

    root_dir = f'./trade_{dataset}'
    if not os.path.exists(root_dir):
        os.makedirs(root_dir)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_trade = preprocessing_utils.read_trade_data(df, dataset, stock_id, sort=True, zero_fill=True)
        stock_dir = os.path.join(root_dir, f'stock_{stock_id}')
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)

        for time_id in sorted(df_trade['time_id'].unique()):
            sequences = df_trade.loc[df_trade['time_id'] == time_id, trade_features].values
            filename = os.path.join(stock_dir, f'time_{time_id}.npy')
            np.save(filename, sequences)


if __name__ == '__main__':

    df_train = pd.read_csv(
        './train.csv',
        dtype=preprocessing_utils.train_test_dtypes['train']
    )
    df_test = pd.read_csv(
        'test.csv',
        usecols=['stock_id', 'time_id'],
        dtype=preprocessing_utils.train_test_dtypes['test']
    )

    write_book_npy_files(df_train, 'train')
    write_book_npy_files(df_test, 'test')
    write_trade_npy_files(df_train, 'train')
    write_trade_npy_files(df_test, 'test')
