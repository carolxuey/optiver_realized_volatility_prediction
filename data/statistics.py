import sys
import numpy as np
import pandas as pd
from tqdm import tqdm

sys.path.append('..')
from src import preprocessing_utils


def get_book_statistics(df):

    """
    Calculate means and stds of book sequences on entire training set after forward filling

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set

    Returns
    -------
    means (dict): Means of specified features on entire training set
    stds (dict): Stds of specified features on entire training set
    """

    book_features = [
        'bid_price1', 'ask_price1', 'bid_price2', 'ask_price2',
        'bid_size1', 'ask_size1', 'bid_size2', 'ask_size2',
        'wap1', 'wap2', 'wap1_squared_log_returns', 'wap2_squared_log_returns'
    ]
    df_books = pd.DataFrame(columns=book_features)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_book = preprocessing_utils.read_book_data('train', stock_id, sort=True, forward_fill=True)
        df_book['wap1'] = (df_book['bid_price1'] * df_book['ask_size1'] + df_book['ask_price1'] * df_book['bid_size1']) /\
                          (df_book['bid_size1'] + df_book['ask_size1'])
        df_book['wap2'] = (df_book['bid_price2'] * df_book['ask_size2'] + df_book['ask_price2'] * df_book['bid_size2']) /\
                          (df_book['bid_size2'] + df_book['ask_size2'])
        df_book['wap1_squared_log_returns'] = np.log(df_book['wap1'] / df_book.groupby('time_id')['wap1'].shift(1)) ** 2
        df_book['wap2_squared_log_returns'] = np.log(df_book['wap2'] / df_book.groupby('time_id')['wap2'].shift(1)) ** 2
        df_books = pd.concat([df_books, df_book.loc[:, book_features]], axis=0, ignore_index=True)

    means = df_books.mean(axis=0).to_dict()
    stds = df_books.std(axis=0).to_dict()
    return means, stds


def get_trade_statistics(df):

    """
    Calculate means and stds of trade sequences on entire training set after zero filling

    Parameters
    ----------
    df [pandas.DataFrame of shape (n_samples, 3)]: Training set

    Returns
    -------
    means (dict): Means of specified features on entire training set
    stds (dict): Stds of specified features on entire training set
    """

    trade_features = ['price', 'size', 'order_count', 'price_squared_log_returns']
    df_trades = pd.DataFrame(columns=trade_features)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_trade = preprocessing_utils.read_trade_data(df, 'train', stock_id, sort=True, zero_fill=False)
        df_trade = preprocessing_utils.read_trade_data(df, 'train', stock_id, sort=True, zero_fill=True)
        df_trade['price_squared_log_returns'] = np.log(df_trade['price'] / df_trade.groupby('time_id')['price'].shift(1)) ** 2
        df_trades = pd.concat([df_trades, df_trade.loc[:, trade_features]], axis=0, ignore_index=True)

    means = df_trades.mean(axis=0).to_dict()
    stds = df_trades.std(axis=0).to_dict()
    return means, stds


if __name__ == '__main__':

    df_train = pd.read_csv(
        './train.csv',
        dtype=preprocessing_utils.train_test_dtypes['train']
    )

    print(get_book_statistics(df_train))
    print(get_trade_statistics(df_train))
