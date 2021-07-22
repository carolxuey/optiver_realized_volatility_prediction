import sys
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
    ]
    df_books = pd.DataFrame(columns=book_features)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_book = preprocessing_utils.read_book_data('train', stock_id, sort=True, forward_fill=True)
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

    trade_features = ['price', 'size', 'order_count']
    df_trades = pd.DataFrame(columns=trade_features)

    for stock_id in tqdm(sorted(df['stock_id'].unique())):
        df_trade = preprocessing_utils.read_trade_data(df, 'train', stock_id, sort=True, zero_fill=True)
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
