import numpy as np
import pandas as pd


train_test_dtypes = {
    'train': {
        'stock_id': np.uint8,  # ID of the stock
        'time_id': np.uint16,  # ID of the time bucket
        'target': np.float64  # Realized volatility of the next 10 minutes
    },
    'test': {
        'stock_id': np.uint8,  # ID of the stock
        'time_id': np.uint16  # ID of the time bucket
    }
}

book_dtypes = {
    'time_id': np.uint16,  # ID of the time bucket
    'seconds_in_bucket': np.uint16,  # Number of seconds passed since the start of the bucket
    'bid_price1': np.float32,  # Highest buy price after normalization
    'ask_price1': np.float32,  # Lowest sell price after normalization
    'bid_price2': np.float32,  # Second highest buy price after normalization
    'ask_price2': np.float32,  # Second lowest sell price after normalization
    'bid_size1': np.uint32,  # Number of shares on the highest buy price
    'ask_size1': np.uint32,  # Number of shares on the lowest sell price
    'bid_size2': np.uint32,  # Number of shares on the second highest buy price
    'ask_size2': np.uint32,  # Number of shares on the second lowest sell price
}

trade_dtypes = {
    'time_id': np.uint16,  # ID of the time bucket
    'seconds_in_bucket': np.uint16,  # Number of seconds passed since the start of the bucket
    'price': np.float32,  # The average price of executed transactions happening in one second
    'size': np.uint16,  # The number of traded shares
    'order_count': np.uint16  # The number of unique trade orders taking place
}


def read_book_data(dataset, stock_id, sort=False, forward_fill=False):

    """
    Read book data of the selected time bucket

    Parameters
    ----------
    dataset (str): Name of the dataset (train or test)
    stock_id (int): ID of the stock (0 <= stock_id <= 126)
    sort (bool): Whether to sort book data by time_id and seconds_in_bucket or not
    forward_fill (bool): Whether to reindex every time bucket to 600 seconds and forward fill missing values or not

    Returns
    -------
    df_book [pandas.DataFrame of shape (n_updates or 600 if forward_fill is True, 10)]: Book data of the selected time bucket
    """

    df_book = pd.read_parquet(f'../data/book_{dataset}.parquet/stock_id={stock_id}')
    for column, dtype in book_dtypes.items():
        df_book[column] = df_book[column].astype(dtype)

    if sort:
        df_book.sort_values(by=['time_id', 'seconds_in_bucket'], inplace=True)

    if forward_fill:
        df_book = df_book.set_index(['time_id', 'seconds_in_bucket'])
        df_book = df_book.reindex(
            pd.MultiIndex.from_product([df_book.index.levels[0], np.arange(0, 600)], names=['time_id', 'seconds_in_bucket']),
            method='ffill'
        )
        df_book.reset_index(inplace=True)

    return df_book


def read_trade_data(df, dataset, stock_id, sort=False, zero_fill=False):

    """
    Read trade data of the selected time bucket

    Parameters
    ----------
    dataset (str): Name of the dataset (train or test)
    stock_id (int): ID of the stock (0 <= stock_id <= 126)
    sort (bool): Whether to sort book data by time_id and seconds_in_bucket or not
    zero_fill (bool): Whether to reindex every time bucket to 600 seconds and zero fill missing values or not

    Returns
    -------
    df_trade [pandas.DataFrame of shape (n_trades or 600 if zero_fill is True, 5)]: Trade data of the selected time bucket
    """

    df_trade = pd.read_parquet(f'../data/trade_{dataset}.parquet/stock_id={stock_id}')

    if zero_fill:
        stock_time_buckets = df.loc[df['stock_id'] == stock_id, 'time_id'].reset_index(drop=True)
        missing_time_buckets = stock_time_buckets[~stock_time_buckets.isin(df_trade['time_id'])]
        df_trade = df_trade.merge(missing_time_buckets, how='outer')

    if sort:
        df_trade.sort_values(by=['time_id', 'seconds_in_bucket'], inplace=True)

    if zero_fill:
        df_trade = df_trade.set_index(['time_id', 'seconds_in_bucket'])
        df_trade = df_trade.reindex(
            pd.MultiIndex.from_product([df_trade.index.levels[0], np.arange(0, 600)], names=['time_id', 'seconds_in_bucket']),
        )
        df_trade.fillna(0, inplace=True)
        df_trade.reset_index(inplace=True)

    for column, dtype in trade_dtypes.items():
        df_trade[column] = df_trade[column].astype(dtype)

    return df_trade
