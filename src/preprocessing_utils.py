import numpy as np
import pandas as pd


train_test_dtypes = {
    'train_dtypes': {
        'stock_id': np.uint8,  # ID of the stock
        'time_id': np.uint16,  # ID of the time bucket
        'target': np.float64  # Realized volatility of the next 10 minutes
    },
    'test_dtypes': {
        'stock_id': np.uint8,  # ID of the stock
        'time_id': np.uint16  # ID of the time bucket
    }
}

book_dtypes = {
    'stock_id': np.uint8,  # ID of the stock
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
    'stock_id': np.uint8,  # ID of the stock
    'time_id': np.uint16,  # ID of the time bucket
    'seconds_in_bucket': np.uint16,  # Number of seconds passed since the start of the bucket
    'price': np.float32,  # The average price of executed transactions happening in one second
    'size': np.uint16,  # The number of traded shares
    'order_count': np.uint16  # The number of unique trade orders taking place
}


def read_book_data(dataset, stock_id=None):

    """
    Read the whole book data or selected stock_id partition from selected dataset

    Parameters
    ----------
    dataset (str): Name of the dataset (train or test)
    stock_id (int or None): ID of the stock (0 <= stock_id <= 126) or None

    Returns
    -------
    df_book [pandas.DataFrame of shape (n_samples, 10 if stock_id is selected else 11)]: Book data
    """

    if stock_id is None:
        df_book = pd.read_parquet(f'../data/book_{dataset}.parquet')
    else:
        df_book = pd.read_parquet(f'../data/book_{dataset}.parquet/stock_id={stock_id}')

    for column, dtype in book_dtypes.items():
        # Skip iteration if parquet file is partitioned by stock_id
        if column == 'stock_id' and stock_id is not None:
            continue
        df_book[column] = df_book[column].astype(dtype)

    if stock_id is None:
        df_book.sort_values(by=['stock_id', 'time_id', 'seconds_in_bucket'], inplace=True)
    else:
        df_book.sort_values(by=['time_id', 'seconds_in_bucket'], inplace=True)

    return df_book


def read_trade_data(dataset, stock_id=None):

    """
    Read the whole trade data or selected stock_id partition from selected dataset

    Parameters
    ----------
    dataset (str): Name of the dataset (train or test)
    stock_id (int or None): ID of the stock (0 <= stock_id <= 126) or None

    Returns
    -------
    df_trade [pandas.DataFrame of shape (n_samples, 5 if stock_id is selected else 6)]: Trade data
    """

    if stock_id is None:
        df_trade = pd.read_parquet(f'../data/trade_{dataset}.parquet')
    else:
        df_trade = pd.read_parquet(f'../data/trade_{dataset}.parquet/stock_id={stock_id}')

    for column, dtype in trade_dtypes.items():
        # Skip iteration if parquet file is partitioned by stock_id
        if column == 'stock_id' and stock_id is not None:
            continue
        df_trade[column] = df_trade[column].astype(dtype)

    if stock_id is None:
        df_trade.sort_values(by=['stock_id', 'time_id', 'seconds_in_bucket'], inplace=True)
    else:
        df_trade.sort_values(by=['time_id', 'seconds_in_bucket'], inplace=True)

    return df_trade
