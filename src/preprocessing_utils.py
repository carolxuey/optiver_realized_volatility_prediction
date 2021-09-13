import numpy as np
import pandas as pd

import path_utils


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
    df [pandas.DataFrame of shape (n_samples, 2)]: DataFrame with stock_id and time_id columns
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


def read_time_bucket(df_train, stock_id, time_id):

    """
    Read book and trade data of the selected time bucket from training set and concatenate them

    Parameters
    ----------
    df_train [pandas.DataFrame of shape (n_samples, 2)]: Training DataFrame with stock_id and time_id columns
    stock_id (int): ID of the stock (0 <= stock_id <= 126)
    time_id (int): ID of the stock (0 <= stock_id <= 32767)

    Returns
    -------
    sequences [array-like of shape (600, n_channels)]: Concatenated sequences from book and trade data
    current_realized_volatility (float): Realized volatility of the current 10-minute window
    target (float): Realized volatility of the next 10-minute window
    """

    # Raw sequences
    book_features = ['bid_price1', 'ask_price1', 'bid_price2', 'ask_price2', 'bid_size1', 'ask_size1', 'bid_size2', 'ask_size2']
    trade_features = ['price', 'size', 'order_count']

    # Normalizing sequences with global means and stds
    book_means = np.array([
        # Raw sequences
        0.99969482421875, 1.000321388244629, 0.9995064735412598, 1.0005191564559937,
        769.990177708821, 766.7345672818379, 959.3416027831918, 928.2202512713748,
        # Absolute log returns of raw sequences
        5.05890857311897e-05, 5.1026330766035244e-05, 5.74059049540665e-05, 5.8218309277435765e-05,
        0.3967152245253066, 0.39100519899866804, 0.3239659116907835, 0.31638538484106116,
        # Weighted average prices
        1.0000068043192514, 1.0000055320253616, 1.000006872969592,
        # Absolute log returns of weighted average prices
        8.211420490291096e-05, 0.00011112522790786203, 8.236187150264073e-05
    ])
    book_stds = np.array([
        # Raw sequences
        0.0036880988627672195, 0.003687119111418724, 0.0037009266670793295, 0.0036990800872445107,
        5354.051690318169, 4954.947103063445, 6683.816183660414, 5735.299917793827,
        # Absolute log returns of raw sequences
        0.00016576898633502424, 0.00016801751917228103, 0.0001837657910073176, 0.0001868011022452265,
        0.9121719707304721, 0.8988021131995019, 0.8415323589617927, 0.8244750862945265,
        # Weighted average prices
        0.003689893218043926, 0.00370745215558702, 0.0036913980961173682,
        # Absolute log returns of weighted average prices
        0.00021108155612872302, 0.00029320157822289604, 0.00019975085953727163
    ])
    trade_means = np.array([0.999971866607666, 352.9736760331942, 4.1732040971227145])
    trade_stds = np.array([0.004607073962688446, 1041.9441951057488, 7.79955795393431])

    # Order book
    df_book = pd.read_parquet(f'{path_utils.DATA_PATH}/book_train.parquet/stock_id={stock_id}')

    # Trade data
    df_trade = pd.read_parquet(f'{path_utils.DATA_PATH}/trade_train.parquet/stock_id={stock_id}')
    stock_time_buckets = df_train.loc[df_train['stock_id'] == stock_id, 'time_id'].reset_index(drop=True)
    missing_time_buckets = stock_time_buckets[~stock_time_buckets.isin(df_trade['time_id'])]
    df_trade = df_trade.merge(missing_time_buckets, how='outer')

    # Resample order book to 600 seconds, forward fill and back fill for edge cases
    df_book_time_bucket = df_book.loc[df_book['time_id'] == time_id]
    df_book_time_bucket = df_book_time_bucket.set_index(['seconds_in_bucket'])
    df_book_time_bucket = df_book_time_bucket.reindex(np.arange(0, 600), method='ffill').fillna(method='bfill')

    # Sequences from book data
    book_sequences = df_book_time_bucket.reset_index(drop=True)[book_features].values

    # Absolute log returns of raw sequences
    book_bid_price1_log = np.log(book_sequences[:, 0])
    book_bid_price1_absolute_log_returns = np.abs(np.diff(book_bid_price1_log, prepend=[book_bid_price1_log[0]]))
    book_ask_price1_log = np.log(book_sequences[:, 1])
    book_ask_price1_absolute_log_returns = np.abs(np.diff(book_ask_price1_log, prepend=[book_ask_price1_log[0]]))
    book_bid_price2_log = np.log(book_sequences[:, 2])
    book_bid_price2_absolute_log_returns = np.abs(np.diff(book_bid_price2_log, prepend=[book_bid_price2_log[0]]))
    book_ask_price2_log = np.log(book_sequences[:, 3])
    book_ask_price2_absolute_log_returns = np.abs(np.diff(book_ask_price2_log, prepend=[book_ask_price2_log[0]]))
    book_bid_size1_log = np.log(book_sequences[:, 4])
    book_bid_size1_absolute_log_returns = np.abs(np.diff(book_bid_size1_log, prepend=[book_bid_size1_log[0]]))
    book_ask_size1_log = np.log(book_sequences[:, 5])
    book_ask_size1_absolute_log_returns = np.abs(np.diff(book_ask_size1_log, prepend=[book_ask_size1_log[0]]))
    book_bid_size2_log = np.log(book_sequences[:, 6])
    book_bid_size2_absolute_log_returns = np.abs(np.diff(book_bid_size2_log, prepend=[book_bid_size2_log[0]]))
    book_ask_size2_log = np.log(book_sequences[:, 7])
    book_ask_size2_absolute_log_returns = np.abs(np.diff(book_ask_size2_log, prepend=[book_ask_size2_log[0]]))

    # Weighted average prices
    book_wap1 = (book_sequences[:, 0] * book_sequences[:, 5] + book_sequences[:, 1] * book_sequences[:, 4]) / \
                (book_sequences[:, 4] + book_sequences[:, 5])
    book_wap2 = (book_sequences[:, 2] * book_sequences[:, 7] + book_sequences[:, 3] * book_sequences[:, 6]) / \
                (book_sequences[:, 6] + book_sequences[:, 7])
    book_wap3 = ((book_sequences[:, 0] * book_sequences[:, 5] + book_sequences[:, 1] * book_sequences[:, 4]) +
                 (book_sequences[:, 2] * book_sequences[:, 7] + book_sequences[:, 3] * book_sequences[:, 6])) / \
                (book_sequences[:, 4] + book_sequences[:, 5] + book_sequences[:, 6] + book_sequences[:, 7])

    # Absolute log returns of weighted average prices
    book_wap1_log = np.log(book_wap1)
    book_wap1_absolute_log_returns = np.abs(np.diff(book_wap1_log, prepend=[book_wap1_log[0]]))
    book_wap2_log = np.log(book_wap2)
    book_wap2_absolute_log_returns = np.abs(np.diff(book_wap2_log, prepend=[book_wap2_log[0]]))
    book_wap3_log = np.log(book_wap3)
    book_wap3_absolute_log_returns = np.abs(np.diff(book_wap3_log, prepend=[book_wap3_log[0]]))

    book_sequences = np.hstack([
        book_sequences,
        book_bid_price1_absolute_log_returns.reshape(-1, 1),
        book_ask_price1_absolute_log_returns.reshape(-1, 1),
        book_bid_price2_absolute_log_returns.reshape(-1, 1),
        book_ask_price2_absolute_log_returns.reshape(-1, 1),
        book_bid_size1_absolute_log_returns.reshape(-1, 1),
        book_ask_size1_absolute_log_returns.reshape(-1, 1),
        book_bid_size2_absolute_log_returns.reshape(-1, 1),
        book_ask_size2_absolute_log_returns.reshape(-1, 1),
        book_wap1.reshape(-1, 1),
        book_wap2.reshape(-1, 1),
        book_wap3.reshape(-1, 1),
        book_wap1_absolute_log_returns.reshape(-1, 1),
        book_wap2_absolute_log_returns.reshape(-1, 1),
        book_wap3_absolute_log_returns.reshape(-1, 1),
    ])
    book_sequences = (book_sequences - book_means) / book_stds

    df_trade_time_bucket = df_trade.loc[df_trade['time_id'] == time_id]
    df_trade_time_bucket = df_trade_time_bucket.set_index(['seconds_in_bucket'])
    df_trade_time_bucket = df_trade_time_bucket.reindex(np.arange(0, 600)).fillna(0)

    # Sequences from trade data
    trade_sequences = df_trade_time_bucket.reset_index(drop=True)[trade_features].values
    # Not normalizing zero values in trade data
    trade_sequences[trade_sequences[:, 0] != 0, :] = (trade_sequences[trade_sequences[:, 0] != 0, :] - trade_means) / trade_stds

    # Concatenate book and trade sequences
    sequences = np.hstack([book_sequences, trade_sequences])

    current_realized_volatility = np.sqrt(np.sum(np.diff(book_wap1_log, prepend=[book_wap1_log[0]]) ** 2))
    target = df_train.loc[(df_train['stock_id'] == stock_id) & (df_train['time_id'] == time_id), 'target'].values[0]

    return sequences, current_realized_volatility, target
