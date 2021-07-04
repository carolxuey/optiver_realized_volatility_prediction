import numpy as np

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
