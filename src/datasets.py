import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

import path_utils


class Optiver2DDataset(Dataset):

    def __init__(self, df, channels, normalization=None, flip_probability=0.):

        self.df = df
        self.channels = channels
        self.normalization = normalization

        if self.normalization == 'global':
            # Normalizing sequences with global means and stds across stocks
            book_means = np.array([
                0.99969482421875, 1.000321388244629, 0.9995064735412598, 1.0005191564559937,
                769.990177708821, 766.7345672818379, 959.3416027831918, 928.2202512713748,
                1.0000068043192514, 1.0000055320253616, 5.129816581143487e-08, 9.831598141593519e-08
            ])
            book_stds = np.array([
                0.0036880988627672195, 0.003687119111418724, 0.0037009266670793295, 0.0036990800872445107,
                5354.051690318169, 4954.947103063445, 6683.816183660414, 5735.299917793827,
                0.003689893218043926, 0.00370745215558702, 6.618708642293018e-07, 1.2508970015188411e-06
            ])
            # Not normalizing trade price and trade price log returns because of the sparsity
            trade_means = np.array([0, 352.9736760331942, 4.1732040971227145, 0])
            trade_stds = np.array([1, 1041.9441951057488, 7.79955795393431, 1])

            self.transforms = {
                'flip': flip_probability,
                'normalize': {
                    'book_means': book_means,
                    'book_stds': book_stds,
                    'trade_means': trade_means,
                    'trade_stds': trade_stds
                }
            }

        elif self.normalization == 'local':
            # Normalizing sequences with stock means and stds
            self.df_book_stock_means = pd.read_csv(f'{path_utils.DATA_PATH}/book_stock_means.csv')
            self.df_book_stock_stds = pd.read_csv(f'{path_utils.DATA_PATH}/book_stock_stds.csv')
            self.df_trade_stock_means = pd.read_csv(f'{path_utils.DATA_PATH}/trade_stock_means.csv')
            self.df_trade_stock_stds = pd.read_csv(f'{path_utils.DATA_PATH}/trade_stock_stds.csv')
            # Not normalizing trade price and trade price log returns because of the sparsity
            self.df_trade_stock_means['price'] = 0
            self.df_trade_stock_means['price_squared_log_returns'] = 0
            self.df_trade_stock_stds['price'] = 1
            self.df_trade_stock_stds['price_squared_log_returns'] = 1

            self.transforms = {
                'flip': flip_probability,
                'normalize': {
                    'book_means': None,
                    'book_stds': None,
                    'trade_means': None,
                    'trade_stds': None
                }
            }

        else:
            # No normalization
            self.transforms = {
                'flip': flip_probability,
                'normalize': {}
            }

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        """
        Get the idxth element in the dataset

        Parameters
        ----------
        idx (int): Index of the sample (0 <= idx < len(self.df))

        Returns
        -------
        stock_id_encoded [torch.LongTensor of shape (1)]: Encoded stock_id for stock embeddings
        sequences [torch.FloatTensor of shape (600, n_channels)]: Concatenated sequences from book and trade data
        target [torch.Tensor of shape (1)]: Target
        """

        sample = self.df.iloc[idx]
        stock_id = int(sample['stock_id'])
        time_id = int(sample['time_id'])

        if self.normalization == 'local':
            # Retrieve precomputed means and stds of the current stock
            book_means = self.df_book_stock_means.loc[self.df_book_stock_means['stock_id'] == stock_id].iloc[0, 1:].values
            book_stds = self.df_book_stock_stds.loc[self.df_book_stock_stds['stock_id'] == stock_id].iloc[0, 1:].values
            trade_means = self.df_trade_stock_means.loc[self.df_trade_stock_means['stock_id'] == stock_id].iloc[0, 1:].values
            trade_stds = self.df_trade_stock_stds.loc[self.df_trade_stock_stds['stock_id'] == stock_id].iloc[0, 1:].values

            self.transforms['normalize'] = {
                'book_means': book_means,
                'book_stds': book_stds,
                'trade_means': trade_means,
                'trade_stds': trade_stds
            }

        # Sequences from book data
        book_sequences = np.load(f'{path_utils.DATA_PATH}/book_train/stock_{stock_id}/time_{time_id}.npy')
        book_wap1 = (book_sequences[:, 0] * book_sequences[:, 5] + book_sequences[:, 1] * book_sequences[:, 4]) /\
                    (book_sequences[:, 4] + book_sequences[:, 5])
        book_wap2 = (book_sequences[:, 2] * book_sequences[:, 7] + book_sequences[:, 3] * book_sequences[:, 6]) /\
                    (book_sequences[:, 6] + book_sequences[:, 7])
        book_wap1_log = np.log(book_wap1)
        book_wap1_log_returns = np.abs(np.diff(book_wap1_log, prepend=[book_wap1_log[0]]))
        book_wap2_log = np.log(book_wap2)
        book_wap2_log_returns = np.abs(np.diff(book_wap2_log, prepend=[book_wap2_log[0]]))
        book_sequences = np.hstack([
            book_sequences,
            book_wap1.reshape(-1, 1),
            book_wap2.reshape(-1, 1),
            book_wap1_log_returns.reshape(-1, 1),
            book_wap2_log_returns.reshape(-1, 1),
        ])

        # Sequences from trade data
        trade_sequences = np.load(f'{path_utils.DATA_PATH}/trade_train/stock_{stock_id}/time_{time_id}.npy')
        trade_price_log1p = np.log1p(trade_sequences[:, 0])
        trade_price_log_returns = np.diff(trade_price_log1p, prepend=trade_price_log1p[0])
        trade_sequences = np.hstack([trade_sequences, trade_price_log_returns.reshape(-1, 1)])

        if self.normalization is not None:
            book_sequences = (book_sequences - self.transforms['normalize']['book_means']) / self.transforms['normalize']['book_stds']
            trade_sequences = (trade_sequences - self.transforms['normalize']['trade_means'] / self.transforms['normalize']['trade_stds'])

        # Concatenate book and trade sequences
        sequences = np.hstack([book_sequences, trade_sequences])
        sequences = torch.as_tensor(sequences[:, self.channels], dtype=torch.float)

        # Flip sequences on zeroth dimension
        if np.random.rand() < self.transforms['flip']:
            sequences = torch.flip(sequences, dims=[0])

        stock_id_encoded = torch.as_tensor(sample['stock_id_encoded'], dtype=torch.long)
        target = sample['target']
        target = torch.as_tensor(target, dtype=torch.float)

        return stock_id_encoded, sequences, target
