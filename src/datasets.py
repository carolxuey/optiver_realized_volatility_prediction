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
        book_wap1 = (book_sequences[:, 0] * book_sequences[:, 5] + book_sequences[:, 1] * book_sequences[:, 4]) /\
                    (book_sequences[:, 4] + book_sequences[:, 5])
        book_wap2 = (book_sequences[:, 2] * book_sequences[:, 7] + book_sequences[:, 3] * book_sequences[:, 6]) /\
                    (book_sequences[:, 6] + book_sequences[:, 7])
        book_wap3 = ((book_sequences[:, 0] * book_sequences[:, 5] + book_sequences[:, 1] * book_sequences[:, 4]) +
                     (book_sequences[:, 2] * book_sequences[:, 7] + book_sequences[:, 3] * book_sequences[:, 6])) /\
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

        # Sequences from trade data
        trade_sequences = np.load(f'{path_utils.DATA_PATH}/trade_train/stock_{stock_id}/time_{time_id}.npy')

        if self.normalization is not None:
            book_sequences = (book_sequences - self.transforms['normalize']['book_means']) / self.transforms['normalize']['book_stds']
            # Not normalizing zero values in trade data
            trade_sequences[trade_sequences[:, 0] != 0, :] = (trade_sequences[trade_sequences[:, 0] != 0, :] - self.transforms['normalize']['trade_means']) / self.transforms['normalize']['trade_stds']

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
