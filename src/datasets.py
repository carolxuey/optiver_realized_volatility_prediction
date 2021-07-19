import numpy as np
import torch
from torch.utils.data import Dataset

import path_utils


class OptiverDataset(Dataset):

    def __init__(self, df, dataset):

        self.df = df
        self.dataset = dataset

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        sample = self.df.iloc[idx]
        stock_id = int(sample['stock_id'])
        time_id = int(sample['time_id'])

        book_means = torch.tensor([
            0.99969482421875, 1.000321388244629, 0.9995064735412598, 1.0005191564559937,
            769.990177708821, 766.7345672818379, 959.3416027831918, 928.2202512713748
        ])
        book_stds = torch.tensor([
            0.0036880988627672195, 0.003687119111418724, 0.0037009266670793295, 0.0036990800872445107,
            5354.051690318169, 4954.947103063445, 6683.816183660414, 5735.299917793827
        ])

        book_sequences = np.load(f'{path_utils.DATA_PATH}/book_{self.dataset}/stock_{stock_id}/time_{time_id}.npy')
        book_sequences = torch.as_tensor(book_sequences, dtype=torch.float)
        book_sequences = (book_sequences - book_means) / book_stds

        if self.dataset == 'train':
            target = sample['target']
            target = torch.as_tensor(target, dtype=torch.float)
            return book_sequences, target

        elif self.dataset == 'test':
            return book_sequences
