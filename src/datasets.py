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
        target = sample['target']

        book_sequences = np.load(f'{path_utils.DATA_PATH}/book_{self.dataset}/stock_{stock_id}/time_{time_id}.npy')
        book_sequences = torch.as_tensor(book_sequences, dtype=torch.float)
        target = torch.as_tensor(target, dtype=torch.double)

        return book_sequences, target
