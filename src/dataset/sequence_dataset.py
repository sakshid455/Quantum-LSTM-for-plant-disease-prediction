import numpy as np
import torch
from torch.utils.data import Dataset


class WheatSequenceDataset(Dataset):

    def __init__(self):

        self.X = np.load("data/sequences/X.npy")
        self.y = np.load("data/sequences/y.npy")

    def __len__(self):

        return len(self.X)

    def __getitem__(self, idx):

        feature = torch.tensor(
            self.X[idx],
            dtype=torch.float32
        )

        target = torch.tensor(
            self.y[idx],
            dtype=torch.float32
        )

        return feature, target