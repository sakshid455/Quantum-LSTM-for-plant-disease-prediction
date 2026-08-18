import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader


class SequenceDataset(Dataset):

    def __init__(self, X_path, y_path):

        self.X = np.load(X_path)
        self.y = np.load(y_path)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):

        x = torch.tensor(self.X[idx], dtype=torch.float32)

        y = torch.tensor(self.y[idx], dtype=torch.float32)

        return x, y


def get_dataloaders(batch_size=8):

    train_dataset = SequenceDataset(
        "data/sequences/X_train.npy",
        "data/sequences/y_train.npy",
    )

    val_dataset = SequenceDataset(
        "data/sequences/X_val.npy",
        "data/sequences/y_val.npy",
    )

    test_dataset = SequenceDataset(
        "data/sequences/X_test.npy",
        "data/sequences/y_test.npy",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
    )

    return train_loader, val_loader, test_loader