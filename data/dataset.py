import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class ImbalancedDataset(Dataset):
    def __init__(self, features: np.ndarray, labels: np.ndarray):
        """
        Custom Dataset for handling class-imbalanced tabular or feature data.
        """
        self.features = torch.tensor(features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

def get_imbalanced_dataloader(features: np.ndarray, labels: np.ndarray, batch_size: int = 64, shuffle: bool = True):
    """
    Utility function to instantiate a DataLoader for training.
    """
    dataset = ImbalancedDataset(features, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=True)
