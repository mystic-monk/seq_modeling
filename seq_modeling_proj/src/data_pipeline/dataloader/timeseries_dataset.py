from torch.utils.data import Dataset
import numpy as np

import torch
class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass for time series data.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, output_size: int):
        """
        Initialize the dataset with the input features and target values.
        """
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len
        self.output_size = output_size

    def __len__(self):
        """
        Returns the number of samples in the dataset.
        """
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """
        Returns a single sample at the given index.
        """
        return (self.X[index:index+self.seq_len], 
                self.y[index: index+self.output_size])