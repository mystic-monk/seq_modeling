import pandas as pd
import numpy as np

class ExpandingWindow:
    """
    Expanding window cross-validation for time series.

    Parameters
    ----------
    initial : int
        Initial training data length.
    horizon : int
        Forecast horizon (validation/test length).
    period : int
        Length by which training data expands in each iteration.
    """

    def __init__(self, initial=1, horizon=1, period=1):
        self.initial = initial
        self.horizon = horizon
        self.period = period

    def split(self, data):
        """
        Generate train-test splits using an expanding window approach.

        Parameters
        ----------
        data : array-like
            Input data to split.

        Returns
        -------
        splits : list of tuples
            A list where each tuple contains (train_index, val_index).
        """
        if isinstance(data, (pd.DataFrame, pd.Series)):
            data = data.to_numpy()

        data_index = np.arange(len(data))
        splits = []


        while True:
            train_end = self.initial + len(splits) * self.period
            val_end = train_end + self.horizon

            # Stop if the validation set goes out of bounds
            if val_end > len(data_index):
                break

            train_indices = data_index[:train_end].tolist()
            val_indices = data_index[train_end:val_end].tolist()

            splits.append((train_indices, val_indices))

        return splits
