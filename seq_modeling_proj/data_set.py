# data_set.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# Sklearn tools
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TimeseriesDataset(Dataset):
    """
    Custom Dataset subclass.
    Serves as input to DataLoader to transform X
      into sequence data using rolling window.
    DataLoader using this dataset will output batches
      of `(batch_size, seq_len, n_features)` shape.
    Suitable as an input to RNNs.
    """

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int = 1):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len

    def __len__(self):
        # return self.X.__len__() - (self.seq_len - 1)
        # return len(self.X) - self.seq_len + 1
        return max(0, len(self.X) - self.seq_len + 1)

    def __getitem__(self, index):
        dx = self.X[index : index + self.seq_len], self.y[index + self.seq_len - 1]

        return dx


class LineListingDataModule(L.LightningDataModule):
    """
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    """

    def __init__(self, seq_len=1, batch_size=32, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.columns = None
        self.preprocessing = None
        self.data_path = "../data/transformed/influenza_all_seasons.parquet"
        # self.data_path = "../data/transformed/influenza_features.parquet"

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        """
        Data is already transformed and so no need to resample.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        """

        if stage == "fit" and self.X_train is not None:
            return
        if stage == "test" and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return


        # data = pd.read_parquet(self.data_path, columns=["event_creation_date", "log_cases_14d_moving_avg", "outlier"])
        data = pd.read_parquet(self.data_path, columns=["event_creation_date", "diff_log_14d"])

           # Check if data is loaded correctly
        if data is None or data.empty:
            logger.error("Data loading failed. The dataframe is empty or None.")
            raise ValueError("Data loading failed.")

        # data.index = pd.to_datetime(data.index)
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data.set_index('event_creation_date', inplace=True)
        df_resample = data.copy()
      
        # X = df_resample.dropna().copy()
        # X = df_resample[["log_cases_14d_moving_avg", "outlier"]].copy()
        X = df_resample[["diff_log_14d"]].copy()
        y = X["diff_log_14d"].shift(-1).ffill()
        y = y.dropna()

        if X.empty or y.empty:
            logger.error("Data after preprocessing is empty.")
            raise ValueError("Data after preprocessing is empty.")

        self.columns = X.columns

        X_cv, X_test, y_cv, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        X_train, X_val, y_train, y_val = train_test_split(
            X_cv, y_cv, test_size=0.25, shuffle=False
        )

        logger.info(f"Len of X_cv: {len(X_cv)}, Len of X_train: {len(X_train)}, Len of X_val: {len(X_val)}, Len of X_test: {len(X_test)}")

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == "fit" or stage is None:
            self.X_train = preprocessing.transform(X_train)
            self.y_train = y_train.values.reshape((-1, 1))
            self.X_val = preprocessing.transform(X_val)
            self.y_val = y_val.values.reshape((-1, 1))

        if stage == "test" or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = y_test.values.reshape((-1, 1))

    def train_dataloader(self):
        print("Train Dataloader is called.")
        train_dataset = TimeseriesDataset(
            self.X_train,
            self.y_train,
            seq_len=self.seq_len,  # type: ignore
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, self.y_val, seq_len=self.seq_len)  # type: ignore
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            #persistent_workers=True
        )

        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.y_test, seq_len=self.seq_len)  # type: ignore
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

        return test_loader
