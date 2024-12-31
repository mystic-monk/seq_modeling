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

    def __init__(self, X: np.ndarray, y: np.ndarray, seq_len: int, output_size: int):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len
        self.output_size = output_size

    def __len__(self):
        return max(0, len(self.X) - self.seq_len - self.output_size + 1)

    def __getitem__(self, index):
        # print(f"Index: {index}, Seq_len: {self.seq_len}, Output_size: {self.output_size}")
        X_seq = self.X[index : index + self.seq_len]
        y_seq = self.y[index + self.seq_len : index + self.seq_len + self.output_size]
        # print(f"X_seq: {X_seq.shape}, y_seq: {y_seq.shape}")
        return X_seq, y_seq
        #dx = self.X[index : index + self.seq_len], self.y[index + self.seq_len : index + self.seq_len + self.output_size]
        #return dx


class LineListingDataModule(L.LightningDataModule):
    """
    PyTorch Lighting DataModule subclass:
    https://pytorch-lightning.readthedocs.io/en/latest/datamodules.html

    Serves the purpose of aggregating all data loading
      and processing work in one place.
    """

    def __init__(self, seq_len=1, output_size=1, batch_size=32, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        self.columns = None
        self.preprocessing = None
        self.data_path = "../data/transformed/influenza_features.parquet"

    def prepare_data(self):
        pass
    
    def load_and_preprocess_data(self):
        data = pd.read_parquet(
            self.data_path, 
            columns=["event_creation_date", "log_cases_14d_moving_avg", "cases_14d_moving_avg","diff_log_14d"]
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data.set_index('event_creation_date', inplace=True)
        X = data[["log_cases_14d_moving_avg", "cases_14d_moving_avg", "diff_log_14d"]].copy()  # Ensure the correct columns are included
        # y = X["diff_log_14d"].shift(-1).ffill().dropna()
        # return X, y
            # Generate y as sequences of output_size
        y = np.array([X["diff_log_14d"].iloc[i:i + self.output_size].values
                  for i in range(len(X) - self.output_size + 1)])
        return X.iloc[:-self.output_size + 1], y

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

        X, y = self.load_and_preprocess_data()


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
            # self.y_train = y_train.values.reshape((-1, 1))
            self.y_train = y_train.reshape((-1, self.output_size))
            self.X_val = preprocessing.transform(X_val)
            # self.y_val = y_val.values.reshape((-1, 1))
            self.y_val = y_val.reshape((-1, self.output_size))

        if stage == "test" or stage is None:
            self.X_test = preprocessing.transform(X_test)

            print(f"y_test shape: {y_test.shape}, output_size: {self.output_size}")

            # self.y_test = y_test.values.reshape((-1, 1))
            self.y_test = y_test.reshape((-1, self.output_size))

    def setup_fold(self, train_idx, val_idx):
        """
        Sets up the training and validation datasets for the given fold.
        """
        X, y = self.load_and_preprocess_data()
        preprocessing = StandardScaler()
        preprocessing.fit(X.iloc[train_idx])

        self.X_train = preprocessing.transform(X.iloc[train_idx])
        self.y_train = y[train_idx].reshape((-1, self.output_size))
        self.X_val = preprocessing.transform(X.iloc[val_idx])
        self.y_val = y[val_idx].reshape((-1, self.output_size))


    def train_dataloader(self):
        print("Train Dataloader is called.")
        train_dataset = TimeseriesDataset(
            self.X_train,
            self.y_train,
            seq_len=self.seq_len,  # type: ignore
            output_size=self.output_size  # Use the output_size from the instance
        )
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return train_loader

    def val_dataloader(self):
        val_dataset = TimeseriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return test_loader
