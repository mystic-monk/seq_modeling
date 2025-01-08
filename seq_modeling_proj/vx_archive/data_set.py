# data_set.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import TimeSeriesSplit
# Sklearn tools
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Import configuration
from config import p , logger

# import logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)


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
        # y_seq = self.y[index + self.seq_len : index + self.seq_len + self.output_size]
        y_seq = self.y[index : index + self.seq_len]
        print(f"X_seq shape: {X_seq.shape}, y_seq shape: {y_seq.shape}")    
        # Ensure y_seq has the correct length by trimming or padding
        # if len(y_seq) < self.output_size:
        #     y_seq = torch.cat([y_seq, torch.zeros(self.output_size - len(y_seq), y_seq.shape[1])])
        # else:
        #     y_seq = y_seq[:self.output_size]
        # 
        return X_seq, y_seq


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
        self.data_path = p["data_path"]
        self.nums_splits = p["nums_splits"]

    def prepare_data(self):
        print("Prepare Data is called.")
        logger.info("Prepare Data is called.")
        pass
    
    def load_and_preprocess_data(self):
        data = pd.read_parquet(
            self.data_path, 
            columns=["event_creation_date", "log_cases_14d_moving_avg"]
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data = data.sort_values(by='event_creation_date')
        data.set_index('event_creation_date', inplace=True)
        
        X = data[["log_cases_14d_moving_avg"]].copy() # type: ignore
        # y = X["log_cases_14d_moving_avg"].shift(-1).dropna()  # type: ignore
        y = X["log_cases_14d_moving_avg"].shift(-self.output_size).dropna()
        print(f"In Load and Preprocees :> X shape: {X.shape}", f"y shape: {y.shape}")
        # Ensure y has enough values to support slicing with output_size
        # y = y.iloc[self.output_size - 1:] 
        
        # Trim X to match the length of y
        X = X.iloc[:len(y)]
        print(f"In Load and Preprocees :>> X shape: {X.shape}", f"y shape: {y.shape}")
        print(f"X element : {X.head(5) }")
        print(f"y element : {y.head(5) }")
        return X, y
    

    def split_data(self, X, y, future_forecast):
        # Ensure the dataset length is divisible by output_size
        total_size = len(X)
        num_of_bins = total_size //  future_forecast

        divisible_size = num_of_bins * future_forecast
        print(f"Total size: {total_size}", f"Divisible size: {divisible_size}")
        X, y = X[len(X) - divisible_size:], y[len(y) - divisible_size:]
        print(f"New X shape: {X.shape}", f"New y shape: {y.shape}")

        train_val_bins = int(num_of_bins * 0.8)  # 80% for training
        print(f"Train and validation bin: {train_val_bins}")
        train_bins = int(train_val_bins * 0.8)  # 80% of 80% for training
        train_size = train_bins * future_forecast

        val_bins = train_val_bins - train_bins  # Remaining 20% for validation
        val_size = val_bins * future_forecast 

        test_bins = num_of_bins - train_val_bins  # Remaining 20% for testing
        test_size = test_bins * future_forecast
        print(f"Train size: {train_size}", f"Test size: {test_size}")

        # Perform the splits
        X_train, X_val, X_test = (
            X[:train_size],
            X[train_size:train_size + val_size],
            X[train_size + val_size:],
        )
        y_train, y_val, y_test = (
            y[:train_size],
            y[train_size:train_size + val_size],
            y[train_size + val_size:],
        )
        print(f"X_train shape: {X_train.shape}", f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}", f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}", f"y_test shape: {y_test.shape}")

        print(f"Total size: {total_size}", f"Divisible size: {divisible_size}")
        print(f"Total size: {train_size + val_size + test_size}")
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def walkForwardSplit(self, X, y, nums_splits):
        tss = TimeSeriesSplit(n_splits = nums_splits)
        
        for train_index, test_index in tss.split(X):
            X_train, X_test = X.iloc[train_index], X.iloc[test_index]
            y_train, y_test = y.iloc[train_index], y.iloc[test_index]
            return X_train, X_test, y_train, y_test

    def setup(self, stage=None):
        """
        Data is already transformed and so no need to resample.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        """
        print("Setup is called.")
        logger.info("Setup is called.")

        if stage == "fit" and self.X_train is not None:
            print("Setup is returned.")
            return
        if stage == "test" and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return
        print("Data is being loaded.")
        X, y = self.load_and_preprocess_data()
        print("Data is loaded.")
        
        # X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, self.output_size)
        
        X_train, X_test, y_train, y_test = self.walkForwardSplit(X, y, self.nums_splits)
        X_val, y_val = X_test, y_test  # Use the test split as validation for simplicity
        print("Data is split.")
        
        print(f"X_train shape: {X_train.shape}", f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}", f"y_val shape: {y_val.shape}")
        print(f"X_test shape: {X_test.shape}", f"y_test shape: {y_test.shape}")
        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == "fit" or stage is None:
            # self.X_train = preprocessing.transform(X_train)
            # self.X_val = preprocessing.transform(X_val)

            self.X_train = np.array(X_train)
            self.X_val = np.array(X_val)

            self.y_train = np.array(y_train)
            self.y_val = np.array(y_val)
  

        if stage == "test" or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = np.array(y_test)

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
        print(f"X_train shape: {self.X_train.shape}", f"y_train shape: {self.y_train.shape}")
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
