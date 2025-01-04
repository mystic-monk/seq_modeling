# data_set.py
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L

# Sklearn tools
from sklearn.model_selection import train_test_split
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
        self.data_path = p["data_path"]

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
        data.set_index('event_creation_date', inplace=True)
        
        # Exclude the last two weeks of data
        # end_date = data.index.max() - pd.Timedelta(weeks=2)
        # data = data[data.index <= end_date]

        X = data[["log_cases_14d_moving_avg"]].copy() # type: ignore
        y = X["log_cases_14d_moving_avg"].shift(-1).dropna()
        
        
        # Ensure y has enough values to support slicing with output_size
        y = y.iloc[:len(y) - self.output_size + 1]  # Trim y to ensure no out-of-bounds slicing
        
        # Trim X to match the length of y
        X = X.iloc[-len(y):]
        
        return X, y
    

    def split_data(self, X, y, future_forecast):
        # Ensure the dataset length is divisible by output_size
        total_size = len(X)
        num_of_bins = total_size //  future_forecast

        divisible_size = num_of_bins * future_forecast
        # print(f"Total size: {total_size}", f"Divisible size: {divisible_size}")
        X, y = X[len(X) - divisible_size:], y[len(y) - divisible_size:]
        # print(f"New X shape: {X.shape}", f"New y shape: {y.shape}")

        train_val_bins = int(num_of_bins * 0.8)  # 80% for training
        # print(f"Train and validation bins: {train_val_bins}")
        train_bins = int(train_val_bins * 0.8)  # 80% of 80% for training
        train_size = train_bins * future_forecast

        val_bins = train_val_bins - train_bins  # Remaining 20% for validation
        val_size = val_bins * future_forecast 

        test_bins = num_of_bins - train_val_bins  # Remaining 20% for testing
        test_size = test_bins * future_forecast

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

        return X_train, X_val, X_test, y_train, y_val, y_test
    
    
    def setup(self, stage=None):
        """
        Data is already transformed and so no need to resample.
        Both 'np.nan' and '?' are converted to 'np.nan'
        'Date' and 'Time' columns are merged into 'dt' index
        """
        print("Setup is called.")
        logger.info("Setup is called.")

        if stage == "fit" and self.X_train is not None:
            return
        if stage == "test" and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return

        X, y = self.load_and_preprocess_data()
        
        print(f"X shape: {X.shape}, y shape: {y.shape}")
        print(f"Output size: {self.output_size}")
        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(X, y, self.output_size)

        # X_cv, X_test, y_cv, y_test = train_test_split(
        #     X, y, test_size=0.2, shuffle=False
        # )

        # X_train, X_val, y_train, y_val = train_test_split(
        #     X_cv, y_cv, test_size=0.25, shuffle=False
        # )
        print(f"Train size: {len(X_train)}, Val size: {len(X_val)}, Test size: {len(X_test)}")
        # logger.info(f"Len of X_cv: {len(X_cv)}, Len of X_train: {len(X_train)}, Len of X_val: {len(X_val)}, Len of X_test: {len(X_test)}")

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if stage == "fit" or stage is None:
            self.X_train = preprocessing.transform(X_train)
            print(f"y_train shape: {y_train.shape}")
            self.y_train = y_train.values.reshape((-1, self.output_size))
            print(f"y_train shape: {y_train.shape}")
            self.X_val = preprocessing.transform(X_val)
            self.y_val = y_val.values.reshape((-1, self.output_size))

        if stage == "test" or stage is None:
            self.X_test = preprocessing.transform(X_test)
            self.y_test = y_test.values.reshape((-1, self.output_size))

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
