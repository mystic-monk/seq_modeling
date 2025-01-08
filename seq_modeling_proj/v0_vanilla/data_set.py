# data_set.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from sklearn.model_selection import TimeSeriesSplit
# Sklearn tools
from sklearn.preprocessing import StandardScaler

# Import configuration
from config import p , logger


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
        return self.X.shape[0]

    def __getitem__(self, index):
        end = min(index + self.seq_len, self.X.shape[0])
        
        X_seq = self.X[index : end]

        y_seq = self.y[index : end]

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
        self.train_val_data_path = p["train_val_data_path"]
        self.test_data_path = p["test_data_path"]
        self.nums_splits = p["nums_splits"]
        self.X_train_val = None 
        self.y_train_val = None
        
        self.X_test_p= None
        self.y_test_p= None

        self.debug = p["debug"]


    def prepare_data(self):
        """
        This method is for any data-related operations that should happen once
        and before setup. For example, loading datasets.
        """
        if self.X_train is not None:
            return
        print("-----------------    -----------------")
        print("Prepare_data: Start")
        print("-----------------    -----------------")
        logger.info("Preparing data...")

        # Example: Check if data files exist, and if not, download or extract them
        if not os.path.exists(self.train_val_data_path) or not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path} and or {self.test_data_path}")
        
        self.X_train_val, self.y_train_val = self.load_and_preprocess_data(self.train_val_data_path)
        self.X_test_p, self.y_test_p = self.load_and_preprocess_data(self.test_data_path)
        
        print("\tData is loaded and preprocessed.")

        print(f"\tX_train_val shape: {self.X_train_val.shape}, y_train_val shape: {self.y_train_val.shape}")
        print(f"\tX_test_p shape: {self.X_test_p.shape}, y_test_p shape: {self.y_test_p.shape}")

        return

    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the data.
        """
        print("-----------------    -----------------")
        print("Data is being loaded...")
        print("-----------------    -----------------")
        data = pd.read_parquet(
            data_path, 
            columns=["event_creation_date", "log_cases_14d_moving_avg"]
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data = data.sort_values(by='event_creation_date')
        data.set_index('event_creation_date', inplace=True)
        
        X = data[["log_cases_14d_moving_avg"]].copy() # type: ignore
        y = X["log_cases_14d_moving_avg"].shift(-self.output_size).dropna()

        # Trim X to match the length of y
        X = X.iloc[:len(y)]


        return X, y
    
    
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


        if stage == "fit" and self.X_train is not None:
            print("Setup is returned.")
            return
        if stage == "test" and self.X_test is not None:
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return
                
        if self.debug:
            print("-----------------    -----------------")
            print("Setup is called.")
            print("-----------------    -----------------")
            logger.info("Setup is called.")
        
        X_train, X_val, y_train, y_val = self.walkForwardSplit(self.X_train_val, self.y_train_val, self.nums_splits)

        print("\tData is split.")
        print(f"\tX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
        print(f"\tX_val shape: {X_val.shape}, y_val shape: {y_val.shape}")   

        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        # if stage == "fit" or stage is None:
        self.X_train = preprocessing.transform(X_train)
        self.X_val = preprocessing.transform(X_val)

        # self.X_train = np.array(X_train)
        # self.X_val = np.array(X_val)

        self.y_train = np.array(y_train)
        self.y_val = np.array(y_val)


    # if stage == "test" or stage is None:
        self.X_test = preprocessing.transform(self.X_test_p)
        self.y_test = np.array(self.y_test_p)

    # def setup_fold(self, train_idx, val_idx):
    #     """
    #     Sets up the training and validation datasets for the given fold.
    #     """
    #     X, y = self.load_and_preprocess_data()
        

    #     preprocessing = StandardScaler()
    #     preprocessing.fit(X.iloc[train_idx])

    #     self.X_train = preprocessing.transform(X.iloc[train_idx])
    #     self.y_train = y[train_idx].reshape((-1, self.output_size))
    #     self.X_val = preprocessing.transform(X.iloc[val_idx])
    #     self.y_val = y[val_idx].reshape((-1, self.output_size))


    def train_dataloader(self):
        print("-----------------    -----------------")
        print("Train Dataloader: Start.")
        print("-----------------    -----------------")
        print(f"\tX_train shape: {self.X_train.shape}", f"y_train shape: {self.y_train.shape}")
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
        print("-----------------    -----------------")
        print("Val Dataloader: Start.")
        print("-----------------    -----------------")
        print(f"\tX_val shape: {self.X_val.shape}", f"y_val shape: {self.y_val.shape}")
        val_dataset = TimeseriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        print(f"\tVal Length {len(val_dataset)}")
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        print(f"\tVal Dataloader Shape {len(val_loader)}")
        return val_loader

    def test_dataloader(self):
        test_dataset = TimeseriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        print("--------------------------------------")
        print("Test Dataloader: Start.")
        print("--------------------------------------") 
        print(f"\tTest Length {len(test_dataset)}")
        print(f"\tX_test shape: {self.X_test.shape}", f"y_test shape: {self.y_test.shape}")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        return test_loader
