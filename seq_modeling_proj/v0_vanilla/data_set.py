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
from rich.console import Console
from rich.text import Text
# Import configuration
from config import p , logger
from sklearn.model_selection import train_test_split

# Rich Console for better terminal output
console = Console()


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
        # print(f" >>> Length calculated : {self.X.__len__() - (self.seq_len-1)}")
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """
        Returns a single sample at the given index.
        """
        # if index ==0 and self.deb:
        #     print(f"Index: {index}, X : {self.X[index:index+self.seq_len]}, y : {self.y[index: index+self.output_size]}")
        # return (self.X[index:index+self.seq_len], self.y[index+self.seq_len-1: index+self.seq_len-1+self.output_size])#.unsqueeze(0))
        return (self.X[index:index+self.seq_len], 
                self.y[index: index+self.output_size])#.unsqueeze(0))

class LineListingDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule subclass for handling data loading and processing.
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

        self.debug = p.get("debug", False)


    def prepare_data(self):
        """
        Prepares the data by loading and preprocessing it.
        """
        if self.X_train is not None:
            return
 
        logger.info("Preparing data...")

        # Example: Check if data files exist, and if not, download or extract them
        if not os.path.exists(self.train_val_data_path) or not os.path.exists(self.test_data_path):
            raise FileNotFoundError(f"Data not found at {self.data_path} and or {self.test_data_path}")
        
        self.X_train_val, self.y_train_val = self.load_and_preprocess_data(self.train_val_data_path)
        self.X_test_p, self.y_test_p = self.load_and_preprocess_data(self.test_data_path)
        
        if self.debug:
            console.print(f"[bold green]Data loaded and preprocessed successfully![/bold green]")
            console.print(f"X_train_val shape: {self.X_train_val.shape}, y_train_val shape: {self.y_train_val.shape}")
            console.print(f"X_test shape: {self.X_test_p.shape}, y_test shape: {self.y_test_p.shape}")


        return
    
    def split_data(self, X, y, window_size):
        # Ensure the dataset length is divisible by output_size
        total_size = len(X)
        num_of_bins = total_size //  window_size

        divisible_size = num_of_bins * window_size
        print(f"Total size: {total_size}", f"Divisible size: {divisible_size}")
        X, y = X[len(X) - divisible_size:], y[len(y) - divisible_size:]
        print(f"New X shape: {X.shape}", f"New y shape: {y.shape}")

        #train_val_bins = int(num_of_bins * 0.8)  # 80% for training
        #print(f"Train and validation bin: {train_val_bins}")
        train_bins = int(num_of_bins * 0.8)  # 80% of 80% for training
        train_size = train_bins * window_size

        val_bins = num_of_bins - train_bins  # Remaining 20% for validation
        val_size = val_bins * window_size 

        # test_bins = num_of_bins - train_val_bins  # Remaining 20% for testing
        # test_size = test_bins * window_size
        print(f"Train size: {train_size}", f"Val size: {val_size}")

        # Perform the splits
        X_train, X_val = (
            X[:train_size],
            X[train_size:train_size + val_size],
            # X[train_size + val_size:],
        )
        y_train, y_val = (
            y[:train_size],
            y[train_size:train_size + val_size],
            # y[train_size + val_size:],
        )
        print(f"X_train shape: {X_train.shape}", f"y_train shape: {y_train.shape}")
        print(f"X_val shape: {X_val.shape}", f"y_val shape: {y_val.shape}")
        # print(f"X_test shape: {X_test.shape}", f"y_test shape: {y_test.shape}")

        print(f"Total size: {total_size}", f"Divisible size: {divisible_size}")
        # print(f"Total size: {train_size + val_size + test_size}")
        return X_train, X_val, y_train, y_val

    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the data.
        """
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_parquet(
            data_path, 
            columns=["event_creation_date", "log_cases_14d_moving_avg"]
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data = data.sort_values(by='event_creation_date')
        data.set_index('event_creation_date', inplace=True)
        
        X = data[["log_cases_14d_moving_avg"]].copy() # type: ignore
        # y = X["log_cases_14d_moving_avg"].shift(-self.output_size).dropna()
        y = X["log_cases_14d_moving_avg"].shift(-self.seq_len).dropna()

        # Trim X to match the length of y
        X = X.iloc[:len(y)]
        return X, y
    
    
    def walkForwardSplit(self, X, y, nums_splits):
        """
        Perform walk-forward split for time series data.
        """
        # tss = TimeSeriesSplit(n_splits = nums_splits)
        
        # for train_index, test_index in tss.split(X):
        #     console.print(f"Train: {train_index}, Test: {test_index}")
        #     X_train, X_test = X.iloc[train_index], X.iloc[test_index]
        #     y_train, y_test = y.iloc[train_index], y.iloc[test_index]
        #     return X_train, X_test, y_train, y_test

        X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=.2, random_state=1, shuffle=False)
        return X_train, X_val, y_train, y_val


    def setup(self, stage=None):
        """
        Setup the data for training, validation, and testing.
        This called from the trainer, during training and testing.
        """


        if stage == "fit" and self.X_train is not None:
            logger.info("Data already setup for training.")
            return
        if stage == "test" and self.X_test is not None:
            logger.info("Data already setup for testing.")
            return
        if stage is None and self.X_train is not None and self.X_test is not None:
            return
                
        if self.debug:
            logger.info("Setup is being called.")
        
        # X_train, X_val, y_train, y_val = self.walkForwardSplit(self.X_train_val, self.y_train_val, self.nums_splits)
        X_train, X_val, y_train, y_val = self.split_data(self.X_train_val, self.y_train_val, self.seq_len+self.batch_size-1)

        logger.info("Data split into training and validation sets.")
        if self.debug:
            console.print(f"[bold cyan]Training data shape: {X_train.shape}, Validation data shape: {X_val.shape}[/bold cyan]")


        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        if self.debug:
            self.X_train = np.array(X_train)
            self.X_val = np.array(X_val)
        else:
            self.X_train = preprocessing.transform(X_train)
            self.X_val = preprocessing.transform(X_val)


        self.y_train = np.array(y_train)
        self.y_val = np.array(y_val)

        self.X_test = preprocessing.transform(self.X_test_p)
        self.y_test = np.array(self.y_test_p)

        if self.debug or True:
            console.print(f"[bold green]Data setup complete![/bold green]")
            console.print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            console.print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            console.print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")


    def train_dataloader(self):
        """
        Returns the train DataLoader.
        """
        if self.debug or True:
            logger.debug("Creating train DataLoader...")

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
        if self.debug or True:
            console.print(f"[bold green]Train DataLoader created with {len(train_loader)} batches[/bold green]")

        return train_loader

    def val_dataloader(self):
        """
        Returns the validation DataLoader.
        """
        if self.debug or True:
            logger.debug("Creating validation DataLoader...")
        val_dataset = TimeseriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        if self.debug or True:
            console.print(f"[bold yellow]Validation DataLoader created with {len(val_loader)} batches[/bold yellow]")

        return val_loader

    def test_dataloader(self):
        """
        Returns the test DataLoader.
        """
        if self.debug or True:
            logger.debug("Creating test DataLoader...")
        test_dataset = TimeseriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        if self.debug or True:
            console.print(f"[bold red]Test DataLoader created with {len(test_loader)} batches[/bold red]")

        return test_loader
