# data_set.py
import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
import lightning as L
from torch.utils.data import Subset

# from expanding_window import ExpandingWindow

# Sklearn tools
from sklearn.preprocessing import StandardScaler
from rich.console import Console

# Import configuration
from config import p , logger

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
        return self.X.__len__() - (self.seq_len-1)

    def __getitem__(self, index):
        """
        Returns a single sample at the given index.
        """
        return (self.X[index:index+self.seq_len], 
                self.y[index: index+self.output_size])

class LineListingDataModule(L.LightningDataModule):
    """
    PyTorch Lightning DataModule subclass for handling data loading and processing.
    """
    def __init__(self, seq_len=14, output_size=1, batch_size=14, num_workers=0, initial=14, horizon=14, period=14):
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
        # self.data_path = p["data_path"]
        self.nums_splits = p["nums_splits"]
        self.data_columns = p["data_columns"]
        self.train_val_data_path = p["train_val_data_path"]
        self.test_data_path = p["test_data_path"]
        self.X_train_val = None 
        self.y_train_val = None
        self.debug = p.get("debug", False)
        self.initial = initial
        self.horizon = horizon
        self.period = period

    def collect_dataloader_info(self, dataloader, dataset_name="Dataset"):
        """
        Collect and print information about a DataLoader.
        """
        total_samples = len(dataloader.dataset)
        num_batches = len(dataloader)
        batch_size = dataloader.batch_size
        
        console.print(f"[bold cyan]{dataset_name} DataLoader Info:[/bold cyan]")
        console.print(f"  - Total samples: {total_samples}")
        console.print(f"  - Number of batches: {num_batches}")
        console.print(f"  - Batch size: {batch_size}")
        # console.print(f"  - Number of workers: {dataloader.num_workers}")
        console.print("-" * 50)

    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the data.
        """
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_parquet(
            data_path, columns=self.data_columns
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data = data.sort_values(by='event_creation_date')
        data.set_index('event_creation_date', inplace=True)
        
        X = data.copy()
        y = X["log_cases_14d_moving_avg"].shift(-self.seq_len).dropna()

        # Trim X to match the length of y
        X = X.iloc[:len(y)]
        return X, y
    

    def prepare_data(self):
        """
        Prepares the data by loading and preprocessing it.
        """
        if self.X_train_val is not None:
            return
 
        logger.info("Preparing data...")

        if not os.path.exists(self.train_val_data_path):
            raise FileNotFoundError(f"Data not found at {self.train_val_data_path}")
        
        self.X_train_val, self.y_train_val = self.load_and_preprocess_data(self.train_val_data_path)
        self.X_test, self.y_test = self.load_and_preprocess_data(self.test_data_path)
        
        if self.debug:
            console.print("[bold green]Data loaded and preprocessed successfully![/bold green]")
            logger.info(f"X_train_val shape: {self.X_train_val.shape}, y_train_val shape: {self.y_train_val.shape}")  
            logger.info(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}") 


        return


    def setup_split(self, train_indices, val_indices):
        """
        Setup data loaders for a specific train-test split.

        Args:
            train_indices (list): Indices for training data.
            test_indices (list): Indices for testing data.
        """

        self.X_train = self.X_train_val.iloc[train_indices]
        self.y_train = self.y_train_val.iloc[train_indices]

        self.X_val = self.X_train_val.iloc[val_indices]
        self.y_val = self.y_train_val.iloc[val_indices]

        self.X_train = np.array(self.X_train)
        self.X_val = np.array(self.X_val)

        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)


        if self.debug:
            console.print(  "[bold green]Data split into training and validation sets.[/bold green]")
            console.print(f"[bold cyan]Training data shape: {self.X_train.shape}, Validation data shape: {self.X_val.shape}[/bold cyan]")
            console.print(f"[bold cyan]Training target shape: {self.y_train}, Validation target shape: {self.y_val}[/bold cyan]")

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
        
        # Prepare the data
        logger.info("Data split into training and validation sets.")
        if self.debug:
            console.print(f"[bold cyan]Training data shape: {self.X_train.shape}, Validation data shape: {self.X_val.shape}[/bold cyan]")


        preprocessing = StandardScaler()
        preprocessing.fit(self.X_train)

        if self.debug:
            self.X_train = np.array(self.X_train)
            self.X_val = np.array(self.X_val)
        else:
            self.X_train = preprocessing.transform(self.X_train)
            self.X_val = preprocessing.transform(self.X_val)


        self.y_train = np.array(self.y_train)
        self.y_val = np.array(self.y_val)

        self.X_test = preprocessing.transform(self.X_test)
        self.y_test = np.array(self.y_test)

        if self.debug:
            console.print(f"[bold green]Data setup complete![/bold green]")
            console.print(f"X_train shape: {self.X_train.shape}, y_train shape: {self.y_train.shape}")
            console.print(f"X_val shape: {self.X_val.shape}, y_val shape: {self.y_val.shape}")
            console.print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")


    def update_splits(self, split_index):
        """
        Update the training and validation sets for a given walk-forward split index.
        """
        splits = self.walk_forward_split(self.X_train_val, self.y_train_val)
        X_train, X_val, y_train, y_val = splits[split_index]

        # Preprocess the new split
        preprocessing = StandardScaler()
        preprocessing.fit(X_train)

        self.X_train = preprocessing.transform(X_train)
        self.X_val = preprocessing.transform(X_val)
        self.y_train = np.array(y_train)
        self.y_val = np.array(y_val)


    def create_dataloader(self, dataset, batch_size, dataset_name: str):
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )
        console.print(f"[bold green]{dataset_name} DataLoader created with {len(loader)} batches[/bold green]")
        self.collect_dataloader_info(loader, dataset_name)
        return loader
    
    def train_dataloader(self):
        """
        Returns the train DataLoader.
        """
        logger.info("Creating train DataLoader...")

        train_dataset = TimeseriesDataset(
            self.X_train,
            self.y_train,
            seq_len=self.seq_len,  # type: ignore
            output_size=self.output_size  
        )


        # return train_loader
        return self.create_dataloader(train_dataset, self.batch_size, "Train")

    def val_dataloader(self):
        """
        Returns the validation DataLoader.
        """

        logger.info("Creating validation DataLoader...")
        val_dataset = TimeseriesDataset(self.X_val, self.y_val, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore
        
        # val_loader = DataLoader(
        #     val_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     drop_last=True,
        # )

        # console.print(f"[bold yellow]Validation DataLoader created with {len(val_loader)} batches[/bold yellow]")

        # Collect and print the DataLoader info
        # self.collect_dataloader_info(val_loader, "Validation")

        return self.create_dataloader(val_dataset, self.batch_size,"Validation")

    def test_dataloader(self):
        """
        Returns the test DataLoader.
        """

        logger.info("Creating test DataLoader...")
        test_dataset = TimeseriesDataset(self.X_test, self.y_test, seq_len=self.seq_len, output_size=self.output_size)  # type: ignore

        # test_loader = DataLoader(
        #     test_dataset,
        #     batch_size=self.batch_size,
        #     shuffle=False,
        #     num_workers=self.num_workers,
        #     drop_last=True,
        # )

        # console.print(f"[bold red]Test DataLoader created with {len(test_loader)} batches[/bold red]")

        # # Collect and print the DataLoader info
        # self.collect_dataloader_info(test_loader, "Test")

        return self.create_dataloader(test_dataset, self.batch_size, "Test")



# class ExpandingWindow:
#     """
#     Expanding window cross-validation for time series.

#     Parameters
#     ----------
#     initial : int
#         Initial training data length.
#     horizon : int
#         Forecast horizon (validation/test length).
#     period : int
#         Length by which training data expands in each iteration.
#     """

#     def __init__(self, initial=1, horizon=1, period=1):
#         self.initial = initial
#         self.horizon = horizon
#         self.period = period

#     def split(self, data):
#         """
#         Generate train-test splits using an expanding window approach.

#         Parameters
#         ----------
#         data : array-like
#             Input data to split.

#         Returns
#         -------
#         splits : list of tuples
#             A list where each tuple contains (train_index, test_index).

#         Example
#         -------
#         >>> data = np.arange(10)
#         >>> ew = ExpandingWindow(initial=3, horizon=2, period=2)
#         >>> splits = ew.split(data)
#         >>> for train, test in splits:
#         ...     print(f"Train: {train}, Test: {test}")
#         """
#         if isinstance(data, (pd.DataFrame, pd.Series)):
#             data = data.to_numpy()

#         data_length = len(data)
#         data_index = np.arange(data_length)

#         output_train, output_test = [], []
#         progress = data_index[self.initial:]

#         # Initial train-test split
#         output_train.append(data_index[:self.initial].tolist())
#         output_test.append(data_index[self.initial:self.initial + self.horizon].tolist())

#         while len(progress) > 0:
#             self.counter = len(output_train)
#             # Expand training set
#             expanded_train = output_train[self.counter - 1] + progress[:self.period].tolist()
#             output_train.append(expanded_train)

#             # Create the test set
#             next_test_start = len(expanded_train)
#             next_test_end = next_test_start + self.horizon
#             output_test.append(data_index[next_test_start:next_test_end].tolist())

#             # Update progress
#             progress = data_index[next_test_end:]

#         # Exclude the last incomplete split
#         output_train = output_train[:-1]
#         output_test = output_test[:-1]

#         # Combine into tuples
#         splits = [(train, test) for train, test in zip(output_train, output_test)]
#         return splits
