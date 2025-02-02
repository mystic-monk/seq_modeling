from scripts.timeseriesdataset import TimeseriesDataset
import lightning as L
from config_predict import input_path
from config import p
import pandas as pd
from config import logger, console
from sklearn.preprocessing import StandardScaler
import numpy as np
from torch.utils.data import DataLoader
import os


class inferenceLightiningDataModule(L.LightningDataModule):
    
    """
    PyTorch Lightning DataModule subclass for handling data loading and processing.
    """

    def __init__(self, seq_len=14, output_size=1, batch_size=14, num_workers=0):
        super().__init__()
        self.seq_len = seq_len
        self.output_size = output_size
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.X_test = None
        self.y_test = None
        self.preprocessing = None
        self.input_data_path = input_path
        self.data_columns = p["data_columns"]
     
        self.debug = p.get("debug", False)

  
    def load_and_preprocess_data(self, data_path):
        """
        Load and preprocess the data.
        """
        logger.info(f"Loading data from {data_path}...")
        data = pd.read_parquet(
            data_path, 
            columns=self.data_columns
            )
        data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
        data = data.sort_values(by='event_creation_date')
        data.set_index('event_creation_date', inplace=True)
        
        X = data.copy()
        y = X["log_cases_14d_moving_avg"].shift(-self.seq_len).dropna()

        # Trim X to match the length of y
        X = X.iloc[:len(y)]

        logger.info(f"Data loaded successfully. X shape: {X.shape}, y shape: {y.shape}")
        logger.info(f"X type : {type(X)}")
        return X, y
    

    def prepare_data(self):
        """
        Prepares the data by loading and preprocessing it.
        """
        if self.X_test is not None:
            return
 
        logger.info("Preparing data...")

        # Example: Check if data files exist, and if not, download or extract them
        if not os.path.exists(self.input_data_path):
            raise FileNotFoundError(f"Data not found at {self.input_data_path}")
        

        self.X_test, self.y_test = self.load_and_preprocess_data(self.input_data_path)
        
        # if self.debug:
        #     console.print("[bold green]Data loaded and preprocessed successfully![/bold green]")
        #     console.print(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")
        logger.info(f"X_test shape: {self.X_test.shape}, y_test shape: {self.y_test.shape}")


        return

    def setup(self, stage=None):
        """
        Setup the data for testing.
        This called from the trainer, during testing.
        """


        # if stage == "test" and self.X_test is not None:
        #     logger.info("Data already setup for testing.")
        #     return
        # if stage is None and self.X_test is not None:
        #     return
                
        
        logger.info("Setup is being called.")
        

        # if self.debug:
        #     console.print(f"[bold cyan]Testing data shape: {self.X_test.shape}[/bold cyan]")


        preprocessing = StandardScaler()
        preprocessing.fit(self.X_test)


        self.X_test = preprocessing.transform(self.X_test)
        self.y_test = np.array(self.y_test)
        

        logger.info("[bold green]Data preprocessed and setup complete![/bold green]")



    def test_dataloader(self):
        """
        Returns the test DataLoader.
        """

        logger.info("Creating test DataLoader...")
        test_dataset = TimeseriesDataset(
            self.X_test, 
            self.y_test, 
            seq_len=self.seq_len, 
            output_size=self.output_size
            )

        test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            drop_last=True,
        )

        logger.info(f"[bold red]Test DataLoader created with {len(test_loader)} batches[/bold red]")

        # Collect and print the DataLoader info
        self.collect_dataloader_info(test_loader, "Test")

        return test_loader


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