# data_pipeline/data_loaders.py
from torch.utils.data import DataLoader
from data_pipeline.dataloader.timeseries_dataset import TimeseriesDataset
# from timeseries_dataset import TimeseriesDataset
from utils.logger_utils import collect_dataloader_info

def train_dataloader(X_train, y_train, seq_len, output_size, batch_size, num_workers):
    """
    Returns the train DataLoader.
    """
    train_dataset = TimeseriesDataset(X_train, y_train, seq_len=seq_len, output_size=output_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    collect_dataloader_info(train_loader, "Train")
    return train_loader

def val_dataloader(X_val, y_val, seq_len, output_size, batch_size, num_workers):
    """
    Returns the validation DataLoader.
    """
    val_dataset = TimeseriesDataset(X_val, y_val, seq_len=seq_len, output_size=output_size)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    collect_dataloader_info(val_loader, "Validation")
    return val_loader

def test_dataloader(X_test, y_test, seq_len, output_size, batch_size, num_workers):
    """
    Returns the test DataLoader.
    """
    test_dataset = TimeseriesDataset(X_test, y_test, seq_len=seq_len, output_size=output_size)
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=True,
    )
    collect_dataloader_info(test_loader, "Test")
    return test_loader
