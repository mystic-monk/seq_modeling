# data_pipeline/data_preprocessing.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
import numpy as np
# import joblib
def load_and_preprocess_data(data_path, data_columns, seq_len, logger):
    """
    Load and preprocess the data.
    """
    logger.info(f"Loading data from {data_path}...")
    data = pd.read_parquet(data_path, columns=data_columns)
    data['event_creation_date'] = pd.to_datetime(data['event_creation_date'])
    data = data.sort_values(by='event_creation_date')
    data.set_index('event_creation_date', inplace=True)

    X = data.copy()
    y = X["log_cases_14d_moving_avg"].shift(-seq_len).dropna()

    # Trim X to match the length of y
    X = X.iloc[:len(y)]
    return X, y

def scale_data(X, y):
    """Handle both DataFrame and array inputs"""
    if isinstance(X, pd.DataFrame):
        X = X.values
    if isinstance(y, pd.Series):
        y = y.values

    
    scaler = StandardScaler()
    scaler.fit(X)
    scaled_X = scaler.transform(X)
    scaled_y = np.array(y)
    
    return scaled_X, scaled_y, scaler


def split_data(X, y, window_size):
    """
    Split the dataset into train and validation.
    """
    total_size = len(X)
    num_of_bins = total_size // window_size
    divisible_size = num_of_bins * window_size

    X, y = X[len(X) - divisible_size:], y[len(y) - divisible_size:]

    train_bins = int(num_of_bins * 0.8)  # 80% for training
    train_size = train_bins * window_size
    val_bins = num_of_bins - train_bins  # Remaining 20% for validation
    val_size = val_bins * window_size 

    X_train, X_val = X[:train_size], X[train_size:train_size + val_size]
    y_train, y_val = y[:train_size], y[train_size:train_size + val_size]

    return X_train, X_val, y_train, y_val
