import os
import torch
import numpy as np
import pandas as pd
import logging
from model import LSTMRegressor
from config import p
from config_predict import last_date, checkpoint_path, output_csv, prediction_days, input_path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def load_model(checkpoint_path: str, config: dict):
    """
    Load the trained model from a checkpoint file.
    Args:
        checkpoint_path (str): Path to the saved model checkpoint.
        config (dict): Model configuration parameters.
    Returns:
        torch.nn.Module: Loaded LSTM model.
    """
    model = LSTMRegressor(
        n_features=config["n_features"],
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
        learning_rate=config["learning_rate"],
        criterion=config["criterion"],
        batch_size=p["batch_size"],
        output_size=config["output_size"],
    )
    checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model

def preprocess_input_data(raw_data: pd.DataFrame, seq_len: int) -> torch.Tensor:
    """
    Preprocess the input data for prediction.
    Args:
        raw_data (pd.DataFrame): Raw input data.
        seq_len (int): Sequence length.
    Returns:
        torch.Tensor: Preprocessed input data tensor.
    """
    data = raw_data.values
    sequences = [data[i:i + seq_len] for i in range(len(data) - seq_len + 1)]
    input_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    return input_tensor

def predict(model, input_data: torch.Tensor):
    """
    Generate predictions using the trained model.
    Args:
        model (torch.nn.Module): Trained LSTM model.
        input_data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, n_features).
    Returns:
        torch.Tensor: Model predictions.
    """
    with torch.no_grad():
        pred, _ = model(input_data)
    return pred

def main():
    """
    Main function to load the model, preprocess input data, and make predictions.
    """
    try:
        # Ensure the predictions directory exists
        os.makedirs(os.path.dirname(output_csv), exist_ok=True)

        logger.info("Loading model...")
        model = load_model(checkpoint_path, p)

        logger.info("Loading and preprocessing input data...")
        raw_data = pd.read_parquet(input_path, columns=["log_cases_14d_moving_avg", "cases_14d_moving_avg", "diff_log_14d"])
        raw_data = raw_data.set_index(pd.to_datetime(raw_data.index))
        raw_data = raw_data.sort_index()

        # Extract the last seq_len rows to create the initial input
        initial_input_data = raw_data.iloc[-p["seq_len"]:]
        initial_input = torch.tensor(np.array(initial_input_data), dtype=torch.float32).unsqueeze(0)

        logger.info("Generating predictions...")
        predictions = predict(model, initial_input)

        # Generate a sequence of dates for the predictions
        last_date_dt = initial_input_data.index[-1]
        prediction_dates = pd.date_range(start=last_date_dt, periods=p["output_size"])

        logger.info("Saving predictions...")
        # Save predictions to a CSV file
        predictions_df = pd.DataFrame({
            "date": prediction_dates,
            "prediction": predictions.numpy().flatten()
        })
        predictions_df.to_csv(output_csv, index=False)

        logger.info(f"Predictions saved to {output_csv}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
