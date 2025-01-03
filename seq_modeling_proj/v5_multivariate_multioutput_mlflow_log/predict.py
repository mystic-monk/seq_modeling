import os
import torch
import numpy as np
import pandas as pd
import logging
from model import LSTMRegressor
from config import p
from config_predict import checkpoint_path, output_file, input_path

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
    return torch.tensor(np.array(sequences), dtype=torch.float32)


def predict(model, input_data: torch.Tensor):
    """
    Generate predictions using the trained model.
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
        os.makedirs(os.path.dirname(output_file), exist_ok=True)

        logger.info("Loading model...")
        model = load_model(checkpoint_path, p)

        logger.info("Loading and preprocessing input data...")
        raw_data = pd.read_parquet(input_path, columns=["event_creation_date", "log_cases_14d_moving_avg", "cases_14d_moving_avg", "diff_log_14d"])
        
        # Set the index of the DataFrame to the event_creation_date column
        raw_data['event_creation_date'] = pd.to_datetime(raw_data['event_creation_date'])
        raw_data.set_index('event_creation_date', inplace=True)
        raw_data = raw_data.sort_index()

        # Extract the last seq_len rows to create the initial input
        if len(raw_data) < p["seq_len"]:
            raise ValueError("Not enough data for the specified sequence length.")
        initial_input_data = raw_data.iloc[-p["seq_len"]:]
        initial_input = torch.tensor(np.array(initial_input_data), dtype=torch.float32).unsqueeze(0)

        logger.info("Generating predictions...")   
        predictions = predict(model, initial_input)

        # Generate a sequence of dates for the predictions
        last_date_dt = initial_input_data.index[-7]
        prediction_dates = pd.date_range(start=last_date_dt + pd.Timedelta(days=1), periods=p["output_size"])

        logger.info("Saving predictions...")
        # Create a DataFrame for predictions
        predictions_df = pd.DataFrame({
            "event_creation_date": prediction_dates,
            "log_cases_14d_moving_avg": np.nan,  # No raw data for future dates
            "cases_14d_moving_avg": np.nan,      # No raw data for future dates
            "diff_log_14d_prediction": predictions.numpy().flatten()
        }).set_index("event_creation_date")

        # Combine raw data and predictions
        combined_data = raw_data.combine_first(predictions_df).sort_index()

        # Save combined data
        combined_data.to_parquet(output_file)
        
        logger.info(f"Predictions saved to {output_file}")
    except Exception as e:
        logger.error(f"An error occurred: {e}")

if __name__ == "__main__":
    main()
