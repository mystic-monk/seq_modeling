import os
import torch
import numpy as np
import pandas as pd
from model import LSTMRegressor
from config import p
from config_predict import last_date, checkpoint_path, output_csv, prediction_days

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
        output_size=config["output_size"],
    )
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"), weights_only=True)["state_dict"])
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
    sequences = []
    for i in range(len(data) - seq_len + 1):
        sequences.append(data[i:i + seq_len])
    input_tensor = torch.tensor(sequences, dtype=torch.float32)
    return input_tensor

def predict(model, input_data: torch.Tensor, prediction_days: int):
    """
    Generate predictions using the trained model.
    Args:
        model (torch.nn.Module): Trained LSTM model.
        input_data (torch.Tensor): Input data tensor of shape (batch_size, seq_len, n_features).
        prediction_days (int): Number of days to predict.
    Returns:
        torch.Tensor: Model predictions.
    """
    predictions = []
    with torch.no_grad():
        for _ in range(prediction_days):
            pred = model(input_data)
            predictions.append(pred)
            # Update input_data with the new prediction
            input_data = torch.cat((input_data[:, 1:, :], pred.unsqueeze(1)), dim=1)
    return torch.cat(predictions, dim=0)

def main():
    """
    Main function to load the model, preprocess input data, and make predictions.
    """
    # Ensure the predictions directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    print("Loading model...")
    model = load_model(checkpoint_path, p)

    print("Generating predictions...")
    # Create an initial input tensor with zeros (or any other initial value)
    initial_input = torch.zeros((1, p["seq_len"], p["n_features"]))

    predictions = predict(model, initial_input, prediction_days)

    # Generate a sequence of dates for the predictions
    last_date_dt = pd.to_datetime(last_date)
    prediction_dates = pd.date_range(start=last_date_dt, periods=prediction_days)

    print("Saving predictions...")
    # Save predictions to a CSV file
    predictions_df = pd.DataFrame({
        "date": prediction_dates,
        "prediction": predictions.numpy().flatten()
    })
    predictions_df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")

if __name__ == "__main__":
    main()
