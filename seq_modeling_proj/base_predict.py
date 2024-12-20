import torch
import numpy as np
from base_model import LSTMRegressor

# from data_set import EpiCountsDataModule
from config import p

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
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device("cpu"))["state_dict"])
    model.eval()  # Set the model to evaluation mode
    return model

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
        predictions = model(input_data)
    return predictions
  
def main(input_csv: str, checkpoint_path: str, output_csv: str):
    """
    Main function to load the model, preprocess input data, and make predictions.
    Args:
        input_csv (str): Path to the input CSV file with raw data.
        checkpoint_path (str): Path to the saved model checkpoint.
        output_csv (str): Path to save the predictions.
    """
    print("Loading model...")
    model = load_model(checkpoint_path, p)

    print("Preprocessing input data...")
    # Preprocess the input data
    raw_data = pd.read_csv(input_csv)
    input_tensor = preprocess_input_data(raw_data, seq_len=p["seq_len"], n_features=p["n_features"])

    print("Generating predictions...")
    predictions = predict(model, input_tensor)

    print("Saving predictions...")
    # Save predictions to a CSV file
    predictions_df = pd.DataFrame(predictions.numpy(), columns=["prediction"])
    predictions_df.to_csv(output_csv, index=False)

    print(f"Predictions saved to {output_csv}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="LSTM Time Series Prediction Script")
    parser.add_argument("--input_csv", type=str, required=True, help="Path to the input CSV file.")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to the model checkpoint file.")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to save the predictions CSV.")

    args = parser.parse_args()

    main(input_csv=args.input_csv, checkpoint_path=args.checkpoint, output_csv=args.output_csv)