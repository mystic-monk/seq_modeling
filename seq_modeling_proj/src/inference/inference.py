
import torch
# import torch.nn as nn
import numpy as np
import pandas as pd
import logging
from utils.config.config_loader import load_config
import joblib
from utils.inference_mlflow import get_best_model
from utils.data.data_utils import test_data_path
from data_pipeline.preprocessing.data_preprocessing import load_and_preprocess_data
from data_pipeline.dataloader.data_loaders import test_dataloader
from pathlib import Path
from utils.logging.logger import setup_logger
from models.lstm import LSTMRegressor


# Load inference-specific configuration
config = load_config("inference_config.json")


log_dir = config['logging']['log_dir']
log_file = config['logging']['log_file']

current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent
abs_path = project_root / Path(log_dir).relative_to("/") if log_dir.startswith("/") else Path(log_dir)

# Set up logging
phase="Inference"
# logger = setup_logger(phase, log_dir, log_file)
logger = setup_logger(abs_path, log_file, logging.INFO, name="inference_logger", phase=phase)

# Load model function
# Load model function
def load_model(checkpoint_path, device):
    """
    Load the LSTM model and hidden states (h0, c0) from checkpoint.
    """
    logger.info("🔍 Loading the model and hidden states...")

    model = LSTMRegressor(
        n_features=config['inference']["input_size"],
        hidden_size=config['inference']["hidden_size"],
        num_layers=config['inference']["num_layers"],
        dropout=config['inference']["dropout"], 
        output_size=config['inference']["output_size"],
        learning_rate=config['inference']["learning_rate"],
        criterion=config['inference']["criterion"],
        batch_size=config['inference']["batch_size"]
    )

    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint)

    # Load hidden and cell states from the checkpoint
    h0 = checkpoint.get("h0", None)
    c0 = checkpoint.get("c0", None)
    # scaler = checkpoint.get("X_scaler", None)

    if h0 is not None and c0 is not None:
        h0, c0 = h0.to(device), c0.to(device)
        model.h0 = h0
        model.c0 = c0
        logger.info("✅ Hidden state (h0) and Cell state (c0) loaded from checkpoint.")
    else:
        logger.warning("⚠️ No saved hidden states found. Initializing to zeros.")
        h0 = torch.zeros(config['inference']["num_layers"], config['inference']["batch_size"], config['inference']["hidden_size"]).to(device)
        c0 = torch.zeros(config['inference']["num_layers"], config['inference']["batch_size"], config['inference']["hidden_size"]).to(device)



    model.to(device)
    model.eval()
    logger.info(f"✅ Model loaded from {checkpoint_path}")
    
    return model, h0, c0

def run_inference(model, device, h0, c0):  
    """
    Run inference to predict the next 14 days of disease counts using the model.
    """
    model.to(device)

    # Prepare initial input (if any) as a dummy value, as we're predicting the next 14 days
    # Here, we don't need actual test data, we only need the model and hidden states
    # You can modify this part if you need to input a specific sequence for predictions.
    initial_input = torch.zeros(1, 1, config['inference']['input_size']).to(device)  # Single timestep input

    # Set the initial hidden and cell states
    # hidden = (h0, c0)
    predictions = []
    # Predict the next 14 days at once
    with torch.no_grad():
        
        input_seq = initial_input  # Starting with initial dummy input
        
        # We predict for 14 steps in a loop (one step per day)
        # for _ in range(14):  # Predicting the next 14 days
            # Forward pass to predict the next time step
        output, (h0, c0) = model(input_seq, h0, c0)
        predictions.append(output.squeeze().cpu().numpy())

            # Prepare the next input as the previous model's prediction
        input_seq = output.unsqueeze(1)  # Use the output for next step prediction

        predictions = np.array(predictions).flatten()

    return predictions



def main():
    """
    Main function to load the model, preprocess input data, and make predictions.
    """
    try:
        # Ensure the predictions directory exists

        logger.info("🔍 Finding the best model checkpoint...")
        checkpoint_path = get_best_model(config, logger)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model,  h0, c0 = load_model(checkpoint_path, device)

        logger.info("📊 Loading and preprocessing test data...")
        predictions = run_inference(model, device, h0, c0)
 
        # Convert predictions to DataFrame for easy saving
        predictions_df = pd.DataFrame({"y_pred": predictions})

        # Save predictions
        output_file = "predictions.csv"
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"✅ Predictions saved to {output_file}")
 

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
