
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
def load_model(checkpoint_path, device):
    """
    Load the LSTM model along with its initial hidden (h0) and cell (c0) states.
    """
    
    logger.info("🔍 Loading the model and hidden states...")

    model = LSTMRegressor(
    n_features=config['inference']["input_size"],
    hidden_size=config['inference']["hidden_size"],
    num_layers=config['inference']["num_layers"],
    dropout=config['inference']["dropout"], 
    output_size=config['inference']["output_size"],   # ✅ Add this
    learning_rate=config['inference']["learning_rate"],  # ✅ Add this
    criterion=config['inference']["criterion"],  # ✅ Add this
    batch_size=config['inference']["batch_size"]  # ✅ Add this
    )
    
    # model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True))
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if "model_state" in checkpoint:
        model.load_state_dict(checkpoint["model_state"])
    else:
        model.load_state_dict(checkpoint) 

    # Load hidden and cell states if available
    h0 = checkpoint.get("h0", None)
    c0 = checkpoint.get("c0", None)
    scaler = checkpoint.get("X_scaler", None)

    if h0 is not None and c0 is not None:
        h0, c0 = h0.to(device), c0.to(device)
        logger.info("✅ Hidden state (h0) and Cell state (c0) loaded from checkpoint.")
    else:
        logger.warning("⚠️ No saved hidden states found. Initializing to zeros.")
        h0 = torch.zeros(config['inference']["num_layers"], config['inference']["batch_size"], config['inference']["hidden_size"]).to(device)
        c0 = torch.zeros(config['inference']["num_layers"], config['inference']["batch_size"], config['inference']["hidden_size"]).to(device)
    

    model.to(device)
    model.eval()
    
    logger.info(f"✅ Model loaded from {checkpoint_path}")
    return model, scaler, h0, c0

def run_inference(model, data_loader, device, h0, c0):  
    model.to(device)
    final_prediction = []
    
    # ✅ Set the initial hidden and cell states
    hidden = (h0, c0)
    
    # number = 1
    with torch.no_grad():
        for x_batch, _ in data_loader:
            x_batch = x_batch.to(device)
            y_pred, _ = model(x_batch, hidden)

            final_prediction.append(y_pred.cpu().numpy())

    return np.concatenate(final_prediction).squeeze()



def main():
    """
    Main function to load the model, preprocess input data, and make predictions.
    """
    try:
        # Ensure the predictions directory exists

        logger.info("🔍 Finding the best model checkpoint...")
        checkpoint_path = get_best_model(config, logger)

        #  = config["experiment"]["checkpoint_path"]
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model, scaler,  h0, c0 = load_model(checkpoint_path, device)

        logger.info("📊 Loading and preprocessing test data...")
        test_data = test_data_path(config)
        data_columns = config['data']["data_columns"]
        seq_len = config['inference']["seq_len"]
        
        X_test, y_test = load_and_preprocess_data(test_data, data_columns, seq_len, logger)

        logger.info("Scale the data from the trained scaler...")

        X_test_scaled = scaler.transform(X_test)
        logger.info("Scaled...")
        
        y_test = np.array(y_test)
        # X_test, y_test = scale_data( X_test, y_test)

        
        batch_size = config['inference']["batch_size"]
        num_workers = config['inference']["num_workers"]
        output_size = config['inference']["output_size"]

        # ✅ Load test DataLoader
        test_loader  = test_dataloader(X_test_scaled, y_test, seq_len, output_size, batch_size,num_workers, logger )

        logger.info("🚀 Running inference on test data...")    
        predictions = run_inference(model, test_loader, device, h0, c0)       
  
        # Convert to DataFrame
        predictions_df = pd.DataFrame({ "y_pred": predictions.flatten()})
        
        # print(predictions)

        # Save predictions
        output_file = "predictions.csv"
        predictions_df.to_csv(output_file, index=False)
        logger.info(f"✅ Predictions saved to {output_file}")

    except Exception as e:
        logger.error(f"❌ An error occurred: {e}")

if __name__ == "__main__":
    main()
