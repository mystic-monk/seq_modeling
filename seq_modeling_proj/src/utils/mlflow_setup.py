import mlflow
import os
from utils.config_loader import load_config
from pathlib import Path
def setup_mlflow(experiment_name="default_experiment", tracking_dir="logs/mlruns"):
    """
    Sets up MLflow to track experiments in a specified directory.
    
    Args:
        experiment_name (str): Name of the MLflow experiment.
        tracking_dir (str): Directory where MLflow runs should be stored.
    
    Returns:
        None
    """

    p = load_config()
    experiment_name = p["experiment"]["experiment_name"]
    tracking_dir = p["experiment"]["tracking_dir"]

    # Get absolute path to store MLflow logs
    tracking_path = Path.cwd() / tracking_dir  # Using Path.cwd() to get the current working directory
    # tracking_path.mkdir(parents=True, exist_ok=True) 

    # Convert tracking path to a file URI format
    tracking_uri = f"file:///{tracking_path.as_posix()}"  

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
 
    # Set or create an experiment
    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        print(f"Warning: Failed to set experiment {experiment_name}. Error: {e}")

    print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    print(f"MLflow experiment set to: {experiment_name}")

