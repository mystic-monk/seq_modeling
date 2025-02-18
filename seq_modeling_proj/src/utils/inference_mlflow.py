import mlflow
from mlflow.tracking import MlflowClient

from pathlib import Path

def get_best_model(config, logger):
    

    # Load configuration parameters
    # experiment_name = config["experiment"]["experiment_name"]
    # tracking_dir = config["experiment"]["mlflow_runs_dir"]
    checkpoint_path = config["experiment"]["checkpoint_path"]



    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    # abs_path = project_root / Path(tracking_dir).relative_to("/") if tracking_dir.startswith("/") else Path(tracking_dir)

    best_checkpoint_path = project_root / Path(checkpoint_path).relative_to("/") if checkpoint_path.startswith("/") else Path(checkpoint_path)

    return best_checkpoint_path    

    # # Get absolute path to store MLflow logs
    # tracking_uri = abs_path.as_uri()  # Using Path.cwd() to get the current working directory

    # # Set tracking URI (update if running an MLflow server)
    # mlflow.set_tracking_uri(tracking_uri)  # Local MLflow tracking
    # logger.info(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")
    
    # # logger.info(f"MLflow tracking URI fetched : {mlflow.get_tracking_uri()}")

    # # logger.info(f"experiment_name : {experiment_name}")
    # # print(mlflow.tracking.MlflowClient().list_experiments())
    # experiment = mlflow.get_experiment_by_name(experiment_name)
    # if experiment is None:
    #     raise ValueError(f"Experiment '{experiment_name}' not found in MLflow.")
    
    # experiment_id = experiment.experiment_id 

    # client = MlflowClient()

    # # Get best run sorted by validation loss
    # runs = client.search_runs(
    #     experiment_ids=[experiment_id],
    #     order_by=["metrics.val_loss ASC"],  # Sort by lowest validation loss
    #     max_results=1
    # )

    # if runs:
        
    #     best_run = runs[0]
    #     best_run_id = best_run.info.run_id

    #     # Path to best model in local MLflow directory
    #     model_path = f"mlruns/{experiment_id}/{best_run_id}/artifacts/model"
    #     logger.info(f"Best model path: {model_path}")

    #     # Load the model
    #     best_model = mlflow.pytorch.load_model(model_path)
    #     best_model.eval()
    #     logger.info("Best model loaded successfully!")

    # else:
    #     logger.info("No runs found for the given experiment.")

    # return best_model