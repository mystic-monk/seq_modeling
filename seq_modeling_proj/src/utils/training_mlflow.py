import mlflow
from pathlib import Path

def setup_mlflow(experiment_name, config, logger):
    """
    Sets up MLflow to track experiments in a specified directory.
    
    Returns:
        None
    """
    logger.info("[bold blue]🚀 Initializing MLflow Setup...[/bold blue]")

    # Load configuration parameters
    # experiment_name = config["experiment"]["experiment_name"]
    tracking_dir = config["experiment"]["tracking_dir"]

    current_dir = Path(__file__).resolve().parent
    project_root = current_dir.parent.parent
    abs_path = project_root / Path(tracking_dir).relative_to("/") if tracking_dir.startswith("/") else Path(tracking_dir)

    tracking_uri = abs_path.as_uri()

    # Set the MLflow tracking URI
    mlflow.set_tracking_uri(tracking_uri)
    logger.info(f"[bold cyan]🔗 MLflow Tracking URI set to: [white]{tracking_uri}[/white][/bold cyan]")
 
    # Check if the experiment already exists
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        try:
            # ✅ Create experiment if it doesn’t exist
            experiment_id = mlflow.create_experiment(name=experiment_name, artifact_location=tracking_uri)
            logger.info(f"[bold green]✨ Created new MLflow experiment: [white]{experiment_name}[/white] (ID: {experiment_id})[/bold green]")
        except Exception as e:
            logger.error(f"[bold red]❌ Failed to create experiment '{experiment_name}'. Error: {e}[/bold red]")
            raise e
    else:
        experiment_id = experiment.experiment_id
        logger.info(f"[bold yellow]⚠️ Using existing MLflow experiment: [white]{experiment_name}[/white] (ID: {experiment.experiment_id})[/bold yellow]")

    # Set or create an experiment
    try:
        mlflow.set_experiment(experiment_name)
        logger.info(f"[bold green]🟢 Experiment set to: {experiment_name}[/bold green]")
    except Exception as e:
        logger.error(f"[bold red]❌ Failed to set experiment '{experiment_name}'. Error: {e}[/bold red]")
        raise e
    
    logger.info("[bold blue]✅ MLflow Setup Complete![/bold blue]\n" + "-" * 60)

    return experiment_name, experiment_id, abs_path

def log_training_results(split_idx, experiment_name, history, model, config, best_val_loss, logger):
    """
    Logs training results to MLflow.
    """

    if mlflow.active_run():
        mlflow.end_run()

    experiment_name = f"{config['experiment']['experiment_name']}_split_{split_idx+1}"
    experiment_name, experiment_id, tracking_uri = setup_mlflow(experiment_name, config, logger)

    with mlflow.start_run(experiment_name=experiment_name):
        

        logger.info(f"Logging to MLflow Experiment ID: [bold green]{experiment_id}[bold green]")

        
        mlflow.log_params({
            "learning_rate": config["learning_rate"],
            "batch_size": config["batch_size"],
            "num_layers": config["num_layers"],
            "hidden_size": config["hidden_size"],
            "dropout": config["dropout"],
            "criterion": config["criterion"],
            "sequence_length": config["seq_len"],
        })

        # Log training metrics
        for epoch, (train_loss, val_loss, r2) in enumerate(
            zip(history["train_loss"], history["val_loss"], history["r2_score"])
        ):
            mlflow.log_metric("train_loss", train_loss, step=epoch)
            mlflow.log_metric("val_loss", val_loss, step=epoch)
            mlflow.log_metric("val_r2", r2, step=epoch)

        # Log the best validation loss
        mlflow.log_metric("best_val_loss", best_val_loss)

        # Log best model
        # mlflow.pytorch.log_model(model, f"best_model_split_{split_idx}")

        mlflow.pytorch.log_model(model, artifact_path=f"models/best_model_split_{split_idx}", registered_model_name="LSTMRegressor")

    # Log best model
    logger.info(f"Logged training run to experiment {experiment_name} (ID: {experiment_id})")



