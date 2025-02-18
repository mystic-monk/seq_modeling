import os
import random
from pathlib import Path

import numpy as np
import torch
# from torch import nn
from lightning import seed_everything
from utils.config.config_loader import load_config

from utils.logging.logger import setup_logger
# Load configuration
config = load_config("baseline_config.json")


# Set random seed for reproducibility
seed = 303
seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# Resolve project paths
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent.parent.parent

# Ensure necessary directories exist (updated to handle exp_dirs)
exp_dirs = config.get("experiment_dirs", {})
for dir_key, relative_path in exp_dirs.items():
    abs_path = project_root / Path(relative_path).relative_to("/") #if relative_path.startswith("/") else Path(relative_path)
    print(abs_path)
    # abs_path = Path(relative_path) if relative_path.startswith("/") else (project_root / relative_path)
    os.makedirs(abs_path, exist_ok=True)
    if dir_key == "log_dir":
        log_dir = abs_path

logger = setup_logger(log_dir=log_dir, log_file=config['paths']['log_file'])
for dir_key, relative_path in exp_dirs.items():
    logger.info(f"[bold blue]📁 Created/Found: {dir_key} → {abs_path}[/bold blue]")

logger.info("[bold green]🎉 Configuration loaded successfully! Ready to go.[/bold green]")
