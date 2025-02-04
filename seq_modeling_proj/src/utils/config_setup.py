# ~/seq_modeling_proj/utils/config_setup.py
import os
import torch
import numpy as np
import random
from torch import nn
from lightning import seed_everything   
from utils.logging_setup import get_logger
# from utils.config_setup import p
from utils.config_loader import load_config

p = load_config()
# Get the logger from the centralized setup
logger = get_logger()

# ✅ Ensure reproducibility
seed = p["seed"]
seed_everything(seed, workers=True)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ✅ Convert objects that cannot be stored in JSON
if isinstance(p["criterion"], str):  
    p["criterion"] = getattr(nn, p["criterion"])()

# ✅ Ensure necessary directories exist (updated to handle exp_dirs)
exp_dirs = p.get("experiment_dirs", {})
for dir_key in exp_dirs:
    dir_path = exp_dirs.get(dir_key)

    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Directory created or already exists: {dir_path}")

logger.info("[bold green]Configuration Setup Completed![/bold green]")
