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

# ✅ Ensure necessary directories exist
for path_key in ["log_dir", "optuna_db_dir"]:
    if path_key in p["paths"]:
        os.makedirs(p["paths"][path_key], exist_ok=True)


logger.info("[bold green]Configuration Setup Completed![/bold green]")
