# ~/seq_modeling_proj/utils/config_inference.py

import os
import torch
import numpy as np
# import random
# from lightning import seed_everything   
from utils.logging_setup import get_logger
from utils.config_loader import load_config

# Load inference-specific configuration
config = load_config("../configs/inference_config.json")
logger = get_logger()

# ✅ Ensure reproducibility

# seed_everything(seed, workers=True)
# torch.manual_seed(seed)
# np.random.seed(seed)
# random.seed(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


# ✅ Ensure necessary directories exist for inference output
inference_dirs = config.get("inference_dirs", {})  # Separate from training dirs
for dir_key, dir_path in inference_dirs.items():
    if dir_path:
        os.makedirs(dir_path, exist_ok=True)
        logger.info(f"Inference directory created or already exists: {dir_path}")

logger.info("[bold green]Inference Configuration Setup Completed![/bold green]")
