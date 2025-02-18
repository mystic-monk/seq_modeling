# ~/seq_modeling_proj/utils/config_loader.py
import json
from pathlib import Path

def load_config(json_name):
    """Load a JSON config file from the ~/configs directory and return it as a dictionary."""
    
    config_path = Path(__file__).resolve().parent.parent.parent / "configs" / json_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)


