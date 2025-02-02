# ~/seq_modeling_proj/utils/config_loader.py
import json
from pathlib import Path

def load_config(config_name="baseline_config.json"):
    """Load a JSON config file from the ~/configs directory and return it as a dictionary."""
    

    # home_dir = Path.home()  # This is the user's home directory
    # config_path = home_dir / "configs" / config_name 
    config_path = Path(__file__).resolve().parent.parent / "configs" / config_name

    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        return json.load(f)

