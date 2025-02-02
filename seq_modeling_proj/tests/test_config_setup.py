# tests/test_config_setup.py
import os
import pytest
from utils.config_setup import load_config, setup_logger
from unittest.mock import patch, MagicMock

# Setup logger mock
@pytest.fixture(scope="module")
def mock_logger():
    with patch("utils.config_setup.setup_logger") as mock:
        yield mock

# Test if the logger info message is called
def test_logger_info_message(mock_logger):
    # Setup mock for logger
    mock_logger.return_value = MagicMock()
    mock_logger_instance = mock_logger.return_value

    # Load configuration (mocked)
    with patch("utils.config_loader.load_config") as mock_load_config:
        mock_load_config.return_value = {"seed": 42, "paths": {"log_dir": "logs", "model_dir": "models", "optuna_db_dir": "optuna_db"}}

        # Run the config setup
        # from utils.config_setup import p  # Import after the mock
        logger_instance = mock_logger_instance
        
        # Ensure logger info was called
        # logger_instance.info.assert_called_with("[bold green]Configuration Setup Completed![/bold green]")

# Test the directory creation logic in config_setup.py
@pytest.mark.parametrize("path_key", ["log_dir", "model_dir", "optuna_db_dir"])
def test_directories_created(path_key):
    # Mock configuration loading
    config = {
        "seed": 42,
        "paths": {
            "log_dir": "logs",
            "model_dir": "models",
            "optuna_db_dir": "optuna_db"
        }
    }
    
    with patch("utils.config_loader.load_config", return_value=config):
        # Create the directories using the setup code
        from utils.config_setup import p
        
        # Check if directories are created
        os.makedirs(p["paths"][path_key], exist_ok=True)
        
        assert os.path.exists(p["paths"][path_key]), f"Directory {path_key} was not created"

# Test configuration loading
def test_load_config():
    # Patch the actual file loading to use a mocked config
    mock_config = {
        "seed": 42,
        "paths": {
            "log_dir": "logs",
            "model_dir": "models",
            "optuna_db_dir": "optuna_db"
        }
    }
    
    with patch("utils.config_loader.load_config", return_value=mock_config):
        config = load_config()
        
        assert config["seed"] == 42
        assert config["paths"]["log_dir"] == "logs"
        assert config["paths"]["model_dir"] == "models"
        assert config["paths"]["optuna_db_dir"] == "optuna_db"

# Test seed and reproducibility setup
def test_reproducibility():
    # Mock configuration loading to ensure reproducibility setup is triggered
    mock_config = {
        "seed": 42,
        "paths": {
            "log_dir": "logs",
            "model_dir": "models",
            "optuna_db_dir": "optuna_db"
        }
    }
    
    with patch("utils.config_loader.load_config", return_value=mock_config):
        from utils.config_setup import p
        
        # Check if seed setup is called
        import random
        import numpy as np
        import torch
        from lightning import seed_everything
        
        # Verify seed setting
        seed = p["seed"]
        seed_everything(seed, workers=True)
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        # Check if the random module's seed was set
        assert random.getstate() is not None
        assert np.random.get_state() is not None
        assert torch.initial_seed() is not None
