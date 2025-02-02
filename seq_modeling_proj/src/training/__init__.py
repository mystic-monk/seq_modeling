import os
import sys
from pathlib import Path
from utils.logging_setup import get_logger

logger = get_logger()

# Set the working directory to the project root
current_dir = Path(__file__).resolve().parent
project_root = current_dir.parent 

os.chdir(project_root)  
sys.path.append(str(project_root))  

logger.info(f"Current Directory: {current_dir}")

logger.info(f"Project Root     : {project_root}")

