# src/data/loader.py

import pandas as pd
from pathlib import Path
from src.utils import config, get_logger

logger = get_logger(__name__)   

def load_raw_data(path: Path = None) -> pd.DataFrame:
    """ Load the raw data from the path specified in the config fila and return as a pandas DataFrame. """
    raw_path = path or config.paths.raw_data
    
    if not raw_path.exists():
        logger.error(f"Raw data file not found at {raw_path}")
        raise FileNotFoundError(f"Raw data file not found at {raw_path}")
    
    df = pd.read_csv(raw_path)
    logger.info(f"Dataset loaded successfully with shape {df.shape}")
    return df

def get_standardize_columns(df:pd.DataFrame)-> pd.DataFrame:
    """Lowercase column names and replace spaces with underscores."""
    df.columns = df.columns.str.lower().str.replace(" ", "_")
    logger.info("Column names standardized")
    return df

def loader_pipeline(path: Path = None)-> pd.DataFrame:
    
    # Step 1: Load dataset
    df = load_raw_data(path)

    # Step 2: Lowercase coulmns names
    df = get_standardize_columns(df)

    return df    