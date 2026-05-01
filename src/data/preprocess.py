# src/data/preprocess.py

import pandas as pd
from pathlib import Path

from src.utils import config, get_logger

logger = get_logger(__name__)


def get_missing_values(df: pd.DataFrame) -> pd.Series:
    """ Return a DataFrame with the count and percentage of missing values for each column in the input DataFrame. """
    missing_count = df.isnull().sum()
    missing_percentage = (missing_count / len(df)) * 100
    total_missing = missing_count.sum()

    if total_missing == 0:
        logger.info("No missing values found.")
    else:
        logger.info(f"Total missing values: {total_missing}")
        missing_summary = missing_percentage[missing_percentage > 0]
        for col, pct in missing_summary.items():
            logger.info(f"  {col}: {pct:.2f}%")

    return missing_percentage

def get_duplicate_rows(df: pd.DataFrame) -> int:
    duplicate_count = df.duplicated().sum()
    logger.info(f"Found {duplicate_count} duplicate rows.")
    return duplicate_count

def encode_columns(df: pd.DataFrame) -> pd.DataFrame:
    columns_to_encode = ["gender", "partner", "dependents", "phoneservice", "paperlessbilling"]

    for column in columns_to_encode:
        if column not in df.columns:
            raise ValueError(f"Column '{column}' not found in DataFrame")
        
        if column == "gender":
            df[column] = df[column].map({"Male": 1, "Female": 0})
        else:
            df[column] = df[column].map({"Yes": 1, "No": 0})

    logger.info("Columns encoded successfully.")
    return df

def preprocess_pipeline(df: pd.DataFrame= None) -> pd.DataFrame:
    """ Perform basic data cleaning steps such as handling missing values and duplicates. """
    
    #Step 1: Drop ID column if exists
    if "customerid" in df.columns:
        df = df.drop(columns=["customerid"])
    
    # Step 2: Handle total charges
    df["totalcharges"] = pd.to_numeric(df["totalcharges"], errors="coerce").fillna(0)

    # Step 3: Missing count
    get_missing_values(df)
    
    # Step 3: Handle duplicates   
    get_duplicate_rows(df)
    df = df.drop_duplicates()

    # Step 4: encode columns
    df = encode_columns(df)

    logger.info("Preprocessing pipeline completed successfully.")
    return df

    
