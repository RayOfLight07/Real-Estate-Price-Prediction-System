"""
Data Loader Module for Indian Real Estate Price Prediction System.

This module handles loading and sampling dataset records from CSV storage,
providing clean dataframes for preprocessing and downstream modeling pipelines.
"""

import os
import logging
import pandas as pd
from typing import Optional

# Configure logging format
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("DataLoader")


def load_data(
    data_path: str = "data/india_housing_prices.csv",
    sample_size: Optional[int] = 100000,
    random_state: int = 42
) -> pd.DataFrame:
    """
    Loads housing dataset from specified CSV file and samples up to sample_size rows.

    Parameters:
    -----------
    data_path : str
        Path to the CSV dataset file.
    sample_size : Optional[int], default=100000
        Maximum number of rows to sample for compute optimization.
        If None or larger than dataset size, all rows are returned.
    random_state : int, default=42
        Seed for reproducible random sampling.

    Returns:
    --------
    pd.DataFrame
        Loaded and optionally sampled DataFrame.
    """
    if not os.path.exists(data_path):
        # Check relative path from root directory if needed
        alt_path = os.path.join(os.path.dirname(__file__), "..", data_path)
        if os.path.exists(alt_path):
            data_path = os.path.abspath(alt_path)
        else:
            logger.error(f"Dataset file not found at path: {data_path}")
            raise FileNotFoundError(f"Dataset file not found at path: {data_path}")

    logger.info(f"Loading dataset from: {data_path}")
    df = pd.read_csv(data_path)
    logger.info(f"Full dataset loaded. Total rows: {len(df):,}, Total columns: {len(df.columns)}")

    if sample_size and sample_size > 0 and len(df) > sample_size:
        logger.info(f"Sampling {sample_size:,} rows with random_state={random_state}...")
        df = df.sample(n=sample_size, random_state=random_state).reset_index(drop=True)
        logger.info(f"Dataset sampled successfully. New shape: {df.shape}")
    else:
        logger.info(f"Using complete dataset without sampling. Shape: {df.shape}")

    mem_usage_mb = df.memory_usage(deep=True).sum() / (1024 * 1024)
    logger.info(f"Memory footprint: {mem_usage_mb:.2f} MB")

    return df


if __name__ == "__main__":
    logger.info("Executing data_loader self-test...")
    df_sample = load_data(sample_size=100000)
    print("\n--- Data Sample Head ---")
    print(df_sample.head(3))
