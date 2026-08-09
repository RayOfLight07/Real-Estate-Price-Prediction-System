"""
Model Training Module for Indian Real Estate Price Prediction System.

This module builds a full end-to-end Scikit-Learn Pipeline combining leakage cleaning,
feature engineering, preprocessor column transformer, and an ensemble regressor (RandomForestRegressor or HistGradientBoostingRegressor).
The trained model pipeline is saved to models/model.pkl.
"""

import os
import logging
import joblib
import pandas as pd
from typing import Tuple, Optional
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor

from src.data_loader import load_data
from src.preprocessing import prepare_data, DataLeakageCleaner, FeatureEngineer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("ModelTraining")


def train_model(
    sample_size: Optional[int] = None,
    model_type: str = "random_forest",
    save_path: str = "models/model.pkl",
    random_state: int = 42
) -> Tuple[Pipeline, pd.DataFrame, pd.Series]:

    """
    Loads complete dataset (250,000 rows), prepares preprocessor, trains an ensemble model
    within a single pipeline, evaluates on test split, and serializes pipeline to models/model.pkl.

    Parameters:
    -----------
    sample_size : Optional[int], default=None
        Number of dataset rows to sample. If None, trains on the full 250,000 dataset.
    model_type : str, default='hist_gradient_boosting'
        Algorithm choice: 'hist_gradient_boosting' or 'random_forest'.
    save_path : str, default='models/model.pkl'
        Output path for the serialized pipeline.
    random_state : int, default=42
        Seed for reproducibility.

    Returns:
    --------
    Tuple[Pipeline, X_test, y_test]
        Trained pipeline and test datasets for immediate evaluation.
    """
    logger.info(f"Starting full dataset model training pipeline with sample_size={sample_size}, model_type='{model_type}'")

    # 1. Load data
    df_raw = load_data(sample_size=sample_size, random_state=random_state)

    # 2. Extract feature sets and split
    X_train, X_test, y_train, y_test, preprocessor, cat_cols, num_cols = prepare_data(
        df_raw, test_size=0.2, random_state=random_state
    )

    # 3. Instantiate model regressor
    if model_type == "hist_gradient_boosting":
        logger.info("Initializing HistGradientBoostingRegressor (max_iter=200, learning_rate=0.08)...")
        regressor = HistGradientBoostingRegressor(
            max_iter=200,
            learning_rate=0.08,
            max_depth=15,
            random_state=random_state
        )
    elif model_type == "random_forest":
        logger.info("Initializing RandomForestRegressor (n_estimators=100, max_depth=20, n_jobs=-1)...")
        regressor = RandomForestRegressor(
            n_estimators=100,
            max_depth=20,
            random_state=random_state,
            n_jobs=-1
        )
    else:
        raise ValueError(f"Unsupported model_type '{model_type}'. Choose 'hist_gradient_boosting' or 'random_forest'.")

    # 4. Construct complete end-to-end Pipeline
    full_pipeline = Pipeline([
        ('leakage_cleaner', DataLeakageCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('preprocessor', preprocessor),
        ('regressor', regressor)
    ])

    raw_train_indices = X_train.index
    raw_test_indices = X_test.index
    df_clean_raw = df_raw.drop(columns=['Price_in_Lakhs'])

    X_train_raw = df_clean_raw.loc[raw_train_indices]
    X_test_raw = df_clean_raw.loc[raw_test_indices]

    logger.info("Fitting master pipeline on raw training data (200,000 training samples)...")
    full_pipeline.fit(X_train_raw, y_train)
    logger.info("Full dataset model training completed successfully!")

    # 5. Save model pipeline to disk
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    logger.info(f"Saving trained pipeline model to: {save_path}")
    joblib.dump(full_pipeline, save_path)
    logger.info("Pipeline serialized and saved successfully!")

    return full_pipeline, X_test_raw, y_test


if __name__ == "__main__":
    logger.info("Executing full dataset model training module...")
    pipeline, X_test, y_test = train_model(sample_size=None, model_type="random_forest")
    logger.info("Full dataset training script execution complete.")


