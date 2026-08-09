"""
Evaluation Module for Indian Real Estate Price Prediction System.

This module evaluates trained models on unseen test data, computing key regression
metrics: Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and R² Score.
"""

import logging
import numpy as np
import pandas as pd
from typing import Dict, Any
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Evaluation")


def evaluate_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, float]:
    """
    Computes key regression metrics (MAE, RMSE, R²) for a trained model pipeline.

    Parameters:
    -----------
    model : Any
        Fitted model or pipeline with a .predict() method.
    X_test : pd.DataFrame
        Test features.
    y_test : pd.Series
        Ground truth target values (Price in Lakhs).

    Returns:
    --------
    Dict[str, float]
        Dictionary containing calculated performance metrics.
    """
    logger.info(f"Evaluating model on test dataset ({len(X_test):,} samples)...")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Compute key metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    mse = float(mean_squared_error(y_test, y_pred))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(y_test, y_pred))

    # Mean percentage error calculation
    mape = float(np.mean(np.abs((y_test - y_pred) / np.maximum(y_test, 1e-5))) * 100)

    metrics = {
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2,
        "MAPE": mape
    }

    # Print formatted metrics summary
    print("\n" + "="*50)
    print("       MODEL PERFORMANCE EVALUATION METRICS       ")
    print("="*50)
    print(f"  Mean Absolute Error (MAE):     Rs. {mae:,.2f} Lakhs")
    print(f"  Root Mean Squared Error (RMSE): Rs. {rmse:,.2f} Lakhs")
    print(f"  R2 Score (Variance Explained):   {r2:.4f} ({r2*100:.2f}%)")
    print(f"  Mean Abs Percentage Error:     {mape:.2f}%")
    print("="*50 + "\n")


    logger.info(f"Evaluation metrics computed: MAE={mae:.2f}, RMSE={rmse:.2f}, R2={r2:.4f}")
    return metrics


if __name__ == "__main__":
    import joblib
    import os

    logger.info("Executing evaluation self-test...")
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at '{model_path}'. Please run model_training.py first.")
    else:
        pipeline = joblib.load(model_path)
        from src.data_loader import load_data
        from src.preprocessing import prepare_data

        df_raw = load_data(sample_size=10000)
        X_train, X_test, y_train, y_test, _, _, _ = prepare_data(df_raw)
        
        # Drop target to get raw features matching pipeline input format
        raw_test_indices = X_test.index
        df_clean_raw = df_raw.drop(columns=['Price_in_Lakhs'], errors='ignore')
        X_test_raw = df_clean_raw.loc[raw_test_indices]

        metrics = evaluate_model(pipeline, X_test_raw, y_test)
