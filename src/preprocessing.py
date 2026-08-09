"""
Preprocessing and Feature Engineering Module for Indian Real Estate Price Prediction.

This module provides data cleaning, data leakage prevention (explicitly dropping Price_per_SqFt),
feature engineering, missing value imputation, categorical encoding, and 80/20 train/test splitting.
"""

import logging
import pandas as pd
import numpy as np
from typing import Tuple, List
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Preprocessing")


class DataLeakageCleaner(BaseEstimator, TransformerMixin):
    """
    Transformer that explicitly removes target-leaking columns (e.g. Price_per_SqFt)
    and non-predictive IDs.
    """
    def __init__(self, leakage_cols: List[str] = None):
        if leakage_cols is None:
            self.leakage_cols = ['Price_per_SqFt', 'ID']
        else:
            self.leakage_cols = leakage_cols

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()
        dropped = []
        for col in self.leakage_cols:
            if col in X.columns:
                X = X.drop(columns=[col])
                dropped.append(col)
        if dropped:
            logger.info(f"Dropped leakage/identifier columns: {dropped}")
        return X


class FeatureEngineer(BaseEstimator, TransformerMixin):
    """
    Custom Transformer for real estate feature engineering:
    - Amenity count, weighted amenity score, and indicator flags.
    - Relative floor position ratio (Floor_No / Total_Floors).
    - Building density ratio and property age grouping.
    """
    def __init__(self):
        self.amenity_weights = {
            'Pool': 2.0,
            'Gym': 2.0,
            'Clubhouse': 2.0,
            'Garden': 1.0,
            'Playground': 1.0
        }

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = X.copy()

        # Amenity features
        if 'Amenities' in X.columns:
            amenities_str = X['Amenities'].astype(str).fillna('')
            X['Amenity_Count'] = amenities_str.apply(
                lambda val: len([item for item in val.split(',') if item.strip()]) if val else 0
            )
            
            # Weighted Amenity Score
            def calc_score(val):
                score = 0.0
                val_lower = val.lower()
                for amen, weight in self.amenity_weights.items():
                    if amen.lower() in val_lower:
                        score += weight
                return score

            X['Amenity_Score'] = amenities_str.apply(calc_score)

            for amenity in self.amenity_weights.keys():
                X[f'Has_{amenity}'] = amenities_str.apply(
                    lambda val: 1 if amenity.lower() in val.lower() else 0
                )

        # Floor position ratio
        if 'Floor_No' in X.columns and 'Total_Floors' in X.columns:
            total = X['Total_Floors'].replace(0, 1)
            X['Floor_Ratio'] = (X['Floor_No'] / total).clip(0, 1)

        # Building density & Age Group
        if 'Total_Floors' in X.columns and 'BHK' in X.columns:
            X['Building_Density'] = X['Total_Floors'] / (X['BHK'] + 1)

        if 'Age_of_Property' in X.columns:
            X['Age_Group'] = pd.cut(
                X['Age_of_Property'],
                bins=[-1, 5, 15, 100],
                labels=[0, 1, 2]
            ).astype(float).fillna(1.0)

        return X



def build_preprocessor_pipeline(categorical_cols: List[str], numerical_cols: List[str]) -> ColumnTransformer:
    """
    Builds a Scikit-Learn ColumnTransformer for numerical imputation & scaling
    and categorical imputation & ordinal encoding.

    Parameters:
    -----------
    categorical_cols : List[str]
        List of categorical feature names.
    numerical_cols : List[str]
        List of numerical feature names.

    Returns:
    --------
    ColumnTransformer
        Configured preprocessor object.
    """
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    cat_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('encoder', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', num_pipeline, numerical_cols),
            ('cat', cat_pipeline, categorical_cols)
        ],
        remainder='drop'
    )

    return preprocessor


def create_full_preprocessing_pipeline(
    categorical_cols: List[str],
    numerical_cols: List[str]
) -> Pipeline:
    """
    Creates a master Pipeline including feature engineering and column transformations.
    """
    preprocessor = build_preprocessor_pipeline(categorical_cols, numerical_cols)
    full_pipeline = Pipeline([
        ('leakage_cleaner', DataLeakageCleaner()),
        ('feature_engineer', FeatureEngineer()),
        ('column_transformer', preprocessor)
    ])
    return full_pipeline


def prepare_data(
    df: pd.DataFrame,
    target_col: str = 'Price_in_Lakhs',
    test_size: float = 0.2,
    random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, Pipeline, List[str], List[str]]:
    """
    Prepares input DataFrame by splitting target, performing feature engineering,
    identifying column types, and performing an 80/20 train/test split.

    Returns:
    --------
    Tuple[X_train, X_test, y_train, y_test, preprocessor_pipeline, categorical_cols, numerical_cols]
    """
    logger.info("Starting data preparation and splitting...")

    if target_col not in df.columns:
        raise KeyError(f"Target column '{target_col}' not found in dataset.")

    # 1. Clean leakage & identifier columns first
    cleaner = DataLeakageCleaner()
    df_clean = cleaner.transform(df)

    # 2. Extract features and target
    y = df_clean[target_col]
    X_raw = df_clean.drop(columns=[target_col])

    # 3. Apply Feature Engineering
    engineer = FeatureEngineer()
    X_engineered = engineer.transform(X_raw)

    # 4. Identify categorical and numerical column lists
    categorical_cols = X_engineered.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_cols = X_engineered.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    logger.info(f"Categorical features ({len(categorical_cols)}): {categorical_cols}")
    logger.info(f"Numerical features ({len(numerical_cols)}): {numerical_cols}")

    # 5. Build preprocessor pipeline
    preprocessor_pipeline = build_preprocessor_pipeline(categorical_cols, numerical_cols)

    # 6. Perform Train/Test Split (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X_engineered, y, test_size=test_size, random_state=random_state
    )

    logger.info(f"Train set shape: X={X_train.shape}, y={y_train.shape}")
    logger.info(f"Test set shape:  X={X_test.shape}, y={y_test.shape}")
    logger.info(f"Target 'Price_in_Lakhs' stats - Train Mean: {y_train.mean():.2f}, Test Mean: {y_test.mean():.2f}")

    return X_train, X_test, y_train, y_test, preprocessor_pipeline, categorical_cols, numerical_cols


if __name__ == "__main__":
    from src.data_loader import load_data

    logger.info("Executing preprocessing self-test...")
    df_raw = load_data(sample_size=10000)
    X_train, X_test, y_train, y_test, prep, cat_cols, num_cols = prepare_data(df_raw)

    logger.info("Fitting preprocessor on training data...")
    X_train_trans = prep.fit_transform(X_train)
    X_test_trans = prep.transform(X_test)

    logger.info(f"Transformed X_train shape: {X_train_trans.shape}")
    logger.info(f"Transformed X_test shape:  {X_test_trans.shape}")
    logger.info("Preprocessing self-test completed successfully!")
