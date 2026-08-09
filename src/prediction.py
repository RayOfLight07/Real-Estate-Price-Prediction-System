"""
Prediction Module for Indian Real Estate Price Prediction System.

Provides inference functions to load serialized models and generate instant price predictions
given input property attributes. Returns results formatted in INR Lakhs and Crores.
"""

import os
import logging
import joblib
import pandas as pd
from typing import Union, Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("Prediction")

_MODEL_CACHE = None


def load_prediction_model(model_path: str = "models/model.pkl") -> Any:
    """
    Loads and caches the serialized model pipeline from disk.

    Parameters:
    -----------
    model_path : str
        Path to the joblib model artifact.

    Returns:
    --------
    Any
        Loaded Scikit-Learn Pipeline.
    """
    global _MODEL_CACHE
    if _MODEL_CACHE is not None:
        return _MODEL_CACHE

    if not os.path.exists(model_path):
        # Fallback check relative to file directory
        alt_path = os.path.join(os.path.dirname(__file__), "..", model_path)
        if os.path.exists(alt_path):
            model_path = os.path.abspath(alt_path)
        else:
            raise FileNotFoundError(f"Model file not found at: {model_path}")

    logger.info(f"Loading prediction pipeline model from: {model_path}")
    _MODEL_CACHE = joblib.load(model_path)
    return _MODEL_CACHE


def format_inr_price(price_lakhs: float) -> str:
    """
    Formats a price value in Lakhs to standard Indian Currency notation (Lakhs / Crores).
    1 Crore = 100 Lakhs = 10,000,000 INR.
    """
    if price_lakhs >= 100.0:
        crores = price_lakhs / 100.0
        return f"Rs. {crores:.2f} Cr (Rs. {price_lakhs:,.2f} Lakhs)"
    else:
        return f"Rs. {price_lakhs:,.2f} Lakhs"



def predict_price(
    input_data: Union[pd.DataFrame, Dict[str, Any]],
    model_path: str = "models/model.pkl",
    model: Any = None
) -> Dict[str, Any]:
    """
    Predicts real estate price for single or multiple property inputs.

    Parameters:
    -----------
    input_data : Union[pd.DataFrame, Dict[str, Any]]
        Property feature dictionary or DataFrame.
    model_path : str, default='models/model.pkl'
        Path to loaded model if model object is not provided directly.
    model : Any, optional
        Pre-loaded model pipeline object.

    Returns:
    --------
    Dict[str, Any]
        Dictionary with prediction metrics: 'price_lakhs', 'price_crores', and 'formatted_price'.
    """
    if model is None:
        model = load_prediction_model(model_path)

    if isinstance(input_data, dict):
        df_input = pd.DataFrame([input_data])
    elif isinstance(input_data, pd.DataFrame):
        df_input = input_data.copy()
    else:
        raise TypeError("input_data must be a pandas DataFrame or a feature dictionary.")

    logger.info(f"Generating price prediction for {len(df_input)} property record(s)...")

    predictions = model.predict(df_input)

    # For single prediction return formatted detailed dictionary
    if len(predictions) == 1:
        pred_lakhs = float(predictions[0])
        pred_crores = float(pred_lakhs / 100.0)
        formatted = format_inr_price(pred_lakhs)

        # Estimate 10% Confidence Interval Range
        min_lakhs = round(pred_lakhs * 0.9, 2)
        max_lakhs = round(pred_lakhs * 1.1, 2)
        
        min_cr = round(min_lakhs / 100.0, 2)
        max_cr = round(max_lakhs / 100.0, 2)

        range_str = f"Rs. {min_cr:.2f} Cr - Rs. {max_cr:.2f} Cr" if pred_lakhs >= 100 else f"Rs. {min_lakhs:,.2f} Lakhs - Rs. {max_lakhs:,.2f} Lakhs"

        return {
            "price_lakhs": round(pred_lakhs, 2),
            "price_crores": round(pred_crores, 4),
            "formatted_price": formatted,
            "min_lakhs": min_lakhs,
            "max_lakhs": max_lakhs,
            "price_range": range_str
        }

    else:
        pred_lakhs_list = [round(float(p), 2) for p in predictions]
        pred_crores_list = [round(float(p) / 100.0, 4) for p in predictions]
        formatted_list = [format_inr_price(p) for p in pred_lakhs_list]

        return {
            "price_lakhs": pred_lakhs_list,
            "price_crores": pred_crores_list,
            "formatted_price": formatted_list
        }


if __name__ == "__main__":
    logger.info("Testing prediction module...")
    sample_input = {
        "State": "Maharashtra",
        "City": "Mumbai",
        "Locality": "Andheri East",
        "Property_Type": "Apartment",
        "BHK": 3,
        "Size_in_SqFt": 1450,
        "Year_Built": 2018,
        "Furnished_Status": "Furnished",
        "Floor_No": 12,
        "Total_Floors": 25,
        "Age_of_Property": 6,
        "Nearby_Schools": 8,
        "Nearby_Hospitals": 6,
        "Public_Transport_Accessibility": "High",
        "Parking_Space": "Yes",
        "Security": "Yes",
        "Amenities": "Gym, Pool, Clubhouse, Garden",
        "Facing": "East",
        "Owner_Type": "Owner",
        "Availability_Status": "Ready_to_Move"
    }

    result = predict_price(sample_input)
    print("\n--- SAMPLE PREDICTION RESULT ---")
    print(f"Predicted Price (Lakhs):  Rs. {result['price_lakhs']} Lakhs")
    print(f"Predicted Price (Crores): Rs. {result['price_crores']} Cr")
    print(f"Formatted Output:         {result['formatted_price']}\n")

