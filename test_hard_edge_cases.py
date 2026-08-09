"""
Hard Edge Cases & Out-of-Distribution Stress Test Suite for INDIAREALESTATES ML Model.

Tests challenging property profiles not typically present in standard training samples:
1. Ultra-Luxury Skyscraper Penthouse (Mumbai: 6 BHK, 7500 sqft, Floor 48/50)
2. Decayed Budget Studio Apartment (Jaipur: 1 BHK, 380 sqft, 32 Years Old)
3. Sprawling Heritage Estate Villa (Bangalore: 5 BHK, 5200 sqft, Gated Estate)
4. Altitude Premium Sensitivity Test (Gurgaon 45th Floor vs Ground Floor)
5. Amenity Contrast Sensitivity Test (Pune 5-Amenity Resort vs Zero Amenity Unit)
"""

import os
import sys
import json
import logging
import joblib
import pandas as pd
import numpy as np

sys.path.append(os.path.abspath('.'))
from src.prediction import predict_price

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("HardEdgeCasesTest")

def main():
    logger.info("Initializing Hard Edge Cases ML Diagnostic Test Suite...")
    
    model_path = "models/model.pkl"
    if not os.path.exists(model_path):
        logger.error("Model file 'models/model.pkl' not found. Please train model first.")
        return

    pipeline = joblib.load(model_path)

    hard_cases = [
        {
            "name": "Hard Case 1: Ultra-Luxury Skyscraper Penthouse (Mumbai)",
            "description": "7,500 SqFt 6 BHK Penthouse on 48th floor of 50-story tower in Mumbai with 5 luxury amenities",
            "payload": {
                "State": "Maharashtra",
                "City": "Mumbai",
                "Locality": "Worli_Coastal_Tower",
                "Property_Type": "Apartment",
                "BHK": 6,
                "Size_in_SqFt": 7500,
                "Year_Built": 2024,
                "Furnished_Status": "Furnished",
                "Floor_No": 48,
                "Total_Floors": 50,
                "Age_of_Property": 2,
                "Nearby_Schools": 5,
                "Nearby_Hospitals": 5,
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Amenities": "Pool, Gym, Garden, Playground, Clubhouse",
                "Facing": "East",
                "Owner_Type": "Builder",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 2: Aged Budget Studio Apartment (Jaipur)",
            "description": "380 SqFt 1 BHK studio on ground floor of 3-story building, 32 years old, zero amenities",
            "payload": {
                "State": "Rajasthan",
                "City": "Jaipur",
                "Locality": "Old_City_Chowk",
                "Property_Type": "Apartment",
                "BHK": 1,
                "Size_in_SqFt": 380,
                "Year_Built": 1994,
                "Furnished_Status": "Unfurnished",
                "Floor_No": 0,
                "Total_Floors": 3,
                "Age_of_Property": 32,
                "Nearby_Schools": 2,
                "Nearby_Hospitals": 1,
                "Public_Transport_Accessibility": "Low",
                "Parking_Space": "No",
                "Security": "No",
                "Amenities": "None",
                "Facing": "West",
                "Owner_Type": "Broker",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 3: Sprawling Heritage Villa Estate (Bangalore)",
            "description": "5,200 SqFt 5 BHK Independent Villa Estate in Bangalore Tech Hub with full amenities",
            "payload": {
                "State": "Karnataka",
                "City": "Bangalore",
                "Locality": "Whitefield_Estate",
                "Property_Type": "Villa",
                "BHK": 5,
                "Size_in_SqFt": 5200,
                "Year_Built": 2011,
                "Furnished_Status": "Furnished",
                "Floor_No": 2,
                "Total_Floors": 2,
                "Age_of_Property": 15,
                "Nearby_Schools": 5,
                "Nearby_Hospitals": 4,
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Amenities": "Pool, Gym, Garden, Clubhouse",
                "Facing": "North",
                "Owner_Type": "Owner",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 4A: Top Floor Skyscraper Unit (Gurgaon Floor 45/45)",
            "description": "1,800 SqFt 3 BHK on 45th floor of 45-story Gurgaon tower",
            "payload": {
                "State": "Delhi",
                "City": "Gurgaon",
                "Locality": "Golf_Course_Road",
                "Property_Type": "Apartment",
                "BHK": 3,
                "Size_in_SqFt": 1800,
                "Year_Built": 2021,
                "Furnished_Status": "Semi-furnished",
                "Floor_No": 45,
                "Total_Floors": 45,
                "Age_of_Property": 5,
                "Nearby_Schools": 4,
                "Nearby_Hospitals": 4,
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Amenities": "Gym, Pool, Clubhouse",
                "Facing": "East",
                "Owner_Type": "Owner",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 4B: Ground Floor Unit (Gurgaon Floor 0/45)",
            "description": "Identical 1,800 SqFt 3 BHK on ground floor of same 45-story Gurgaon tower",
            "payload": {
                "State": "Delhi",
                "City": "Gurgaon",
                "Locality": "Golf_Course_Road",
                "Property_Type": "Apartment",
                "BHK": 3,
                "Size_in_SqFt": 1800,
                "Year_Built": 2021,
                "Furnished_Status": "Semi-furnished",
                "Floor_No": 0,
                "Total_Floors": 45,
                "Age_of_Property": 5,
                "Nearby_Schools": 4,
                "Nearby_Hospitals": 4,
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Amenities": "Gym, Pool, Clubhouse",
                "Facing": "East",
                "Owner_Type": "Owner",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 5A: Full Luxury Amenity Resort Unit (Pune)",
            "description": "1,200 SqFt 2 BHK in Pune with 5 luxury amenities + security",
            "payload": {
                "State": "Maharashtra",
                "City": "Pune",
                "Locality": "Koregaon_Park",
                "Property_Type": "Apartment",
                "BHK": 2,
                "Size_in_SqFt": 1200,
                "Year_Built": 2020,
                "Furnished_Status": "Furnished",
                "Floor_No": 5,
                "Total_Floors": 12,
                "Age_of_Property": 6,
                "Nearby_Schools": 4,
                "Nearby_Hospitals": 4,
                "Public_Transport_Accessibility": "High",
                "Parking_Space": "Yes",
                "Security": "Yes",
                "Amenities": "Pool, Gym, Garden, Playground, Clubhouse",
                "Facing": "East",
                "Owner_Type": "Owner",
                "Availability_Status": "Ready_to_Move"
            }
        },
        {
            "name": "Hard Case 5B: Zero Amenity Bare Unit (Pune)",
            "description": "Identical 1,200 SqFt 2 BHK in Pune with zero amenities and no parking",
            "payload": {
                "State": "Maharashtra",
                "City": "Pune",
                "Locality": "Koregaon_Park",
                "Property_Type": "Apartment",
                "BHK": 2,
                "Size_in_SqFt": 1200,
                "Year_Built": 2020,
                "Furnished_Status": "Unfurnished",
                "Floor_No": 5,
                "Total_Floors": 12,
                "Age_of_Property": 6,
                "Nearby_Schools": 4,
                "Nearby_Hospitals": 4,
                "Public_Transport_Accessibility": "Low",
                "Parking_Space": "No",
                "Security": "No",
                "Amenities": "None",
                "Facing": "East",
                "Owner_Type": "Owner",
                "Availability_Status": "Ready_to_Move"
            }
        }
    ]

    results = []

    print("\n" + "="*80)
    print("      HARD OUT-OF-DISTRIBUTION EDGE CASES MODEL DIAGNOSTIC TEST       ")
    print("="*80)

    for idx, case in enumerate(hard_cases, start=1):
        res = predict_price(case["payload"], model=pipeline)
        size = case["payload"]["Size_in_SqFt"]
        lakhs = res["price_lakhs"]
        rate = int((lakhs * 100000) / size)

        results.append({
            "case_name": case["name"],
            "formatted_price": res["formatted_price"],
            "lakhs": lakhs,
            "rate_per_sqft": rate,
            "range": res.get("price_range", "N/A")
        })

        print(f"\n[{idx}] {case['name']}")
        print(f"    Profile:      {case['description']}")
        print(f"    Valuation:    {res['formatted_price']}")
        print(f"    Unit Rate:    Rs. {rate:,} / sqft")
        print(f"    Val Range:    {res.get('price_range', 'N/A')}")

    print("\n" + "="*80)
    print("      SENSITIVITY & BEHAVIOR DIAGNOSTIC ANALYSIS       ")
    print("="*80)

    p_top = results[3]["lakhs"]
    p_gnd = results[4]["lakhs"]
    alt_diff = p_top - p_gnd
    alt_pct = (alt_diff / p_gnd) * 100

    p_lux = results[5]["lakhs"]
    p_bare = results[6]["lakhs"]
    amen_diff = p_lux - p_bare
    amen_pct = (amen_diff / p_bare) * 100

    print(f"1. Altitude Floor Premium Sensitivity (Gurgaon 45th Floor vs 0 Floor):")
    print(f"   - 45th Floor Valuation: Rs. {p_top:.2f} Lakhs")
    print(f"   - Ground Floor Valuation: Rs. {p_gnd:.2f} Lakhs")
    print(f"   - Altitude Premium:     +Rs. {alt_diff:.2f} Lakhs (+{alt_pct:.2f}% floor multiplier)\n")

    print(f"2. Amenity Score Sensitivity (Pune Resort Unit vs Bare Unit):")
    print(f"   - 5-Amenity Resort Unit: Rs. {p_lux:.2f} Lakhs")
    print(f"   - Zero Amenity Bare Unit: Rs. {p_bare:.2f} Lakhs")
    print(f"   - Luxury Amenity Premium: +Rs. {amen_diff:.2f} Lakhs (+{amen_pct:.2f}% amenity value)\n")

    print(f"3. Domain Extrapolation Sanity Check:")
    print(f"   - Mumbai Penthouse (7500 sqft):  {results[0]['formatted_price']} (Rate: Rs. {results[0]['rate_per_sqft']:,}/sqft)")
    print(f"   - Jaipur Old Studio (380 sqft):  {results[1]['formatted_price']} (Rate: Rs. {results[1]['rate_per_sqft']:,}/sqft)")
    print("="*80 + "\n")

    logger.info("Hard Edge Cases diagnostic test execution completed successfully!")

if __name__ == "__main__":
    main()
