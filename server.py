"""
REST API Backend Server for INDIAREALESTATES System.

Exposes endpoints for Vite/TypeScript frontend:
- POST /api/predict
- GET /api/metadata
- GET /api/analytics
- GET /api/health
"""

import os
import sys
import json
import logging
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import parse_qs, urlparse

# Ensure src modules can be imported
sys.path.append(os.path.abspath(os.path.dirname(__file__)))

from src.prediction import predict_price, load_prediction_model, format_inr_price
from src.data_loader import load_data

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("APIServer")

# Global Cached Metadata, Analytics & Model
METADATA_CACHE = None
ANALYTICS_CACHE = None
MODEL_INSTANCE = None

CITY_COORDINATES = {
    "Vijayawada": {"state": "Andhra Pradesh", "lat": 16.5062, "lon": 80.6480},
    "Vishakhapatnam": {"state": "Andhra Pradesh", "lat": 17.6868, "lon": 83.2185},
    "Guwahati": {"state": "Assam", "lat": 26.1445, "lon": 91.7362},
    "Silchar": {"state": "Assam", "lat": 24.8333, "lon": 92.7789},
    "Gaya": {"state": "Bihar", "lat": 24.7914, "lon": 85.0002},
    "Patna": {"state": "Bihar", "lat": 25.5941, "lon": 85.1376},
    "Bilaspur": {"state": "Chhattisgarh", "lat": 22.0797, "lon": 82.1391},
    "Raipur": {"state": "Chhattisgarh", "lat": 21.2514, "lon": 81.6296},
    "Dwarka": {"state": "Delhi", "lat": 28.5921, "lon": 77.0460},
    "New Delhi": {"state": "Delhi", "lat": 28.6139, "lon": 77.2090},
    "Ahmedabad": {"state": "Gujarat", "lat": 23.0225, "lon": 72.5714},
    "Surat": {"state": "Gujarat", "lat": 21.1702, "lon": 72.8311},
    "Faridabad": {"state": "Haryana", "lat": 28.4089, "lon": 77.3178},
    "Gurgaon": {"state": "Haryana", "lat": 28.4595, "lon": 77.0266},
    "Jamshedpur": {"state": "Jharkhand", "lat": 22.8046, "lon": 86.2029},
    "Ranchi": {"state": "Jharkhand", "lat": 23.3441, "lon": 85.3096},
    "Bangalore": {"state": "Karnataka", "lat": 12.9716, "lon": 77.5946},
    "Mangalore": {"state": "Karnataka", "lat": 12.9141, "lon": 74.8560},
    "Mysore": {"state": "Karnataka", "lat": 12.2958, "lon": 76.6394},
    "Kochi": {"state": "Kerala", "lat": 9.9312, "lon": 76.2673},
    "Trivandrum": {"state": "Kerala", "lat": 8.5241, "lon": 76.9366},
    "Bhopal": {"state": "Madhya Pradesh", "lat": 23.2599, "lon": 77.4126},
    "Indore": {"state": "Madhya Pradesh", "lat": 22.7196, "lon": 75.8577},
    "Mumbai": {"state": "Maharashtra", "lat": 19.0760, "lon": 72.8777},
    "Nagpur": {"state": "Maharashtra", "lat": 21.1458, "lon": 79.0882},
    "Pune": {"state": "Maharashtra", "lat": 18.5204, "lon": 73.8567},
    "Bhubaneswar": {"state": "Odisha", "lat": 20.2961, "lon": 85.8245},
    "Cuttack": {"state": "Odisha", "lat": 20.4625, "lon": 85.8828},
    "Amritsar": {"state": "Punjab", "lat": 31.6340, "lon": 74.8723},
    "Ludhiana": {"state": "Punjab", "lat": 30.9010, "lon": 75.8573},
    "Jaipur": {"state": "Rajasthan", "lat": 26.9124, "lon": 75.7873},
    "Jodhpur": {"state": "Rajasthan", "lat": 26.2389, "lon": 73.0243},
    "Chennai": {"state": "Tamil Nadu", "lat": 13.0827, "lon": 80.2707},
    "Coimbatore": {"state": "Tamil Nadu", "lat": 11.0168, "lon": 76.9558},
    "Hyderabad": {"state": "Telangana", "lat": 17.3850, "lon": 78.4867},
    "Warangal": {"state": "Telangana", "lat": 17.9689, "lon": 79.5941},
    "Lucknow": {"state": "Uttar Pradesh", "lat": 26.8467, "lon": 80.9462},
    "Noida": {"state": "Uttar Pradesh", "lat": 28.5355, "lon": 77.3910},
    "Dehradun": {"state": "Uttarakhand", "lat": 30.3165, "lon": 78.0322},
    "Haridwar": {"state": "Uttarakhand", "lat": 29.9457, "lon": 78.1642},
    "Durgapur": {"state": "West Bengal", "lat": 23.5204, "lon": 87.3119},
    "Kolkata": {"state": "West Bengal", "lat": 22.5726, "lon": 88.3639}
}


def initialize_server_context():
    """Initializes dataset metadata, analytics, and pre-loads prediction model."""
    global METADATA_CACHE, ANALYTICS_CACHE, MODEL_INSTANCE
    logger.info("Initializing API Server context...")

    try:
        MODEL_INSTANCE = load_prediction_model("models/model.pkl")
        logger.info("ML Model Pipeline successfully loaded into server memory.")
    except Exception as e:
        logger.warning(f"Could not pre-load model pipeline: {e}")

    try:
        df = load_data("data/india_housing_prices.csv", sample_size=100000)

        states = sorted(df['State'].dropna().unique().tolist())
        cities_by_state = {}
        localities_by_city = {}

        for state in states:
            state_df = df[df['State'] == state]
            cities_in_state = sorted(state_df['City'].dropna().unique().tolist())
            cities_by_state[state] = cities_in_state

        all_cities = sorted(df['City'].dropna().unique().tolist())
        for city in all_cities:
            city_df = df[df['City'] == city]
            localities_by_city[city] = sorted(city_df['Locality'].dropna().unique().tolist())

        # Compute Market Analytics Stats
        city_stats = {}
        for city, grp in df.groupby('City'):
            st_name = grp['State'].iloc[0]
            avg_price = round(float(grp['Price_in_Lakhs'].mean()), 2)
            avg_sqft = round(float(grp['Size_in_SqFt'].mean()), 1)
            avg_rate = int((avg_price * 100000) / avg_sqft) if avg_sqft > 0 else 0
            coords = CITY_COORDINATES.get(city, {"lat": 20.5937, "lon": 78.9629})
            
            city_stats[city] = {
                "state": st_name,
                "lat": coords["lat"],
                "lon": coords["lon"],
                "avg_price_lakhs": avg_price,
                "avg_sqft": avg_sqft,
                "avg_rate_per_sqft": avg_rate,
                "total_listings": len(grp)
            }

        state_stats = {}
        for state_name, grp in df.groupby('State'):
            avg_price = round(float(grp['Price_in_Lakhs'].mean()), 2)
            avg_sqft = round(float(grp['Size_in_SqFt'].mean()), 1)
            avg_rate = int((avg_price * 100000) / avg_sqft) if avg_sqft > 0 else 0
            state_stats[state_name] = {
                "avg_price_lakhs": avg_price,
                "avg_rate_per_sqft": avg_rate,
                "total_listings": len(grp)
            }

        bhk_stats = {}
        for bhk_val, grp in df.groupby('BHK'):
            bhk_stats[int(bhk_val)] = round(float(grp['Price_in_Lakhs'].mean()), 2)

        prop_type_stats = {}
        for ptype, grp in df.groupby('Property_Type'):
            prop_type_stats[ptype] = round(float(grp['Price_in_Lakhs'].mean()), 2)

        ANALYTICS_CACHE = {
            "city_stats": city_stats,
            "state_stats": state_stats,
            "bhk_stats": bhk_stats,
            "prop_type_stats": prop_type_stats
        }

        METADATA_CACHE = {
            "states": states,
            "cities_by_state": cities_by_state,
            "localities_by_city": localities_by_city,
            "property_types": sorted(df['Property_Type'].dropna().unique().tolist()),
            "furnishing_options": sorted(df['Furnished_Status'].dropna().unique().tolist()),
            "facing_options": sorted(df['Facing'].dropna().unique().tolist()),
            "owner_options": sorted(df['Owner_Type'].dropna().unique().tolist()),
            "availability_options": sorted(df['Availability_Status'].dropna().unique().tolist()),
            "transport_options": sorted(df['Public_Transport_Accessibility'].dropna().unique().tolist())
        }
        logger.info(f"Metadata & Analytics initialized: {len(states)} states, {len(all_cities)} cities.")
    except Exception as e:
        logger.error(f"Failed to build metadata/analytics cache: {e}")


class APIRequestHandler(BaseHTTPRequestHandler):
    """Custom HTTP Request Handler with full CORS support for Vite frontend."""

    def _set_cors_headers(self):
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type, Authorization')

    def do_OPTIONS(self):
        self.send_response(200)
        self._set_cors_headers()
        self.end_headers()

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/health":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            response = {
                "status": "healthy",
                "model_loaded": MODEL_INSTANCE is not None,
                "dataset_rows": 250000
            }
            self.wfile.write(json.dumps(response).encode('utf-8'))

        elif path == "/api/metadata":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(METADATA_CACHE).encode('utf-8'))

        elif path == "/api/analytics":
            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps(ANALYTICS_CACHE).encode('utf-8'))

        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()
            self.wfile.write(json.dumps({"error": "Endpoint not found"}).encode('utf-8'))

    def do_POST(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/predict":
            content_length = int(self.headers.get('Content-Length', 0))
            post_data = self.rfile.read(content_length)
            
            try:
                payload = json.loads(post_data.decode('utf-8'))
                logger.info(f"Received prediction request for: {payload.get('City')}, {payload.get('State')}")

                global MODEL_INSTANCE
                if MODEL_INSTANCE is None:
                    MODEL_INSTANCE = load_prediction_model("models/model.pkl")

                res = predict_price(payload, model=MODEL_INSTANCE)
                
                size_sqft = payload.get('Size_in_SqFt', 1000)
                price_lakhs = res['price_lakhs']
                rate_per_sqft = int((price_lakhs * 100000) / size_sqft) if size_sqft > 0 else 0

                response = {
                    "success": True,
                    "price_lakhs": res['price_lakhs'],
                    "price_crores": res['price_crores'],
                    "formatted_price": res['formatted_price'],
                    "price_range": res.get('price_range', ''),
                    "rate_per_sqft": rate_per_sqft,
                    "bhk": payload.get('BHK'),
                    "property_type": payload.get('Property_Type'),
                    "city": payload.get('City'),
                    "state": payload.get('State')
                }


                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps(response).encode('utf-8'))

            except Exception as e:
                logger.error(f"Prediction failed: {e}")
                self.send_response(500)
                self.send_header('Content-Type', 'application/json')
                self._set_cors_headers()
                self.end_headers()
                self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))
        else:
            self.send_response(404)
            self._set_cors_headers()
            self.end_headers()


def run_server(port=8000):
    initialize_server_context()
    server_address = ('', port)
    httpd = HTTPServer(server_address, APIRequestHandler)
    logger.info(f"🚀 INDIAREALESTATES API Server running on http://localhost:{port}")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("Server stopped.")


if __name__ == "__main__":
    run_server(port=8000)
