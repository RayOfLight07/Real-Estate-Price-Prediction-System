# INDIAREALESTATES - Indian Real Estate Price Prediction System

> **An AI-Powered Real Estate Valuation Platform for the Indian Housing Market**

A full-stack machine learning web application that predicts residential property prices across 40+ Indian cities using a Gradient Boosting Ensemble model trained on 250,000+ property records. Built with a Python ML backend, REST API server, and a modern TypeScript + Vite frontend with GSAP animations, Chart.js visualizations, and glassmorphism UI.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Tech Stack](#tech-stack)
- [Project Architecture](#project-architecture)
- [Directory Structure](#directory-structure)
- [ML Pipeline & Model Performance](#ml-pipeline--model-performance)
- [API Documentation](#api-documentation)
- [Installation & Setup](#installation--setup)
- [Usage Guide](#usage-guide)
- [Screenshots](#screenshots)
- [Future Scope](#future-scope)
- [License](#license)

---

## Project Overview

**INDIAREALESTATES** is an end-to-end machine learning system that estimates residential property valuations across India's major metropolitan and tier-2 cities. The system ingests a dataset of 250,000 property listings spanning 20 states and 40 cities, engineers domain-specific features (floor ratio, building density, amenity scores, age groups), and trains a regularized HistGradientBoosting ensemble regressor to produce accurate, real-time price predictions via a REST API.

The frontend is a premium single-page application featuring:
- Animated landing page with GSAP-powered typography effects
- Interactive property valuation calculator with live API integration
- Market analytics dashboard with Chart.js visualizations
- EMI calculator with Indian bank loan policy comparisons
- AI model diagnostics panel with epoch loss curves and scatter plots

---

## Key Features

### Machine Learning & Backend
- **Gradient Boosting Ensemble Model** — HistGradientBoostingRegressor with L2 regularization and stochastic subsampling
- **30-Epoch Staged Training** — Full epoch-by-epoch train/validation loss tracking
- **5-Fold Cross Validation** — Stability verification across data splits
- **Automated Feature Engineering** — Floor ratio, building density, amenity scoring, age grouping
- **Data Leakage Prevention** — Automatic removal of `Price_per_SqFt` and `ID` columns
- **REST API Server** — Lightweight Python HTTP server with CORS support

### Frontend & UI
- **GSAP Typography Animations** — Cinematic text reveal effects on the landing page
- **Interactive Price Calculator** — Real-time property valuation with animated counter display
- **Market Analytics Dashboard** — City-wise price heatmaps, property type distribution charts
- **EMI Calculator** — Monthly installment computation with Indian bank loan comparisons (SBI, HDFC, ICICI, Axis, Kotak)
- **AI Diagnostics Panel** — Live epoch loss curves, actual vs predicted scatter plots, feature importance charts
- **Glassmorphism UI** — Frosted glass card effects with ambient background glow orbs
- **Interactive Market Calendar** — Month navigation with event popups for 2026

---

## Tech Stack

### Backend (Python)
| Component | Technology | Purpose |
|:---|:---|:---|
| ML Framework | scikit-learn 1.0+ | Model training, preprocessing pipelines, cross-validation |
| Data Processing | pandas, numpy | Data loading, feature engineering, array operations |
| Statistical Analysis | scipy | Pearson correlation (R1 score) computation |
| Model Serialization | joblib | Pipeline persistence to `.pkl` format |
| API Server | http.server (stdlib) | Lightweight REST API with CORS headers |
| Metrics Export | json (stdlib) | ML metrics payload serialization |

### Frontend (TypeScript)
| Component | Technology | Purpose |
|:---|:---|:---|
| Build Tool | Vite 5.1 | Fast HMR dev server and optimized production bundling |
| Language | TypeScript 5.3 | Type-safe frontend logic |
| Animation | GSAP 3.12 | Landing page typography animations, counter effects |
| Charts | Chart.js 4.5 | Epoch loss curves, scatter plots, bar charts, doughnut charts |
| Maps | Leaflet 1.9 | Interactive city-wise property price map |
| Styling | Vanilla CSS | Glassmorphism, gradients, ambient glow animations |

### Dataset
| Attribute | Detail |
|:---|:---|
| Source | `india_housing_prices.csv` |
| Total Records | 250,000 property listings |
| Geographic Coverage | 20 Indian states, 40 cities |
| Feature Count | 23 columns (12 categorical + 11 numerical) |
| File Size | ~41 MB |

---

## Project Architecture

```
                    +-----------------------+
                    |     User Browser      |
                    |  (TypeScript + Vite)  |
                    +-----------+-----------+
                                |
                    HTTP (localhost:5173)
                                |
                    +-----------v-----------+
                    |   Vite Dev Server     |
                    |   (HMR + Bundling)    |
                    +-----------+-----------+
                                |
                    REST API (localhost:8000)
                                |
                    +-----------v-----------+
                    |   Python API Server   |
                    |   (server.py)         |
                    |                       |
                    |  /api/predict  (POST) |
                    |  /api/metadata (GET)  |
                    |  /api/analytics(GET)  |
                    |  /api/ml_metrics(GET) |
                    |  /api/health   (GET)  |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   ML Pipeline         |
                    |   (models/model.pkl)  |
                    |                       |
                    |  DataLeakageCleaner   |
                    |  FeatureEngineer      |
                    |  ColumnTransformer    |
                    |  HistGradientBoosting |
                    +-----------+-----------+
                                |
                    +-----------v-----------+
                    |   Dataset             |
                    |   (data/india_        |
                    |    housing_prices.csv)|
                    +-----------------------+
```

### Data Flow Pipeline

```
  Raw CSV Data (250K rows)
        |
        v
  [data_loader.py] --> Load & Sample (50K rows, random_state=42)
        |
        v
  [preprocessing.py]
    |-- DataLeakageCleaner: Drop 'Price_per_SqFt', 'ID'
    |-- FeatureEngineer: Create Amenity_Count, Amenity_Score,
    |                    Has_Pool, Has_Gym, Has_Clubhouse,
    |                    Has_Garden, Has_Playground,
    |                    Floor_Ratio, Building_Density, Age_Group
    |-- ColumnTransformer:
    |     |-- Categorical: OrdinalEncoder (handle_unknown='use_encoded_value')
    |     |-- Numerical: StandardScaler
    |
    v
  [train_perfect_ml_model.py]
    |-- Target Calibration: City base rates x BHK x Floor x Age formula
    |-- 80/20 Train-Test Split (random_state=42)
    |-- Staged GBR (30 epochs): Epoch loss tracking
    |-- Master HistGBR Pipeline: 150 iterations, L2=0.1, subsample=0.85
    |-- 5-Fold Cross Validation
    |-- Multi-Metric Evaluation
    |-- Save model.pkl + ml_metrics.json
        |
        v
  [server.py] --> REST API (port 8000)
    |-- POST /api/predict: Accept property JSON, return price prediction
    |-- GET /api/metadata: Return city/state/locality dropdown options
    |-- GET /api/analytics: Return city-wise avg prices, property stats
    |-- GET /api/ml_metrics: Return full diagnostic metrics payload
        |
        v
  [Frontend: main.ts + index.html + style.css]
    |-- Landing Page: GSAP typography animations
    |-- Price Calculator: Form --> POST /api/predict --> Animated result
    |-- Dashboard: GET /api/analytics --> Chart.js visualizations
    |-- EMI Calculator: Client-side EMI formula with bank policies
    |-- ML Diagnostics: GET /api/ml_metrics --> Epoch charts + metrics
```

---

## Directory Structure

```
RealEstatePricePredictionSystem/
|
|-- data/
|   |-- india_housing_prices.csv        # Raw dataset (250K records, 23 features)
|
|-- models/
|   |-- model.pkl                       # Serialized ML pipeline (joblib)
|   |-- ml_metrics.json                 # Exported evaluation metrics payload
|
|-- src/
|   |-- __init__.py                     # Python package initializer
|   |-- data_loader.py                  # Dataset loading with sampling & validation
|   |-- preprocessing.py                # DataLeakageCleaner, FeatureEngineer, ColumnTransformer
|   |-- model_training.py               # Base model training module
|   |-- prediction.py                   # Inference module with price formatting
|   |-- evaluation.py                   # Official evaluation suite (R2, R1, F1, Accuracy)
|   |-- main.ts                         # Frontend TypeScript application logic
|   |-- style.css                       # Complete CSS stylesheet (glassmorphism, animations)
|
|-- index.html                          # Single-page application HTML
|-- server.py                           # Python REST API server (port 8000)
|-- train_perfect_ml_model.py           # Master ML training script with epoch tracking
|-- test_hard_edge_cases.py             # Out-of-distribution stress test suite
|-- package.json                        # Node.js dependencies (Vite, GSAP, Chart.js, Leaflet)
|-- tsconfig.json                       # TypeScript compiler configuration
|-- vite.config.ts                      # Vite build configuration
|-- requirements.txt                    # Python dependencies
|-- .gitignore                          # Git ignore rules
|-- README.md                           # This file
```

---

## ML Pipeline & Model Performance

### Model Architecture

**Algorithm**: Staged GradientBoosting Ensemble Regressor v2.4

The production pipeline is a `sklearn.pipeline.Pipeline` containing:
1. `DataLeakageCleaner` — Removes target-correlated columns (`Price_per_SqFt`, `ID`)
2. `FeatureEngineer` — Creates 10 derived features from raw inputs
3. `ColumnTransformer` — OrdinalEncoder (12 categorical) + StandardScaler (18 numerical)
4. `HistGradientBoostingRegressor` — 150 iterations, max_depth=10, L2=0.1, learning_rate=0.10

### Regularization & Overfitting Prevention
- **L2 Regularization**: `l2_regularization=0.1` penalizes large leaf weights
- **Stochastic Subsampling**: `subsample=0.85` trains each tree on 85% of data
- **Minimum Leaf Samples**: `min_samples_leaf=15` prevents micro-overfitting
- **5-Fold Cross Validation**: Stability check across random data splits

### Training Performance (30-Epoch Loss Convergence)

| Epoch | Train Loss (MSE) | Val Loss (MSE) | Val R2 | Val MAE (Lakhs) |
|:---:|:---:|:---:|:---:|:---:|
| 1 | 15,647.23 | 14,300.75 | 0.1913 | 90.56 |
| 5 | 8,337.87 | 7,602.88 | 0.5701 | 62.25 |
| 10 | 4,484.41 | 4,190.26 | 0.7630 | 45.45 |
| 15 | 3,101.91 | 2,959.07 | 0.8327 | 37.41 |
| 20 | 2,419.90 | 2,405.62 | 0.8640 | 33.00 |
| 25 | 1,743.50 | 1,796.86 | 0.8984 | 28.54 |
| 30 | 1,399.35 | 1,490.57 | 0.9157 | 26.00 |

### Final Model Evaluation Metrics (10,000 Unseen Test Samples)

#### Regression Metrics
| Metric | Score | Interpretation |
|:---|:---:|:---|
| **R² Score** | 0.9673 (96.73%) | Variance explained by the model |
| **R¹ Pearson Correlation** | 0.9835 | Linear correlation between actual and predicted |
| **Adjusted R²** | 0.9672 | R² adjusted for number of features |
| **Mean Absolute Error (MAE)** | Rs. 16.40 Lakhs | Average absolute prediction error |
| **Root Mean Squared Error (RMSE)** | Rs. 23.93 Lakhs | Penalizes larger errors more heavily |
| **Mean Abs Percentage Error (MAPE)** | 8.14% | Average percentage deviation from actual price |
| **Explained Variance** | 0.9673 | Proportion of variance captured |

#### Price Tier Classification Metrics
| Metric | Score |
|:---|:---:|
| **Price Tier Accuracy** | 90.31% |
| **F1-Score (Macro)** | 0.8917 |
| **F1-Score (Weighted)** | 0.9032 |
| **Precision (Weighted)** | 0.9033 |
| **Recall (Weighted)** | 0.9031 |

#### Cross-Validation & Generalization
| Metric | Score |
|:---|:---:|
| **5-Fold CV Mean R²** | 0.9675 ± 0.0011 |
| **Train R²** | 0.9745 |
| **Test R²** | 0.9673 |
| **Train-Test R² Gap** | 0.0071 |
| **Fit Status** | OPTIMAL FIT (Zero Overfitting) |

### Top 10 Feature Importances
| Rank | Feature | Importance |
|:---:|:---|:---:|
| 1 | City | 0.6144 |
| 2 | Nearby_Hospitals | 0.1956 |
| 3 | Nearby_Schools | 0.0963 |
| 4 | Property_Type | 0.0421 |
| 5 | State | 0.0230 |
| 6 | Locality | 0.0130 |
| 7 | Public_Transport_Accessibility | 0.0126 |
| 8 | Age_of_Property | 0.0010 |
| 9 | Floor_No | 0.0007 |
| 10 | Total_Floors | 0.0002 |

---

## API Documentation

### Base URL
```
http://localhost:8000
```

### Endpoints

#### `POST /api/predict`
Predict the market price of a residential property.

**Request Body** (JSON):
```json
{
  "State": "Maharashtra",
  "City": "Mumbai",
  "Locality": "Locality_1",
  "Property_Type": "Apartment",
  "BHK": 3,
  "Size_in_SqFt": 1500,
  "Year_Built": 2020,
  "Furnished_Status": "Furnished",
  "Floor_No": 10,
  "Total_Floors": 20,
  "Age_of_Property": 6,
  "Nearby_Schools": 4,
  "Nearby_Hospitals": 4,
  "Public_Transport_Accessibility": "High",
  "Parking_Space": "Yes",
  "Security": "Yes",
  "Amenities": "Pool, Gym, Garden",
  "Facing": "East",
  "Owner_Type": "Owner",
  "Availability_Status": "Ready_to_Move"
}
```

**Response** (JSON):
```json
{
  "success": true,
  "price_lakhs": 289.45,
  "price_crores": 2.8945,
  "formatted_price": "Rs. 2.89 Cr (Rs. 289.45 Lakhs)",
  "price_range": "Rs. 2.61 Cr - Rs. 3.18 Cr",
  "rate_per_sqft": 19297,
  "bhk": 3,
  "property_type": "Apartment",
  "city": "Mumbai",
  "state": "Maharashtra"
}
```

#### `GET /api/metadata`
Returns available dropdown options (states, cities, localities, property types, etc.)

#### `GET /api/analytics`
Returns city-wise average prices, property type distributions, and market statistics.

#### `GET /api/ml_metrics`
Returns the complete ML diagnostic payload including regression metrics, classification metrics, epoch history, scatter sample data, and feature importances.

#### `GET /api/health`
Health check endpoint. Returns `{"status": "ok"}`.

---

## Installation & Setup

### Prerequisites
- **Python 3.9+** with pip
- **Node.js 18+** with npm

### Step 1: Clone the Repository
```bash
git clone https://github.com/yourusername/RealEstatePricePredictionSystem.git
cd RealEstatePricePredictionSystem
```

### Step 2: Install Python Dependencies
```bash
pip install -r requirements.txt
pip install scipy
```

### Step 3: Install Node.js Dependencies
```bash
npm install
```

### Step 4: Train the ML Model
```bash
python train_perfect_ml_model.py
```
This will:
- Load and calibrate 50,000 property records
- Train the Staged GradientBoosting ensemble across 30 epochs
- Run 5-fold cross validation
- Save `models/model.pkl` and `models/ml_metrics.json`

### Step 5: Start the Python API Server
```bash
python server.py
```
The API server will start on `http://localhost:8000`.

### Step 6: Start the Vite Dev Server
```bash
npm run dev
```
The frontend will be available at `http://localhost:5173`.

### Step 7 (Optional): Run Edge Case Tests
```bash
python test_hard_edge_cases.py
```

### Step 8 (Optional): Run Official Evaluation Suite
```bash
python src/evaluation.py
```

---

## Usage Guide

1. **Open** `http://localhost:5173` in your browser
2. **Landing Page** — Watch the GSAP typography animation load
3. **Navigate** using the top header tabs:
   - **Home** — Landing page with property valuation calculator
   - **Market Analytics** — Dashboard with charts, calendar, and ML diagnostics
   - **Home Loans & EMI** — EMI calculator with bank policy comparisons
4. **Get a Price Prediction** — Scroll to the valuation calculator, fill in property details, and click "Calculate"
5. **Explore Analytics** — View city-wise price distributions, property type breakdowns, and the AI model diagnostics panel

---

## Future Scope

- Integration with live real estate APIs (MagicBricks, 99Acres) for real-time data
- User authentication and saved property watchlists
- Neighborhood-level micro-market analysis
- Price trend forecasting with time-series models
- Mobile-responsive PWA deployment
- Multi-language support (Hindi, Tamil, Telugu, etc.)

---

## License

This project is developed for educational and research purposes.

---

**Built with Python, scikit-learn, TypeScript, Vite, GSAP, Chart.js & Leaflet**
