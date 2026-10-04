"""
Central configuration for the Regime-Aware Tourism Forecasting project.

This module defines all paths, hyperparameters, feature groups, and evaluation
settings used throughout the project. Import this module to ensure consistency
across all scripts and notebooks.
"""

from pathlib import Path

# ============================================================
# 1. PATHS
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "tourism_eu27.xlsx"
DATA_DIR = PROJECT_ROOT / "data"
DATA_PROCESSED = DATA_DIR / "processed"
DATA_FEATURES = DATA_DIR / "features"
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = RESULTS_DIR / "models"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"

# ============================================================
# 2. DATA SETTINGS
# ============================================================
SHEET_NAME = "Monthly"
DATE_COL = "date"
GEO_COL = "geo"
COUNTRY_COL = "country"

# Target columns
TARGET_ARRIVALS = "arrivals_total"
TARGET_NIGHTS = "nights_total"
TARGETS = [TARGET_ARRIVALS, TARGET_NIGHTS]

# Countries to exclude (too much missing data)
EXCLUDE_COUNTRIES = ["IE"]  # Ireland has inf values due to missing 2019 data

# ============================================================
# 3. TEMPORAL SPLIT
# ============================================================
TRAIN_END_YEAR = 2022      # Train: 2012–2022
VAL_START_YEAR = 2023      # Validation: 2023
VAL_END_YEAR = 2023
TEST_START_YEAR = 2024     # Test: 2024–2025
TEST_END_YEAR = 2025

# ============================================================
# 4. REGIME SETTINGS
# ============================================================
N_REGIMES = 3  # pre-COVID, COVID, recovery

# Period group mapping (ground truth for validation)
REGIME_MAPPING = {
    "pre_2020": 0,     # Normal
    "2020_2021": 1,    # COVID crisis
    "from_2022": 2,    # Recovery
}

REGIME_NAMES = {
    0: "Pre-COVID (Normal)",
    1: "COVID Crisis",
    2: "Post-COVID Recovery",
}

# HMM settings
HMM_N_ITER = 200
HMM_COVARIANCE_TYPE = "full"
HMM_RANDOM_STATE = 42

# PELT settings
PELT_PENALTY = 10  # Penalty for change point detection

# ============================================================
# 5. FEATURE ENGINEERING
# ============================================================
# Lag features
LAG_PERIODS = [1, 2, 3, 6, 12]

# Rolling window sizes
ROLLING_WINDOWS = [3, 6, 12]

# Calendar features
CALENDAR_FEATURES = [
    "month_sin", "month_cos",
    "year",
    "is_summer",       # June-August
    "is_winter",       # December-February
    "quarter",
]

# Composition features (from raw data)
COMPOSITION_FEATURES = [
    "foreign_nights_share",
    "nights_per_arrival_ratio",
]

# Supply features
SUPPLY_FEATURES = [
    "hotel_room_occupancy_pct",
    "hotel_bed_occupancy_pct",
]

# Component features
COMPONENT_FEATURES = [
    "arrivals_domestic",
    "arrivals_foreign",
    "nights_domestic",
    "nights_foreign",
]

# Regime features (added after regime detection)
REGIME_FEATURES = [
    "regime_label",
    "months_since_regime_change",
    "regime_duration",
]

# ============================================================
# 6. MODEL HYPERPARAMETERS
# ============================================================
RANDOM_STATE = 42

# XGBoost
XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "random_state": RANDOM_STATE,
    "n_jobs": -1,
}

# LSTM / BiLSTM
LSTM_PARAMS = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,       # BiLSTM
    "attention_heads": 4,        # Multi-head attention
    "regime_embedding_dim": 8,   # Regime embedding dimension
    "learning_rate": 1e-3,
    "batch_size": 64,
    "epochs": 100,
    "patience": 15,              # Early stopping patience
    "sequence_length": 12,       # 12 months lookback
}

# TFT (Temporal Fusion Transformer)
TFT_PARAMS = {
    "hidden_size": 64,
    "attention_head_size": 4,
    "dropout": 0.1,
    "learning_rate": 1e-3,
    "batch_size": 64,
    "max_epochs": 100,
    "patience": 10,
    "max_encoder_length": 24,    # 24 months lookback
    "max_prediction_length": 6,  # Up to 6 months ahead
}

# ============================================================
# 7. EVALUATION
# ============================================================
FORECAST_HORIZONS = [1, 3, 6]  # months ahead

METRICS = ["MAE", "RMSE", "MAPE", "sMAPE", "MdAPE"]

# Country groups for analysis
COUNTRY_GROUPS = {
    "Western Europe": ["DE", "FR", "NL", "BE", "LU", "AT"],
    "Southern Europe": ["ES", "IT", "PT", "GR", "MT", "CY", "HR", "SI"],
    "Northern Europe": ["SE", "DK", "FI", "EE", "LV", "LT"],
    "Eastern Europe": ["PL", "CZ", "SK", "HU", "RO", "BG"],
}

# ============================================================
# 8. VISUALIZATION
# ============================================================
FIGURE_DPI = 300
FIGURE_FORMAT = "png"

# Color palette for regimes
REGIME_COLORS = {
    0: "#2ecc71",  # Green  - Pre-COVID (Normal)
    1: "#e74c3c",  # Red    - COVID Crisis
    2: "#3498db",  # Blue   - Recovery
}

# Color palette for models
MODEL_COLORS = {
    "SARIMA": "#95a5a6",
    "Prophet": "#9b59b6",
    "XGBoost": "#e67e22",
    "LSTM": "#1abc9c",
    "BiLSTM-Attention": "#e74c3c",
    "TFT": "#3498db",
}
