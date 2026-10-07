"""Read the workbook's prepared annual forecasting table as a standalone dataset."""

from pathlib import Path

import numpy as np
import pandas as pd


WORKBOOK_SHEET = "1Y Forecast"
TARGET_COLUMN = "Tourist_Arrivals"
LAGGED_PREDICTOR_COLUMNS = [
    "Tourist_Arrivals_lag1",
    "Tourism_Revenue_lag1",
    "GDP_lag1",
    "GDP_Growth_lag1",
    "Population_lag1",
    "Inflation_lag1",
    "Unemployment_lag1",
    "Exchange_Rate_lag1",
    "Domestic_Tourism_Trips_lag1",
    "Domestic_Accommodation_Nights_lag1",
]
CATEGORICAL_FEATURE_COLUMNS = ["Country Code", "Region"]
MODEL_FEATURE_COLUMNS = [
    *CATEGORICAL_FEATURE_COLUMNS,
    "Year",
    *LAGGED_PREDICTOR_COLUMNS,
]


def load_world_bank_training_data(workbook_path: Path) -> pd.DataFrame:
    """Load every valid country/area row in the workbook's one-year-ahead table."""
    workbook_path = Path(workbook_path)
    if not workbook_path.is_file():
        raise FileNotFoundError(f"World Bank workbook not found: {workbook_path}")

    data = pd.read_excel(workbook_path, sheet_name=WORKBOOK_SHEET)
    required = [
        "Country Code", "Country", "Region", "Income Group", "Year",
        "Date", TARGET_COLUMN, *LAGGED_PREDICTOR_COLUMNS,
    ]
    missing_columns = [column for column in required if column not in data.columns]
    if missing_columns:
        raise ValueError(
            f"Sheet {WORKBOOK_SHEET!r} is missing required columns: {missing_columns}"
        )

    data["Country Code"] = data["Country Code"].astype(str).str.upper()
    data["Year"] = pd.to_numeric(data["Year"], errors="coerce")
    data[TARGET_COLUMN] = pd.to_numeric(data[TARGET_COLUMN], errors="coerce")
    for column in LAGGED_PREDICTOR_COLUMNS:
        data[column] = pd.to_numeric(data[column], errors="coerce")

    data = data[
        data["Year"].notna()
        & data[TARGET_COLUMN].notna()
        & np.isfinite(data[TARGET_COLUMN])
        & (data[TARGET_COLUMN] >= 0)
    ].copy()
    data["Year"] = data["Year"].astype(int)
    for column in CATEGORICAL_FEATURE_COLUMNS:
        data[column] = data[column].fillna("Unknown").astype(str)

    duplicate_key = data.duplicated(["Country Code", "Year"], keep=False)
    if duplicate_key.any():
        sample = data.loc[duplicate_key, ["Country Code", "Year"]].head().to_dict("records")
        raise ValueError(f"Duplicate country-year targets in {WORKBOOK_SHEET}: {sample}")

    return data.sort_values(["Year", "Country Code"]).reset_index(drop=True)


def regression_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    """Compute interpretable error metrics for annual tourist arrivals."""
    actual = np.asarray(actual, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    error = predicted - actual
    denominator = np.maximum(np.abs(actual), 1.0)
    return {
        "n": int(len(actual)),
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "WAPE_pct": float(np.sum(np.abs(error)) / max(np.sum(np.abs(actual)), 1.0) * 100.0),
        "MAPE_pct": float(np.mean(np.abs(error) / denominator) * 100.0),
    }
