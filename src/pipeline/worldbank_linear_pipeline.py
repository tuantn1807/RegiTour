"""Annual World Bank tourist-arrivals forecasting with a regularized linear model."""

import os
from pathlib import Path
from typing import Optional, Union

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features.world_bank import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_world_bank_training_data,
    regression_metrics,
)
from src.pipeline.worldbank_outputs import write_world_bank_forecast_outputs


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_WORKBOOK = PROJECT_ROOT / "data" / "raw" / "tourism_economics_worldbank-3.xlsx"
TRAIN_END_YEAR = 2015
VALIDATION_START_YEAR = 2016
VALIDATION_END_YEAR = 2017
TEST_START_YEAR = 2018


def run_world_bank_linear_pipeline(
    workbook_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Train and evaluate a linear annual arrivals model on the WB workbook only."""
    if workbook_path is None:
        workbook_path = os.environ.get("REGITOUR_WORLD_BANK_WORKBOOK", DEFAULT_WORKBOOK)
    workbook_path = Path(workbook_path)

    data = load_world_bank_training_data(workbook_path)
    train = data[data["Year"] <= TRAIN_END_YEAR].copy()
    validation = data[
        data["Year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)
    ].copy()
    test = data[data["Year"] >= TEST_START_YEAR].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The workbook must contain train rows through 2015, validation rows "
            "for 2016–2017, and test rows from 2018 onward."
        )

    numeric_features = [
        column for column in MODEL_FEATURE_COLUMNS
        if column not in CATEGORICAL_FEATURE_COLUMNS
    ]

    def make_model(alpha: float) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    Pipeline([
                        (
                            "imputer",
                            SimpleImputer(
                                strategy="median",
                                add_indicator=True,
                                keep_empty_features=True,
                            ),
                        ),
                        ("scaler", StandardScaler()),
                    ]),
                    numeric_features,
                ),
                (
                    "categorical",
                    Pipeline([
                        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
                        ("onehot", OneHotEncoder(handle_unknown="ignore")),
                    ]),
                    CATEGORICAL_FEATURE_COLUMNS,
                ),
            ],
            remainder="drop",
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            ("regressor", Ridge(alpha=alpha)),
        ])

    y_train_log = np.log1p(train[TARGET_COLUMN].to_numpy(dtype=np.float64))
    y_validation_log = np.log1p(validation[TARGET_COLUMN].to_numpy(dtype=np.float64))

    best_alpha = None
    best_validation_rmse = float("inf")
    for alpha in (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0):
        candidate = make_model(alpha)
        candidate.fit(train[MODEL_FEATURE_COLUMNS], y_train_log)
        validation_log_prediction = candidate.predict(validation[MODEL_FEATURE_COLUMNS])
        rmse = float(np.sqrt(np.mean((y_validation_log - validation_log_prediction) ** 2)))
        if rmse < best_validation_rmse:
            best_alpha = alpha
            best_validation_rmse = rmse

    # Select regularization only on validation, then refit through 2017 before testing.
    fit_data = pd.concat([train, validation], ignore_index=True)
    model = make_model(best_alpha)
    model.fit(fit_data[MODEL_FEATURE_COLUMNS], np.log1p(fit_data[TARGET_COLUMN]))

    validation_model = make_model(best_alpha)
    validation_model.fit(train[MODEL_FEATURE_COLUMNS], y_train_log)
    validation_prediction = np.maximum(
        np.expm1(np.clip(validation_model.predict(validation[MODEL_FEATURE_COLUMNS]), -20, 40)),
        0.0,
    )
    test_prediction = np.maximum(
        np.expm1(np.clip(model.predict(test[MODEL_FEATURE_COLUMNS]), -20, 40)),
        0.0,
    )
    test_actual = test[TARGET_COLUMN].to_numpy(dtype=np.float64)
    persistence = test["Tourist_Arrivals_lag1"].to_numpy(dtype=np.float64)
    persistence_mask = np.isfinite(persistence)

    metrics_rows = [
        {
            "Split": "validation",
            "Model": "Ridge log-target",
            **regression_metrics(
                validation[TARGET_COLUMN].to_numpy(dtype=np.float64),
                validation_prediction,
            ),
        },
        {
            "Split": "test",
            "Model": "Ridge log-target (all test rows)",
            **regression_metrics(test_actual, test_prediction),
        },
    ]
    if persistence_mask.any():
        metrics_rows.extend([
            {
                "Split": "test",
                "Model": "Ridge (persistence-comparable rows)",
                **regression_metrics(test_actual[persistence_mask], test_prediction[persistence_mask]),
            },
            {
                "Split": "test",
                "Model": "One-year persistence baseline",
                **regression_metrics(test_actual[persistence_mask], persistence[persistence_mask]),
            },
        ])
    metrics = pd.DataFrame(metrics_rows)

    prediction_columns = [
        "Country Code", "Country", "Region", "Income Group", "Year", "Date",
        TARGET_COLUMN, "Tourist_Arrivals_lag1",
    ]
    predictions = test[prediction_columns].copy()
    predictions["Predicted_Tourist_Arrivals"] = test_prediction
    predictions["Absolute_Error"] = np.abs(test_actual - test_prediction)
    predictions["Persistence_Prediction"] = persistence
    predictions = predictions.sort_values(["Year", "Country Code"])
    compatibility_outputs = write_world_bank_forecast_outputs(
        predictions,
        model_label="World Bank Ridge Linear",
        model_slug="ridge",
        legacy_filename="forecast_results_ra_he_dlinear.csv",
    )

    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    models_dir = results_dir / "models"
    for directory in (results_dir, tables_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "worldbank_linear_tourist_arrivals_test_predictions.csv"
    metrics_path = tables_dir / "worldbank_linear_tourist_arrivals_metrics.csv"
    model_path = models_dir / "worldbank_linear_tourist_arrivals.joblib"
    predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    joblib.dump(model, model_path)

    print("=" * 76)
    print("WORLD BANK / UN TOURISM — ANNUAL LINEAR ARRIVALS FORECAST")
    print(f"Workbook: {workbook_path}")
    print(
        f"Rows: {len(data):,} | countries: {data['Country Code'].nunique()} | "
        f"years: {data['Year'].min()}–{data['Year'].max()}"
    )
    print(f"Selected Ridge alpha: {best_alpha:g} (validation log-RMSE {best_validation_rmse:.4f})")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Saved model: {model_path}")
    print(f"Familiar forecast CSV: {compatibility_outputs['legacy_predictions']}")
    print(f"Evaluation report: {compatibility_outputs['report']}")
    print(f"Model comparison table: {compatibility_outputs['comparison_outputs']['table']}")
    print(f"Model comparison chart: {compatibility_outputs['comparison_outputs']['chart']}")

    return {
        "model": model,
        "predictions": predictions,
        "metrics": metrics,
        "model_path": model_path,
        "best_alpha": best_alpha,
        "compatibility_outputs": compatibility_outputs,
    }
