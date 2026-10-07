"""Train and evaluate an annual international tourist-arrivals model from the WB workbook."""

import os
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

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


def run_world_bank_pipeline(
    workbook_path: Optional[Union[str, Path]] = None,
) -> dict:
    """Fit on the World Bank workbook only; do not ingest or join the Eurostat panel."""
    if workbook_path is None:
        workbook_path = os.environ.get("REGITOUR_WORLD_BANK_WORKBOOK", DEFAULT_WORKBOOK)
    workbook_path = Path(workbook_path)

    print("=" * 76)
    print("WORLD BANK / UN TOURISM — ANNUAL INTERNATIONAL ARRIVALS FORECAST")
    print(f"Source workbook: {workbook_path}")
    print("=" * 76)

    data = load_world_bank_training_data(workbook_path)
    train = data[data["Year"] <= TRAIN_END_YEAR].copy()
    validation = data[
        data["Year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)
    ].copy()
    test = data[data["Year"] >= TEST_START_YEAR].copy()
    if train.empty or validation.empty or test.empty:
        raise ValueError(
            "The workbook does not contain enough labeled rows for the configured "
            "1995–2015 train, 2016–2017 validation, and 2018+ test split."
        )

    print(
        f"Rows: {len(data):,} | countries: {data['Country Code'].nunique()} | "
        f"years: {data['Year'].min()}–{data['Year'].max()}"
    )
    print(
        f"Split: train {len(train):,} (up to {TRAIN_END_YEAR}), "
        f"validation {len(validation):,} ({VALIDATION_START_YEAR}-{VALIDATION_END_YEAR}), "
        f"test {len(test):,} (from {TEST_START_YEAR})"
    )
    print(
        "Predictors: annual one-year lags from the workbook plus country and region; "
        "the target is Tourist_Arrivals."
    )

    numeric_features = [
        column for column in MODEL_FEATURE_COLUMNS
        if column not in CATEGORICAL_FEATURE_COLUMNS
    ]

    def make_model(n_estimators: int) -> Pipeline:
        preprocessor = ColumnTransformer(
            transformers=[
                (
                    "numeric",
                    SimpleImputer(strategy="median", add_indicator=True),
                    numeric_features,
                ),
                (
                    "categorical",
                    OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                    CATEGORICAL_FEATURE_COLUMNS,
                ),
            ],
            remainder="drop",
        )
        return Pipeline([
            ("preprocessor", preprocessor),
            (
                "regressor",
                GradientBoostingRegressor(
                    n_estimators=n_estimators,
                    learning_rate=0.03,
                    max_depth=3,
                    min_samples_leaf=10,
                    loss="huber",
                    random_state=42,
                ),
            ),
        ])

    y_train_log = np.log1p(train[TARGET_COLUMN].to_numpy(dtype=np.float64))
    y_validation_log = np.log1p(validation[TARGET_COLUMN].to_numpy(dtype=np.float64))

    # Select the boosting iteration on later validation years, never on the test period.
    selection_model = make_model(n_estimators=600)
    selection_model.fit(train[MODEL_FEATURE_COLUMNS], y_train_log)
    validation_matrix = selection_model.named_steps["preprocessor"].transform(
        validation[MODEL_FEATURE_COLUMNS]
    )
    best_iteration = 1
    best_validation_rmse = float("inf")
    stale_iterations = 0
    for iteration, validation_prediction in enumerate(
        selection_model.named_steps["regressor"].staged_predict(validation_matrix),
        start=1,
    ):
        rmse = float(np.sqrt(np.mean((y_validation_log - validation_prediction) ** 2)))
        if rmse < best_validation_rmse:
            best_iteration = iteration
            best_validation_rmse = rmse
            stale_iterations = 0
        else:
            stale_iterations += 1
        if iteration >= 100 and stale_iterations >= 50:
            break

    model = make_model(n_estimators=best_iteration)
    model.fit(train[MODEL_FEATURE_COLUMNS], y_train_log)

    validation_prediction = np.maximum(
        np.expm1(model.predict(validation[MODEL_FEATURE_COLUMNS])), 0.0
    )
    log_predictions = model.predict(test[MODEL_FEATURE_COLUMNS])
    predictions = np.maximum(np.expm1(log_predictions), 0.0)
    actual = test[TARGET_COLUMN].to_numpy(dtype=np.float64)

    persistence = test["Tourist_Arrivals_lag1"].to_numpy(dtype=np.float64)
    persistence_mask = np.isfinite(persistence)
    metrics_rows = [
        {
            "Split": "validation",
            "Model": "GradientBoosting log-target",
            **regression_metrics(
                validation[TARGET_COLUMN].to_numpy(dtype=np.float64),
                validation_prediction,
            ),
        },
        {
            "Split": "test",
            "Model": "GradientBoosting log-target (all test rows)",
            **regression_metrics(actual, predictions),
        },
    ]
    if persistence_mask.any():
        metrics_rows.extend([
            {
                "Split": "test",
                "Model": "GradientBoosting (persistence-comparable rows)",
                **regression_metrics(actual[persistence_mask], predictions[persistence_mask]),
            },
            {
                "Split": "test",
                "Model": "One-year persistence baseline",
                **regression_metrics(actual[persistence_mask], persistence[persistence_mask]),
            },
        ])
    metrics = pd.DataFrame(metrics_rows)

    prediction_columns = [
        "Country Code", "Country", "Region", "Income Group", "Year", "Date",
        TARGET_COLUMN, "Tourist_Arrivals_lag1",
    ]
    test_predictions = test[prediction_columns].copy()
    test_predictions["Predicted_Tourist_Arrivals"] = predictions
    test_predictions["Absolute_Error"] = np.abs(actual - predictions)
    test_predictions["Persistence_Prediction"] = persistence
    test_predictions = test_predictions.sort_values(["Year", "Country Code"])
    compatibility_outputs = write_world_bank_forecast_outputs(
        test_predictions,
        model_label="World Bank Gradient Boosting",
        model_slug="gradientboost",
        legacy_filename="forecast_results_ra_he_bilstm.csv",
    )

    yearly_metrics = []
    for year, group in test_predictions.groupby("Year", sort=True):
        yearly_metrics.append({"Year": int(year), **regression_metrics(
            group[TARGET_COLUMN].to_numpy(),
            group["Predicted_Tourist_Arrivals"].to_numpy(),
        )})
    yearly_metrics = pd.DataFrame(yearly_metrics)

    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    models_dir = results_dir / "models"
    for directory in (results_dir, tables_dir, models_dir):
        directory.mkdir(parents=True, exist_ok=True)

    predictions_path = results_dir / "worldbank_tourist_arrivals_test_predictions.csv"
    metrics_path = tables_dir / "worldbank_tourist_arrivals_metrics.csv"
    yearly_path = tables_dir / "worldbank_tourist_arrivals_metrics_by_year.csv"
    importance_path = tables_dir / "worldbank_tourist_arrivals_feature_importance.csv"
    model_path = models_dir / "worldbank_tourist_arrivals.joblib"

    test_predictions.to_csv(predictions_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    yearly_metrics.to_csv(yearly_path, index=False)
    importances = permutation_importance(
        model,
        validation[MODEL_FEATURE_COLUMNS],
        y_validation_log,
        scoring="neg_mean_squared_error",
        n_repeats=3,
        random_state=42,
    )
    pd.DataFrame({
        "Feature": MODEL_FEATURE_COLUMNS,
        "Importance_Mean": importances.importances_mean,
        "Importance_Std": importances.importances_std,
    }).sort_values("Importance_Mean", ascending=False).to_csv(importance_path, index=False)
    joblib.dump(model, model_path)

    print("\nTest metrics (raw arrivals scale):")
    print(metrics.to_string(index=False, float_format=lambda value: f"{value:,.3f}"))
    print(f"\nBest boosting iteration: {best_iteration}")
    print(f"Predictions: {predictions_path}")
    print(f"Metrics: {metrics_path}")
    print(f"Feature importance: {importance_path}")
    print(f"Saved model: {model_path}")
    print(f"Familiar forecast CSV: {compatibility_outputs['legacy_predictions']}")
    print(f"Evaluation report: {compatibility_outputs['report']}")
    print(f"Model comparison table: {compatibility_outputs['comparison_outputs']['table']}")
    print(f"Model comparison chart: {compatibility_outputs['comparison_outputs']['chart']}")
    return {
        "model": model,
        "predictions": test_predictions,
        "metrics": metrics,
        "yearly_metrics": yearly_metrics,
        "model_path": model_path,
        "compatibility_outputs": compatibility_outputs,
    }
