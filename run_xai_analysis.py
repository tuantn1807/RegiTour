#!/usr/bin/env python3
"""Explain the World Bank annual tourist-arrivals model with SHAP and permutation tests."""

import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src.features.world_bank import (
    CATEGORICAL_FEATURE_COLUMNS,
    MODEL_FEATURE_COLUMNS,
    TARGET_COLUMN,
    load_world_bank_training_data,
)
from src.pipeline.worldbank_pipeline import (
    DEFAULT_WORKBOOK,
    VALIDATION_END_YEAR,
    VALIDATION_START_YEAR,
    run_world_bank_pipeline,
)


ROOT_DIR = Path(__file__).resolve().parent


def _markdown_table(frame: pd.DataFrame) -> str:
    """Render a compact Markdown table without pandas' optional tabulate dependency."""
    columns = list(frame.columns)

    def cell(value) -> str:
        if isinstance(value, (float, np.floating)):
            return f"{value:.4f}"
        if isinstance(value, (int, np.integer)):
            return str(value)
        return str(value).replace("|", "\\|")

    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join("---" for _ in columns) + " |"
    rows = ["| " + " | ".join(cell(value) for value in row) + " |" for row in frame.itertuples(index=False, name=None)]
    return "\n".join([header, separator, *rows])


def _source_feature(transformed_name: str) -> str:
    """Map a transformed preprocessing column back to its workbook field."""
    name = transformed_name.split("__", 1)[-1]
    if name.startswith("missingindicator_"):
        name = name.removeprefix("missingindicator_")
    for column in CATEGORICAL_FEATURE_COLUMNS:
        if name == column or name.startswith(f"{column}_"):
            return column
    return name


def _build_scenario_sensitivity(model, validation: pd.DataFrame) -> pd.DataFrame:
    """Run small one-variable model sensitivity scenarios, not causal estimates."""
    scenario_types = {
        "Tourism_Revenue_lag1": "relative",
        "GDP_lag1": "relative",
        "GDP_Growth_lag1": "percentage_point",
        "Population_lag1": "relative",
        "Inflation_lag1": "percentage_point",
        "Unemployment_lag1": "percentage_point",
        "Exchange_Rate_lag1": "relative",
        "Domestic_Tourism_Trips_lag1": "relative",
        "Domestic_Accommodation_Nights_lag1": "relative",
    }
    rows = []
    for feature, scenario_type in scenario_types.items():
        if feature not in validation.columns:
            continue
        values = pd.to_numeric(validation[feature], errors="coerce")
        observed = values.notna()
        if not observed.any():
            continue

        base_rows = validation.loc[observed, MODEL_FEATURE_COLUMNS].copy()
        base_prediction = np.maximum(
            np.expm1(np.clip(model.predict(base_rows), -20, 40)), 0.0
        )
        source_values = values.loc[observed].to_numpy(dtype=np.float64)
        for direction in (-1, 1):
            scenario_rows = base_rows.copy()
            if scenario_type == "relative":
                adjusted = np.maximum(source_values * (1.0 + 0.10 * direction), 0.0)
                label = "-10%" if direction < 0 else "+10%"
            else:
                adjusted = source_values + float(direction)
                label = "-1 pp" if direction < 0 else "+1 pp"
            scenario_rows[feature] = pd.to_numeric(
                scenario_rows[feature], errors="coerce"
            ).astype(np.float64)
            scenario_rows.loc[:, feature] = adjusted
            scenario_prediction = np.maximum(
                np.expm1(np.clip(model.predict(scenario_rows), -20, 40)), 0.0
            )
            country_rows = validation.loc[observed]
            base_denominator = np.maximum(np.abs(base_prediction), 1.0)
            for row_number, (_, row) in enumerate(country_rows.iterrows()):
                rows.append({
                    "Country Code": row["Country Code"],
                    "Country": row["Country"],
                    "Year": int(row["Year"]),
                    "Feature": feature,
                    "Scenario": label,
                    "Baseline_Predicted_Arrivals": base_prediction[row_number],
                    "Scenario_Predicted_Arrivals": scenario_prediction[row_number],
                    "Change_Pct": (
                        (scenario_prediction[row_number] - base_prediction[row_number])
                        / base_denominator[row_number] * 100.0
                    ),
                })
    return pd.DataFrame(rows)


def main():
    workbook_path = Path(
        os.environ.get("REGITOUR_WORLD_BANK_WORKBOOK", DEFAULT_WORKBOOK)
    )
    # Train the same World Bank-only GradientBoosting model used by run_system_pipeline.py.
    training_result = run_world_bank_pipeline(workbook_path)
    model = training_result["model"]

    data = load_world_bank_training_data(workbook_path)
    validation = data[
        data["Year"].between(VALIDATION_START_YEAR, VALIDATION_END_YEAR)
    ].copy()
    if validation.empty:
        raise ValueError("No World Bank validation rows are available for XAI analysis.")

    X_validation = validation[MODEL_FEATURE_COLUMNS]
    y_validation_log = np.log1p(validation[TARGET_COLUMN].to_numpy(dtype=np.float64))
    permutation = permutation_importance(
        model,
        X_validation,
        y_validation_log,
        scoring="neg_mean_squared_error",
        n_repeats=5,
        random_state=42,
    )
    permutation_df = pd.DataFrame({
        "Feature": MODEL_FEATURE_COLUMNS,
        "Importance_Mean": permutation.importances_mean,
        "Importance_Std": permutation.importances_std,
    }).sort_values("Importance_Mean", ascending=False)

    xai_dir = ROOT_DIR / "results" / "xai"
    tables_dir = ROOT_DIR / "results" / "tables"
    figure_dir = ROOT_DIR / "paper" / "figures"
    for directory in (xai_dir, tables_dir, figure_dir):
        directory.mkdir(parents=True, exist_ok=True)

    permutation_path = tables_dir / "worldbank_xai_permutation_importance.csv"
    permutation_df.to_csv(permutation_path, index=False)
    permutation_df.rename(columns={
        "Feature": "feature",
        "Importance_Mean": "importance_mean",
        "Importance_Std": "importance_std",
    }).to_csv(tables_dir / "xai_permutation_importance.csv", index=False)

    shap_df = pd.DataFrame(columns=["Feature", "Mean_Abs_SHAP", "Importance_Pct"])
    shap_error = None
    try:
        import shap

        preprocessor = model.named_steps["preprocessor"]
        tree_model = model.named_steps["regressor"]
        transformed = preprocessor.transform(X_validation)
        transformed_names = preprocessor.get_feature_names_out()
        shap_values = np.asarray(shap.TreeExplainer(tree_model).shap_values(transformed))
        if shap_values.ndim == 3:
            shap_values = shap_values[..., 0]
        mean_abs = np.mean(np.abs(shap_values), axis=0)
        grouped = pd.DataFrame({
            "Feature": [_source_feature(str(name)) for name in transformed_names],
            "Mean_Abs_SHAP": mean_abs,
        }).groupby("Feature", as_index=False)["Mean_Abs_SHAP"].sum()
        total = float(grouped["Mean_Abs_SHAP"].sum())
        grouped["Importance_Pct"] = (
            grouped["Mean_Abs_SHAP"] / total * 100.0 if total > 0 else 0.0
        )
        shap_df = grouped.sort_values("Mean_Abs_SHAP", ascending=False)
    except Exception as exc:  # Keep permutation-based XAI usable if SHAP has a runtime issue.
        shap_error = f"{type(exc).__name__}: {exc}"

    shap_path = tables_dir / "worldbank_xai_shap_importance.csv"
    shap_df.to_csv(shap_path, index=False)
    shap_df.rename(columns={
        "Feature": "feature",
        "Mean_Abs_SHAP": "mean_abs_shap",
        "Importance_Pct": "importance_pct",
    }).to_csv(tables_dir / "xai_tree_shap.csv", index=False)

    scenarios = _build_scenario_sensitivity(model, validation)
    scenario_path = tables_dir / "xai_what_if_scenarios.csv"
    scenarios.to_csv(scenario_path, index=False)

    if not shap_df.empty:
        plot_data = shap_df.head(15).sort_values("Importance_Pct")
        values = plot_data["Importance_Pct"]
        x_label = "Mean absolute SHAP importance (%) on log arrivals"
        title = "World Bank Annual Tourist Arrivals — TreeSHAP"
        plot_path = xai_dir / "worldbank_shap_feature_importance.png"
    else:
        plot_data = permutation_df.head(15).sort_values("Importance_Mean")
        values = plot_data["Importance_Mean"]
        x_label = "Permutation importance (increase in log-target MSE)"
        title = "World Bank Annual Tourist Arrivals — Permutation Importance"
        plot_path = xai_dir / "worldbank_permutation_importance.png"

    fig, ax = plt.subplots(figsize=(9, 6), dpi=180)
    ax.barh(plot_data["Feature"], values, color="#2b5c8f", edgecolor="black", alpha=0.85)
    ax.set_xlabel(x_label)
    ax.set_title(title)
    fig.tight_layout()
    fig.savefig(plot_path, bbox_inches="tight")
    legacy_plot_name = (
        "shap_feature_importance.png" if not shap_df.empty
        else "permutation_feature_importance.png"
    )
    legacy_plot_path = xai_dir / legacy_plot_name
    fig.savefig(legacy_plot_path, bbox_inches="tight")
    fig.savefig(figure_dir / plot_path.name, bbox_inches="tight")
    fig.savefig(figure_dir / legacy_plot_name, bbox_inches="tight")
    plt.close(fig)

    report_path = xai_dir / "worldbank_xai_report.md"
    report = [
        "# World Bank Annual Tourism Forecast — XAI Report",
        "",
        f"- Workbook: `{workbook_path}`",
        f"- Target: `{TARGET_COLUMN}` (annual international tourist arrivals)",
        f"- Country/area codes in workbook scope: {data['Country Code'].nunique()}",
        f"- Validation period: {VALIDATION_START_YEAR}–{VALIDATION_END_YEAR}",
        f"- Validation rows explained: {len(validation):,}",
        "- The model predicts `log1p(Tourist_Arrivals)`; SHAP values are in that model-output space.",
        "- Feature attributions describe this fitted model and are not causal effects.",
        "",
        "## Permutation importance",
        "",
        _markdown_table(permutation_df),
        "",
        "## TreeSHAP by source variable",
        "",
    ]
    if not shap_df.empty:
        report.append(_markdown_table(shap_df))
    else:
        report.extend([
            "TreeSHAP could not be computed in this environment; the permutation results and chart are available.",
            f"Runtime detail: `{shap_error}`",
        ])
    report.extend([
        "",
        "## One-variable scenario sensitivity",
        "",
        "The scenarios perturb one available lagged predictor at a time. They are model sensitivity checks, not causal policy effects.",
        "",
        _markdown_table(scenarios.head(12)) if not scenarios.empty else "No complete scenario rows were available.",
        "",
        "The World Bank workbook contains annual arrivals but no accommodation-night series, HICP accommodation prices, or attention weights.",
        "The former Eurostat-only night/price tables and synthetic attention heatmap are not regenerated.",
        "Existing files such as `xai_price_elasticity.csv` or `attention_cross_temporal_heatmap.png` may remain from an earlier Eurostat run and are not current World Bank outputs.",
        "",
        f"![Feature importance]({plot_path.name})",
        "",
    ])
    report_text = "\n".join(report)
    report_path.write_text(report_text, encoding="utf-8")
    legacy_report_path = ROOT_DIR / "results" / "xai_report.md"
    legacy_report_path.write_text(
        report_text.replace(f"({plot_path.name})", f"(xai/{legacy_plot_name})"),
        encoding="utf-8",
    )

    print(f"Workbook: {workbook_path}")
    print(f"Permutation importance: {permutation_path}")
    print(f"Familiar TreeSHAP table: {tables_dir / 'xai_tree_shap.csv'}")
    print(f"Familiar scenario table: {scenario_path}")
    print(f"TreeSHAP table: {shap_path}")
    print(f"Importance figure: {legacy_plot_path}")
    print(f"Familiar report: {legacy_report_path}")
    print(f"Detailed report: {report_path}")
    if shap_error:
        print(f"TreeSHAP unavailable; permutation analysis completed. Detail: {shap_error}")


if __name__ == "__main__":
    main()
