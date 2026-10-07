"""Create familiar project result files for annual World Bank forecasts."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.features.world_bank import TARGET_COLUMN, regression_metrics


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def write_world_bank_model_comparison() -> dict:
    """Combine saved full-workbook test scores into one table and WAPE chart."""
    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    tables_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    model_sources = [
        ("worldbank_ridge_overall_metrics.csv", "Linear (Ridge)"),
        ("worldbank_gradientboost_overall_metrics.csv", "Gradient Boosting"),
    ]
    comparison_rows = []
    baseline_row = None
    for filename, display_name in model_sources:
        source = tables_dir / filename
        if not source.is_file():
            continue
        saved = pd.read_csv(source)
        test_rows = saved[saved["Split"].eq("test")]
        model_rows = test_rows[~test_rows["Model"].str.contains("persistence baseline", case=False)]
        if not model_rows.empty:
            row = model_rows.iloc[0]
            comparison_rows.append({
                "Model": display_name,
        "Split": "test (2018-2020)",
                "n": int(row["n"]),
                "MAE": float(row["MAE"]),
                "RMSE": float(row["RMSE"]),
                "WAPE_pct": float(row["WAPE_pct"]),
                "MAPE_pct": float(row["MAPE_pct"]),
            })
        baseline_rows = test_rows[test_rows["Model"].str.contains("persistence baseline", case=False)]
        if baseline_row is None and not baseline_rows.empty:
            baseline_row = baseline_rows.iloc[0]

    if baseline_row is not None:
        comparison_rows.append({
            "Model": "Previous-year persistence baseline",
            "Split": "test (2018-2020)",
            "n": int(baseline_row["n"]),
            "MAE": float(baseline_row["MAE"]),
            "RMSE": float(baseline_row["RMSE"]),
            "WAPE_pct": float(baseline_row["WAPE_pct"]),
            "MAPE_pct": float(baseline_row["MAPE_pct"]),
        })

    comparison = pd.DataFrame(comparison_rows)
    comparison_path = tables_dir / "benchmark_model_comparison.csv"
    comparison.to_csv(comparison_path, index=False)

    chart_path = figures_dir / "benchmark_model_comparison_wape.png"
    report_path = results_dir / "model_comparison_report.md"
    if not comparison.empty:
        fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
        bars = ax.bar(
            comparison["Model"], comparison["WAPE_pct"],
            color=["#4776a8", "#40a07b", "#999999"][:len(comparison)],
            edgecolor="black",
        )
        ax.set_ylabel("WAPE (%) - lower is better")
        ax.set_title("World Bank annual tourist arrivals: test-set comparison")
        ax.tick_params(axis="x", rotation=12)
        ax.grid(axis="y", alpha=0.25)
        for bar, score in zip(bars, comparison["WAPE_pct"]):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), f"{score:.1f}%",
                    ha="center", va="bottom")
        fig.tight_layout()
        fig.savefig(chart_path, bbox_inches="tight")
        plt.close(fig)

        table_lines = [
            "| Model | Test records | MAE | RMSE | WAPE | MAPE |",
            "| --- | ---: | ---: | ---: | ---: | ---: |",
        ]
        for row in comparison.itertuples(index=False):
            table_lines.append(
                f"| {row.Model} | {row.n:,} | {row.MAE:,.0f} | {row.RMSE:,.0f} | "
                f"{row.WAPE_pct:.1f}% | {row.MAPE_pct:.1f}% |"
            )
        report_path.write_text(
            "# World Bank full-country model comparison\n\n"
            "All models are evaluated on the same annual 2018–2020 test rows. "
            "WAPE is calculated on raw tourist-arrival counts; lower is better.\n\n"
            + "\n".join(table_lines)
            + "\n\n![Test-set WAPE comparison](figures/benchmark_model_comparison_wape.png)\n",
            encoding="utf-8",
        )

    return {"table": comparison_path, "chart": chart_path, "report": report_path}


def _compatibility_view(predictions: pd.DataFrame, model_label: str) -> pd.DataFrame:
    """Give annual arrivals predictions stable, easy-to-find output column names."""
    output = predictions.copy()
    if "Date" in output.columns:
        output["date"] = pd.to_datetime(output["Date"], errors="coerce")
        output = output.drop(columns=["Date"])
    elif "Year" in output.columns:
        output["date"] = pd.to_datetime(output["Year"].astype(str) + "-01-01")

    output = output.rename(columns={
        "Country Code": "geo",
        "Country": "country",
        "Region": "region",
        "Income Group": "income_group",
        "Year": "year",
        TARGET_COLUMN: "arrivals_total",
        "Tourist_Arrivals_lag1": "arrivals_lag_1",
        "Predicted_Tourist_Arrivals": "pred_arrivals_point",
        "Absolute_Error": "arrivals_absolute_error",
        "Persistence_Prediction": "pred_arrivals_persistence",
    })
    output["Model"] = model_label
    first_columns = [
        "date", "year", "geo", "country", "region", "income_group",
        "arrivals_total", "pred_arrivals_point", "arrivals_absolute_error",
        "arrivals_lag_1", "pred_arrivals_persistence", "Model",
    ]
    ordered = [column for column in first_columns if column in output.columns]
    ordered.extend(column for column in output.columns if column not in ordered)
    sort_columns = [c for c in ("year", "geo") if c in output.columns]
    return output[ordered].sort_values(sort_columns)


def write_world_bank_forecast_outputs(
    predictions: pd.DataFrame,
    model_label: str,
    model_slug: str,
    legacy_filename: str,
) -> dict:
    """Write prediction CSVs, comparable annual summaries, a chart, and a report."""
    results_dir = PROJECT_ROOT / "results"
    tables_dir = results_dir / "tables"
    figures_dir = results_dir / "figures"
    for directory in (results_dir, tables_dir, figures_dir):
        directory.mkdir(parents=True, exist_ok=True)

    output = _compatibility_view(predictions, model_label)
    legacy_path = results_dir / legacy_filename
    named_path = results_dir / f"forecast_results_worldbank_{model_slug}.csv"
    output.to_csv(legacy_path, index=False)
    output.to_csv(named_path, index=False)

    actual = predictions[TARGET_COLUMN].to_numpy(dtype=np.float64)
    predicted = predictions["Predicted_Tourist_Arrivals"].to_numpy(dtype=np.float64)
    overall_rows = [{
        "Split": "test",
        "Model": model_label,
        **regression_metrics(actual, predicted),
    }]
    baseline_col = "Persistence_Prediction"
    if baseline_col in predictions.columns:
        baseline = predictions[baseline_col].to_numpy(dtype=np.float64)
        comparable = np.isfinite(baseline)
        if comparable.any():
            overall_rows.append({
                "Split": "test",
                "Model": "One-year persistence baseline",
                **regression_metrics(actual[comparable], baseline[comparable]),
            })
    overall = pd.DataFrame(overall_rows)

    by_country = []
    group_column = "Country Code"
    for code, group in predictions.groupby(group_column, sort=True):
        by_country.append({
            "Country Code": code,
            "Country": group["Country"].iloc[0],
            **regression_metrics(
                group[TARGET_COLUMN].to_numpy(),
                group["Predicted_Tourist_Arrivals"].to_numpy(),
            ),
        })
    country_metrics = pd.DataFrame(by_country).sort_values("WAPE_pct", ascending=False)

    by_year = []
    for year, group in predictions.groupby("Year", sort=True):
        by_year.append({
            "Year": int(year),
            **regression_metrics(
                group[TARGET_COLUMN].to_numpy(),
                group["Predicted_Tourist_Arrivals"].to_numpy(),
            ),
        })
    year_metrics = pd.DataFrame(by_year)

    overall_path = tables_dir / "overall_metrics.csv"
    country_path = tables_dir / "metrics_by_country.csv"
    year_path = tables_dir / "metrics_by_year.csv"
    overall.to_csv(overall_path, index=False)
    country_metrics.to_csv(country_path, index=False)
    year_metrics.to_csv(year_path, index=False)
    # Keep model-specific copies so running the other pipeline does not erase these summaries.
    overall.to_csv(tables_dir / f"worldbank_{model_slug}_overall_metrics.csv", index=False)
    country_metrics.to_csv(tables_dir / f"worldbank_{model_slug}_metrics_by_country.csv", index=False)
    year_metrics.to_csv(tables_dir / f"worldbank_{model_slug}_metrics_by_year.csv", index=False)
    comparison_outputs = write_world_bank_model_comparison()

    figure_path = figures_dir / f"worldbank_{model_slug}_actual_vs_predicted_by_year.png"
    actual_by_year = predictions.groupby("Year")[TARGET_COLUMN].sum()
    predicted_by_year = predictions.groupby("Year")["Predicted_Tourist_Arrivals"].sum()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=180)
    ax.plot(actual_by_year.index, actual_by_year.values, marker="o", label="Actual")
    ax.plot(predicted_by_year.index, predicted_by_year.values, marker="o", label=model_label)
    ax.set_xlabel("Year")
    ax.set_ylabel("International tourist arrivals")
    ax.set_title(f"World Bank country/area codes: actual vs {model_label}")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(figure_path, bbox_inches="tight")
    plt.close(fig)

    report_path = results_dir / "evaluation_report.md"
    metrics_table = overall.to_string(index=False, float_format=lambda value: f"{value:,.3f}")
    country_count = predictions["Country Code"].nunique()
    report = f"""# Annual World Bank tourism forecast evaluation

- Model: {model_label}
- Scope: all {country_count} country/area codes represented in the World Bank test data
- Test years: {int(predictions['Year'].min())}–{int(predictions['Year'].max())}
- Test records: {len(predictions):,}

## Overall test metrics

```text
{metrics_table}
```

## Output files

- Forecast rows: `{legacy_path.name}`
- Metrics by country: `tables/{country_path.name}`
- Metrics by year: `tables/{year_path.name}`
- Actual vs predicted chart: `figures/{figure_path.name}`

The workbook provides annual arrivals only. It has no monthly seasonality, accommodation categories,
or accommodation nights, so this report does not include those Eurostat-only breakdowns or interval coverage.
Any existing `metrics_by_accommodation.csv` from an earlier Eurostat run is left untouched and is not a current World Bank result.
"""
    report_path.write_text(report, encoding="utf-8")

    return {
        "legacy_predictions": legacy_path,
        "named_predictions": named_path,
        "overall_metrics": overall_path,
        "country_metrics": country_path,
        "year_metrics": year_path,
        "figure": figure_path,
        "report": report_path,
        "comparison_outputs": comparison_outputs,
    }
