"""
Execution script to run the Unified RA-HE-DLinear Tourism Forecasting Pipeline.
Supports both:
  1. Eurostat European Regional Monthly Panel (38 countries, 4 accommodation types)
  2. World Bank Global Macroeconomic Annual Panel (203 countries)

Usage:
    python run_linear_pipeline.py --dataset eurostat
    python run_linear_pipeline.py --dataset worldbank
    python run_linear_pipeline.py --dataset all
"""

import argparse
from pathlib import Path
import pandas as pd
from src.pipeline.linear_pipeline import TourismLinearForecastingPipeline


def run_single_dataset(dataset_name: str, results_dir: Path) -> dict:
    print("\n" + "#" * 80)
    print(f"  KHỞI ĐỘNG RA-HE-DLinear TRÊN TẬP DỮ LIỆU: {dataset_name.upper()}")
    print("#" * 80)

    pipeline = TourismLinearForecastingPipeline(dataset=dataset_name)
    output = pipeline.run_pipeline()

    out_csv = results_dir / f"forecast_results_ra_he_dlinear_{dataset_name}.csv"
    output["results_df"].to_csv(out_csv, index=False)
    print(f"\n💾 Đã lưu bảng kết quả chi tiết tại: {out_csv}")
    return output


def main():
    parser = argparse.ArgumentParser(description="Run Unified RA-HE-DLinear Pipeline")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eurostat",
        choices=["eurostat", "worldbank", "all"],
        help="Target dataset: 'eurostat', 'worldbank', or 'all'",
    )
    args = parser.parse_args()

    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    tables_dir = results_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)

    summary_rows = []

    if args.dataset in ["eurostat", "all"]:
        res_eur = run_single_dataset("eurostat", results_dir)
        summary_rows.append({
            "Dataset": "Eurostat (Monthly Regional)",
            "Frequency": "Monthly (H=12, L=24)",
            "Entities": "38 Countries, 4 NACE Types",
            "Test Period": "2024 - 2026",
            "Arrivals WAPE (%)": res_eur["metrics_arrivals"]["WAPE"],
            "Arrivals MAE": res_eur["metrics_arrivals"]["MAE"],
            "Nights WAPE (%)": res_eur["metrics_nights"]["WAPE"],
            "Nights MAE": res_eur["metrics_nights"]["MAE"],
        })

    if args.dataset in ["worldbank", "all"]:
        res_wb = run_single_dataset("worldbank", results_dir)
        summary_rows.append({
            "Dataset": "World Bank (Annual Global)",
            "Frequency": "Annual (H=1, L=4)",
            "Entities": "203 Economies",
            "Test Period": "2018 - 2020",
            "Arrivals WAPE (%)": res_wb["metrics_arrivals"]["WAPE"],
            "Arrivals MAE": res_wb["metrics_arrivals"]["MAE"],
            "Nights WAPE (%)": res_wb["metrics_nights"]["WAPE"],
            "Nights MAE": res_wb["metrics_nights"]["MAE"],
        })

    if len(summary_rows) > 0:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = tables_dir / "unified_cross_dataset_benchmark.csv"
        summary_df.to_csv(summary_path, index=False)
        print("\n" + "=" * 80)
        print("🏆 [TỔNG KẾT ĐỐI SÁNH KHOA HỌC RA-HE-DLinear TRÊN CẢ 2 BỘ DỮ LIỆU]")
        print("=" * 80)
        print(summary_df.to_string(index=False))
        print(f"\n💾 Đã lưu bảng tổng kết đối sánh tại: {summary_path}")
        print("=" * 80)


if __name__ == "__main__":
    main()
