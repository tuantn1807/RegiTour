"""
Execution script to run the full RA-HE-BiLSTM European Tourism Forecasting Pipeline.
Usage:
    python run_system_pipeline.py
"""

from pathlib import Path
import pandas as pd
from src.pipeline.end_to_end_pipeline import TourismForecastingPipeline

def main():
    print("=" * 80)
    print("  KHỞI ĐỘNG HỆ THỐNG TRÍ TUỆ NHÂN TẠO DỰ BÁO DU LỊCH & LƯU TRÚ CHÂU ÂU (RA-HE-BiLSTM)")
    print("  Theo kiến trúc Master Blueprint tại overview_system.md")
    print("=" * 80)

    pipeline = TourismForecastingPipeline()
    output = pipeline.run_pipeline()

    # Save output predictions to results directory
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "forecast_results_ra_he_bilstm.csv"
    output["results_df"].to_csv(out_csv, index=False)
    print(f"\n💾 Đã lưu bảng kết quả dự báo chi tiết tại: {out_csv}")

    # Automatically generate scientific evaluation tables and report
    import evaluate_results
    evaluate_results.main()

    # Automatically run baseline benchmark comparisons
    import benchmark_models
    benchmark_models.main()

    print("🎉 Hoàn tất quy trình huấn luyện, đánh giá và đối sánh mô hình thành công!")

if __name__ == "__main__":
    main()
