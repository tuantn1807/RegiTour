"""
Execution script to run the full RA-HE-DLinear European Tourism Forecasting Pipeline.
Usage:
    python run_linear_pipeline.py
"""

from pathlib import Path
import pandas as pd
from src.pipeline.linear_pipeline import TourismLinearForecastingPipeline

def main():
    print("=" * 80)
    print("  KHỞI ĐỘNG HỆ THỐNG TRÍ TUỆ NHÂN TẠO DỰ BÁO DU LỊCH & LƯU TRÚ CHÂU ÂU (RA-HE-DLinear)")
    print("  Kiến trúc Modern Linear-Centric (Không chồng chéo, Tinh giản, Green AI)")
    print("=" * 80)

    pipeline = TourismLinearForecastingPipeline()
    output = pipeline.run_pipeline()

    # Save output predictions to results directory
    results_dir = Path(__file__).resolve().parent / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_csv = results_dir / "forecast_results_ra_he_dlinear.csv"
    output["results_df"].to_csv(out_csv, index=False)
    print(f"\n💾 Đã lưu bảng kết quả dự báo chi tiết tại: {out_csv}")
    print("🎉 Hoàn tất quy trình huấn luyện, đánh giá và đối chứng thành công!")

if __name__ == "__main__":
    main()
