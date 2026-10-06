"""
Script: compare_eurostat_oecd.py
Mục đích: So sánh đối chuẩn (Benchmark & Discrepancy Analysis) giữa hai nguồn dữ liệu
          độc lập: Eurostat (dữ liệu chính) và OECD (dữ liệu đối chứng trong data_oced).
          KHÔNG gộp hay làm thay đổi bộ dữ liệu Eurostat gốc.
"""

from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent
EUROSTAT_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_europe_expanded.csv"
OECD_CLEAN_CSV = PROJECT_ROOT / "data_oced" / "oecd_tourism_benchmark_clean.csv"
OUTPUT_REPORT_CSV = PROJECT_ROOT / "data_oced" / "comparison_eurostat_vs_oecd.csv"


def run_comparison():
    print("=" * 70)
    print("  ĐỐI CHUẨN DỮ LIỆU ĐỘC LẬP: EUROSTAT vs OECD (TOURISM BENCHMARK)")
    print("=" * 70)

    if not EUROSTAT_CSV.exists():
        print(f"[!] Không tìm thấy dữ liệu Eurostat tại: {EUROSTAT_CSV}")
        return

    if not OECD_CLEAN_CSV.exists():
        print(f"[!] Không tìm thấy dữ liệu OECD tại: {OECD_CLEAN_CSV}")
        print("    Vui lòng chạy 'python fetch_oecd_data.py' trước.")
        return

    # 1. Đọc dữ liệu hai nguồn
    print("[1] Đang nạp dữ liệu Eurostat và OECD...")
    df_euro = pd.read_csv(EUROSTAT_CSV, low_memory=False)
    df_oecd = pd.read_csv(OECD_CLEAN_CSV, low_memory=False)

    # 2. Xử lý thời gian và tổng hợp năm của Eurostat
    # Eurostat đã có sẵn cột 'year', 'geo', 'accommodation_type', 'nights_total'
    df_euro["year"] = df_euro["year"].astype(str)
    
    # Lấy tổng số đêm lưu trú (total nights) theo từng quốc gia theo năm
    euro_annual = (
        df_euro[df_euro["accommodation_type"] == "total"]
        .groupby(["geo", "year"])["nights_total"]
        .sum()
        .reset_index()
        .rename(columns={"geo": "country_code", "nights_total": "eurostat_annual_nights"})
    )

    # 3. Lọc dữ liệu OECD đối với chỉ tiêu số đêm lưu trú (NIGHTS_ACCOM)
    # Filter: Loại hình lưu trú tổng cộng (_T hoặc PRIVATE)
    oecd_nights = (
        df_oecd[df_oecd["oecd_measure"] == "NIGHTS_ACCOM"]
        .dropna(subset=["country_code_2", "oecd_value"])
        .copy()
    )
    oecd_nights["time_period"] = oecd_nights["time_period"].astype(str)
    
    oecd_summary = (
        oecd_nights.groupby(["country_code_2", "time_period"])["oecd_value"]
        .sum()
        .reset_index()
        .rename(columns={"country_code_2": "country_code", "time_period": "year", "oecd_value": "oecd_annual_nights"})
    )

    # 4. Ghép cặp đối sánh tạm thời (KHÔNG lưu đè lên dữ liệu gốc)
    merged = pd.merge(euro_annual, oecd_summary, on=["country_code", "year"], how="inner")
    
    if merged.empty:
        print("[!] Chưa tìm thấy giao điểm thời gian/quốc gia giữa 2 bộ dữ liệu.")
        return

    # 5. Tính toán các chỉ số thống kê đối sánh
    # Sai lệch phần trăm tuyệt đối (Absolute Percentage Discrepancy)
    merged["abs_diff"] = (merged["eurostat_annual_nights"] - merged["oecd_annual_nights"]).abs()
    merged["pct_discrepancy"] = (merged["abs_diff"] / merged["eurostat_annual_nights"]) * 100

    # Lưu báo cáo đối chuẩn
    merged.to_csv(OUTPUT_REPORT_CSV, index=False)
    
    # Hiển thị tổng quan kết quả
    avg_discrepancy = merged["pct_discrepancy"].median()
    corr = merged["eurostat_annual_nights"].corr(merged["oecd_annual_nights"])

    print(f"\n[+] Số cặp quan sát đối sánh được (Nước x Năm): {len(merged):,}")
    print(f"[+] Hệ số tương quan toàn diện (Pearson Correlation): {corr:.4f}")
    print(f"[+] Trung vị độ lệch giữa 2 nguồn (Median Discrepancy): {avg_discrepancy:.2f}%")
    print(f"\n[+] Mẫu 10 bản ghi đối sánh:")
    print(
        merged[["country_code", "year", "eurostat_annual_nights", "oecd_annual_nights", "pct_discrepancy"]]
        .head(10)
        .to_string(index=False)
    )
    print(f"\n-> Báo cáo chi tiết đã lưu tại: {OUTPUT_REPORT_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    run_comparison()
