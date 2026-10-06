"""
Script: build_expanded_dataset.py
Purpose: Tự động tải, trích xuất và xây dựng Dataset Du lịch & Lưu trú Châu Âu mở rộng
         kết hợp đồng thời CẤP ĐỘ 1 (Phân rã 4 loại hình lưu trú NACE: hotel, short_stay, camping, total)
         và CẤP ĐỘ 2 (Mở rộng 39 quốc gia: 27 nước EU + 12 nước Châu Âu lân cận).
         Thời gian: 2000 - 2025 (26 năm = 312 tháng).
         Tổng số bản ghi: 39 nước x 312 tháng x 4 loại hình = 48.672 records.
Sources: Eurostat (tour_occ_arm, tour_occ_nim, tour_occ_mnor)
Outputs: tourism_europe_expanded.xlsx và data/processed/tourism_europe_expanded.csv
"""

import io
import gzip
import time
import requests
import pandas as pd
import numpy as np
from pathlib import Path

# ============================================================
# 1. CẤU HÌNH THỜI GIAN, QUỐC GIA & PHÂN LOẠI NACE
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_XLSX = PROJECT_ROOT / "tourism_europe_expanded.xlsx"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_europe_expanded.csv"
OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)

# Cấu hình giai đoạn tối ưu theo Phương án 4:
# Từ 2010 đến các tháng mới nhất của năm 2026 (hiện tại đến 2026-07, tự động bắt 2026-08 nếu có)
START_YEAR = 2010
START_MONTH = "2010-01"

# 38 Quốc gia Châu Âu (Đã loại bỏ Moldova - MD do 0% dữ liệu báo cáo trên Eurostat)
COUNTRY_MAP = {
    # 27 Nước thành viên Liên minh Châu Âu (EU27)
    "AT": {"country": "Austria", "country_vi": "Áo"},
    "BE": {"country": "Belgium", "country_vi": "Bỉ"},
    "BG": {"country": "Bulgaria", "country_vi": "Bulgaria"},
    "CY": {"country": "Cyprus", "country_vi": "Síp"},
    "CZ": {"country": "Czechia", "country_vi": "Séc"},
    "DE": {"country": "Germany", "country_vi": "Đức"},
    "DK": {"country": "Denmark", "country_vi": "Đan Mạch"},
    "EE": {"country": "Estonia", "country_vi": "Estonia"},
    "EL": {"country": "Greece", "country_vi": "Hy Lạp"},
    "ES": {"country": "Spain", "country_vi": "Tây Ban Nha"},
    "FI": {"country": "Finland", "country_vi": "Phần Lan"},
    "FR": {"country": "France", "country_vi": "Pháp"},
    "HR": {"country": "Croatia", "country_vi": "Croatia"},
    "HU": {"country": "Hungary", "country_vi": "Hungary"},
    "IE": {"country": "Ireland", "country_vi": "Ireland"},
    "IT": {"country": "Italy", "country_vi": "Ý"},
    "LT": {"country": "Lithuania", "country_vi": "Litva"},
    "LU": {"country": "Luxembourg", "country_vi": "Luxembourg"},
    "LV": {"country": "Latvia", "country_vi": "Latvia"},
    "MT": {"country": "Malta", "country_vi": "Malta"},
    "NL": {"country": "Netherlands", "country_vi": "Hà Lan"},
    "PL": {"country": "Poland", "country_vi": "Ba Lan"},
    "PT": {"country": "Portugal", "country_vi": "Bồ Đào Nha"},
    "RO": {"country": "Romania", "country_vi": "Romania"},
    "SE": {"country": "Sweden", "country_vi": "Thụy Điển"},
    "SI": {"country": "Slovenia", "country_vi": "Slovenia"},
    "SK": {"country": "Slovakia", "country_vi": "Slovakia"},
    # 6 Nước Châu Âu ngoài EU (Đợt 1)
    "UK": {"country": "United Kingdom", "country_vi": "Vương quốc Anh"},
    "CH": {"country": "Switzerland", "country_vi": "Thụy Sĩ"},
    "NO": {"country": "Norway", "country_vi": "Na Uy"},
    "IS": {"country": "Iceland", "country_vi": "Iceland"},
    "TR": {"country": "Türkiye", "country_vi": "Thổ Nhĩ Kỳ"},
    "RS": {"country": "Serbia", "country_vi": "Serbia"},
    # 5 Nước Châu Âu mở rộng bổ sung (Cấp độ 2 - Đã loại bỏ MD)
    "AL": {"country": "Albania", "country_vi": "Albania"},
    "ME": {"country": "Montenegro", "country_vi": "Montenegro"},
    "MK": {"country": "North Macedonia", "country_vi": "Bắc Macedonia"},
    "XK": {"country": "Kosovo", "country_vi": "Kosovo"},
    "LI": {"country": "Liechtenstein", "country_vi": "Liechtenstein"},
}
GEO_LIST = list(COUNTRY_MAP.keys())

# Phân loại cơ sở lưu trú NACE Rev. 2 (Cấp độ 1)
NACE_MAP = {
    "I551": {"type": "hotel", "type_vi": "Khách sạn"},
    "I552": {"type": "short_stay", "type_vi": "Căn hộ và nhà nghỉ dưỡng"},
    "I553": {"type": "camping", "type_vi": "Khu cắm trại"},
    "I551-I553": {"type": "total", "type_vi": "Tổng hợp"},
}


# ============================================================
# 2. HÀM TẢI VÀ GIẢI NÉN EUROSTAT TSV
# ============================================================
def download_eurostat_tsv(dataset_code: str) -> pd.DataFrame:
    """Tải trực tiếp file TSV nén từ Eurostat API (tốc độ cao)."""
    url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/{dataset_code}?format=TSV&compressed=true"
    print(f"[*] Đang tải dữ liệu {dataset_code} từ Eurostat...")
    t0 = time.time()
    resp = requests.get(url, timeout=90)
    resp.raise_for_status()
    with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
        df = pd.read_csv(f, sep="\t", low_memory=False)
    print(f"    -> Đã tải xong {dataset_code} ({df.shape[0]} dòng) trong {time.time()-t0:.1f}s")
    return df


def parse_cell(val):
    """
    Tách giá trị số và cờ chất lượng từ chuỗi Eurostat.
    Trả về (num_val, flag_val). Nếu không có cờ bất thường thì flag là np.nan.
    """
    if pd.isna(val):
        return np.nan, np.nan
    val_str = str(val).strip()
    if val_str.startswith(":"):
        flag = val_str[1:].strip()
        return np.nan, (flag if flag else np.nan)
    parts = val_str.split()
    if not parts:
        return np.nan, np.nan
    try:
        num = float(parts[0])
    except ValueError:
        return np.nan, val_str
    flag = parts[1] if len(parts) > 1 else np.nan
    return num, flag


# ============================================================
# 3. TRÍCH XUẤT VÀ CHUYỂN ĐỔI CHUỖI THỜI GIAN
# ============================================================
def process_demand_table(df_raw: pd.DataFrame, metric_prefix: str, target_months: list) -> pd.DataFrame:
    """
    Xử lý bảng tour_occ_arm hoặc tour_occ_nim.
    Trích xuất 4 phân loại NACE (hotel, short_stay, camping, total)
    và 3 phân nhóm cư trú: TOTAL, DOM, FOR với freq=M và unit=NR.
    """
    header_col = df_raw.columns[0]
    meta_df = df_raw[header_col].str.split(",", expand=True)
    meta_df.columns = ["freq", "c_resid", "unit", "nace_r2", "geo"]

    mask = (
        (meta_df["freq"] == "M") &
        (meta_df["unit"] == "NR") &
        (meta_df["nace_r2"].isin(list(NACE_MAP.keys()))) &
        (meta_df["c_resid"].isin(["TOTAL", "DOM", "FOR"])) &
        (meta_df["geo"].isin(GEO_LIST))
    )
    filtered_df = df_raw[mask].copy()
    filtered_meta = meta_df[mask].copy()

    data_cols = [c for c in df_raw.columns[1:] if c.strip() in target_months]

    records = []
    for idx, row in filtered_df.iterrows():
        geo = filtered_meta.loc[idx, "geo"]
        nace = filtered_meta.loc[idx, "nace_r2"]
        accom_type = NACE_MAP[nace]["type"]
        c_resid = filtered_meta.loc[idx, "c_resid"]
        field_type = "total" if c_resid == "TOTAL" else ("domestic" if c_resid == "DOM" else "foreign")
        val_name = f"{metric_prefix}_{field_type}"
        flag_name = f"{val_name}_flag"

        for col in data_cols:
            month_str = col.strip()
            num_val, flag_val = parse_cell(row[col])
            records.append({
                "geo": geo,
                "date": f"{month_str}-01",
                "accommodation_type": accom_type,
                "metric": val_name,
                "value": num_val,
                "flag_metric": flag_name,
                "flag": flag_val
            })

    df_records = pd.DataFrame(records)
    pivot_val = df_records.pivot(index=["geo", "date", "accommodation_type"], columns="metric", values="value").reset_index()
    pivot_flag = df_records.pivot(index=["geo", "date", "accommodation_type"], columns="flag_metric", values="flag").reset_index()

    merged = pd.merge(pivot_val, pivot_flag, on=["geo", "date", "accommodation_type"])
    return merged


def process_occupancy_table(df_raw: pd.DataFrame, target_months: list) -> pd.DataFrame:
    """
    Xử lý bảng tour_occ_mnor.
    Trích xuất BEDRM (phòng) và BEDPL (giường) với freq=M và unit=PC cho 38 quốc gia.
    """
    header_col = df_raw.columns[0]
    meta_df = df_raw[header_col].str.split(",", expand=True)
    meta_df.columns = ["freq", "accomunit", "unit", "geo"]

    mask = (
        (meta_df["freq"] == "M") &
        (meta_df["unit"] == "PC") &
        (meta_df["accomunit"].isin(["BEDRM", "BEDPL"])) &
        (meta_df["geo"].isin(GEO_LIST))
    )
    filtered_df = df_raw[mask].copy()
    filtered_meta = meta_df[mask].copy()

    data_cols = [c for c in df_raw.columns[1:] if c.strip() in target_months]

    records = []
    for idx, row in filtered_df.iterrows():
        geo = filtered_meta.loc[idx, "geo"]
        accom = filtered_meta.loc[idx, "accomunit"]
        val_name = "hotel_room_occupancy_pct" if accom == "BEDRM" else "hotel_bed_occupancy_pct"
        flag_name = f"{val_name}_flag"

        for col in data_cols:
            month_str = col.strip()
            num_val, flag_val = parse_cell(row[col])
            records.append({
                "geo": geo,
                "date": f"{month_str}-01",
                "metric": val_name,
                "value": num_val,
                "flag_metric": flag_name,
                "flag": flag_val
            })

    df_records = pd.DataFrame(records)
    pivot_val = df_records.pivot(index=["geo", "date"], columns="metric", values="value").reset_index()
    pivot_flag = df_records.pivot(index=["geo", "date"], columns="flag_metric", values="flag").reset_index()

    merged = pd.merge(pivot_val, pivot_flag, on=["geo", "date"])
    return merged


# ============================================================
# 4. HÀM MAIN GHÉP VÀ KHÔI PHỤC DỮ LIỆU TOÁN HỌC
# ============================================================
def main():
    print("=" * 80)
    print("XÂY DỰNG DATASET DU LỊCH CHÂU ÂU: PHƯƠNG ÁN 4 (2010 – 2026, LOẠI BỎ MOLDOVA)")
    print("Mở rộng 4 loại hình lưu trú NACE x 38 Quốc gia Châu Âu x Chuỗi tháng đến mới nhất 2026")
    print("=" * 80)

    start_time = time.time()

    # 1. Tải 3 bảng gốc từ Eurostat
    df_arm_raw = download_eurostat_tsv("tour_occ_arm")
    df_nim_raw = download_eurostat_tsv("tour_occ_nim")
    df_mnor_raw = download_eurostat_tsv("tour_occ_mnor")

    # Xác định danh sách các tháng chung từ START_MONTH trở đi (tự động lấy đến 2026-08 nếu có)
    arm_cols = {c.strip() for c in df_arm_raw.columns[1:] if c.strip() >= START_MONTH and len(c.strip().split("-")) == 2}
    nim_cols = {c.strip() for c in df_nim_raw.columns[1:] if c.strip() >= START_MONTH and len(c.strip().split("-")) == 2}
    mnor_cols = {c.strip() for c in df_mnor_raw.columns[1:] if c.strip() >= START_MONTH and len(c.strip().split("-")) == 2}
    target_months = sorted(list(arm_cols & nim_cols & mnor_cols))

    print(f"\n[*] Đã xác định {len(target_months)} tháng liên tục từ {target_months[0]} đến {target_months[-1]}.")
    expected_rows = len(GEO_LIST) * len(target_months) * len(NACE_MAP)
    print(f"[*] Quy mô dự kiến: {len(GEO_LIST)} quốc gia x {len(target_months)} tháng x {len(NACE_MAP)} loại hình = {expected_rows:,} dòng.")

    # 2. Xử lý từng bảng
    print("\n[*] Đang lọc và chuẩn hóa dữ liệu lượt khách (Arrivals)...")
    df_arm = process_demand_table(df_arm_raw, metric_prefix="arrivals", target_months=target_months)

    print("[*] Đang lọc và chuẩn hóa dữ liệu số đêm lưu trú (Nights)...")
    df_nim = process_demand_table(df_nim_raw, metric_prefix="nights", target_months=target_months)

    print("[*] Đang lọc và chuẩn hóa dữ liệu công suất khách sạn (Occupancy)...")
    df_occ = process_occupancy_table(df_mnor_raw, target_months=target_months)

    # 3. Tạo khung dữ liệu nền đầy đủ (Cartesian Grid: 38 nước x số tháng x 4 loại hình)
    print(f"\n[*] Đang tạo lưới chuỗi thời gian đầy đủ ({expected_rows:,} dòng)...")
    grid_rows = []
    for geo in GEO_LIST:
        country_en = COUNTRY_MAP[geo]["country"]
        country_vi = COUNTRY_MAP[geo]["country_vi"]
        for _, n_info in NACE_MAP.items():
            accom_type = n_info["type"]
            accom_type_vi = n_info["type_vi"]
            for m_str in target_months:
                y, m = map(int, m_str.split("-"))
                grid_rows.append({
                    "date": f"{m_str}-01",
                    "geo": geo,
                    "country": country_en,
                    "country_vi": country_vi,
                    "accommodation_type": accom_type,
                    "accommodation_type_vi": accom_type_vi,
                    "year": y,
                    "month": m,
                })
    df_base = pd.DataFrame(grid_rows)
    df_base["month_index"] = (df_base["year"] - START_YEAR) * 12 + df_base["month"] - 1

    # 4. Ghép các bảng
    print("[*] Ghép dữ liệu Arrivals và Nights...")
    merged = pd.merge(df_base, df_arm, on=["geo", "date", "accommodation_type"], how="left")
    merged = pd.merge(merged, df_nim, on=["geo", "date", "accommodation_type"], how="left")

    # Ghép Occupancy (gán cho phân loại 'hotel' và 'total' theo chuẩn định nghĩa Eurostat)
    print("[*] Ghép dữ liệu Occupancy (cho phân loại Khách sạn & Tổng hợp)...")
    df_occ_hotel = df_occ.copy()
    df_occ_hotel["accommodation_type"] = "hotel"
    df_occ_total = df_occ.copy()
    df_occ_total["accommodation_type"] = "total"
    df_occ_combined = pd.concat([df_occ_hotel, df_occ_total], ignore_index=True)

    merged = pd.merge(merged, df_occ_combined, on=["geo", "date", "accommodation_type"], how="left")

    # ============================================================
    # 5. KHÔI PHỤC TOÁN HỌC & ĐỒNG NHẤT THỨC CẤU TRÚC (RECOVERY ENGINE)
    # ============================================================
    print("\n[*] KHÔI PHỤC DỮ LIỆU THIẾU BẰNG ĐỒNG NHẤT THỨC KẾ TOÁN DU LỊCH:")

    # 5.1 Khôi phục Lượt khách (Arrivals: TOTAL = DOM + FOR)
    mask_dom_arr = merged["arrivals_domestic"].isna() & merged["arrivals_total"].notna() & merged["arrivals_foreign"].notna()
    merged.loc[mask_dom_arr, "arrivals_domestic"] = merged.loc[mask_dom_arr, "arrivals_total"] - merged.loc[mask_dom_arr, "arrivals_foreign"]
    merged.loc[mask_dom_arr, "arrivals_domestic_flag"] = "d"  # derived
    print(f"    -> Đã khôi phục {mask_dom_arr.sum():,} ô lượt khách nội địa (arrivals_domestic) bị thiếu!")

    mask_for_arr = merged["arrivals_foreign"].isna() & merged["arrivals_total"].notna() & merged["arrivals_domestic"].notna()
    merged.loc[mask_for_arr, "arrivals_foreign"] = merged.loc[mask_for_arr, "arrivals_total"] - merged.loc[mask_for_arr, "arrivals_domestic"]
    merged.loc[mask_for_arr, "arrivals_foreign_flag"] = "d"
    print(f"    -> Đã khôi phục {mask_for_arr.sum():,} ô lượt khách quốc tế (arrivals_foreign) bị thiếu!")

    mask_tot_arr = merged["arrivals_total"].isna() & merged["arrivals_domestic"].notna() & merged["arrivals_foreign"].notna()
    merged.loc[mask_tot_arr, "arrivals_total"] = merged.loc[mask_tot_arr, "arrivals_domestic"] + merged.loc[mask_tot_arr, "arrivals_foreign"]
    merged.loc[mask_tot_arr, "arrivals_total_flag"] = "d"
    print(f"    -> Đã khôi phục {mask_tot_arr.sum():,} ô tổng lượt khách (arrivals_total) bị thiếu!")

    # 5.2 Khôi phục Số đêm lưu trú (Nights: TOTAL = DOM + FOR)
    mask_dom_nim = merged["nights_domestic"].isna() & merged["nights_total"].notna() & merged["nights_foreign"].notna()
    merged.loc[mask_dom_nim, "nights_domestic"] = merged.loc[mask_dom_nim, "nights_total"] - merged.loc[mask_dom_nim, "nights_foreign"]
    merged.loc[mask_dom_nim, "nights_domestic_flag"] = "d"
    print(f"    -> Đã khôi phục {mask_dom_nim.sum():,} ô đêm lưu trú nội địa (nights_domestic) bị thiếu!")

    mask_for_nim = merged["nights_foreign"].isna() & merged["nights_total"].notna() & merged["nights_domestic"].notna()
    merged.loc[mask_for_nim, "nights_foreign"] = merged.loc[mask_for_nim, "nights_total"] - merged.loc[mask_for_nim, "nights_domestic"]
    merged.loc[mask_for_nim, "nights_foreign_flag"] = "d"
    print(f"    -> Đã khôi phục {mask_for_nim.sum():,} ô đêm lưu trú quốc tế (nights_foreign) bị thiếu!")

    mask_tot_nim = merged["nights_total"].isna() & merged["nights_domestic"].notna() & merged["nights_foreign"].notna()
    merged.loc[mask_tot_nim, "nights_total"] = merged.loc[mask_tot_nim, "nights_domestic"] + merged.loc[mask_tot_nim, "nights_foreign"]
    merged.loc[mask_tot_nim, "nights_total_flag"] = "d"
    print(f"    -> Đã khôi phục {mask_tot_nim.sum():,} ô tổng số đêm lưu trú (nights_total) bị thiếu!")

    # 5.3 Ước tính Công suất phòng từ Công suất giường (Room Occupancy Imputation)
    print("\n[*] Ước tính công suất phòng (hotel_room_occupancy_pct) từ công suất giường...")
    imputed_room_occ = 0
    for geo in GEO_LIST:
        for a_type in ["hotel", "total"]:
            geo_mask = (merged["geo"] == geo) & (merged["accommodation_type"] == a_type)
            geo_df = merged[geo_mask]
            valid_both = geo_df[geo_df["hotel_room_occupancy_pct"].notna() & geo_df["hotel_bed_occupancy_pct"].notna()]
            if len(valid_both) >= 12:
                ratio = (valid_both["hotel_room_occupancy_pct"] / valid_both["hotel_bed_occupancy_pct"]).mean()
                fill_room = geo_mask & merged["hotel_room_occupancy_pct"].isna() & merged["hotel_bed_occupancy_pct"].notna()
                cnt = fill_room.sum()
                if cnt > 0:
                    merged.loc[fill_room, "hotel_room_occupancy_pct"] = (merged.loc[fill_room, "hotel_bed_occupancy_pct"] * ratio).clip(0, 100).round(2)
                    merged.loc[fill_room, "hotel_room_occupancy_pct_flag"] = "e"  # estimated
                    imputed_room_occ += cnt

    print(f"    -> Đã ước tính thành công {imputed_room_occ:,} ô công suất phòng từ công suất giường!")

    # 6. Tính toán các đặc trưng tỷ số và cờ kiểm soát
    print("\n[*] Tính toán các đặc trưng tỷ số và cờ kiểm soát...")
    merged["nights_per_arrival_ratio"] = np.where(
        (merged["arrivals_total"] > 0) & merged["arrivals_total"].notna() & merged["nights_total"].notna(),
        (merged["nights_total"] / merged["arrivals_total"]).round(4),
        np.nan
    )
    merged["foreign_nights_share"] = np.where(
        (merged["nights_total"] > 0) & merged["nights_total"].notna() & merged["nights_foreign"].notna(),
        (merged["nights_foreign"] / merged["nights_total"]).round(4),
        np.nan
    )

    # Phân nhóm thời kỳ (Period Group)
    def assign_period(dt_str):
        if dt_str < "2020-01-01":
            return "pre_2020"
        elif dt_str <= "2021-12-31":
            return "2020_2021"
        else:
            return "from_2022"

    merged["period_group"] = merged["date"].apply(assign_period)

    # Cờ kiểm tra chất lượng
    raw_num_cols = [
        "arrivals_total", "arrivals_domestic", "arrivals_foreign",
        "nights_total", "nights_domestic", "nights_foreign",
        "hotel_room_occupancy_pct", "hotel_bed_occupancy_pct"
    ]
    merged["missing_raw_count"] = merged[raw_num_cols].isna().sum(axis=1)
    merged["both_targets_present"] = (
        merged["arrivals_total"].notna() & merged["nights_total"].notna()
    ).astype(int)

    def count_occupancy_out_of_range(row):
        cnt = 0
        r_occ = row.get("hotel_room_occupancy_pct")
        b_occ = row.get("hotel_bed_occupancy_pct")
        if pd.notna(r_occ) and (r_occ < 0 or r_occ > 100):
            cnt += 1
        if pd.notna(b_occ) and (b_occ < 0 or b_occ > 100):
            cnt += 1
        return cnt

    merged["occupancy_out_of_range_count"] = merged.apply(count_occupancy_out_of_range, axis=1)

    # Định danh thứ tự cột chuẩn tắc
    FINAL_COLUMNS = [
        "date", "geo", "country", "country_vi",
        "accommodation_type", "accommodation_type_vi",
        "year", "month", "month_index",
        "arrivals_total", "arrivals_domestic", "arrivals_foreign",
        "nights_total", "nights_domestic", "nights_foreign",
        "hotel_room_occupancy_pct", "hotel_bed_occupancy_pct",
        "arrivals_total_flag", "arrivals_domestic_flag", "arrivals_foreign_flag",
        "nights_total_flag", "nights_domestic_flag", "nights_foreign_flag",
        "hotel_room_occupancy_pct_flag", "hotel_bed_occupancy_pct_flag",
        "nights_per_arrival_ratio", "foreign_nights_share", "period_group",
        "missing_raw_count", "both_targets_present", "occupancy_out_of_range_count"
    ]

    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = np.nan

    merged = merged[FINAL_COLUMNS]
    merged = merged.sort_values(by=["geo", "accommodation_type", "date"]).reset_index(drop=True)

    print(f"\n[+] TỔNG SỐ DÒNG SAU MỞ RỘNG (PHƯƠNG ÁN 4): {len(merged):,} dòng")
    print(f"[+] Cơ cấu: {len(GEO_LIST)} quốc gia x {len(target_months)} tháng ({target_months[0]} – {target_months[-1]}) x 4 loại hình lưu trú")
    print(f"[+] Thời gian bao phủ: Từ {merged['date'].min()} đến {merged['date'].max()}")
    print(f"[+] Số quan sát có đầy đủ 2 Target (both_targets_present=1): {merged['both_targets_present'].sum():,} dòng ({merged['both_targets_present'].mean()*100:.2f}%)")

    # 7. Xuất ra file CSV và Excel
    print(f"\n[*] Đang lưu file CSV vào: {OUTPUT_CSV}")
    merged.to_csv(OUTPUT_CSV, index=False)

    print(f"[*] Đang ghi file Excel vào: {OUTPUT_XLSX} (Sheet: Monthly)...")
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as writer:
        merged.to_excel(writer, sheet_name="Monthly", index=False)

        # Tạo thêm Sheet Readme tóm tắt
        readme_data = [
            ("Phạm vi bộ dữ liệu", f"{len(GEO_LIST)} quốc gia Châu Âu (27 EU + 11 ngoài EU; đã loại bỏ Moldova MD do 0% dữ liệu)."),
            ("Thời gian bao phủ", f"{target_months[0]} đến {target_months[-1]} ({len(target_months)} tháng liên tục); Tổng cộng {len(merged):,} bản ghi."),
            ("Mục tiêu khoa học", "Dự báo nhu cầu du lịch lưu trú đa quốc gia Châu Âu thích ứng trạng thái thị trường (Regime-Aware)."),
            ("Cấp độ 1 (NACE)", "Phân rã 4 loại hình: hotel (I551), short_stay (I552), camping (I553), total (I551-I553)."),
            ("Cấp độ 2 (Địa lý)", "Mở rộng 38 quốc gia chuẩn tắc (EU27 + UK, CH, NO, IS, TR, RS, AL, ME, MK, XK, LI)."),
            ("Khôi phục toán học", "Áp dụng đồng nhất thức DOM = TOTAL - FOR và ước tính công suất phòng từ công suất giường."),
            ("Nguồn dữ liệu", "Eurostat: tour_occ_arm (Lượt đến), tour_occ_nim (Số đêm), tour_occ_mnor (Công suất)."),
            ("Thời điểm trích xuất", pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")),
        ]
        pd.DataFrame(readme_data, columns=["Tiêu chí", "Thông tin"]).to_excel(writer, sheet_name="Readme", index=False)

    total_time = time.time() - start_time
    print(f"\n🎉 HOÀN TẤT THÀNH CÔNG TRONG {total_time:.1f} GIÂY!")
    print(f"File mới đã sẵn sàng: {OUTPUT_XLSX.name} và {OUTPUT_CSV.name}")
    print("=" * 80)


if __name__ == "__main__":
    main()
