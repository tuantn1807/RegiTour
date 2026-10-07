"""
Script: enrich_dataset_exogenous.py
Mục đích: Tự động tải, trích xuất và tích hợp các biến ngoại sinh (Exogenous Variables)
         vào bộ dữ liệu Du lịch & Lưu trú Châu Âu mở rộng:
         1. Lịch biểu & Ngày lễ chính thức (Calendar, Weekends & National Holidays)
         2. Chỉ số Giá Tiêu dùng & Dịch vụ Lưu trú HICP từ Eurostat (CP112 & CP00)
         3. Chỉ số Nghiêm ngặt Phòng dịch Oxford (OxCGRT Stringency Index)
Đầu vào: data/processed/tourism_europe_expanded.csv
Đầu ra: 
  - data/processed/tourism_europe_enriched.csv (Bộ dữ liệu mở rộng đầy đủ biến ngoại sinh)
  - data/features/tourism_europe_features.parquet (Định dạng Parquet chuẩn hoá cho AI Model)
"""

import io
import gzip
import time
import calendar
from datetime import date
from pathlib import Path
import requests
import pandas as pd
import numpy as np

# Thư viện ngày lễ quốc tế
try:
    import holidays
    HAS_HOLIDAYS = True
except ImportError:
    HAS_HOLIDAYS = False
    print("[!] Cảnh báo: Thư viện 'holidays' chưa cài đặt, dùng fallback tính ngày lễ.")

# ============================================================
# CẤU HÌNH ĐƯỜNG DẪN & BẢNG MÃ
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_europe_expanded.csv"
OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_europe_enriched.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "features" / "tourism_europe_features.parquet"
OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

# Bản đồ ánh xạ mã Eurostat 2 ký tự sang ISO-3166-1 alpha-3 (dùng cho Oxford OxCGRT)
GEO_TO_ISO3 = {
    "AT": "AUT", "BE": "BEL", "BG": "BGR", "CY": "CYP", "CZ": "CZE",
    "DE": "DEU", "DK": "DNK", "EE": "EST", "EL": "GRC", "ES": "ESP",
    "FI": "FIN", "FR": "FRA", "HR": "HRV", "HU": "HUN", "IE": "IRL",
    "IT": "ITA", "LT": "LTU", "LU": "LUX", "LV": "LVA", "MT": "MLT",
    "NL": "NLD", "PL": "POL", "PT": "PRT", "RO": "ROU", "SE": "SWE",
    "SI": "SVN", "SK": "SVK", "UK": "GBR", "CH": "CHE", "NO": "NOR",
    "IS": "ISL", "TR": "TUR", "RS": "SRB", "AL": "ALB", "ME": "MNE",
    "MK": "MKD", "XK": "XKX", "LI": "LIE"
}

# Ánh xạ mã Eurostat sang mã 2 ký tự chuẩn ISO cho thư viện holidays
GEO_TO_HOLIDAY_ALPHA2 = {
    "EL": "GR",  # Hy Lạp trong Eurostat là EL, chuẩn ISO là GR
    "UK": "GB",  # Vương quốc Anh trong Eurostat là UK, chuẩn ISO là GB
    "XK": "XK",
}


# Import module xây dựng cơ sở dữ liệu ngày nghỉ lễ
try:
    from build_holidays_database import build_holidays_database, aggregate_monthly_holiday_features
    HAS_HOLIDAY_DB = True
except ImportError:
    HAS_HOLIDAY_DB = False

# ============================================================
# 1. HÀM TẠO ĐẶC TRƯNG LỊCH BIỂU & PHÂN RÃ NGÀY NGHỈ LỄ
# ============================================================
def generate_calendar_and_holiday_features(df_input: pd.DataFrame) -> pd.DataFrame:
    """Tạo số ngày cuối tuần (T6, T7, CN) và phân rã chi tiết các loại ngày nghỉ lễ theo tháng."""
    print("[*] Đang sinh các đặc trưng Lịch biểu & Phân rã Ngày nghỉ lễ chi tiết...")
    
    unique_dates = df_input[["date", "year", "month", "geo"]].drop_duplicates()
    
    # 1. Sinh các biến lịch biểu thuần (days_in_month, weekend_days, working_days, season flags)
    cal_list = []
    for _, row in unique_dates.iterrows():
        y = int(row["year"])
        m = int(row["month"])
        geo = row["geo"]
        dt_str = row["date"]
        
        days_in_month = calendar.monthrange(y, m)[1]
        weekend_days = 0
        for d in range(1, days_in_month + 1):
            cur_d = date(y, m, d)
            if cur_d.weekday() in [4, 5, 6]:
                weekend_days += 1
                
        working_days = days_in_month - weekend_days
        is_peak_summer = 1 if m in [7, 8] else 0
        is_shoulder_season = 1 if m in [5, 6, 9, 10] else 0
        is_winter_holiday = 1 if m == 12 else 0
        
        cal_list.append({
            "date": dt_str,
            "geo": geo,
            "days_in_month": days_in_month,
            "weekend_days": weekend_days,
            "working_days": working_days,
            "is_peak_summer": is_peak_summer,
            "is_shoulder_season": is_shoulder_season,
            "is_winter_holiday": is_winter_holiday,
        })
    df_cal = pd.DataFrame(cal_list)
    
    # 2. Xây dựng Database ngày lễ và phân rã các lý do nghỉ lễ
    min_year = int(df_input["year"].min())
    max_year = int(df_input["year"].max())
    df_hol_db = build_holidays_database(min_year, max_year)
    df_hol_agg = aggregate_monthly_holiday_features(df_hol_db)
    
    # Ghép thông tin ngày lễ chi tiết
    df_result = pd.merge(df_cal, df_hol_agg.drop(columns=["year", "month"]), on=["date", "geo"], how="left")
    
    # Điền 0 cho các tháng không có ngày lễ nào
    hol_num_cols = [
        "holiday_days", "holiday_religious_days", "holiday_national_days",
        "holiday_memorial_days", "holiday_labor_days", "holiday_long_weekends_count",
        "holiday_bridge_days_count", "holiday_has_easter", "holiday_has_christmas"
    ]
    for col in hol_num_cols:
        df_result[col] = df_result[col].fillna(0).astype(int)
    df_result["holiday_names_list"] = df_result["holiday_names_list"].fillna("None")
    
    print(f"    -> Đã tạo và phân rã đặc trưng lịch biểu cho {len(df_result)} cặp (date, geo).")
    return df_result


# ============================================================
# 2. HÀM TẢI & XỬ LÝ CHỈ SỐ GIÁ HICP TỪ EUROSTAT
# ============================================================
def fetch_eurostat_hicp(coicop_code: str, feature_name: str) -> pd.DataFrame:
    """
    Tải chỉ số giá HICP (Index 2015=100) theo mã COICOP qua SDMX API.
    - CP112: Accommodation services (Dịch vụ lưu trú)
    - CP00: All-items HICP (Lạm phát chung)
    """
    url = f"https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/prc_hicp_midx/M.I15.{coicop_code}.?format=TSV&compressed=true"
    print(f"[*] Đang tải chỉ số giá HICP ({coicop_code} -> {feature_name}) từ Eurostat...")
    
    try:
        resp = requests.get(url, timeout=45)
        resp.raise_for_status()
        with gzip.open(io.BytesIO(resp.content), "rt", encoding="utf-8") as f:
            df_raw = pd.read_csv(f, sep="\t", low_memory=False)
    except Exception as e:
        print(f"    [!] Lỗi tải HICP {coicop_code}: {e}. Sẽ dùng fallback.")
        return pd.DataFrame()
        
    first_col = df_raw.columns[0]
    time_cols = [c.strip() for c in df_raw.columns[1:]]
    df_raw.columns = [first_col] + time_cols
    
    # Tách mã geo từ cột đầu tiên (freq,unit,coicop,geo)
    records = []
    for _, row in df_raw.iterrows():
        parts = row[first_col].split(",")
        if len(parts) >= 4:
            geo = parts[3].strip()
            for t_col in time_cols:
                # time_col có dạng YYYY-MM
                if "-" in t_col:
                    val_str = str(row[t_col]).strip()
                    # Loại bỏ cờ Eurostat (ví dụ: "105.2 p" -> 105.2)
                    num_str = "".join([c for c in val_str.split()[0] if c.isdigit() or c in ".-"])
                    try:
                        val = float(num_str) if num_str else np.nan
                    except ValueError:
                        val = np.nan
                    
                    records.append({
                        "geo": geo,
                        "date": f"{t_col}-01",
                        feature_name: val
                    })
                    
    df_hicp = pd.DataFrame(records)
    print(f"    -> Đã trích xuất {len(df_hicp)} bản ghi HICP {feature_name}.")
    return df_hicp


# ============================================================
# 3. HÀM TẢI & XỬ LÝ CHỈ SỐ NGHIÊM NGẶT OXFORD (OxCGRT)
# ============================================================
def fetch_oxford_stringency() -> pd.DataFrame:
    """Tải và tính trung bình theo tháng chỉ số nghiêm ngặt phòng dịch COVID-19 của Oxford."""
    url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_national_v1.csv"
    print("[*] Đang tải Oxford COVID-19 Government Response Tracker (OxCGRT)...")
    
    try:
        # Đọc trực tiếp các cột cần thiết
        use_cols = [
            "CountryCode", "Jurisdiction", "Date",
            "StringencyIndex_Average",
            "GovernmentResponseIndex_Average",
            "ContainmentHealthIndex_Average"
        ]
        df_ox = pd.read_csv(url, usecols=use_cols, low_memory=False)
        # Chỉ lấy cấp độ quốc gia (NAT_TOTAL)
        df_ox = df_ox[df_ox["Jurisdiction"] == "NAT_TOTAL"].copy()
        
        # Tạo cột ngày YYYY-MM-01
        df_ox["Date_dt"] = pd.to_datetime(df_ox["Date"].astype(str), format="%Y%m%d")
        df_ox["date"] = df_ox["Date_dt"].dt.strftime("%Y-%m-01")
        
        # Gom nhóm theo tháng và tính trung bình
        monthly = df_ox.groupby(["CountryCode", "date"])[
            ["StringencyIndex_Average", "GovernmentResponseIndex_Average", "ContainmentHealthIndex_Average"]
        ].mean().reset_index()
        
        # Đổi tên cột
        monthly = monthly.rename(columns={
            "StringencyIndex_Average": "covid_stringency_index",
            "GovernmentResponseIndex_Average": "covid_gov_response_index",
            "ContainmentHealthIndex_Average": "covid_containment_index",
        })
        
        # Tạo bản đồ ngược từ ISO3 sang Geo (Eurostat)
        iso3_to_geo = {v: k for k, v in GEO_TO_ISO3.items()}
        monthly["geo"] = monthly["CountryCode"].map(iso3_to_geo)
        monthly = monthly.dropna(subset=["geo"]).drop(columns=["CountryCode"])
        
        print(f"    -> Đã trích xuất {len(monthly)} bản ghi chỉ số Oxford theo tháng.")
        return monthly
    except Exception as e:
        print(f"    [!] Lỗi tải OxCGRT: {e}")
        return pd.DataFrame()


# ============================================================
# 4. CHƯƠNG TRÌNH TỔNG HỢP VÀ GHÉP NỐI (DATA ENRICHMENT PIPELINE)
# ============================================================
def main():
    start_time = time.time()
    print("=" * 70)
    print("🚀 BẮT ĐẦU QUY TRÌNH BỔ SUNG DỮ LIỆU NGOẠI SINH CHO ĐỀ TÀI NCKH")
    print("=" * 70)
    
    # 1. Đọc bộ dữ liệu hiện có
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"Không tìm thấy file đầu vào: {INPUT_CSV}")
    
    print(f"[*] Đang nạp dataset cơ sở: {INPUT_CSV.name}")
    df_base = pd.read_csv(INPUT_CSV, low_memory=False)
    print(f"    -> Kích thước cơ sở: {df_base.shape[0]:,} dòng × {df_base.shape[1]} cột.")
    
    # 2. Sinh đặc trưng Lịch biểu & Phân rã Ngày nghỉ lễ chi tiết
    df_cal = generate_calendar_and_holiday_features(df_base)
    df_merged = pd.merge(df_base, df_cal, on=["date", "geo"], how="left")
    
    # 3. Tải và ghép HICP (Giá lưu trú & Lạm phát chung)
    df_hicp_acc = fetch_eurostat_hicp("CP112", "hicp_accommodation")
    if not df_hicp_acc.empty:
        df_merged = pd.merge(df_merged, df_hicp_acc, on=["date", "geo"], how="left")
    else:
        df_merged["hicp_accommodation"] = np.nan
        
    df_hicp_cpi = fetch_eurostat_hicp("CP00", "hicp_overall")
    if not df_hicp_cpi.empty:
        df_merged = pd.merge(df_merged, df_hicp_cpi, on=["date", "geo"], how="left")
    else:
        df_merged["hicp_overall"] = np.nan
        
    # Tính tốc độ tăng trưởng giá MoM & YoY cho dịch vụ lưu trú
    print("[*] Đang tính toán tỷ lệ tăng giá dịch vụ lưu trú (MoM & YoY Inflation)...")
    df_merged = df_merged.sort_values(by=["geo", "accommodation_type", "date"]).reset_index(drop=True)
    
    # Nội suy nhẹ tuyến tính cho HICP bị khuyết cục bộ trước khi tính %
    df_merged["hicp_accommodation"] = df_merged.groupby(["geo", "accommodation_type"])["hicp_accommodation"].transform(
        lambda s: s.interpolate(method="linear", limit=6).bfill().ffill()
    )
    df_merged["hicp_overall"] = df_merged.groupby(["geo", "accommodation_type"])["hicp_overall"].transform(
        lambda s: s.interpolate(method="linear", limit=6).bfill().ffill()
    )
    
    df_merged["hicp_acc_mom_pct"] = df_merged.groupby(["geo", "accommodation_type"])["hicp_accommodation"].pct_change(1) * 100
    df_merged["hicp_acc_yoy_pct"] = df_merged.groupby(["geo", "accommodation_type"])["hicp_accommodation"].pct_change(12) * 100
    
    # 4. Tải và ghép chỉ số Oxford COVID-19 Stringency
    df_ox = fetch_oxford_stringency()
    if not df_ox.empty:
        df_merged = pd.merge(df_merged, df_ox, on=["date", "geo"], how="left")
        # Giai đoạn trước 2020 hoặc sau khi dỡ bỏ (NaN): mức hạn chế bằng 0
        for col in ["covid_stringency_index", "covid_gov_response_index", "covid_containment_index"]:
            df_merged[col] = df_merged[col].fillna(0.0)
    else:
        for col in ["covid_stringency_index", "covid_gov_response_index", "covid_containment_index"]:
            df_merged[col] = 0.0
            
    # 5. Lưu trữ kết quả ra CSV và Parquet
    print(f"[*] Đang lưu tập dữ liệu mở rộng ra CSV: {OUTPUT_CSV.name}...")
    df_merged.to_csv(OUTPUT_CSV, index=False)
    
    print(f"[*] Đang lưu định dạng chuẩn Apache Parquet: {OUTPUT_PARQUET.name}...")
    df_merged.to_parquet(OUTPUT_PARQUET, index=False, compression="snappy")
    
    # 6. Báo cáo tổng kết
    elapsed = time.time() - start_time
    print("=" * 70)
    print("✅ HOÀN THÀNH QUY TRÌNH BỔ SUNG DỮ LIỆU NGOẠI SINH & PHÂN RÃ NGÀY NGHỈ LỄ!")
    print(f"⏱️  Thời gian thực thi: {elapsed:.2f} giây")
    print(f"📊 Kích thước dữ liệu mới: {df_merged.shape[0]:,} dòng × {df_merged.shape[1]} cột")
    print(f"📁 Tệp CSV: {OUTPUT_CSV}")
    print(f"📁 Tệp Parquet: {OUTPUT_PARQUET}")
    
    new_cols = [
        "days_in_month", "weekend_days", "working_days",
        "holiday_days", "holiday_religious_days", "holiday_national_days",
        "holiday_memorial_days", "holiday_labor_days", "holiday_long_weekends_count",
        "holiday_bridge_days_count", "holiday_has_easter", "holiday_has_christmas",
        "is_peak_summer", "is_shoulder_season", "is_winter_holiday",
        "hicp_accommodation", "hicp_overall", "hicp_acc_mom_pct", "hicp_acc_yoy_pct",
        "covid_stringency_index", "covid_gov_response_index", "covid_containment_index",
        "holiday_names_list"
    ]
    print("\n📋 Các đặc trưng ngoại sinh và phân rã ngày nghỉ lễ đã tích hợp:")
    for c in new_cols:
        non_null_cnt = df_merged[c].notna().sum()
        pct = (non_null_cnt / len(df_merged)) * 100
        print(f"   • {c:<28}: {non_null_cnt:,}/{len(df_merged):,} giá trị ({pct:.1f}%)")
    print("=" * 70)

if __name__ == "__main__":
    main()
