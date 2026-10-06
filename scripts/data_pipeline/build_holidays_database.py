"""
Script: build_holidays_database.py
Mục đích:
  1. Xây dựng Cơ sở dữ liệu Tra cứu Chi tiết Ngày Nghỉ Lễ Châu Âu (2010 - 2026) cho 38 quốc gia.
     Lưu từng ngày lễ cụ thể, tên ngày lễ, lý do/phân loại kỳ nghỉ, thứ trong tuần, và cờ kỳ nghỉ dài (Long Weekend).
     -> Xuất ra: data/processed/holidays_europe_detailed.csv
  2. Cung cấp hàm tổng hợp theo tháng để phân rã (decompose) các cột ngày nghỉ trong bộ dữ liệu chuỗi thời gian chính.
"""

import calendar
from datetime import date
from pathlib import Path
import pandas as pd
import numpy as np
import holidays

PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_LOOKUP_CSV = PROJECT_ROOT / "data" / "processed" / "holidays_europe_detailed.csv"
OUTPUT_LOOKUP_CSV.parent.mkdir(parents=True, exist_ok=True)

# 38 Quốc gia Châu Âu trong Đề tài
COUNTRY_MAP = {
    "AT": "Austria", "BE": "Belgium", "BG": "Bulgaria", "CY": "Cyprus", "CZ": "Czechia",
    "DE": "Germany", "DK": "Denmark", "EE": "Estonia", "EL": "Greece", "ES": "Spain",
    "FI": "Finland", "FR": "France", "HR": "Croatia", "HU": "Hungary", "IE": "Ireland",
    "IT": "Italy", "LT": "Lithuania", "LU": "Luxembourg", "LV": "Latvia", "MT": "Malta",
    "NL": "Netherlands", "PL": "Poland", "PT": "Portugal", "RO": "Romania", "SE": "Sweden",
    "SI": "Slovenia", "SK": "Slovakia", "UK": "United Kingdom", "CH": "Switzerland",
    "NO": "Norway", "IS": "Iceland", "TR": "Türkiye", "RS": "Serbia", "AL": "Albania",
    "ME": "Montenegro", "MK": "North Macedonia", "XK": "Kosovo", "LI": "Liechtenstein"
}

# Ánh xạ mã Eurostat sang mã chuẩn ISO của thư viện holidays
GEO_TO_HOLIDAYS_ISO = {
    "EL": "GR",  # Hy Lạp
    "UK": "GB",  # Vương quốc Anh
    "XK": "XK",  # Kosovo
}

WEEKDAY_NAMES_EN = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
WEEKDAY_NAMES_VI = ["Thứ Hai", "Thứ Ba", "Thứ Tư", "Thứ Năm", "Thứ Sáu", "Thứ Bảy", "Chủ Nhật"]


def classify_holiday(name: str) -> tuple[str, str]:
    """
    Phân loại lý do nghỉ lễ dựa trên tên ngày lễ:
    Trả về (mã_phân_loại_en, tên_phân_loại_vi)
    """
    n_lower = name.lower()
    
    # 1. Tôn giáo (Religious)
    religious_keywords = [
        "easter", "good friday", "easter monday", "palm sunday", "pentecost", "whit",
        "ascension", "corpus christi", "assumption", "all saints", "immaculate conception",
        "christmas", "st. stephen", "boxing day", "epiphany", "st. joseph", "st. john",
        "holy", "orthodox", "saint", "midsummer", "annunciation", "nativity"
    ]
    if any(k in n_lower for k in religious_keywords):
        return "religious", "Tôn giáo"
        
    # 2. Quốc gia, Thể chế & Lịch sử (National & Civic)
    national_keywords = [
        "national", "republic", "constitution", "independence", "statehood", "sovereignty",
        "unification", "unity", "liberation", "revolution", "king", "queen", "flag",
        "monarchy", "royal", "accession", "proclamation"
    ]
    if any(k in n_lower for k in national_keywords):
        return "national_civic", "Quốc khánh / Lịch sử / Thể chế"
        
    # 3. Tưởng niệm & Hòa bình (Memorial & Peace)
    memorial_keywords = [
        "victory", "armistice", "remembrance", "memorial", "veterans", "defenders",
        "uprising", "holocaust", "heroes", "fallen", "peace"
    ]
    if any(k in n_lower for k in memorial_keywords):
        return "memorial", "Tưởng niệm / Hòa bình"
        
    # 4. Lao động & Xã hội (Labor & Social)
    labor_keywords = ["labor", "labour", "may day", "worker", "work"]
    if any(k in n_lower for k in labor_keywords):
        return "labor", "Quốc tế Lao động"
        
    # 5. Năm mới & Lễ hội văn hóa (New Year & Cultural)
    new_year_keywords = ["new year", "eve", "cultural", "carnival"]
    if any(k in n_lower for k in new_year_keywords):
        return "new_year_cultural", "Năm mới / Lễ hội Văn hóa"
        
    return "other", "Khác"


def build_holidays_database(start_year: int = 2010, end_year: int = 2026) -> pd.DataFrame:
    """Tạo bảng tra cứu chi tiết ngày nghỉ lễ toàn bộ 38 nước Châu Âu."""
    print(f"[*] Đang khởi tạo Cơ sở dữ liệu Ngày nghỉ lễ Châu Âu ({start_year} - {end_year})...")
    years = list(range(start_year, end_year + 1))
    records = []
    
    for geo, country_name in COUNTRY_MAP.items():
        iso_code = GEO_TO_HOLIDAYS_ISO.get(geo, geo)
        try:
            geo_holidays = holidays.country_holidays(iso_code, years=years)
        except NotImplementedError:
            continue
            
        for dt, raw_name in sorted(geo_holidays.items()):
            # Loại bỏ các mục chỉ ghi là Sunday ở Thụy Điển
            clean_name = raw_name.replace("; Sunday", "").strip()
            if clean_name == "Sunday":
                continue
                
            y = dt.year
            m = dt.month
            d = dt.day
            weekday_num = dt.weekday() # 0 = Thứ 2 ... 6 = Chủ Nhật
            weekday_en = WEEKDAY_NAMES_EN[weekday_num]
            weekday_vi = WEEKDAY_NAMES_VI[weekday_num]
            
            category_en, category_vi = classify_holiday(clean_name)
            
            # Kỳ nghỉ dài (Long Weekend): Ngày lễ rơi vào Thứ 6 hoặc Thứ 2
            is_long_weekend = 1 if weekday_num in [0, 4] else 0
            
            # Cầu nối (Bridge Day): Ngày lễ rơi vào Thứ 5 (nghỉ thêm Thứ 6) hoặc Thứ 3
            is_bridge_potential = 1 if weekday_num in [1, 3] else 0
            
            # Kiểm tra xem có phải lễ Phục sinh hoặc Giáng sinh không
            name_low = clean_name.lower()
            is_easter = 1 if "easter" in name_low or "good friday" in name_low else 0
            is_christmas = 1 if "christmas" in name_low or "st. stephen" in name_low or "boxing" in name_low else 0
            
            records.append({
                "geo": geo,
                "country": country_name,
                "holiday_date": dt.strftime("%Y-%m-%d"),
                "year": y,
                "month": m,
                "day": d,
                "weekday_num": weekday_num,
                "weekday_name_en": weekday_en,
                "weekday_name_vi": weekday_vi,
                "holiday_name": clean_name,
                "holiday_category": category_en,
                "holiday_category_vi": category_vi,
                "is_long_weekend": is_long_weekend,
                "is_bridge_potential": is_bridge_potential,
                "is_easter": is_easter,
                "is_christmas": is_christmas,
            })
            
    df_hol = pd.DataFrame(records)
    df_hol.to_csv(OUTPUT_LOOKUP_CSV, index=False)
    print(f"✅ Đã tạo xong Database tra cứu ngày lễ: {OUTPUT_LOOKUP_CSV.name}")
    print(f"    -> Tổng số ngày lễ: {len(df_hol):,} bản ghi.")
    return df_hol


def aggregate_monthly_holiday_features(df_hol: pd.DataFrame) -> pd.DataFrame:
    """
    Gom nhóm theo (geo, year, month) để tạo các cột đặc trưng phân rã cho chuỗi thời gian:
    - holiday_days: tổng số ngày lễ
    - holiday_religious_days: số ngày lễ tôn giáo
    - holiday_national_days: số ngày lễ quốc gia / lịch sử
    - holiday_long_weekends_count: số kỳ nghỉ dài nối cuối tuần
    - holiday_has_easter: tháng có kỳ nghỉ Phục sinh (0/1)
    - holiday_has_christmas: tháng có kỳ nghỉ Giáng sinh (0/1)
    - holiday_names_list: danh sách tên các ngày lễ trong tháng
    """
    print("[*] Đang phân rã các đặc trưng ngày nghỉ lễ theo tháng...")
    
    def agg_group(group):
        names = "; ".join(group["holiday_name"].unique())
        return pd.Series({
            "holiday_days": len(group),
            "holiday_religious_days": (group["holiday_category"] == "religious").sum(),
            "holiday_national_days": (group["holiday_category"] == "national_civic").sum(),
            "holiday_memorial_days": (group["holiday_category"] == "memorial").sum(),
            "holiday_labor_days": (group["holiday_category"] == "labor").sum(),
            "holiday_long_weekends_count": group["is_long_weekend"].sum(),
            "holiday_bridge_days_count": group["is_bridge_potential"].sum(),
            "holiday_has_easter": int(group["is_easter"].any()),
            "holiday_has_christmas": int(group["is_christmas"].any()),
            "holiday_names_list": names,
        })
        
    df_monthly = df_hol.groupby(["geo", "year", "month"]).apply(agg_group, include_groups=False).reset_index()
    # Tạo cột date YYYY-MM-01 để khớp với bảng du lịch
    df_monthly["date"] = df_monthly.apply(lambda r: f"{int(r['year']):04d}-{int(r['month']):02d}-01", axis=1)
    return df_monthly


if __name__ == "__main__":
    df_db = build_holidays_database(2010, 2026)
    print("\nVí dụ 5 ngày lễ đầu tiên trong Database:")
    print(df_db[["geo", "country", "holiday_date", "weekday_name_en", "holiday_name", "holiday_category_vi", "is_long_weekend"]].head())
    
    df_agg = aggregate_monthly_holiday_features(df_db)
    print("\nVí dụ tổng hợp phân rã theo tháng (Pháp năm 2024):")
    fr_2024 = df_agg[(df_agg["geo"] == "FR") & (df_agg["year"] == 2024)].head(5)
    print(fr_2024[["date", "holiday_days", "holiday_religious_days", "holiday_long_weekends_count", "holiday_has_easter", "holiday_names_list"]])
