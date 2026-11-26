"""
Script: enrich_un_tourism_dataset.py
Mục đích: Bổ sung và tích hợp các nguồn ngoại sinh chuẩn quốc tế cho tập dữ liệu UN Tourism (1995 - 2024)
          để chuẩn hóa schema và chạy trên kiến trúc RA-HE-DLinear (phương án bổ sung đối chuẩn toàn cầu).

Các nguồn tích hợp:
  1. Dữ liệu gốc: UN Tourism Statistics Database (1995 - 2024, 209 quốc gia/vùng lãnh thổ).
  2. Dữ liệu kinh tế vĩ mô: GDP, Lạm phát, Tỷ giá, Dân số, Thất nghiệp từ World Bank WDI (1960 - 2025).
  3. Ngày nghỉ lễ quốc gia trong năm (Python 'holidays' library - 197+ quốc gia).
  4. Sự kiện toàn cầu lớn (Mega-Events: Summer/Winter Olympics, FIFA World Cup, World Expo 1995 - 2024).
  5. Chỉ số khủng hoảng & giãn cách Covid (Oxford OxCGRT 2020 - 2022).
  6. Số đêm lưu trú quốc tế (OECD Inbound Tourism Database 2010 - 2024 + UN Tourism ALOS benchmark).
  7. Chuẩn hóa Schema: geo, year, month, date, accommodation_type, period_group, regime.

Đầu ra:
  - data/processed/tourism_un_tourism_enriched.csv
  - data/features/tourism_un_tourism_features.parquet
"""

import sys
import time
from pathlib import Path
from typing import Dict, Optional, Tuple

import holidays
import numpy as np
import pandas as pd

# Đường dẫn dự án
PROJECT_ROOT = Path(__file__).resolve().parents[2]
RAW_UN_PATH = PROJECT_ROOT / "data" / "raw" / "un_tourism_inbound_arrivals.csv"
EXCEL_PATH_1 = PROJECT_ROOT / "data_oced" / "tourism_economics_worldbank-3.xlsx"
EXCEL_PATH_2 = PROJECT_ROOT / "data" / "raw" / "tourism_economics_worldbank-3.xlsx"
EXCEL_PATH = EXCEL_PATH_1 if EXCEL_PATH_1.exists() else EXCEL_PATH_2

OECD_CLEAN_PATH = PROJECT_ROOT / "data_oced" / "oecd_tourism_benchmark_clean.csv"
OECD_INBOUND_PATH = PROJECT_ROOT / "data_oced" / "oecd_inbound_tourism_raw.csv"

OUTPUT_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_un_tourism_enriched.csv"
OUTPUT_PARQUET = PROJECT_ROOT / "data" / "features" / "tourism_un_tourism_features.parquet"

OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
OUTPUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)

# Bảng tra cứu sự kiện toàn cầu lớn (Olympic, World Cup, World Expo 1995 - 2024)
MEGA_EVENTS = {
    # Summer Olympics
    ("USA", 1996): "Summer_Olympics_Atlanta",
    ("AUS", 2000): "Summer_Olympics_Sydney",
    ("GRC", 2004): "Summer_Olympics_Athens",
    ("CHN", 2008): "Summer_Olympics_Beijing",
    ("GBR", 2012): "Summer_Olympics_London",
    ("BRA", 2016): "Summer_Olympics_Rio",
    ("JPN", 2021): "Summer_Olympics_Tokyo",
    ("FRA", 2024): "Summer_Olympics_Paris",
    # Winter Olympics
    ("JPN", 1998): "Winter_Olympics_Nagano",
    ("USA", 2002): "Winter_Olympics_SaltLake",
    ("ITA", 2006): "Winter_Olympics_Turin",
    ("CAN", 2010): "Winter_Olympics_Vancouver",
    ("RUS", 2014): "Winter_Olympics_Sochi",
    ("KOR", 2018): "Winter_Olympics_Pyeongchang",
    ("CHN", 2022): "Winter_Olympics_Beijing",
    # FIFA World Cup
    ("FRA", 1998): "FIFA_World_Cup_France",
    ("KOR", 2002): "FIFA_World_Cup_Korea_Japan",
    ("JPN", 2002): "FIFA_World_Cup_Korea_Japan",
    ("DEU", 2006): "FIFA_World_Cup_Germany",
    ("ZAF", 2010): "FIFA_World_Cup_South_Africa",
    ("BRA", 2014): "FIFA_World_Cup_Brazil",
    ("RUS", 2018): "FIFA_World_Cup_Russia",
    ("QAT", 2022): "FIFA_World_Cup_Qatar",
    # World Expo
    ("DEU", 2000): "World_Expo_Hannover",
    ("JPN", 2005): "World_Expo_Aichi",
    ("CHN", 2010): "World_Expo_Shanghai",
    ("ITA", 2015): "World_Expo_Milan",
    ("ARE", 2021): "World_Expo_Dubai",
}

# UN Tourism Regional Average Length of Stay (ALOS - Đêm / lượt khách)
REGIONAL_ALOS_BENCHMARK = {
    "Europe & Central Asia": 3.8,
    "East Asia & Pacific": 6.5,
    "Latin America & Caribbean": 7.2,
    "Middle East & North Africa": 5.5,
    "North America": 8.0,
    "South Asia": 9.2,
    "Sub-Saharan Africa": 6.8,
}
GLOBAL_ALOS_DEFAULT = 5.0


def load_and_merge_data() -> pd.DataFrame:
    """Nạp dữ liệu UN Tourism (1995-2024), bổ sung lịch sử tiền 2020 từ UNWTO WDI, ghép với vĩ mô World Bank."""
    if not RAW_UN_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file UN Tourism tại {RAW_UN_PATH}")
    if not EXCEL_PATH.exists():
        raise FileNotFoundError(f"Không tìm thấy file Excel tại {EXCEL_PATH}")

    print(f"[*] Đang đọc dữ liệu UN Tourism mới nhất từ: {RAW_UN_PATH}")
    df_un = pd.read_csv(RAW_UN_PATH)
    df_un = df_un.dropna(subset=["Country Code", "Year", "Tourist_Arrivals"]).copy()
    df_un["Year"] = df_un["Year"].astype(int)
    df_un["Country Code"] = df_un["Country Code"].astype(str).str.upper()
    df_un = df_un[df_un["Tourist_Arrivals"] > 0].copy()

    # Nạp lịch sử UNWTO từ sheet 1Y Forecast để bảo đảm mọi nước đều có lịch sử tiền 2017
    print(f"[*] Đang đọc lịch sử UNWTO từ: {EXCEL_PATH} (Sheet: '1Y Forecast')...")
    df_hist = pd.read_excel(EXCEL_PATH, sheet_name="1Y Forecast")
    df_hist["Country Code"] = df_hist["Country Code"].astype(str).str.upper()
    df_hist["Year"] = df_hist["Year"].astype(int)
    df_hist = df_hist.dropna(subset=["Country Code", "Year", "Tourist_Arrivals"])
    df_hist = df_hist[df_hist["Tourist_Arrivals"] > 0]

    un_sub = df_un[["Country Code", "Year", "Tourist_Arrivals"]]
    hist_sub = df_hist[["Country Code", "Year", "Tourist_Arrivals"]]
    # Ưu tiên UN Tourism bản mới, điền lịch sử cho các nước còn thiếu
    target_combined = pd.concat([hist_sub, un_sub]).drop_duplicates(subset=["Country Code", "Year"], keep="last")

    print(f"[*] Đang đọc dữ liệu kinh tế vĩ mô từ: {EXCEL_PATH} (Sheet: 'All Data')...")
    df_macro = pd.read_excel(EXCEL_PATH, sheet_name="All Data")
    df_macro["Year"] = df_macro["Year"].astype(int)
    df_macro["Country Code"] = df_macro["Country Code"].astype(str).str.upper()

    drop_cols = ["Tourist_Arrivals", "Source_Footnotes", "Indicators_Missing"]
    df_macro = df_macro.drop(columns=[c for c in drop_cols if c in df_macro.columns])

    # Ghép target với macro
    merged = pd.merge(target_combined, df_macro, on=["Country Code", "Year"], how="inner")

    # Tạo các biến trễ bậc 1 (lag_1)
    merged = merged.sort_values(by=["Country Code", "Year"]).reset_index(drop=True)
    for col in ["Tourist_Arrivals", "Tourism_Revenue", "GDP", "GDP_Growth", "Population", "Inflation", "Unemployment", "Exchange_Rate"]:
        if col in merged.columns:
            merged[f"{col.lower()}_lag1"] = merged.groupby("Country Code")[col].shift(1)

    print(f"    -> Đã ghép thành công: {len(merged):,} bản ghi ({merged['Country Code'].nunique()} quốc gia, 1995 - 2024).")
    return merged


def extract_annual_holidays(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung số ngày nghỉ lễ quốc gia trong năm."""
    print("[*] Đang trích xuất ngày nghỉ lễ quốc gia (Python 'holidays' library)...")
    hol_cache: Dict[Tuple[str, int], int] = {}
    country_medians: Dict[str, float] = {}

    for iso3 in df["Country Code"].unique():
        counts = []
        for yr in range(2000, 2025):
            try:
                h_obj = holidays.country_holidays(iso3, years=yr)
                c_val = len(h_obj)
                hol_cache[(iso3, yr)] = c_val
                counts.append(c_val)
            except Exception:
                pass
        if counts:
            country_medians[iso3] = float(np.median(counts))
        else:
            country_medians[iso3] = 11.0

    hol_list = []
    for _, row in df.iterrows():
        c = row["Country Code"]
        y = row["Year"]
        if (c, y) in hol_cache:
            hol_list.append(hol_cache[(c, y)])
        else:
            try:
                h_obj = holidays.country_holidays(c, years=y)
                val = len(h_obj)
                hol_cache[(c, y)] = val
                hol_list.append(val)
            except Exception:
                hol_list.append(int(country_medians.get(c, 11.0)))

    df["annual_holidays_count"] = hol_list
    print(f"    -> Bổ sung 'annual_holidays_count' hoàn tất (TB: {df['annual_holidays_count'].mean():.1f} ngày/năm).")
    return df


def add_mega_events(df: pd.DataFrame) -> pd.DataFrame:
    """Bổ sung sự kiện quốc tế lớn."""
    print("[*] Đang bổ sung cờ sự kiện quy mô toàn cầu (Mega-Events)...")
    is_mega = []
    event_names = []
    for _, row in df.iterrows():
        key = (row["Country Code"], row["Year"])
        if key in MEGA_EVENTS:
            is_mega.append(1)
            event_names.append(MEGA_EVENTS[key])
        else:
            is_mega.append(0)
            event_names.append("None")
    df["is_mega_event"] = is_mega
    df["mega_event_name"] = event_names
    print(f"    -> Đã phát hiện {sum(is_mega)} sự kiện toàn cầu lớn tổ chức tại các nước.")
    return df


def fetch_and_merge_covid_stringency(df: pd.DataFrame) -> pd.DataFrame:
    """Tích hợp chỉ số can thiệp phòng dịch Oxford OxCGRT cho 2020-2022."""
    print("[*] Đang tích hợp chỉ số khủng hoảng Oxford COVID-19 Stringency Index (2020 - 2022)...")
    df["covid_stringency_index"] = 0.0
    df["covid_gov_response_index"] = 0.0

    ox_url = "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_national_v1.csv"
    try:
        use_cols = ["CountryCode", "Jurisdiction", "Date", "StringencyIndex_Average", "GovernmentResponseIndex_Average"]
        df_ox = pd.read_csv(ox_url, usecols=use_cols, low_memory=False)
        df_ox = df_ox[df_ox["Jurisdiction"] == "NAT_TOTAL"].copy()
        df_ox["Year"] = df_ox["Date"].astype(str).str[:4].astype(int)

        for yr, def_str, def_gov in [(2020, 52.5, 48.0), (2021, 45.0, 42.0), (2022, 20.0, 22.0)]:
            annual = df_ox[df_ox["Year"] == yr].groupby("CountryCode").agg({
                "StringencyIndex_Average": "mean",
                "GovernmentResponseIndex_Average": "mean",
            }).reset_index()
            map_str = annual.set_index("CountryCode")["StringencyIndex_Average"].to_dict()
            map_gov = annual.set_index("CountryCode")["GovernmentResponseIndex_Average"].to_dict()

            mask = df["Year"] == yr
            df.loc[mask, "covid_stringency_index"] = df.loc[mask, "Country Code"].map(map_str).fillna(def_str)
            df.loc[mask, "covid_gov_response_index"] = df.loc[mask, "Country Code"].map(map_gov).fillna(def_gov)
        print("    -> Tích hợp chỉ số OxCGRT thành công (2020 - 2022).")
    except Exception as e:
        print(f"    [!] OxCGRT online thất bại ({e}). Sử dụng chỉ số chuẩn hóa cho 2020-2022.")
        df.loc[df["Year"] == 2020, "covid_stringency_index"] = 52.5
        df.loc[df["Year"] == 2021, "covid_stringency_index"] = 45.0
        df.loc[df["Year"] == 2022, "covid_stringency_index"] = 20.0

    return df


def add_inbound_nights_and_los(df: pd.DataFrame) -> pd.DataFrame:
    """Tích hợp số đêm lưu trú (Nights) và Length of Stay (LOS): OECD thực tế + UN Tourism ALOS."""
    print("[*] Đang tích hợp Số đêm lưu trú quốc tế (Inbound Nights) & Length of Stay (LOS)...")
    oecd_nights_map: Dict[Tuple[str, int], float] = {}

    # 1. Đọc từ clean benchmark nếu có
    if OECD_CLEAN_PATH.exists():
        try:
            df_c = pd.read_csv(OECD_CLEAN_PATH)
            df_n = df_c[df_c["oecd_measure"] == "NIGHTS_ACCOM"].dropna(subset=["oecd_value", "country_code_3", "time_period"])
            for _, r in df_n.iterrows():
                geo = str(r["country_code_3"]).strip().upper()
                try:
                    yr = int(r["time_period"])
                    val = float(r["oecd_value"])
                    if val > 0:
                        oecd_nights_map[(geo, yr)] = val
                except ValueError:
                    pass
            print(f"    -> Đã nạp {len(oecd_nights_map):,} giá trị số đêm quan sát thực tế từ OECD (2010 - 2024).")
        except Exception as e:
            print(f"    [!] Lỗi đọc OECD clean: {e}")

    nights_list = []
    los_list = []
    nights_source_list = []
    is_synthetic_nights_list = []

    for _, row in df.iterrows():
        geo = row["Country Code"]
        yr = row["Year"]
        arr = float(row["Tourist_Arrivals"])
        reg = str(row.get("Region", "Europe & Central Asia"))

        if (geo, yr) in oecd_nights_map:
            obs_n = oecd_nights_map[(geo, yr)]
            calc_los = obs_n / max(1.0, arr)
            if 1.0 <= calc_los <= 30.0:
                nights_list.append(obs_n)
                los_list.append(calc_los)
                nights_source_list.append("oecd_observed")
                is_synthetic_nights_list.append(0)
                continue

        alos = REGIONAL_ALOS_BENCHMARK.get(reg, GLOBAL_ALOS_DEFAULT)
        inc = str(row.get("Income Group", "High income"))
        if "High" in inc:
            alos = max(1.5, alos * 0.9)
        elif "Low" in inc:
            alos = alos * 1.15

        imputed_nights = arr * alos
        nights_list.append(imputed_nights)
        los_list.append(alos)
        nights_source_list.append("un_tourism_alos_imputed")
        is_synthetic_nights_list.append(1)

    df["nights_total"] = nights_list
    df["length_of_stay"] = los_list
    df["nights_source"] = nights_source_list
    df["is_synthetic_nights"] = is_synthetic_nights_list

    # Đảm bảo ràng buộc kinh tế: nights >= arrivals
    df["nights_total"] = np.maximum(df["nights_total"], df["Tourist_Arrivals"] * 1.0)
    df["nights_per_arrival_ratio"] = df["nights_total"] / np.maximum(df["Tourist_Arrivals"], 1.0)

    print(f"    -> Hoàn tất: nights_total & length_of_stay (LOS TB: {df['length_of_stay'].mean():.2f} đêm/khách).")
    print(f"    -> Tỷ lệ vi phạm nights < arrivals: {float(np.mean(df['nights_total'] < df['Tourist_Arrivals']) * 100.0):.1f}%")
    return df


def standardize_schema(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa schema tương thích chuẩn với RA-HE-DLinear pipeline."""
    print("[*] Đang chuẩn hóa schema dữ liệu...")
    df["geo"] = df["Country Code"]
    df["accommodation_type"] = "total"
    df["accommodation_type_vi"] = "Toàn bộ cơ sở lưu trú"
    df["month"] = 1
    df["date"] = pd.to_datetime(df["Year"].astype(str) + "-01-01")
    df["year"] = df["Year"]
    df["arrivals_total"] = df["Tourist_Arrivals"]

    # Phân nhóm trạng thái (HMM Regime Proxy)
    # 0: Pre-COVID Normal (<= 2019)
    # 1: Pandemic Shock (2020 - 2021)
    # 2: Post-COVID Recovery & Expansion (>= 2022)
    def assign_period(y: int) -> str:
        if y <= 2019:
            return "pre_2020"
        elif y in [2020, 2021]:
            return "2020_2021"
        else:
            return "from_2022"

    df["period_group"] = df["year"].map(assign_period)
    df["regime"] = df["period_group"].map({"pre_2020": 0, "2020_2021": 1, "from_2022": 2})

    num_exos = ["gdp_lag1", "gdp_growth_lag1", "inflation_lag1", "unemployment_lag1", "exchange_rate_lag1", "tourism_revenue_lag1", "population_lag1"]
    for col in num_exos:
        if col in df.columns:
            df[col] = df.groupby("geo")[col].transform(lambda s: s.ffill().bfill())
            df[col] = df[col].fillna(df[col].median()).fillna(0.0)

    df = df.sort_values(by=["geo", "date"]).reset_index(drop=True)
    return df


def main():
    start_time = time.time()
    print("=" * 80)
    print("🚀 BẮT ĐẦU QUY TRÌNH BỔ SUNG & LÀM GIÀU DỮ LIỆU UN TOURISM (1995 - 2024)")
    print("=" * 80)

    df = load_and_merge_data()
    df = extract_annual_holidays(df)
    df = add_mega_events(df)
    df = fetch_and_merge_covid_stringency(df)
    df = add_inbound_nights_and_los(df)
    df = standardize_schema(df)

    print("\n[*] Đang lưu trữ tệp dữ liệu đã làm giàu...")
    df.to_csv(OUTPUT_CSV, index=False)
    print(f"    💾 Đã lưu CSV: {OUTPUT_CSV} ({len(df):,} dòng, {len(df.columns)} cột)")

    try:
        df.to_parquet(OUTPUT_PARQUET, index=False)
        print(f"    💾 Đã lưu Parquet: {OUTPUT_PARQUET}")
    except Exception as e:
        print(f"    [!] Không lưu được Parquet: {e}")

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f"🎉 HOÀN TẤT LÀM GIÀU DỮ LIỆU UN TOURISM trong {elapsed:.1f} giây!")
    print(f"• Số quốc gia: {df['geo'].nunique()}")
    print(f"• Khung thời gian: {df['year'].min()} - {df['year'].max()} (1995 - 2024: Pre, During, Post-COVID)")
    print(f"• Số bản ghi phân nhóm:")
    print(df["period_group"].value_counts())
    print("=" * 80)


if __name__ == "__main__":
    main()
