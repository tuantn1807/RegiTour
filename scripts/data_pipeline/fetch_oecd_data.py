"""
Script: fetch_oecd_data.py
Mục đích: Tự động tải dữ liệu Thống kê Du lịch & Kinh tế Vĩ mô từ OECD SDMX REST API
          và lưu trữ độc lập vào thư mục 'data_oced' phục vụ đối sánh / benchmark.
Nguồn: OECD Data Explorer (https://data-explorer.oecd.org/)
Dataflows:
  1. OECD.CFE.TOU,DSD_TOURISM_INTER@DF_INBOUND,1.0  (Inbound tourism: Arrivals, Nights)
  2. OECD.CFE.TOU,DSD_TOURISM_DOM@DF_DOMESTIC,1.0    (Domestic tourism: Trips, Nights)
  3. OECD.CFE.TOU,DSD_TOURISM_KEY@DF_KEY_IND_PC,1.0  (Key indicators: GDP share, Employment)
  4. OECD.SDD.STES,DSD_STES@DF_CLI,4.0              (Consumer Confidence Index - CCI)
"""

import io
import time
from pathlib import Path
import pandas as pd
import requests

# 1. Đường dẫn thư mục đích theo yêu cầu: 'data_oced'
PROJECT_ROOT = Path(__file__).resolve().parent
DEST_DIR = PROJECT_ROOT / "data_oced"
DEST_DIR.mkdir(parents=True, exist_ok=True)

# 2. Bảng ánh xạ mã quốc gia ISO-3 sang ISO-2 (để so sánh với Eurostat)
ISO3_TO_ISO2 = {
    "AUT": "AT", "BEL": "BE", "BGR": "BG", "CYP": "CY", "CZE": "CZ",
    "DEU": "DE", "DNK": "DK", "EST": "EE", "GRC": "EL", "ESP": "ES",
    "FIN": "FI", "FRA": "FR", "HRV": "HR", "HUN": "HU", "IRL": "IE",
    "ITA": "IT", "LTU": "LT", "LUX": "LU", "LVA": "LV", "MLT": "MT",
    "NLD": "NL", "POL": "PL", "PRT": "PT", "ROU": "RO", "SWE": "SE",
    "SVN": "SI", "SVK": "SK", "GBR": "UK", "ISL": "IS", "LIE": "LI",
    "NOR": "NO", "CHE": "CH", "MNE": "ME", "MKD": "MK", "ALB": "AL",
    "SRB": "RS", "TUR": "TR", "USA": "US", "JPN": "JP", "AUS": "AU",
    "CAN": "CA", "KOR": "KR"
}

# 3. Cấu hình các endpoint dữ liệu OECD SDMX
DATASETS = [
    {
        "name": "Du lịch Quốc tế (Inbound Tourism: Arrivals & Nights)",
        "file_name": "oecd_inbound_tourism_raw.csv",
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.CFE.TOU,DSD_TOURISM_INTER@DF_INBOUND,1.0/all?startPeriod=2010",
    },
    {
        "name": "Du lịch Nội địa (Domestic Tourism: Trips & Nights)",
        "file_name": "oecd_domestic_tourism_raw.csv",
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.CFE.TOU,DSD_TOURISM_DOM@DF_DOMESTIC,1.0/all?startPeriod=2010",
    },
    {
        "name": "Chỉ số Kinh tế Du lịch Then chốt (Tỷ trọng GDP & Việc làm)",
        "file_name": "oecd_tourism_key_indicators_raw.csv",
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.CFE.TOU,DSD_TOURISM_KEY@DF_KEY_IND_PC,1.0/all?startPeriod=2010",
    },
    {
        "name": "Chỉ số Niềm tin Tiêu dùng (Consumer Confidence Index - CCI)",
        "file_name": "oecd_consumer_confidence_index_raw.csv",
        "url": "https://sdmx.oecd.org/public/rest/data/OECD.SDD.STES,DSD_STES@DF_CLI,4.0/.M.LI...AA...H?startPeriod=2010",
    },
]

HEADERS = {
    "Accept": "application/vnd.sdmx.data+csv; charset=utf-8",
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
}


def download_dataset(item: dict) -> Path | None:
    """Tải và lưu raw dataset từ OECD SDMX REST API."""
    name = item["name"]
    file_path = DEST_DIR / item["file_name"]
    url = item["url"]
    
    print(f"\n[+] Đang tải: {name} ...")
    print(f"    URL: {url}")
    
    try:
        start_time = time.time()
        response = requests.get(url, headers=HEADERS, timeout=90)
        elapsed = time.time() - start_time
        
        if response.status_code == 200 and len(response.text) > 100:
            file_path.write_text(response.text, encoding="utf-8")
            file_size_kb = file_path.stat().st_size / 1024
            print(f"    -> Đã lưu thành công: {file_path.name} ({file_size_kb:.1f} KB, mất {elapsed:.1f}s)")
            return file_path
        else:
            print(f"    [!] Lỗi tải API: Mã trạng thái {response.status_code}")
            return None
    except Exception as e:
        print(f"    [X] Lỗi kết nối khi tải {name}: {e}")
        return None


def build_clean_benchmark_table():
    """
    Trích xuất và chuẩn hóa bảng dữ liệu đối sánh tinh gọn:
    country_code_3, country_code_2, time_period, measure, oecd_value, unit
    """
    inbound_file = DEST_DIR / "oecd_inbound_tourism_raw.csv"
    if not inbound_file.exists():
        print("[!] Chưa có file inbound tourism raw để chuẩn hóa benchmark.")
        return

    print("\n[*] Đang xử lý chuẩn hóa dữ liệu đối sánh OECD...")
    df_raw = pd.read_csv(inbound_file, low_memory=False)
    
    # Chỉ giữ các dòng có số liệu thực tế
    df_valid = df_raw[df_raw["OBS_VALUE"].notnull()].copy()
    
    # Chuẩn hóa cột
    clean_cols = {
        "REF_AREA": "country_code_3",
        "TIME_PERIOD": "time_period",
        "MEASURE": "oecd_measure",
        "OBS_VALUE": "oecd_value",
        "UNIT_MEASURE": "unit",
        "ACCOMMODATION_TYPE": "accom_type"
    }
    
    available_cols = [c for c in clean_cols.keys() if c in df_valid.columns]
    df_clean = df_valid[available_cols].rename(columns=clean_cols)
    
    # Thêm mã ISO-2 để tiện map với Eurostat
    df_clean["country_code_2"] = df_clean["country_code_3"].map(ISO3_TO_ISO2)
    
    # Lưu file chuẩn hóa
    clean_output = DEST_DIR / "oecd_tourism_benchmark_clean.csv"
    df_clean.to_csv(clean_output, index=False)
    print(f"    -> Bảng dữ liệu đối sánh chuẩn hóa đã lưu tại: {clean_output}")
    print(f"    -> Số bản ghi đối sánh: {len(df_clean):,}")
    print("    -> Các chỉ tiêu đối sánh chính:", df_clean["oecd_measure"].unique().tolist())


def main():
    print("=" * 70)
    print("  OECD DATA FETCHER - BỘ DỮ LIỆU ĐỐI SÁNH ĐỘC LẬP")
    print(f"  Thư mục lưu trữ: {DEST_DIR}")
    print("=" * 70)
    
    # 1. Tải các dataset
    for item in DATASETS:
        download_dataset(item)
        time.sleep(1) # Tránh rate limit của OECD API
        
    # 2. Xây dựng bảng đối sánh sạch
    build_clean_benchmark_table()
    
    print("\n" + "=" * 70)
    print(f"[HOÀN TẤT] Toàn bộ dữ liệu OECD đã được lưu riêng trong thư mục: {DEST_DIR}")
    print("Không ảnh hưởng hay trộn lẫn vào bộ dữ liệu Eurostat 54 cột hiện có.")
    print("=" * 70)


if __name__ == "__main__":
    main()
