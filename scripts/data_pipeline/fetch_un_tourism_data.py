"""
Script: fetch_un_tourism_data.py
Mục đích: Tự động tải dữ liệu gốc UN Tourism (1995 - 2024, 209 quốc gia/vùng lãnh thổ)
          từ nguồn UN Tourism Statistics Database (Open Data) vào 'data/raw/'.
Cách dùng:
    python scripts/data_pipeline/fetch_un_tourism_data.py
"""

import sys
import urllib.request
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEST_DIR = PROJECT_ROOT / "data" / "raw"
DEST_DIR.mkdir(parents=True, exist_ok=True)

CSV_URL = "https://ourworldindata.org/grapher/international-tourist-trips.csv?v=1&csvType=full&useColumnShortNames=false"
JSON_URL = "https://ourworldindata.org/grapher/international-tourist-trips.metadata.json?v=1&csvType=full&useColumnShortNames=false"

DEST_CSV = DEST_DIR / "un_tourism_inbound_arrivals.csv"
DEST_JSON = DEST_DIR / "un_tourism_inbound_arrivals.metadata.json"


def download_file(url: str, dest_path: Path, desc: str):
    print(f"[*] Đang tải {desc} từ: {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dest_path, "wb") as out:
        out.write(resp.read())
    size_kb = dest_path.stat().st_size / 1024
    print(f"    -> Đã lưu: {dest_path} ({size_kb:.1f} KB)")


def main():
    print("=" * 80)
    print("🚀 TẢI DỮ LIỆU GỐC UN TOURISM (1995 - 2024)")
    print("=" * 80)
    download_file(CSV_URL, DEST_CSV, "Dữ liệu CSV lượt khách quốc tế")
    download_file(JSON_URL, DEST_JSON, "Metadata JSON")
    print("=" * 80)
    print(f"✅ HOÀN TẤT! Dữ liệu đã sẵn sàng tại: {DEST_CSV}")
    print("=" * 80)


if __name__ == "__main__":
    main()
