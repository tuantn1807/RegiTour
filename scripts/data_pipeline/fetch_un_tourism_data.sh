#!/usr/bin/env bash
# ==============================================================================
# Script: fetch_un_tourism_data.sh
# Mục đích: Tải bộ dữ liệu UN Tourism (Inbound Arrivals 1995 - 2024, 209 quốc gia)
# Nguồn: UN Tourism Statistics Database (xử lý & chuẩn hóa bởi Our World in Data)
# Cách dùng:
#     bash scripts/data_pipeline/fetch_un_tourism_data.sh
# ==============================================================================

set -euo pipefail

# Xác định thư mục gốc của project
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST_DIR="${PROJECT_ROOT}/data/raw"
DEST_FILE="${DEST_DIR}/un_tourism_inbound_arrivals.csv"
META_FILE="${DEST_DIR}/un_tourism_inbound_arrivals.metadata.json"

mkdir -p "${DEST_DIR}"

CSV_URL="https://ourworldindata.org/grapher/international-tourist-trips.csv?v=1&csvType=full&useColumnShortNames=false"
JSON_URL="https://ourworldindata.org/grapher/international-tourist-trips.metadata.json?v=1&csvType=full&useColumnShortNames=false"

echo "================================================================================"
echo "🌐 ĐANG TẢI DỮ LIỆU GỐC UN TOURISM (1995 - 2024)"
echo "   Nguồn: UN Tourism Statistics Database"
echo "   Đích:  ${DEST_FILE}"
echo "================================================================================"

echo "[*] Tải tệp dữ liệu CSV..."
curl -fsSL "${CSV_URL}" -o "${DEST_FILE}"

echo "[*] Tải tệp Metadata JSON..."
curl -fsSL "${JSON_URL}" -o "${META_FILE}"

echo "================================================================================"
echo "✅ TẢI THÀNH CÔNG!"
echo "   - Tệp dữ liệu: ${DEST_FILE} ($(wc -l < "${DEST_FILE}") dòng)"
echo "   - Tệp mô tả:   ${META_FILE}"
echo "================================================================================"
