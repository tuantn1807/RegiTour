# 📁 THƯ MỤC DỮ LIỆU THÔ (RAW DATA SOURCES & DOWNLOAD COMMANDS)

Thư mục này lưu trữ các tệp dữ liệu thô tải về từ các cơ quan thống kê chính thức quốc tế.

---

## 1. Dữ liệu UN Tourism (Toàn cầu 1995 – 2024, 209 quốc gia)
* **Nguồn:** *UN Tourism Statistics Database* (Tổ chức Du lịch Liên Hợp Quốc).
* **Chỉ số:** Lượt khách du lịch quốc tế qua biên giới (*Arrivals of tourists from abroad*).
* **Đích lưu:** `data/raw/un_tourism_inbound_arrivals.csv`

### Cách 1: Chạy script bash có sẵn
```bash
bash scripts/data_pipeline/fetch_un_tourism_data.sh
```

### Cách 2: Chạy script python có sẵn
```bash
python scripts/data_pipeline/fetch_un_tourism_data.py
```

### Cách 3: Lệnh terminal curl trực tiếp
```bash
curl -fsSL "https://ourworldindata.org/grapher/international-tourist-trips.csv?v=1&csvType=full&useColumnShortNames=false" \
  -o data/raw/un_tourism_inbound_arrivals.csv
```

### Lệnh tải Metadata JSON kèm theo:
```bash
curl -fsSL "https://ourworldindata.org/grapher/international-tourist-trips.metadata.json?v=1&csvType=full&useColumnShortNames=false" \
  -o data/raw/un_tourism_inbound_arrivals.metadata.json
```

---

## 2. Dữ liệu Eurostat (Châu Âu 2010 – 2026, 38 điểm đến theo tháng)
* **Nguồn:** *Eurostat Tourism Database* (`https://ec.europa.eu/eurostat/en/web/tourism/information-data`).
* **Các tệp:**
  - `tour_occ_arm__custom_22994996_spreadsheet.xlsx`: Lượt khách (`ARRIVALS`).
  - `tour_occ_nim__custom_22994437_linear_2_0.csv`: Số đêm lưu trú (`NIGHTS`).
  - `estat_tour_occ_mnor.tsv.gz`: Công suất phòng lưu trú.

---

## 3. Dữ liệu OECD Tourism
* **Nguồn:** *OECD Data Explorer* (chuẩn SDMX REST API).
* **Lệnh tải:**
```bash
python scripts/data_pipeline/fetch_oecd_data.py
```

---

## 4. Dữ liệu chính sách phòng dịch Covid-19 (Oxford OxCGRT)
* **Nguồn:** Đại học Oxford (*Oxford COVID-19 Government Response Tracker*).
* **Lệnh tải trực tiếp:**
```bash
curl -fsSL "https://raw.githubusercontent.com/OxCGRT/covid-policy-dataset/main/data/OxCGRT_compact_national_v1.csv" \
  -o data/raw/OxCGRT_compact_national_v1.csv
```
