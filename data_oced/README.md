# 📊 THƯ MỤC DỮ LIỆU ĐỐI SÁNH OECD (OECD BENCHMARK DATA)

Thư mục này được tổ chức **hoàn toàn độc lập** với dữ liệu huấn luyện Eurostat (`data/processed/`), dùng cho mục đích **so sánh, kiểm định chéo (Cross-source Validation)** và **đối chuẩn vĩ mô**.

---

## 1. Danh mục các tệp dữ liệu đã tải

| Tên tệp | Kích thước | Nguồn dữ liệu OECD (SDMX Dataflow) | Nội dung / Chỉ số chính |
| :--- | :--- | :--- | :--- |
| `oecd_inbound_tourism_raw.csv` | ~1.65 MB | `OECD.CFE.TOU,DSD_TOURISM_INTER@DF_INBOUND` | Lượt khách quốc tế (`ARRIVALS`) và Đêm lưu trú quốc tế (`NIGHTS_ACCOM`). |
| `oecd_domestic_tourism_raw.csv` | ~1.11 MB | `OECD.CFE.TOU,DSD_TOURISM_DOM@DF_DOMESTIC` | Lượt khách du lịch nội địa (`TRIPS`) và Đêm lưu trú (`NIGHTS_ACCOM`). |
| `oecd_tourism_key_indicators_raw.csv` | ~211 KB | `OECD.CFE.TOU,DSD_TOURISM_KEY@DF_KEY_IND_PC` | Tỷ trọng du lịch trong GDP (`GDP_SH`), GVA (`GVA_SH`), Việc làm (`EMP_SH`). |
| `oecd_consumer_confidence_index_raw.csv` | ~312 KB | `OECD.SDD.STES,DSD_STES@DF_CLI` | Chỉ số niềm tin người tiêu dùng theo tháng (`CCI / CLI`). |
| `oecd_tourism_benchmark_clean.csv` | ~520 KB | Đã chuẩn hóa qua `fetch_oecd_data.py` | Bảng đối chuẩn tinh gọn (10,231 records) đã map mã ISO-2 tương thích với Eurostat. |
| `comparison_eurostat_vs_oecd.csv` | ~35 KB | Sinh ra từ `compare_eurostat_oecd.py` | Bảng so sánh trực tiếp từng cặp (Nước x Năm) giữa Eurostat và OECD. |

---

## 2. Cách chạy và cập nhật

### Tải lại dữ liệu mới nhất từ OECD SDMX API:
```bash
python fetch_oecd_data.py
```

### Chạy so sánh đối chuẩn giữa Eurostat và OECD:
```bash
python compare_eurostat_oecd.py
```
