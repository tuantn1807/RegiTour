# 🏛️ RegiTour Software Architecture & Enterprise Integration Guide

## 1. Tổng Quan Kiến Trúc Phần Mềm (Software Architecture Overview)

RegiTour được chuẩn hóa và nâng cấp từ một nguyên mẫu thực nghiệm học thuật thành một **Hệ Thống Phần Mềm Quyết Định Thông Minh (Enterprise Decision Support System — EDSS)** hoàn chỉnh, bao gồm 4 tầng dịch vụ phần mềm:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       GIAO DIỆN NGƯỜI DÙNG & TÍCH HỢP                       │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │  Streamlit DSS Web App  │  │ FastAPI REST Services │  │ RegiTour CLI   │ │
│  │   (Port 8501 / UI Hub)  │  │  (Port 8000 / Swagger)│  │   (Terminal)   │ │
│  └────────────┬────────────┘  └───────────┬───────────┘  └────────┬───────┘ │
└───────────────┼───────────────────────────┼───────────────────────┼─────────┘
                │                           │                       │
┌───────────────▼───────────────────────────▼───────────────────────▼─────────┐
│                       TẦNG NGHIỆP VỤ & QUYẾT ĐỊNH (SERVICES)                 │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │  Executive Report Engine│  │ Operational Alerting  │  │ DMO Decision   │ │
│  │   (Markdown / JSON)     │  │   (Threshold & Rules) │  │ Engine (4 Lớp) │ │
│  └────────────┬────────────┘  └───────────┬───────────┘  └────────┬───────┘ │
└───────────────┼───────────────────────────┼───────────────────────┼─────────┘
                │                           │                       │
┌───────────────▼───────────────────────────▼───────────────────────▼─────────┐
│                    TẦNG THUẬT TOÁN KHOA HỌC & BẢO CHỨNG TOÁN HỌC             │
│  ┌─────────────────────────┐  ┌───────────────────────┐  ┌────────────────┐ │
│  │ RA-Exo-DLinear Core     │  │ Coupled Optimizer Π_C │  │ Dual TRMF / HMM│ │
│  │ (Decomposition + Exo)   │  │ (Zero Business Contrad│  │ & DCQF Conformal│ │
│  └────────────┬────────────┘  └───────────┬───────────┘  └────────┬───────┘ │
└───────────────┼───────────────────────────┼───────────────────────┼─────────┘
                │                           │                       │
┌───────────────▼───────────────────────────▼───────────────────────▼─────────┐
│                         TẦNG DỮ LIỆU & KIỂM THỬ                             │
│  • Eurostat Panel (38 Destinations x 4 NACE Categories x 2010–2026)         │
│  • Bộ kiểm thử tự động toàn diện (45 Unit & Integration Tests — 100% Pass)  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Các Chức Năng Phần Mềm Đã Được Bổ Sung

### 2.1. Cổng Dịch Vụ REST API Doanh Nghiệp (`src/api/`)
Xây dựng trên nền **FastAPI**, **Starlette** và **Pydantic V2**:
- **Tài liệu tự động**: Cung cấp giao diện Swagger UI tương tác tại `/docs`, ReDoc tại `/redoc`, và schema JSON chuẩn tại `/openapi.json`.
- **CORS Middleware**: Hỗ trợ gọi API an toàn từ mọi ứng dụng Web/Mobile bên ngoài (PMS khách sạn, ứng dụng sân bay, cổng thông tin du lịch quốc gia).
- **Danh mục 8 API Endpoints chuẩn hóa:**
  1. `GET /api/v1/health`: Kiểm tra sức khỏe hệ thống, số lượng điểm đến (38 quốc gia), các mốc origin có sẵn.
  2. `GET /api/v1/destinations`: Tra cứu danh sách 38 quốc gia Châu Âu, bảng mã NACE Rev. 2 (I551, I552, I553, Total).
  3. `POST /api/v1/forecast/plan`: Hoạch định đồng bộ lưu lượng khách đến ($Y^{arr}$) và số đêm lưu trú ($Y^{nig}$), tính toán LOS thực tế và số ca vi phạm được toán tử $\Pi_C$ bảo vệ.
  4. `POST /api/v1/capacity/staffing`: Phân tích công suất buồng phòng và điều độ nhân sự thời vụ (FTE), tự động cảnh báo quá tải ($\ge 85\%$) và gợi ý cửa sổ bảo trì cơ sở vật chất.
  5. `POST /api/v1/simulation/stress-test`: Mô phỏng kịch bản căng thẳng kinh tế vĩ mô (lạm phát giá phòng HICP shock, siết chặt biên giới Oxford stringency).
  6. `GET /api/v1/market-regime/timeline`: Phân tích chuỗi xác suất trạng thái thị trường du lịch qua Gaussian HMM (Bình thường, Khủng hoảng, Phục hồi).
  7. `GET /api/v1/recovery/benchmark`: Đối sánh tỷ lệ phục hồi của 38 quốc gia Châu Âu so với mốc cơ sở 2019 pre-crisis.
  8. `POST /api/v1/optimizer/pi-c`: Dịch vụ tối ưu toán học độc lập: Chiếu trực tiếp các cặp chuỗi thô bất kỳ vào tập lồi khả thi $\mathcal{C} = \{(a, n) \mid n \ge a \ge 0\}$, đảm bảo $LOS \ge 1.0$.
  9. `POST /api/v1/reports/generate`: Xuất báo cáo điều hành chuẩn hóa (Markdown hoặc JSON).

### 2.2. Công Cụ Dòng Lệnh Đa Năng (`cli.py`)
Giao diện dòng lệnh chuẩn POSIX/GNU phục vụ DevOps, tự động hóa cronjob và sysadmin:
```bash
# Khởi chạy REST API server
python cli.py api --host 0.0.0.0 --port 8000

# Khởi chạy Streamlit Dashboard
python cli.py dashboard --port 8501

# Chạy kế hoạch điều hành lưu lượng và lưu file JSON/CSV
python cli.py plan --geo FR --accom total --horizon 12 --output plan_fr.json

# Phân bổ nhân sự NACE và cửa sổ bảo trì
python cli.py staffing --geo FR --horizon 12 --output staffing_fr.json

# Chạy mô phỏng Stress Test vĩ mô
python cli.py stress --geo CY --hicp 20 --stringency 30

# Tạo báo cáo điều hành DMO chuẩn Markdown
python cli.py report --geo DK --horizon 12 --format markdown --output dmo_dk.md

# Kiểm toán toán học toàn diện trên toàn bộ 38 quốc gia
python cli.py audit
```

### 2.3. Động Cơ Khởi Tạo Báo Cáo Điều Hành DMO (`src/services/report_generator.py`)
- Tổng hợp tự động các chỉ số KPI vận hành hàng đầu.
- Kiểm toán toán học $\Pi_C$: đếm số vi phạm thô được khắc phục, xác nhận 0% mâu thuẫn nghiệp vụ.
- Bảng lịch trình nhân sự theo phân khúc NACE và cửa sổ sửa chữa bảo trì.
- Xuất dữ liệu đa định dạng: Markdown (dành cho cấp lãnh đạo) và JSON (dành cho tích hợp phần mềm).

### 2.4. Phân Hệ Tích Hợp Trên Web Dashboard (`dashboard/app.py`)
Bổ sung tab điều hành thứ 7: **"🔌 Tích hợp API & Báo cáo Tự động (Integration Hub)"**:
- **Cảnh báo vận hành tự động (Alert Engine)**: Tự động phát hiện nguy cơ quá tải tải trọng phục vụ ($\ge 85\%$), kích hoạt khuyến nghị phân luồng cửa khẩu và cảnh báo thiếu hụt nhân sự trước 60 ngày.
- **1-Click Executive Report**: Người dùng có thể nhấn nút để render trực tiếp báo cáo điều hành và tải về dưới dạng `.md` hoặc `.json`.
- **Developer Integration Sandbox**: Cung cấp mẫu mã nguồn trực tiếp (Python `requests`, cURL bash, JavaScript `fetch`) tương ứng với từng điểm đến và cấu hình đang chọn trên giao diện.

### 2.5. Hệ Thống Kiểm Thử Tự Động (Test Suite)
- Bổ sung `tests/test_api.py`, `tests/test_cli.py`, `tests/test_report_generator.py`.
- Tổng cộng **45 bài kiểm thử đơn vị & tích hợp (Unit & Integration Tests)**, bao gồm cả smoke test giả lập giao diện Streamlit qua `streamlit.testing.v1.AppTest`. Đạt tỷ lệ vượt qua **100% (45/45 tests OK)**.

