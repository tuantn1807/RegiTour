# 📚 TRUNG TÂM TÀI LIỆU KHOA HỌC DỰ ÁN REGITOUR
> Hệ thống tài liệu đặc tả kiến trúc, dữ liệu và hướng dẫn kỹ thuật cho Đề tài NCKH 2027.

---

## 🗂️ MỤC LỤC TÀI LIỆU

### 1. Dữ liệu & Kỹ thuật đặc trưng (`docs/02_data/`)
* [DATA_GUIDE.md](02_data/DATA_GUIDE.md): Cẩm nang chi tiết 54 biến dữ liệu Eurostat, HICP, Oxford Covid và 38 quốc gia Châu Âu.

### 2. Thiết kế Kiến trúc Mô hình (`docs/03_architecture/`)
* [overview_system.md](03_architecture/overview_system.md): Tổng quan hệ thống và các nền tảng khoa học cốt lõi (System Overview & Scientific Foundations).
* [MODEL_ARCHITECTURE.md](03_architecture/MODEL_ARCHITECTURE.md): Đặc tả kiến trúc học sâu phân tầng đa mô hình **RA-HE-BiLSTM** (Nhánh `main`).
* [RA_EXO_DLINEAR_SPEC.md](03_architecture/RA_EXO_DLINEAR_SPEC.md): Đặc tả kiến trúc tuyến tính hiện đại **RA-Exo-DLinear** và framework toàn diện **RA-HE-DLinear** (Nhánh `feature/linear`).

### 3. Quy chuẩn Lập trình (`docs/`)
* [CODING_CONVENTIONS.md](CODING_CONVENTIONS.md): Quy chuẩn viết mã nguồn Python, cấu trúc module, và bảo đảm tính tái lập của thực nghiệm.

---

## 🏆 TỔNG KẾT HIỆU NĂNG MÔ HÌNH TRÊN TẬP TEST (2024–2026)

| Framework / Mô hình | Lớp mô hình | Arrivals WAPE | Nights WAPE | Tỷ lệ vi phạm logic | Tài liệu đặc tả |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **RA-HE-DLinear ⭐** | **Linear Framework (Proposed ⭐)** | **14.67%** | **25.00%** | **0.0%** | [RA_EXO_DLINEAR_SPEC.md](03_architecture/RA_EXO_DLINEAR_SPEC.md) |
| **RA-Exo-DLinear ⭐** | Linear Model (Proposed ⭐) | **15.00%** | **25.17%** | **0.0%** | [RA_EXO_DLINEAR_SPEC.md](03_architecture/RA_EXO_DLINEAR_SPEC.md) |
| **RA-HE-BiLSTM ⭐** | Deep Hybrid (Proposed ⭐) | **19.88%** | **26.16%** | **0.0%** | [MODEL_ARCHITECTURE.md](03_architecture/MODEL_ARCHITECTURE.md) |
