# 🏛️ RegiTour: AI Framework for Multi-Country Tourism Demand Forecasting
> **Đề tài NCKH 2027**: Xây dựng mô hình trí tuệ nhân tạo dự báo nhu cầu du lịch và lưu trú dựa trên dữ liệu chuỗi thời gian  
> **Kiến trúc cốt lõi**: Modern Linear Framework (**RA-HE-DLinear / RA-Exo-DLinear**) & Deep Hybrid Framework (**RA-HE-BiLSTM**)

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **RegiTour** (**RA-HE-DLinear**) is an end-to-end hierarchical regime-aware linear decomposition forecasting framework designed for multi-frequency tourism demand econometrics across 38 European countries (Eurostat monthly panel) and 203 global economies (World Bank annual panel).

---

## 📌 Key Methodological Innovations

1. **Unsupervised Market Regime Identification (Gaussian HMM):** Classifies structural market conditions into *Normal Growth*, *Disruption Shock*, and *Post-Crisis Recovery*, injecting dynamic context embeddings across downstream forecasting modules.
2. **Core Temporal-Exogenous Engine (RA-Exo-DLinear):** Extends moving-average series decomposition (DLinear) via Generalized Additive Model (GAM) formulation with Feature-wise Linear Modulation (FiLM) on trend and dual-stream linear projections for deterministic future calendar events and past macroeconomic covariates.
3. **Graph-Free Spatial Factorization (TRMF):** Decomposes multi-country panel series into low-rank spatial clusters ($r=4$) with autoregressive temporal regularization, capturing cross-border tourism spillovers without subjective graph priors.
4. **Coupled Consistency Convex Optimization:** Enforces strict physical domain bounds ($\text{Nights} \ge \text{Arrivals} \times \text{LOS}_{\min}$) via anchor projection $\Pi_{\mathcal{C}}$ and coupled training loss $\mathcal{L}_{\text{cons}}$, controlling raw neural violations to $0.15\%$ and minimizing economic band deviation to **16.0%**.
5. **Distribution-Free Uncertainty Quantification (DCQF):** Delivers calibrated conformal prediction intervals $[P_{10}, P_{90}]$ with an empirical finite-sample coverage of **94.2%**.
6. **Multi-Frequency Panel Adaptability:** Seamlessly accommodates both monthly regional panels (Eurostat: $H=12, L=24$) and annual macroeconomic panels (World Bank: $H=1, L=4$).

---

## 📊 Empirical Benchmark Results

Evaluated on the out-of-sample test partition (**2024–2026**, 4,712 observations across 38 European countries, trích xuất tự động từ [`results/tables/benchmark_model_comparison.csv`](results/tables/benchmark_model_comparison.csv)):

| Mô hình đối sánh | Lớp mô hình (Family) | Arrivals WAPE | Nights WAPE | Cận vật lý (< 1.0) | Lệch dải kinh tế | Đánh giá khoa học |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **Seasonal Naive (S-Naive)** | Thống kê cổ điển | 27.30% | 32.73% | 0.05% | 28.8% | Baseline kinh tế lượng |
| **Standard LightGBM** | Machine Learning | 30.37% | 34.28% | 0.19% | 49.2% | Bị phân mảnh dải tỷ lệ kinh tế (49.2%) |
| **Standard XGBoost** | Machine Learning | 34.13% | 42.36% | 0.34% | 37.7% | Sai số cao & lệch dải kinh tế |
| **Vanilla DLinear (AAAI 2023)** | Modern Linear LTSF | 45.38% | 25.61% | 0.05% | 28.8% | Mù biến ngoại sinh & cú sốc thị trường |
| **Vanilla NLinear (AAAI 2023)** | Modern Linear LTSF | 32.10% | 49.38% | 0.69% | 29.2% | Khử trôi tốt nhưng không có liên kết kinh tế |
| **Classical LSTM (1997)** | Classical Deep Learning | 108.53% | 113.16% | 0.05% | 99.4% | Quá khớp & bão hòa trạng thái sau Covid |
| **Vanilla BiLSTM (2013)** | Classical Deep Learning | 108.53% | 113.16% | 0.05% | 99.4% | Quá khớp & bão hòa trạng thái sau Covid |
| **CatBoost-Regime (Ablation)** | Machine Learning | 40.27% | 44.86% | 0.82% | 44.9% | Vi phạm logic kinh tế và dải lưu trú |
| **RA-Exo-DLinear (Proposed Linear ⭐)** | Linear LTSF (Proposed ⭐) | **18.29%** | **25.87%** | 0.21% | 21.5% | Lõi Linear độc lập siêu nhẹ |
| **RA-HE-DLinear (Proposed Linear Framework ⭐)** | **Linear Framework (Proposed ⭐)** | **`14.59%`** | **`25.00%`** | **0.00%*** | **`16.0%`** | **🏆 Khung phân rã tuyến tính tối ưu toàn diện** |

*\* Ghi chú: Sau chiếu, vi phạm cận vật lý đạt chính xác 0.00% (0/4,712). Tỷ lệ dự đoán cần can thiệp sửa của mô hình thô trước chiếu là 0.15% (7/4,712). Thước đo bản chất kinh tế là độ lệch dải lưu trú lịch sử (16.0%, tối ưu toàn bảng).*

---

## 🚀 Quick Start

### 1. Installation
```bash
git clone https://github.com/tuantn1807/RegiTour.git
cd RegiTour
pip install -r requirements.txt
```

### 2. Run End-to-End Forecasting Pipeline
Khởi chạy quy trình dự báo du lịch với kiến trúc tuyến tính tối ưu **RA-HE-DLinear**:
```bash
python run_linear_pipeline.py --dataset eurostat
python run_linear_pipeline.py --dataset worldbank
```

### 3. Run Baseline Benchmark Evaluation
So sánh đối đầu giữa các mô hình qua các thế hệ y văn:
```bash
python benchmark_models.py
```

### 4. Run XAI and Policy Simulation
```bash
python run_xai_analysis.py
```

---

## 📄 Scientific Manuscripts
The complete academic manuscripts are located in the `paper/` directory:
- [English Manuscript (`paper/paper_en.tex`)](paper/paper_en.tex) (IEEE Transactions format)
- [Vietnamese Manuscript (`paper/paper_vi.tex`)](paper/paper_vi.tex) (IEEE format)

---

## 👥 Authors
- **Nguyen-Tuan Tran** (Department of Computer Science and Information Systems)
- **Hoai-Bao Phan** (Department of Computer Science and Information Systems)
- Research Project on AI-Driven Tourism Econometrics (NCKH 2027)

Mọi chi tiết chuyên sâu được tổ chức tại thư mục [`docs/`](docs/):
* 🌐 **Tổng quan hệ thống & cơ sở lý thuyết:** Xem [`docs/03_architecture/overview_system.md`](docs/03_architecture/overview_system.md)
* 📊 **Hướng dẫn 54 biến dữ liệu:** Xem [`docs/02_data/DATA_GUIDE.md`](docs/02_data/DATA_GUIDE.md)
* 🏛️ **Đặc tả kiến trúc mô hình Hybrid:** Xem [`docs/03_architecture/MODEL_ARCHITECTURE.md`](docs/03_architecture/MODEL_ARCHITECTURE.md)
* ⚡ **Đặc tả mô hình Linear mới:** Xem [`docs/03_architecture/RA_EXO_DLINEAR_SPEC.md`](docs/03_architecture/RA_EXO_DLINEAR_SPEC.md)
* 📐 **Quy chuẩn lập trình dự án:** Xem [`docs/CODING_CONVENTIONS.md`](docs/CODING_CONVENTIONS.md)

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
