<<<<<<< Updated upstream
# RegiTour: Hierarchical Regime-Aware Hybrid Deep Learning Framework for Tourism Demand Econometrics & Forecasting
=======
# 🏛️ RegiTour: AI Framework for European Tourism Demand Forecasting
> **Đề tài NCKH 2027**: Xây dựng mô hình trí tuệ nhân tạo dự báo nhu cầu du lịch và lưu trú dựa trên dữ liệu chuỗi thời gian  
> **Kiến trúc cốt lõi**: Modern Linear Framework (**RA-HE-DLinear / RA-Exo-DLinear**) & Deep Hybrid Framework (**RA-HE-BiLSTM**)
>>>>>>> Stashed changes

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> **RegiTour** (**RA-HE-BiLSTM**) is an end-to-end hierarchical regime-aware hybrid forecasting framework designed for continental-scale tourism demand econometrics across 38 European countries, 4 NACE accommodation sectors, and 54 macroeconomic, crisis, and calendar variables (2010–2026).

---

## 📌 Key Methodological Innovations

1. **Unsupervised Market Regime Identification (Gaussian HMM):** Classifies structural market conditions into *Normal Growth*, *Disruption Shock*, and *Post-Crisis Recovery*, injecting dynamic context embeddings across all downstream forecasting engines.
2. **Multi-Resolution Wavelet Signal Decomposition (MODWT):** Applies Maximal Overlap Discrete Wavelet Transform (Symlet-4) to separate series into orthogonal low-frequency macroeconomic trends ($\mathbf{Y}_{\text{trend}}$) and high-frequency seasonal oscillations ($\mathbf{Y}_{\text{seasonal}}$).
3. **M-RA-BiLSTM Deep Backbone with Cross-Temporal Attention:**
   - *Regime-Conditioned Forget Gates:* Injects regime embeddings directly into LSTM forget gates, preventing hidden state saturation and catastrophic error compounding following exogenous shocks (e.g., COVID-19).
   - *Dual-Stream Cross-Temporal Attention:* Queries past hidden state trajectories using known deterministic future holiday and calendar schedules ($H=12$ months ahead) to accurately capture summer peak shifts without recursive error drift.
4. **Graph-Free Spatial Factorization (TRMF):** Decomposes continental tourism series ($38 \times T$) into low-rank natural spatial clusters ($r=4$) with autoregressive temporal regularization, scaled by historical accommodation shares $s_{geo, accom}$ to eliminate distorted geographic border graphs.
5. **Coupled Consistency Convex Optimization:** Enforces strict domain compliance ($\text{nights}/\text{arrivals} \ge 1.0$), reducing logical economic violations from 17.8% down to **0.0%**.
6. **Distribution-Free Uncertainty Quantification (DCQF):** Delivers calibrated conformal prediction bounds $[P_{10}, P_{90}]$ with an empirical finite-sample coverage of **91.4%**.
7. **Dual-Layer Explainable AI (XAI):** Incorporates native TreeSHAP attribution, cross-temporal attention heatmaps, and counterfactual What-If policy simulations for accommodation price interventions (HICP CP112).

---

## 📊 Empirical Benchmark Results

Evaluated on the out-of-sample test partition (**2024–2026**, 4,712 observations across 38 European countries):

| Model Architecture | Methodology Family | Arrivals WAPE | Arrivals MAE | Nights WAPE | Nights MAE | Inconsistency Rate (EIR) | $\Delta$ vs S-Naive |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Seasonal Naive (S-Naive)** | Statistical Baseline | 27.30% | 397,503 | 32.73% | 1,287,411 | 0.0% | Baseline |
| **Standard LightGBM** | Gradient Tree Boosting | 30.37% | 442,199 | 34.28% | 1,348,103 | 7.7% | -3.07% |
| **Standard XGBoost** | Gradient Tree Boosting | 34.13% | 496,861 | 42.36% | 1,666,128 | 6.2% | -6.83% |
| **Classical LSTM** | Recurrent Neural Net | 108.53% | 1,580,071 | 113.16% | 4,450,857 | 0.0% | -81.23% |
| **Vanilla BiLSTM** | Recurrent Neural Net | 108.53% | 1,580,071 | 113.16% | 4,450,856 | 0.0% | -81.23% |
| **CatBoost-Regime** | Tabular Standalone | 40.27% | 586,231 | 44.86% | 1,764,508 | 17.8% | -12.97% |
| **RegiTour (RA-HE-BiLSTM)** | **Proposed Hierarchical Hybrid** | **18.61%** | **271,010** | **25.30%** | **994,918** | **0.0%** | **+8.69%** |

### Breakdown by NACE Accommodation Type (RegiTour)
- **Hotels and similar (`hotel`):** Arrivals WAPE **14.70%** | Nights WAPE **22.66%**
- **Total Industry (`total`):** Arrivals WAPE **18.25%** | Nights WAPE **22.78%**
- **Campsites (`camping`):** Arrivals WAPE **26.10%** | Nights WAPE **35.00%**
- **Short-stay rentals (`short_stay`):** Arrivals WAPE **34.27%** | Nights WAPE **38.43%**

---

## 🔍 Explainable AI (XAI) & Policy Simulations

- **TreeSHAP Feature Attribution:** Identifies the accommodation price index (`hicp_accommodation`) as the primary driver of demand (37.85% relative importance), followed by pandemic stringency (28.07%) and calendar weekend distribution (19.49%).
- **Cross-Temporal Attention Heatmaps:** Confirms seasonal synchronization with a 10.37% attention concentration on historical 12-month and 24-month lags.
- **Counterfactual What-If Simulations:** Evaluates price elasticity ($\epsilon = -0.72$), indicating that a $+10\%$ surge in the accommodation price index induces a $-7.20\%$ contraction in hotel arrivals across key European markets (France, Spain, Italy, Germany).

---

## 📂 Repository Structure

```text
<<<<<<< Updated upstream
RegiTour/
├── data/
│   ├── features/
│   │   └── tourism_europe_features.parquet   # ML feature store (38 countries, 54 features)
│   └── processed/
│       └── holidays_europe_detailed.csv      # European public and school holiday registry
├── data_oced/                                # OECD validation benchmark datasets
├── paper/
│   ├── paper_en.tex                          # IEEE / Elsevier format English manuscript
│   ├── paper_vi.tex                          # Vietnamese scientific draft manuscript
│   └── figures/                              # High-resolution publication figures (300 DPI)
├── results/
│   ├── tables/                               # Benchmark comparison and error tables (CSV)
│   ├── xai/                                  # SHAP plots and Attention Heatmaps
│   ├── evaluation_report.md                  # Comprehensive empirical evaluation report
│   └── xai_report.md                         # Detailed XAI and What-If analysis report
├── src/
│   ├── data/                                 # Panel loader, temporal split, sliding window
│   ├── features/                             # MODWT wavelet decomposition module
│   ├── models/                               # HMM, CatBoost, M-RA-BiLSTM, TRMF, DCQF, MoE
│   ├── explainability/                       # TreeSHAP, Attention Visualizer, What-If Engine
│   └── pipeline/                             # End-to-end forecasting pipeline runner
├── benchmark_models.py                       # Out-of-sample baseline benchmark suite
├── run_system_pipeline.py                    # Master CLI pipeline execution script
├── run_xai_analysis.py                       # Master XAI & What-If evaluation script
├── evaluate_results.py                       # Metrics aggregation and error diagnostics
├── requirements.txt                          # Python dependencies
└── README.md
=======
nckh_2027/
├── docs/                                 # 📚 TRUNG TÂM TÀI LIỆU KHOA HỌC
│   ├── README.md                         # Mục lục tra cứu tài liệu
│   ├── 02_data/                          # Hướng dẫn dữ liệu 54 biến Eurostat & API (DATA_GUIDE.md)
│   └── 03_architecture/                  # Đặc tả kiến trúc mô hình (overview_system, MODEL_ARCHITECTURE, RA_EXO_DLINEAR_SPEC)
├── data/                                 # 📊 CƠ SỞ HẠ TẦNG DỮ LIỆU
│   ├── raw/                              # Dữ liệu thô Eurostat (.tsv.gz, .xlsx)
│   ├── processed/                        # Dữ liệu đã làm sạch & phân rã ngày lễ (.csv, .xlsx)
│   ├── features/                         # Feature Store nén chuẩn Parquet (tourism_europe_features.parquet)
│   └── data_oced/                        # Dữ liệu đối chuẩn độc lập từ OECD
├── scripts/                              # ⚙️ SCRIPTS THU THẬP & ETL DỮ LIỆU
│   └── data_pipeline/                    # Pipeline trích xuất Eurostat, Holidays, HICP, Oxford
├── src/                                  # 🧠 MÃ NGUỒN CỐT LÕI (PYTHON MODULES)
│   ├── data/                             # Dataset loader, sliding window temporal splits
│   ├── features/                         # Phân rã tín hiệu Wavelet (MODWT)
│   ├── models/                           # Các mô hình AI:
│   │   ├── ra_exo_dlinear.py             # Proposed Linear: RA-Exo-DLinear & Vanilla DLinear/NLinear ⭐
│   │   ├── baselines.py                  # Baselines: S-Naive, XGBoost, LightGBM, Classical LSTM
│   │   ├── catboost_regime.py            # Trụ cột dữ liệu bảng vĩ mô
│   │   ├── trmf_spatial.py               # Phân rã ma trận lan tỏa không gian Schengen
│   │   ├── dcqf_conformal.py             # Dự báo khoảng tin cậy không phụ thuộc phân phối
│   │   ├── moe_gating.py                 # Mạng gating phân bổ trọng số động MoE
│   │   ├── coupled_optimizer.py          # Bộ tối ưu ràng buộc logic kinh tế
│   │   ├── m_ra_bilstm.py                # Mô hình Deep BiLSTM Attention
│   │   └── regime_hmm.py                 # Phân đoạn 3 trạng thái thị trường (Gaussian HMM)
│   ├── explainability/                   # Engine giải thích XAI (TreeSHAP, DiCE What-If)
│   └── pipeline/                         # Bộ điều phối Pipeline đầu-cuối
├── paper/                                # 📝 BẢN THẢO BÀI BÁO KHOA HỌC SCOPUS/ISI (TeX)
├── results/                              # 📈 BẢNG SỐ LIỆU ĐỐI SÁNH & BÁO CÁO THỰC NGHIỆM
├── run_system_pipeline.py                # 🚀 Entrypoint: Chạy toàn bộ pipeline hệ thống
├── benchmark_models.py                   # 🏁 Entrypoint: Chạy thực nghiệm đối kháng 10 mô hình
├── evaluate_results.py                   # 📊 Entrypoint: Đánh giá chỉ số sai số & chẩn đoán
├── run_xai_analysis.py                   # 🔍 Entrypoint: Chạy phân tích giải thích SHAP & What-If
├── config.py                             # ⚙️ File cấu hình tham số hệ thống
└── requirements.txt                      # Danh mục thư viện phụ thuộc
>>>>>>> Stashed changes
```

---

## 🚀 Quickstart

### 1. Installation
Clone the repository and install required dependencies:
```bash
git clone https://github.com/tuantn1807/RegiTour.git
cd RegiTour
pip install -r requirements.txt
```

<<<<<<< Updated upstream
### 2. Run End-to-End Forecasting Pipeline
Train the five-layer architecture and output multi-horizon predictions:
```bash
python run_system_pipeline.py
=======
### Bước 2: Chạy Pipeline Dự báo Hệ thống
Khởi chạy toàn bộ quy trình dự báo du lịch Châu Âu với kiến trúc tuyến tính tối ưu **RA-HE-DLinear**:
```bash
python3 run_linear_pipeline.py
>>>>>>> Stashed changes
```
*(Hoặc chạy pipeline Deep Learning nguyên bản trên nhánh main: `python3 run_system_pipeline.py`)*

<<<<<<< Updated upstream
### 3. Run Baseline Benchmark Evaluation
Compare against Seasonal Naive, XGBoost, LightGBM, Classical LSTM, and Vanilla BiLSTM:
=======
### Bước 3: Chạy Thực nghiệm Đối kháng Đối đầu (Benchmark 11 Mô hình)
So sánh đối đầu giữa **11 mô hình** qua 4 thế hệ y văn (S-Naive, XGBoost, LightGBM, LSTM, Vanilla DLinear/NLinear, CatBoost, RA-Exo-DLinear, RA-HE-BiLSTM, RA-HE-DLinear):
>>>>>>> Stashed changes
```bash
python benchmark_models.py
```

### 4. Run XAI and What-If Policy Simulation
Compute TreeSHAP attributions, generate cross-temporal attention heatmaps, and run counterfactual scenarios:
```bash
python run_xai_analysis.py
```

---

<<<<<<< Updated upstream
## 📄 Scientific Manuscripts
The complete academic drafts detailing the mathematical formulation, failure mode analysis of RNNs, and empirical validation are located in the `paper/` directory:
- [English Manuscript (`paper/paper_en.tex`)](paper/paper_en.tex)
- [Vietnamese Manuscript (`paper/paper_vi.tex`)](paper/paper_vi.tex)
=======
## 📊 4. BẢNG TỔNG HỢP HIỆU NĂNG ĐỐI SÁNH (BENCHMARK RESULTS)

Kết quả thực nghiệm trên tập Test ngoại mẫu độc lập (**2024–2026**, 4.712 quan sát, 38 quốc gia Châu Âu, trích xuất tự động từ [`results/tables/benchmark_model_comparison.csv`](results/tables/benchmark_model_comparison.csv)):

| Mô hình đối sánh | Lớp mô hình (Family) | Arrivals WAPE | Nights WAPE | Tỷ lệ vi phạm logic | Đánh giá khoa học |
| :--- | :--- | :---: | :---: | :---: | :--- |
| **Seasonal Naive (S-Naive)** | Thống kê cổ điển | 27.30% | 32.73% | 0.0% | Baseline kinh tế lượng |
| **Standard XGBoost** | Machine Learning | 34.13% | 42.36% | 6.2% | Bị lỗi vi phạm logic (nights < arrivals) |
| **Standard LightGBM** | Machine Learning | 30.37% | 34.28% | 7.7% | Bị lỗi vi phạm logic (nights < arrivals) |
| **Vanilla DLinear (AAAI 2023)** | Modern Linear LTSF | 45.38% | 49.36% | 0.0% | Mù biến ngoại sinh & cú sốc thị trường |
| **Vanilla NLinear (AAAI 2023)** | Modern Linear LTSF | 32.10% | 49.38% | 0.7% | Khử trôi tốt nhưng không có liên kết kinh tế |
| **Classical LSTM (1997)** | Classical Deep Learning | 108.53% | 113.16% | 0.0% | Quá khớp & bão hòa trạng thái sau Covid |
| **Vanilla BiLSTM (2013)** | Classical Deep Learning | 108.53% | 113.16% | 0.0% | Quá khớp & bão hòa trạng thái sau Covid |
| **CatBoost-Regime (Ablation)** | Machine Learning | 40.27% | 44.86% | 17.8% | Vi phạm logic kinh tế nghiêm trọng (17.8%) |
| **RA-HE-BiLSTM (Proposed Hybrid ⭐)** | Deep Learning (Proposed) | **19.88%** | **26.16%** | **0.0%** | **Mô hình Deep Hybrid từ nhánh main** |
| **RA-Exo-DLinear (Proposed Linear ⭐)** | Linear LTSF (Proposed ⭐) | **15.00%** | **25.17%** | **0.0%** | **Lõi Linear độc lập siêu nhẹ & khóa logic** |
| **RA-HE-DLinear (Proposed Linear Framework ⭐)** | **Linear Framework (Proposed ⭐)** | **`14.67%`** | **`25.00%`** | **`0.0%`** | **🏆 SOTA toàn bảng, train < 40s (Green AI)** |
>>>>>>> Stashed changes

---

## 👥 Authors
- **Nguyen-Tuan Tran** (Department of Computer Science and Information Systems)
- **Hoai-Bao Phan** (Department of Computer Science and Information Systems)
- Research Project on AI-Driven Tourism Econometrics (NCKH 2027)

<<<<<<< Updated upstream
---
=======
Mọi chi tiết chuyên sâu được tổ chức tại thư mục [`docs/`](docs/):
* 🌐 **Tổng quan hệ thống & cơ sở lý thuyết:** Xem [`docs/03_architecture/overview_system.md`](docs/03_architecture/overview_system.md)
* 📊 **Hướng dẫn 54 biến dữ liệu:** Xem [`docs/02_data/DATA_GUIDE.md`](docs/02_data/DATA_GUIDE.md)
* 🏛️ **Đặc tả kiến trúc mô hình Hybrid (Main):** Xem [`docs/03_architecture/MODEL_ARCHITECTURE.md`](docs/03_architecture/MODEL_ARCHITECTURE.md)
* ⚡ **Đặc tả mô hình Linear mới (Feature/Linear):** Xem [`docs/03_architecture/RA_EXO_DLINEAR_SPEC.md`](docs/03_architecture/RA_EXO_DLINEAR_SPEC.md)
* 📐 **Quy chuẩn lập trình dự án:** Xem [`docs/CODING_CONVENTIONS.md`](docs/CODING_CONVENTIONS.md)
>>>>>>> Stashed changes

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
