# RegiTour: Hierarchical Regime-Aware Hybrid Deep Learning Framework for Tourism Demand Econometrics & Forecasting

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

### 2. Run End-to-End Forecasting Pipeline
Train the five-layer architecture and output multi-horizon predictions:
```bash
python run_system_pipeline.py
```

### 3. Run Baseline Benchmark Evaluation
Compare against Seasonal Naive, XGBoost, LightGBM, Classical LSTM, and Vanilla BiLSTM:
```bash
python benchmark_models.py
```

### 4. Run XAI and What-If Policy Simulation
Compute TreeSHAP attributions, generate cross-temporal attention heatmaps, and run counterfactual scenarios:
```bash
python run_xai_analysis.py
```

---

## 📄 Scientific Manuscripts
The complete academic drafts detailing the mathematical formulation, failure mode analysis of RNNs, and empirical validation are located in the `paper/` directory:
- [English Manuscript (`paper/paper_en.tex`)](paper/paper_en.tex)
- [Vietnamese Manuscript (`paper/paper_vi.tex`)](paper/paper_vi.tex)

---

## 👥 Authors
- **Nguyen-Tuan Tran** (Department of Computer Science and Information Systems)
- **Hoai-Bao Phan** (Department of Computer Science and Information Systems)
- Research Project on AI-Driven Tourism Econometrics (NCKH 2027)

---

## 📜 License
This project is licensed under the MIT License - see the LICENSE file for details.
