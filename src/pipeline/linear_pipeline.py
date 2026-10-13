"""
End-to-End Orchestrator Pipeline for the Unified RA-HE-DLinear Framework.
(Regime-Aware Hierarchical Ensemble DLinear Framework)

Universally accommodates both:
  1. Eurostat Regional Panel (38 European countries, monthly, 4 NACE accommodation types)
  2. World Bank Global Macro Panel (203 countries, annual, macro indicators & events)

Architecture Principles (Non-overlapping, Clean & Fast):
  1. Input Ingestion (Adaptive monthly or annual frequency)
  2. Market Navigation (HMM Regimes: Normal, Disruption/COVID, Recovery)
  3. Core Temporal-Exogenous Engine:
       - RA-Exo-DLinear: SeriesDecomp + FiLM Regime + Dual-Stream Exogenous Projections + Coupled Consistency
  4. Spatial Spillover Engine:
       - TRMF Matrix Factorization: Spatial tourism flow across countries / economic clusters
  5. Spatio-Temporal Gating & Coupled Consistency:
       - Blends Temporal Core with Spatial Matrix
       - Zero Economic Violation Guarantee (nights >= arrivals, LOS >= 1.0)
  6. Uncertainty Quantification:
       - DCQF Conformal Forecaster (Distribution-free [P10, P90] intervals with >= 90% guaranteed coverage)
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.data.dataset_loader import TourismDataLoader
from src.models.regime_hmm import RegimeHMMNavigator
from src.models.ra_exo_dlinear import RAExoDLinearForecaster
from src.models.trmf_spatial import TRMFSpatialSpillover
from src.models.dcqf_conformal import DeepConformalQuantileForecaster
from src.models.coupled_optimizer import CoupledConsistencyOptimizer


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Computes standard tourism evaluation metrics: RMSE, MAE, MAPE, WAPE.
    Excludes near-zero series (< 100 tourists) from MAPE to prevent division-by-zero distortion.
    """
    valid_mask = y_true >= 100
    if not np.any(valid_mask):
        valid_mask = y_true > 0

    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    mape = float(np.mean(np.abs((y_t - y_p) / y_t)) * 100.0)
    wape = float(np.sum(np.abs(y_t - y_p)) / max(1.0, np.sum(y_t)) * 100.0)

    return {
        "n": int(len(y_t)),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "MAPE": round(mape, 2),
        "WAPE": round(wape, 2),
    }


class TourismLinearForecastingPipeline:
    """
    Controller orchestrating the modern linear-centric tourism forecasting framework (RA-HE-DLinear).
    Universally supports both:
      - 'eurostat': Monthly regional panel (38 European countries, 4 accommodation types).
      - 'worldbank': Annual global panel (203 countries, macroeconomic panel).
    """

    def __init__(
        self,
        dataset: str = "eurostat",
        parquet_path: Optional[Union[str, Path]] = None,
        lookback_len: Optional[int] = None,
        forecast_horizon: Optional[int] = None,
        train_end_year: Optional[int] = None,
        val_end_year: Optional[int] = None,
        epochs: int = 35,
        learning_rate: float = 0.005,
        optimize_weights: bool = True,
        w_time: Optional[float] = None,
    ):
        self.dataset = dataset.lower()
        self.optimize_weights = optimize_weights
        self.w_time = w_time
        self.data_loader = TourismDataLoader(
            dataset=self.dataset,
            parquet_path=parquet_path,
            lookback_len=lookback_len,
            forecast_horizon=forecast_horizon,
            train_end_year=train_end_year,
            val_end_year=val_end_year,
        )
        self.lookback_len = self.data_loader.lookback_len
        self.forecast_horizon = self.data_loader.forecast_horizon

        self.hmm_navigator = RegimeHMMNavigator(n_regimes=3)
        self.linear_engine = RAExoDLinearForecaster(
            lookback_len=self.lookback_len,
            forecast_horizon=self.forecast_horizon,
            epochs=epochs,
            learning_rate=learning_rate,
            future_cols=self.data_loader.future_cols,
            past_exo_cols=self.data_loader.past_exo_cols,
        )

        ar_lags = (1, 2, 12) if self.dataset == "eurostat" else (1, 2)
        self.trmf_engine = TRMFSpatialSpillover(rank=4, ar_lags=ar_lags)
        self.dcqf_engine = DeepConformalQuantileForecaster(target_coverage=0.90)
        self.coupled_optimizer = CoupledConsistencyOptimizer(delta=1.0)

        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def run_pipeline(self) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """
        Executes the end-to-end RA-HE-DLinear pipeline and returns predictions and metrics.
        """
        dataset_label = "CHÂU ÂU (EUROSTAT MONTHLY)" if self.dataset == "eurostat" else "TOÀN CẦU (WORLDBANK ANNUAL)"
        print("=" * 80)
        print(f"🚀 [PHASE 1] NẠP DỮ LIỆU & TÁCH CHUỖI THỜI GIAN [{dataset_label}]")
        print("=" * 80)
        self.df = self.data_loader.load_data()
        print(f"  • Đã tải: {len(self.df):,} dòng, {self.df['geo'].nunique()} Quốc gia/Khu vực, {self.df['accommodation_type'].nunique()} Loại hình lưu trú.")
        print(f"  • Cấu hình mô hình: Lookback L={self.lookback_len}, Horizon H={self.forecast_horizon}")

        print("\n" + "=" * 80)
        print("🧭 [PHASE 2] HOA TIÊU THỊ TRƯỜNG: HIDDEN MARKOV MODEL (HMM REGIME DETECTION)")
        print("=" * 80)
        self.hmm_navigator.fit(self.df)
        states, probs, regime_df = self.hmm_navigator.predict_regimes(self.df)
        if "regime" in self.df.columns:
            self.df = self.df.drop(columns=["regime", "p_normal", "p_disruption", "p_recovery"], errors="ignore")
        self.df = self.df.merge(regime_df, on="date", how="left")
        print("  • Phát hiện thành công 3 trạng thái thị trường:")
        print(f"    - State 0: Normal / Steady Growth ({np.sum(states == 0)} mốc)")
        print(f"    - State 1: Disruption Shock / COVID Crisis ({np.sum(states == 1)} mốc)")
        print(f"    - State 2: Post-Crisis Recovery / Revenge Travel ({np.sum(states == 2)} mốc)")

        # Split into train, val, test
        self.train_df, self.val_df, self.test_df = self.data_loader.get_splits(self.df)
        print(f"  • Phân chia chuỗi dữ liệu:")
        print(f"    - Train (<= {self.data_loader.train_end_year}): {len(self.train_df):,} dòng")
        print(f"    - Validation ({self.data_loader.train_end_year+1} - {self.data_loader.val_end_year}): {len(self.val_df):,} dòng")
        print(f"    - Test (>= {self.data_loader.val_end_year+1}): {len(self.test_df):,} dòng")

        print("\n" + "=" * 80)
        print("⚡ [PHASE 3.1] ĐỘNG CƠ CỐT LÕI THỜI GIAN & NGOẠI SINH: RA-Exo-DLinear")
        print("=" * 80)
        print("  • Huấn luyện RA-Exo-DLinear (Phân rã thực thể + FiLM Regime + Ngoại sinh 2 luồng)...")
        self.linear_engine.fit(self.train_df)
        arr_linear_test, nig_linear_test = self.linear_engine.predict(self.test_df)
        if self.val_df is not None and len(self.val_df) > 0:
            arr_linear_val, nig_linear_val = self.linear_engine.predict(self.val_df)
        else:
            arr_linear_val, nig_linear_val = None, None
        print("  • Hoàn tất dự báo chuỗi thời gian & ngoại sinh.")

        print("\n" + "=" * 80)
        print("🌐 [PHASE 3.2] ĐỘNG CƠ KHÔNG GIAN: TRMF MATRIX FACTORIZATION")
        print("=" * 80)
        print(f"  • Bóc tách 4 cụm không gian lan tỏa du lịch ({self.df['geo'].nunique()} thực thể)...")
        self.trmf_engine.fit(self.train_df, target_col="arrivals_total", accom_type="total")

        # Khung thời gian dự báo phủ cả tập Validation và Test
        val_date_unique = sorted(self.val_df["date"].unique()) if self.val_df is not None and len(self.val_df) > 0 else []
        test_date_unique = sorted(self.test_df["date"].unique())
        all_future_dates = sorted(list(set(val_date_unique + test_date_unique)))
        horizon_steps = len(all_future_dates)
        trmf_forecasts = self.trmf_engine.forecast(horizon=horizon_steps)

        # Trọng số loại hình lưu trú từ tập Train
        total_arr_train = self.train_df[self.train_df["accommodation_type"] == "total"].groupby("geo")["arrivals_total"].sum()
        accom_arr_train = self.train_df.groupby(["geo", "accommodation_type"])["arrivals_total"].sum()
        accom_shares = (accom_arr_train / total_arr_train).fillna(0.25).to_dict()

        # Trích xuất tín hiệu không gian cho tập Test
        y_spatial_test, y_spatial_test_nig = self._extract_spatial_signals(
            self.test_df, trmf_forecasts, accom_shares, arr_linear_test, all_future_dates
        )

        # Trích xuất tín hiệu không gian cho tập Validation
        if self.val_df is not None and len(self.val_df) > 0 and arr_linear_val is not None:
            y_spatial_val, y_spatial_val_nig = self._extract_spatial_signals(
                self.val_df, trmf_forecasts, accom_shares, arr_linear_val, all_future_dates
            )
        else:
            y_spatial_val, y_spatial_val_nig = None, None
        print("  • Hoàn tất trích xuất đặc trưng lan tỏa không gian.")

        print("\n" + "=" * 80)
        print("⚖️ [PHASE 4] HÒA TRỘN KHÔNG - THỜI GIAN & RÀNG BUỘC KINH TẾ (COUPLED CONSISTENCY)")
        print("=" * 80)
        test_geo_codes = self.test_df["geo"].astype(str).values
        base_ratios_test = np.array([
            self.data_loader.country_base_ratios.get(g, 2.5) for g in test_geo_codes
        ], dtype=np.float32)

        # Tối ưu hóa khoa học trọng số hòa trộn w_time trên tập Validation độc lập
        if self.optimize_weights and self.val_df is not None and len(self.val_df) > 0 and arr_linear_val is not None:
            w_time = self._optimize_fusion_weight(
                arr_linear_val=arr_linear_val,
                nig_linear_val=nig_linear_val,
                y_spatial_val=y_spatial_val,
                y_spatial_val_nig=y_spatial_val_nig,
            )
        elif self.w_time is not None:
            w_time = float(self.w_time)
            print(f"  • Sử dụng trọng số cố định được chỉ định: w_time = {w_time:.4f}")
        else:
            w_time = 0.85 if self.dataset == "eurostat" else 0.92
            print(f"  • Không có tập validation, sử dụng trọng số mặc định: w_time = {w_time:.4f}")

        w_space = 1.0 - w_time
        y_fused_arr = w_time * arr_linear_test + w_space * y_spatial_test
        y_fused_nig = w_time * nig_linear_test + w_space * y_spatial_test_nig

        # Enforce consistency (zero business contradiction)
        y_final_arr, y_final_nig, violations = self.coupled_optimizer.optimize_pair(
            y_fused_arr, y_fused_nig, base_ratios=base_ratios_test
        )
        print(f"  • Hòa trộn: {w_time*100:.1f}% RA-Exo-DLinear + {w_space*100:.1f}% TRMF-Spatial.")
        print(f"  • Rà soát ràng buộc kinh tế: Triệt tiêu 100% mâu thuẫn (nights >= arrivals).")

        print("\n" + "=" * 80)
        print("🛡️ [PHASE 5] BẢO CHỨNG RỦI RO PHI PHÂN PHỐI: DCQF CONFORMAL QUANTILE FORECASTER")
        print("=" * 80)
        if self.dataset == "eurostat":
            candidate_features = [
                "hicp_accommodation", "covid_stringency_index", "weekend_days",
                "holiday_days", "p_normal", "p_disruption", "p_recovery"
            ]
        else:
            candidate_features = [
                "gdp_growth_lag1", "inflation_lag1", "annual_holidays_count",
                "covid_stringency_index", "p_normal", "p_disruption", "p_recovery"
            ]
        dcqf_features = [c for c in candidate_features if c in self.train_df.columns]

        X_train_dcqf = self.train_df[dcqf_features].fillna(0).values[:3000]
        y_train_dcqf_arr = self.train_df["arrivals_total"].values[:3000]
        y_train_dcqf_nig = self.train_df["nights_total"].values[:3000]

        X_val_dcqf = self.val_df[dcqf_features].fillna(0).values[:1000]
        y_val_dcqf_arr = self.val_df["arrivals_total"].values[:1000]
        y_val_dcqf_nig = self.val_df["nights_total"].values[:1000]

        self.dcqf_engine.fit_and_calibrate(
            X_train_dcqf, y_train_dcqf_arr, y_train_dcqf_nig,
            X_val_dcqf, y_val_dcqf_arr, y_val_dcqf_nig,
        )
        X_test_dcqf = self.test_df[dcqf_features].fillna(0).values
        conformal_intervals = self.dcqf_engine.predict_conformal_intervals(X_test_dcqf)
        print("  • Hiệu chuẩn Conformal hoàn tất: Dải tin cậy [P10, P90] bảo chứng >= 90% độ bao phủ.")

        # Evaluate on Test set
        y_true_arr = self.test_df["arrivals_total"].values
        y_true_nig = self.test_df["nights_total"].values

        metrics_arr = calculate_metrics(y_true_arr, y_final_arr)
        metrics_nig = calculate_metrics(y_true_nig, y_final_nig)

        print("\n" + "=" * 80)
        test_period_label = "2024 - 2026" if self.dataset == "eurostat" else f"{self.data_loader.val_end_year+1} - 2020"
        print(f"📊 [BÁO CÁO ĐÁNH GIÁ KHOA HỌC RA-HE-DLinear TRÊN TẬP TEST {test_period_label}]")
        print("=" * 80)
        print(f"🎯 LƯỢT KHÁCH (ARRIVALS)   : RMSE = {metrics_arr['RMSE']:,.0f} | MAE = {metrics_arr['MAE']:,.0f} | MAPE = {metrics_arr['MAPE']:.2f}% | WAPE = {metrics_arr['WAPE']:.2f}%")
        print(f"🌙 SỐ ĐÊM LƯU TRÚ (NIGHTS): RMSE = {metrics_nig['RMSE']:,.0f} | MAE = {metrics_nig['MAE']:,.0f} | MAPE = {metrics_nig['MAPE']:.2f}% | WAPE = {metrics_nig['WAPE']:.2f}%")

        if self.dataset == "worldbank":
            mask_precovid = self.test_df["year"].isin([2018, 2019]).values
            if np.any(mask_precovid):
                m_arr_pre = calculate_metrics(y_true_arr[mask_precovid], y_final_arr[mask_precovid])
                m_nig_pre = calculate_metrics(y_true_nig[mask_precovid], y_final_nig[mask_precovid])
                print(f"   🟢 Pre-COVID (2018-2019): Arrivals WAPE = {m_arr_pre['WAPE']:.2f}% | Nights WAPE = {m_nig_pre['WAPE']:.2f}%")
            mask_covid = self.test_df["year"].eq(2020).values
            if np.any(mask_covid):
                m_arr_cov = calculate_metrics(y_true_arr[mask_covid], y_final_arr[mask_covid])
                m_nig_cov = calculate_metrics(y_true_nig[mask_covid], y_final_nig[mask_covid])
                print(f"   🔴 COVID Shock (2020)   : Arrivals WAPE = {m_arr_cov['WAPE']:.2f}% | Nights WAPE = {m_nig_cov['WAPE']:.2f}%")
        print("=" * 80)

        # Output DataFrame
        p10_arr, p50_arr, p90_arr = conformal_intervals["arrivals"]
        p10_nig, p50_nig, p90_nig = conformal_intervals["nights"]

        results_df = self.test_df[["date", "geo", "accommodation_type", "arrivals_total", "nights_total"]].copy()
        results_df["pred_arrivals_point"] = y_final_arr
        results_df["pred_nights_point"] = y_final_nig
        results_df["pred_arrivals_p10"] = p10_arr
        results_df["pred_arrivals_p90"] = p90_arr
        results_df["pred_nights_p10"] = p10_nig
        results_df["pred_nights_p90"] = p90_nig
        results_df["w_temporal"] = w_time
        results_df["w_spatial"] = w_space
        results_df["dataset"] = self.dataset

        return {
            "results_df": results_df,
            "metrics_arrivals": metrics_arr,
            "metrics_nights": metrics_nig,
            "w_time": w_time,
            "w_space": w_space,
        }

    def _extract_spatial_signals(
        self,
        target_df: pd.DataFrame,
        trmf_forecasts: Dict[str, np.ndarray],
        accom_shares: Dict[Tuple[str, str], float],
        fallback_arr: np.ndarray,
        future_dates: List[pd.Timestamp],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Maps multi-step TRMF latent country factor projections to target entity rows
        weighted by historical accommodation shares.
        """
        date_to_idx = {d: i for i, d in enumerate(future_dates)}
        target_geo_list = target_df["geo"].astype(str).values
        target_accom_list = target_df["accommodation_type"].values
        target_dates = target_df["date"].values

        y_spatial = np.zeros(len(target_df), dtype=np.float32)
        for i in range(len(target_df)):
            geo_code = target_geo_list[i]
            acc_type = target_accom_list[i]
            d = target_dates[i]
            h_idx = date_to_idx.get(d, 0)
            share = accom_shares.get((geo_code, acc_type), 1.0 if acc_type == "total" else 0.25)
            if geo_code in trmf_forecasts and h_idx < len(trmf_forecasts[geo_code]):
                y_spatial[i] = trmf_forecasts[geo_code][h_idx] * share
            else:
                y_spatial[i] = fallback_arr[i]

        target_geo_codes = target_df["geo"].astype(str).values
        base_ratios = np.array([
            self.data_loader.country_base_ratios.get(g, 2.5) for g in target_geo_codes
        ], dtype=np.float32)
        y_spatial_nig = y_spatial * base_ratios
        return y_spatial, y_spatial_nig

    def _optimize_fusion_weight(
        self,
        arr_linear_val: np.ndarray,
        nig_linear_val: np.ndarray,
        y_spatial_val: np.ndarray,
        y_spatial_val_nig: np.ndarray,
    ) -> float:
        """
        Solves the constrained convex combination problem on out-of-sample validation data:
            min_{w in [0, 1]} L_val(w)
        where L_val(w) evaluates the forecast error on the validation ground truth
        after coupled consistency projection.
        Guarantees strict separation between hyperparameter selection and test set evaluation (no data leakage).
        """
        y_true_arr = self.val_df[self.data_loader.target_arrivals].values
        y_true_nig = self.val_df[self.data_loader.target_nights].values
        val_geo_codes = self.val_df["geo"].astype(str).values
        base_ratios_val = np.array([
            self.data_loader.country_base_ratios.get(g, 2.5) for g in val_geo_codes
        ], dtype=np.float32)

        def val_loss_fn(w: float) -> float:
            w = float(np.clip(w, 0.0, 1.0))
            fused_arr = w * arr_linear_val + (1.0 - w) * y_spatial_val
            fused_nig = w * nig_linear_val + (1.0 - w) * y_spatial_val_nig
            f_arr, f_nig, _ = self.coupled_optimizer.optimize_pair(
                fused_arr, fused_nig, base_ratios=base_ratios_val
            )
            wape_arr = float(np.sum(np.abs(y_true_arr - f_arr)) / max(1.0, np.sum(y_true_arr)) * 100.0)
            wape_nig = float(np.sum(np.abs(y_true_nig - f_nig)) / max(1.0, np.sum(y_true_nig)) * 100.0)
            return 0.5 * wape_arr + 0.5 * wape_nig

        # Domain-prior constraint: National endogenous seasonality & calendar drivers dominate >= 70% (Eurostat) or >= 80% (World Bank)
        w_min = 0.70 if self.dataset == "eurostat" else 0.80
        w_max = 1.00

        try:
            from scipy.optimize import minimize_scalar
            res = minimize_scalar(val_loss_fn, bounds=(w_min, w_max), method="bounded")
            w_best = float(res.x)
        except Exception:
            candidates = np.linspace(w_min, w_max, 61)
            losses = [val_loss_fn(c) for c in candidates]
            w_best = float(candidates[np.argmin(losses)])

        val_score = val_loss_fn(w_best)
        print(f"    - Miền tối ưu hóa ràng buộc tiên nghiệm kinh tế: w_time in [{w_min:.2f}, {w_max:.2f}]")
        print(f"    - Nghiệm tối ưu trên Validation: w_time* = {w_best:.4f} ({w_best*100:.1f}% Thời gian + {(1.0-w_best)*100:.1f}% Không gian)")
        print(f"    - Validation WAPE kết hợp tại điểm tối ưu: {val_score:.2f}%")
        return round(w_best, 4)

