"""
Pipeline: un_tourism_dlinear_pipeline.py
Mục đích: Thực thi framework RA-HE-DLinear trên tập dữ liệu thường niên toàn cầu UN Tourism (1995 - 2024)
          đã được bổ sung và làm giàu (tourism_un_tourism_features.parquet / enriched.csv).
Phục vụ đối sánh khoa học:
  - Dataset A: Eurostat Monthly Regional Panel (38 nước, 4 loại hình, chu kỳ tháng).
  - Dataset B: UN Tourism Annual Global Panel (197 nước, vĩ mô năm 1995 - 2024: Pre, During, Post-COVID).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.ra_exo_dlinear import RAExoDLinearForecaster
from src.models.regime_hmm import RegimeHMMNavigator
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.dcqf_conformal import (
    ResidualConformalForecaster,
    split_validation_temporally,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PARQUET = PROJECT_ROOT / "data" / "features" / "tourism_un_tourism_features.parquet"
ENRICHED_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_un_tourism_enriched.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số đánh giá sai số chuẩn."""
    valid_mask = (y_true > 0) & np.isfinite(y_true) & np.isfinite(y_pred)
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

    if len(y_t) == 0:
        return {"n": 0, "RMSE": 0.0, "MAE": 0.0, "WAPE": 0.0, "MAPE": 0.0}

    rmse = float(np.sqrt(np.mean((y_t - y_p) ** 2)))
    mae = float(np.mean(np.abs(y_t - y_p)))
    wape = float(np.sum(np.abs(y_t - y_p)) / max(1.0, np.sum(y_t)) * 100.0)
    mape = float(np.mean(np.abs((y_t - y_p) / np.maximum(y_t, 100.0))) * 100.0)

    return {
        "n": int(len(y_t)),
        "RMSE": round(rmse, 2),
        "MAE": round(mae, 2),
        "WAPE": round(wape, 2),
        "MAPE": round(mape, 2),
    }


class UNTourismLinearForecastingPipeline:
    """
    Bộ điều phối thực thi RA-HE-DLinear trên dữ liệu thường niên UN Tourism (1995 - 2024).
    """

    FUTURE_COLS = ["annual_holidays_count", "is_mega_event"]
    PAST_EXO_COLS = [
        "gdp_lag1", "gdp_growth_lag1", "inflation_lag1",
        "unemployment_lag1", "exchange_rate_lag1", "covid_stringency_index"
    ]

    def __init__(
        self,
        train_end_year: int = 2016,
        val_end_year: int = 2018,
        lookback_len: int = 4,
        forecast_horizon: int = 1,
        epochs: int = 30,
        learning_rate: float = 0.005,
        kernel_size: int = 13,
    ):
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.lookback_len = lookback_len
        self.forecast_horizon = forecast_horizon
        self.kernel_size = kernel_size

        self.hmm_navigator = RegimeHMMNavigator(n_regimes=3)
        self.linear_engine = RAExoDLinearForecaster(
            lookback_len=lookback_len,
            forecast_horizon=forecast_horizon,
            epochs=epochs,
            learning_rate=learning_rate,
            kernel_size=kernel_size,
            future_cols=self.FUTURE_COLS,
            past_exo_cols=self.PAST_EXO_COLS,
        )
        self.dcqf_engine = ResidualConformalForecaster(target_coverage=0.90)
        self.coupled_optimizer = CoupledConsistencyOptimizer(delta=1.0)

        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Tải dữ liệu UN Tourism đã được làm giàu."""
        if FEATURES_PARQUET.exists():
            df = pd.read_parquet(FEATURES_PARQUET)
        elif ENRICHED_CSV.exists():
            df = pd.read_csv(ENRICHED_CSV)
        else:
            raise FileNotFoundError(
                f"Chưa có tệp dữ liệu đã làm giàu! Vui lòng chạy: "
                f"python scripts/data_pipeline/enrich_un_tourism_dataset.py"
            )

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["geo", "date"]).reset_index(drop=True)
        self.df = df
        return df

    def run_pipeline(self) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """Thực thi pipeline hoàn chỉnh trên UN Tourism Panel."""
        print("=" * 80)
        print("🌍 KHỞI ĐỘNG RA-HE-DLinear TRÊN BỘ DỮ LIỆU THƯỜNG NIÊN UN TOURISM (1995 - 2024)")
        print("   Mục đích: Đánh giá độ bền bỉ đa chế độ (Pre-COVID, COVID Shock, Post-COVID Recovery)")
        print("=" * 80)

        self.df = self.load_data()
        print(f"  • Đã nạp: {len(self.df):,} bản ghi từ {self.df['geo'].nunique()} nền kinh tế.")

        # 1. Phân chia tập dữ liệu
        self.train_df = self.df[self.df["year"] <= self.train_end_year].copy()
        self.val_df = self.df[(self.df["year"] > self.train_end_year) & (self.df["year"] <= self.val_end_year)].copy()
        self.test_df = self.df[self.df["year"] > self.val_end_year].copy()

        val_tuning_df, val_calibration_df = split_validation_temporally(self.val_df)

        print("\n[*] Thiết kế phân chia thời gian (Temporal Split without Lookahead):")
        print(f"    - Train (1995 - {self.train_end_year}): {len(self.train_df):,} dòng ({self.train_df['date'].nunique()} năm)")
        print(f"    - Validation ({self.train_end_year+1} - {self.val_end_year}): {len(self.val_df):,} dòng ({self.val_df['date'].nunique()} năm)")
        print(f"    - Test ({self.val_end_year+1} - 2024): {len(self.test_df):,} dòng (Bao gồm Pre-Covid, Covid Shock và Post-Covid Recovery!)")

        # 2. Huấn luyện HMM phân tầng chế độ
        print("\n[*] Giai đoạn 1: Huấn luyện bộ điều hướng HMM (Regime Navigator)...")
        self.hmm_navigator.fit(self.train_df)

        # 3. Huấn luyện Động cơ tuyến tính RA-Exo-DLinear
        print("\n[*] Giai đoạn 2: Tối ưu mạng nơ-ron RA-Exo-DLinear...")
        self.linear_engine.fit(self.train_df)

        # 4. Dự báo & Ràng buộc nhất quán vận hành Pi_C
        print("\n[*] Giai đoạn 3: Suy luận dự báo & Chiếu ràng buộc kinh tế (Coupled Projection)...")
        history_before_test = pd.concat([self.train_df, self.val_df], ignore_index=True)
        arr_pred_test, nig_pred_test = self.linear_engine.predict(
            self.test_df, history_df=history_before_test
        )
        test_history_diagnostics = self.linear_engine.last_prediction_diagnostics.copy()

        # Chiếu cận kinh tế
        arr_final_test, nig_final_test, violations = self.coupled_optimizer.project_physical_pair(
            arr_pred_test, nig_pred_test
        )
        n_violations = int(np.sum(violations))
        pvr = float(n_violations / max(1, len(arr_pred_test)) * 100.0)
        print(f"  • Rà soát ràng buộc kinh tế: Triệt tiêu vi phạm ({n_violations} trường hợp nights < arrivals) -> PVR: {pvr:.3f}% -> 0.000% sau Pi_C.")

        # Hiệu chuẩn Conformal DCQF
        if len(val_calibration_df) > 0:
            history_before_cal = pd.concat([self.train_df, val_tuning_df], ignore_index=True)
            arr_pred_cal, nig_pred_cal = self.linear_engine.predict(
                val_calibration_df, history_df=history_before_cal
            )
            arr_final_cal, nig_final_cal, _ = self.coupled_optimizer.project_physical_pair(
                arr_pred_cal, nig_pred_cal
            )
            self.dcqf_engine.calibrate_residuals(
                val_calibration_df["arrivals_total"].to_numpy(),
                arr_final_cal,
                val_calibration_df["nights_total"].to_numpy(),
                nig_final_cal,
            )
            print(f"  • Residual Conformal hiệu chuẩn trên {self.dcqf_engine.calibration_size_arr:,} dòng validation.")
            print(f"  • Biên conformal: arrivals ±{self.dcqf_engine.conformal_q_arr:,.0f}; nights ±{self.dcqf_engine.conformal_q_nig:,.0f}")

        conformal_intervals = self.dcqf_engine.predict_conformal_intervals(
            arr_final_test, nig_final_test
        )
        lower_arr, _, upper_arr = conformal_intervals["arrivals"]
        lower_nig, _, upper_nig = conformal_intervals["nights"]

        # 5. Đánh giá phân rã khoa học
        print("\n" + "=" * 80)
        print("📊 [BÁO CÁO ĐÁNH GIÁ THỰC NGHIỆM RA-HE-DLinear TRÊN UN TOURISM TEST 2019 - 2024]")
        print("=" * 80)
        y_true_arr = self.test_df["arrivals_total"].values
        y_true_nig = self.test_df["nights_total"].values

        m_arr_all = calculate_metrics(y_true_arr, arr_final_test)
        m_nig_all = calculate_metrics(y_true_nig, nig_final_test)

        print(f"📌 [TOÀN TẬP TEST 2019-2024 ({m_arr_all['n']:,} bản ghi)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_all['WAPE']:.2f}% | MAE = {m_arr_all['MAE']:,.0f} | RMSE = {m_arr_all['RMSE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_nig_all['WAPE']:.2f}% | MAE = {m_nig_all['MAE']:,.0f} | RMSE = {m_nig_all['RMSE']:,.0f}")

        # Phân rã 3 chế độ: Pre-Covid (2019), Pandemic Shock (2020-2021), Post-Covid Recovery (2022-2024)
        mask_pre = self.test_df["year"].eq(2019).values
        mask_shock = self.test_df["year"].isin([2020, 2021]).values
        mask_post = self.test_df["year"] >= 2022

        m_arr_pre = calculate_metrics(y_true_arr[mask_pre], arr_final_test[mask_pre])
        m_arr_shock = calculate_metrics(y_true_arr[mask_shock], arr_final_test[mask_shock])
        m_arr_post = calculate_metrics(y_true_arr[mask_post], arr_final_test[mask_post])

        m_nig_pre = calculate_metrics(y_true_nig[mask_pre], nig_final_test[mask_pre])
        m_nig_shock = calculate_metrics(y_true_nig[mask_shock], nig_final_test[mask_shock])
        m_nig_post = calculate_metrics(y_true_nig[mask_post], nig_final_test[mask_post])

        print(f"\n🟢 [1. GIAI ĐOẠN TIỀN ĐẠI DỊCH PRE-COVID 2019 ({m_arr_pre['n']:,} bản ghi)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_pre['WAPE']:.2f}% | MAE = {m_arr_pre['MAE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_nig_pre['WAPE']:.2f}% | MAE = {m_nig_pre['MAE']:,.0f}")

        print(f"\n🔴 [2. GIAI ĐOẠN CÚ SỐC ĐẠI DỊCH 2020 - 2021 ({m_arr_shock['n']:,} bản ghi)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_shock['WAPE']:.2f}% | MAE = {m_arr_shock['MAE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_nig_shock['WAPE']:.2f}% | MAE = {m_nig_shock['MAE']:,.0f}")

        print(f"\n🔵 [3. GIAI ĐOẠN PHỤC HỒI HẬU ĐẠI DỊCH 2022 - 2024 ({m_arr_post['n']:,} bản ghi)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_post['WAPE']:.2f}% | MAE = {m_arr_post['MAE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_nig_post['WAPE']:.2f}% | MAE = {m_nig_post['MAE']:,.0f}")
        print("=" * 80)

        # Baseline: 1-Year Persistence Naive
        # y_hat(t) = y(t-1)
        test_history_df = pd.concat([history_before_test, self.test_df], ignore_index=True).sort_values(by=["geo", "year"])
        test_history_df["naive_arr"] = test_history_df.groupby("geo")["arrivals_total"].shift(1)
        test_history_df["naive_nig"] = test_history_df.groupby("geo")["nights_total"].shift(1)
        test_naive_df = test_history_df[test_history_df["year"] > self.val_end_year].copy()
        
        m_naive_arr = calculate_metrics(test_naive_df["arrivals_total"].values, test_naive_df["naive_arr"].values)
        m_naive_nig = calculate_metrics(test_naive_df["nights_total"].values, test_naive_df["naive_nig"].values)
        print(f"\n📉 [BASELINE PERSISTENCE (1-Year Naive)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_naive_arr['WAPE']:.2f}% | MAE = {m_naive_arr['MAE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_naive_nig['WAPE']:.2f}% | MAE = {m_naive_nig['MAE']:,.0f}")

        # 6. Lưu file kết quả
        out_df = self.test_df[["Country Code", "Country", "Region", "Income Group", "year", "date", "arrivals_total", "nights_total"]].copy()
        out_df["pred_arrivals_point"] = arr_final_test
        out_df["pred_nights_point"] = nig_final_test
        out_df["pred_arrivals_lower_90"] = lower_arr
        out_df["pred_arrivals_upper_90"] = upper_arr
        out_df["pred_nights_lower_90"] = lower_nig
        out_df["pred_nights_upper_90"] = upper_nig
        out_df["model"] = "RA-HE-DLinear (UN Tourism Annual)"
        for column in ("forecast_origin_count", "history_observations_min", "history_padding_max"):
            out_df[column] = test_history_diagnostics[column].to_numpy()

        out_csv = RESULTS_DIR / "forecast_results_un_tourism_dlinear.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\n💾 Đã lưu kết quả chi tiết tại: {out_csv}")

        # Lưu bảng đối sánh metrics
        comparison_rows = [
            {"Split": "test_all (2019-2024)", "Target": "Arrivals", **m_arr_all},
            {"Split": "test_precovid (2019)", "Target": "Arrivals", **m_arr_pre},
            {"Split": "test_covid_shock (2020-2021)", "Target": "Arrivals", **m_arr_shock},
            {"Split": "test_postcovid_recovery (2022-2024)", "Target": "Arrivals", **m_arr_post},
            {"Split": "test_all (2019-2024)", "Target": "Nights", **m_nig_all},
            {"Split": "test_precovid (2019)", "Target": "Nights", **m_nig_pre},
            {"Split": "test_covid_shock (2020-2021)", "Target": "Nights", **m_nig_shock},
            {"Split": "test_postcovid_recovery (2022-2024)", "Target": "Nights", **m_nig_post},
            {"Split": "baseline_persistence_all", "Target": "Arrivals", **m_naive_arr},
            {"Split": "baseline_persistence_all", "Target": "Nights", **m_naive_nig},
        ]
        metrics_df = pd.DataFrame(comparison_rows)
        metrics_csv = TABLES_DIR / "un_tourism_dlinear_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"💾 Đã lưu bảng chỉ số khoa học tại: {metrics_csv}")

        return {
            "results_df": out_df,
            "metrics": metrics_df,
        }


def main():
    pipeline = UNTourismLinearForecastingPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
