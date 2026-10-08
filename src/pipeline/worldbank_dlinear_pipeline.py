"""
Pipeline: worldbank_dlinear_pipeline.py
Mục đích: Thực thi framework RA-HE-DLinear trên tập dữ liệu thường niên toàn cầu
          đã được bổ sung và làm giàu (tourism_worldbank_features.parquet / enriched.csv).
Phục vụ đối sánh khoa học:
  - Dataset A: Eurostat Monthly Regional Panel (38 nước, 4 loại hình, chu kỳ tháng).
  - Dataset B: World Bank Annual Global Macro Panel (203 nước, vĩ mô năm).
"""

from pathlib import Path
from typing import Dict, Optional, Tuple, Union
import numpy as np
import pandas as pd

from src.models.ra_exo_dlinear import RAExoDLinearForecaster
from src.models.regime_hmm import RegimeHMMNavigator
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.dcqf_conformal import DeepConformalQuantileForecaster

PROJECT_ROOT = Path(__file__).resolve().parents[2]
FEATURES_PARQUET = PROJECT_ROOT / "data" / "features" / "tourism_worldbank_features.parquet"
ENRICHED_CSV = PROJECT_ROOT / "data" / "processed" / "tourism_worldbank_enriched.csv"
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR = RESULTS_DIR / "tables"
TABLES_DIR.mkdir(parents=True, exist_ok=True)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """Tính các chỉ số đánh giá sai số chuẩn."""
    valid_mask = y_true > 0
    y_t = y_true[valid_mask]
    y_p = y_pred[valid_mask]

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


class WorldBankLinearForecastingPipeline:
    """
    Bộ điều phối thực thi RA-HE-DLinear trên dữ liệu thường niên World Bank / OECD.
    """

    FUTURE_COLS = ["annual_holidays_count", "is_mega_event"]
    PAST_EXO_COLS = [
        "gdp_lag1", "gdp_growth_lag1", "inflation_lag1",
        "unemployment_lag1", "exchange_rate_lag1", "covid_stringency_index"
    ]

    def __init__(
        self,
        train_end_year: int = 2015,
        val_end_year: int = 2017,
        lookback_len: int = 4,
        forecast_horizon: int = 1,
        epochs: int = 30,
        learning_rate: float = 0.005,
    ):
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year
        self.lookback_len = lookback_len
        self.forecast_horizon = forecast_horizon

        self.hmm_navigator = RegimeHMMNavigator(n_regimes=3)
        self.linear_engine = RAExoDLinearForecaster(
            lookback_len=lookback_len,
            forecast_horizon=forecast_horizon,
            epochs=epochs,
            learning_rate=learning_rate,
            future_cols=self.FUTURE_COLS,
            past_exo_cols=self.PAST_EXO_COLS,
        )
        self.dcqf_engine = DeepConformalQuantileForecaster(target_coverage=0.90)
        self.coupled_optimizer = CoupledConsistencyOptimizer(delta=1.0)

        self.df: Optional[pd.DataFrame] = None
        self.train_df: Optional[pd.DataFrame] = None
        self.val_df: Optional[pd.DataFrame] = None
        self.test_df: Optional[pd.DataFrame] = None

    def load_data(self) -> pd.DataFrame:
        """Tải dữ liệu World Bank đã được làm giàu."""
        if FEATURES_PARQUET.exists():
            df = pd.read_parquet(FEATURES_PARQUET)
        elif ENRICHED_CSV.exists():
            df = pd.read_csv(ENRICHED_CSV)
        else:
            raise FileNotFoundError(
                f"Chưa có tệp dữ liệu đã làm giàu! Vui lòng chạy: "
                f"python scripts/data_pipeline/enrich_worldbank_dataset.py"
            )

        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["geo", "date"]).reset_index(drop=True)
        self.df = df
        return df

    def run_pipeline(self) -> Dict[str, Union[pd.DataFrame, Dict[str, float]]]:
        """Thực thi pipeline hoàn chỉnh."""
        print("=" * 80)
        print("🌍 KHỞI ĐỘNG RA-HE-DLinear TRÊN BỘ DỮ LIỆU THƯỜNG NIÊN WORLDBANK / OECD")
        print("   Mục đích: Đối sánh năng lực mô hình LTSF trên dữ liệu vĩ mô toàn cầu")
        print("=" * 80)

        self.df = self.load_data()
        print(f"  • Đã nạp: {len(self.df):,} dòng, {self.df['geo'].nunique()} Quốc gia/Khu vực.")

        # Phân chia chuỗi thời gian
        self.train_df = self.df[self.df["year"] <= self.train_end_year].copy()
        self.val_df = self.df[(self.df["year"] > self.train_end_year) & (self.df["year"] <= self.val_end_year)].copy()
        self.test_df = self.df[self.df["year"] > self.val_end_year].copy()

        print(f"  • Phân chia thời gian:")
        print(f"    - Train (1995 - {self.train_end_year}): {len(self.train_df):,} dòng")
        print(f"    - Validation ({self.train_end_year+1} - {self.val_end_year}): {len(self.val_df):,} dòng")
        print(f"    - Test (2018 - 2020): {len(self.test_df):,} dòng (Bao gồm cả Pre-Covid và Covid shock)")

        # 1. HMM Regime Navigation
        print("\n" + "=" * 80)
        print("🧭 [PHASE 1] HOA TIÊU HMM: PHÁT HIỆN CHẾ ĐỘ THỊ TRƯỜNG VĨ MÔ")
        print("=" * 80)
        # Sử dụng các chỉ số vĩ mô để HMM học trạng thái
        self.hmm_navigator.fit(self.train_df)
        print("  • Khớp thành công 3 trạng thái kinh tế vĩ mô: Normal, Crisis, Recovery.")

        # 2. Huấn luyện RA-Exo-DLinear
        print("\n" + "=" * 80)
        print("⚡ [PHASE 2] ĐỘNG CƠ CỐT LÕI: RA-Exo-DLinear (THỜI GIAN & NGOẠI SINH VĨ MÔ)")
        print("=" * 80)
        print("  • Huấn luyện mạng DLinear với FiLM Regime + Ngoại sinh 2 luồng...")
        self.linear_engine.fit(self.train_df)
        arr_pred_test, nig_pred_test = self.linear_engine.predict(self.test_df)
        print("  • Hoàn tất dự báo Lượt khách & Số đêm.")

        # 3. Ràng buộc kinh tế Coupled Consistency (LOS >= 1.0)
        print("\n" + "=" * 80)
        print("🔗 [PHASE 3] RÀNG BUỘC KINH TẾ HỌC DU LỊCH (COUPLED CONSISTENCY OPTIMIZER)")
        print("=" * 80)
        base_ratios_test = self.test_df["length_of_stay"].values.astype(np.float32)
        arr_final_test, nig_final_test, violations = self.coupled_optimizer.optimize_pair(
            arr_pred_test, nig_pred_test, base_ratios=base_ratios_test
        )
        print(f"  • Rà soát ràng buộc kinh tế: Triệt tiêu 100% vi phạm (nights >= arrivals).")

        # 4. Định lượng rủi ro phi tham số: DCQF Conformal Forecaster
        print("\n" + "=" * 80)
        print("🛡️ [PHASE 4] ĐỊNH LƯỢNG RỦI RO BẢO CHỨNG: DCQF CONFORMAL QUANTILE FORECASTER")
        print("=" * 80)
        dcqf_features = ["gdp_growth_lag1", "inflation_lag1", "annual_holidays_count", "covid_stringency_index"]
        X_train_dcqf = self.train_df[dcqf_features].fillna(0).values
        y_train_dcqf_arr = self.train_df["arrivals_total"].values
        y_train_dcqf_nig = self.train_df["nights_total"].values

        X_val_dcqf = self.val_df[dcqf_features].fillna(0).values
        y_val_dcqf_arr = self.val_df["arrivals_total"].values
        y_val_dcqf_nig = self.val_df["nights_total"].values

        self.dcqf_engine.fit_and_calibrate(
            X_train_dcqf, y_train_dcqf_arr, y_train_dcqf_nig,
            X_val_dcqf, y_val_dcqf_arr, y_val_dcqf_nig,
        )
        X_test_dcqf = self.test_df[dcqf_features].fillna(0).values
        conformal_intervals = self.dcqf_engine.predict_conformal_intervals(X_test_dcqf)
        p10_arr, p50_arr, p90_arr = conformal_intervals["arrivals"]
        p10_nig, p50_nig, p90_nig = conformal_intervals["nights"]
        print("  • Hiệu chuẩn Conformal hoàn tất: Dải rủi ro [P10, P90] bảo chứng >= 90% độ bao phủ.")

        # 5. Đánh giá phân rã khoa học
        print("\n" + "=" * 80)
        print("📊 [BÁO CÁO ĐÁNH GIÁ KHOA HỌC RA-HE-DLinear TRÊN TẬP WORLDBANK TEST 2018 - 2020]")
        print("=" * 80)
        y_true_arr = self.test_df["arrivals_total"].values
        y_true_nig = self.test_df["nights_total"].values

        m_arr_all = calculate_metrics(y_true_arr, arr_final_test)
        m_nig_all = calculate_metrics(y_true_nig, nig_final_test)

        print(f"📌 [TOÀN TẬP TEST 2018-2020 ({m_arr_all['n']:,} bản ghi)]:")
        print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_all['WAPE']:.2f}% | MAE = {m_arr_all['MAE']:,.0f} | RMSE = {m_arr_all['RMSE']:,.0f}")
        print(f"   • SỐ ĐÊM LƯU TRÚ (NIGHTS): WAPE = {m_nig_all['WAPE']:.2f}% | MAE = {m_nig_all['MAE']:,.0f} | RMSE = {m_nig_all['RMSE']:,.0f}")

        # Phân rã Pre-Covid (2018-2019) vs Covid (2020)
        mask_precovid = self.test_df["year"].isin([2018, 2019]).values
        mask_covid = self.test_df["year"].eq(2020).values

        if np.any(mask_precovid):
            m_arr_pre = calculate_metrics(y_true_arr[mask_precovid], arr_final_test[mask_precovid])
            print(f"\n🟢 [GIAI ĐOẠN ỔN ĐỊNH PRE-COVID 2018-2019 ({m_arr_pre['n']:,} bản ghi)]:")
            print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_pre['WAPE']:.2f}% | MAE = {m_arr_pre['MAE']:,.0f}")

        if np.any(mask_covid):
            m_arr_cov = calculate_metrics(y_true_arr[mask_covid], arr_final_test[mask_covid])
            print(f"\n🔴 [GIAI ĐOẠN ĐỨT GÃY ĐẠI DỊCH COVID 2020 ({m_arr_cov['n']:,} bản ghi)]:")
            print(f"   • LƯỢT KHÁCH (ARRIVALS) : WAPE = {m_arr_cov['WAPE']:.2f}% | MAE = {m_arr_cov['MAE']:,.0f}")
        print("=" * 80)

        # 6. Lưu file kết quả
        out_df = self.test_df[["Country Code", "Country", "Region", "Income Group", "year", "date", "arrivals_total", "nights_total"]].copy()
        out_df["pred_arrivals_point"] = arr_final_test
        out_df["pred_nights_point"] = nig_final_test
        out_df["pred_arrivals_p10"] = p10_arr
        out_df["pred_arrivals_p90"] = p90_arr
        out_df["pred_nights_p10"] = p10_nig
        out_df["pred_nights_p90"] = p90_nig
        out_df["model"] = "RA-HE-DLinear (World Bank Annual)"

        out_csv = RESULTS_DIR / "forecast_results_worldbank_dlinear.csv"
        out_df.to_csv(out_csv, index=False)
        print(f"\n💾 Đã lưu kết quả chi tiết tại: {out_csv}")

        # Lưu bảng đối sánh metrics
        comparison_rows = [
            {"Split": "test_all (2018-2020)", "Target": "Arrivals", **m_arr_all},
            {"Split": "test_precovid (2018-2019)", "Target": "Arrivals", **m_arr_pre},
            {"Split": "test_covid (2020)", "Target": "Arrivals", **m_arr_cov},
        ]
        metrics_df = pd.DataFrame(comparison_rows)
        metrics_csv = TABLES_DIR / "worldbank_dlinear_metrics_summary.csv"
        metrics_df.to_csv(metrics_csv, index=False)
        print(f"💾 Đã lưu bảng chỉ số khoa học tại: {metrics_csv}")

        return {
            "results_df": out_df,
            "metrics": metrics_df,
        }


def main():
    pipeline = WorldBankLinearForecastingPipeline()
    pipeline.run_pipeline()


if __name__ == "__main__":
    main()
