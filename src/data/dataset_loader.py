"""
Dataset Loader & Preprocessor for European Tourism Forecasting.
Supports multi-horizon windowing, feature normalization, and temporal train/val/test splits.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

class TourismDataLoader:
    """
    Manages loading, cleaning, feature scaling, and multi-horizon sequence generation
    for the 38 European countries dataset (54 exogenous features).
    """

    DEFAULT_FEATURE_FILE = Path(__file__).resolve().parents[2] / "data" / "features" / "tourism_europe_features.parquet"

    # Known future calendar features (known 6-12 months ahead)
    FUTURE_CALENDAR_COLS = [
        "month",
        "days_in_month",
        "weekend_days",
        "working_days",
        "is_peak_summer",
        "is_shoulder_season",
        "is_winter_holiday",
        "holiday_days",
        "holiday_religious_days",
        "holiday_national_days",
        "holiday_long_weekends_count",
        "holiday_bridge_days_count",
        "holiday_has_easter",
        "holiday_has_christmas",
    ]

    # Exogenous macro/crisis past features
    EXOGENOUS_MACRO_COLS = [
        "hicp_accommodation",
        "hicp_overall",
        "hicp_acc_mom_pct",
        "hicp_acc_yoy_pct",
        "covid_stringency_index",
        "covid_gov_response_index",
        "covid_containment_index",
    ]

    def __init__(
        self,
        parquet_path: Optional[Union[str, Path]] = None,
        target_arrivals: str = "arrivals_total",
        target_nights: str = "nights_total",
        lookback_len: int = 24,
        forecast_horizon: int = 12,
        train_end_year: int = 2022,
        val_end_year: int = 2023,
    ):
        self.parquet_path = Path(parquet_path) if parquet_path else self.DEFAULT_FEATURE_FILE
        self.target_arrivals = target_arrivals
        self.target_nights = target_nights
        self.lookback_len = lookback_len
        self.forecast_horizon = forecast_horizon
        self.train_end_year = train_end_year
        self.val_end_year = val_end_year

        self.df: Optional[pd.DataFrame] = None
        self.country_mapping: Dict[str, int] = {}
        self.country_inv_mapping: Dict[int, str] = {}
        self.accom_mapping: Dict[str, int] = {}
        self.accom_inv_mapping: Dict[int, str] = {}
        self.country_base_ratios: Dict[str, float] = {}

    def load_data(self) -> pd.DataFrame:
        """Loads and sorts the parquet feature store."""
        if not self.parquet_path.exists():
            raise FileNotFoundError(f"Feature file not found at: {self.parquet_path}")

        df = pd.read_parquet(self.parquet_path)
        df["date"] = pd.to_datetime(df["date"])
        df = df.sort_values(by=["geo", "accommodation_type", "date"]).reset_index(drop=True)

        # Handle infinite and fill missing numerical values
        num_cols = df.select_dtypes(include=[np.number]).columns
        df[num_cols] = df[num_cols].replace([np.inf, -np.inf], np.nan)
        for col in num_cols:
            df[col] = df.groupby(["geo", "accommodation_type"])[col].transform(
                lambda s: s.ffill().bfill().fillna(0)
            )

        # Ensure targets are non-negative
        df[self.target_arrivals] = df[self.target_arrivals].clip(lower=0)
        df[self.target_nights] = df[self.target_nights].clip(lower=0)

        # Create integer IDs for entity embeddings
        unique_countries = sorted(df["geo"].astype(str).unique())
        self.country_mapping = {c: i for i, c in enumerate(unique_countries)}
        self.country_inv_mapping = {i: c for c, i in self.country_mapping.items()}
        df["country_id"] = df["geo"].map(self.country_mapping)

        unique_accom = sorted(df["accommodation_type"].astype(str).unique())
        self.accom_mapping = {a: i for i, a in enumerate(unique_accom)}
        self.accom_inv_mapping = {i: a for a, i in self.accom_mapping.items()}
        df["accom_id"] = df["accommodation_type"].map(self.accom_mapping)

        # Precalculate empirical Nights / Arrivals base ratio per country
        valid_pairs = df[df[self.target_arrivals] > 100]
        base_ratios = valid_pairs.groupby("geo").apply(
            lambda sub: (sub[self.target_nights].sum() / max(1.0, sub[self.target_arrivals].sum()))
        )
        self.country_base_ratios = base_ratios.to_dict()

        self.df = df
        return df

    def get_splits(
        self, df: Optional[pd.DataFrame] = None
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Splits into Train (<=2022), Val (2023), Test (>=2024)."""
        if df is None:
            if self.df is None:
                df = self.load_data()
            else:
                df = self.df

        train_df = df[df["year"] <= self.train_end_year].copy()
        val_df = df[(df["year"] > self.train_end_year) & (df["year"] <= self.val_end_year)].copy()
        test_df = df[df["year"] > self.val_end_year].copy()

        return train_df, val_df, test_df

    def create_sliding_windows(
        self,
        df: pd.DataFrame,
        past_feature_cols: List[str],
        future_feature_cols: Optional[List[str]] = None,
    ) -> Dict[str, np.ndarray]:
        """
        Builds temporal tensors for Deep Learning models:
        - X_past: (N, L, D_past)
        - X_future: (N, H, D_future)
        - X_static: (N, 2) -> [country_id, accom_id]
        - Y_arr: (N, H)
        - Y_nig: (N, H)
        - base_ratio: (N,)
        """
        if future_feature_cols is None:
            future_feature_cols = self.FUTURE_CALENDAR_COLS

        X_past_list = []
        X_future_list = []
        X_static_list = []
        Y_arr_list = []
        Y_nig_list = []
        base_ratio_list = []

        total_window = self.lookback_len + self.forecast_horizon

        for (geo, accom), group in df.groupby(["geo", "accommodation_type"]):
            group = group.sort_values("date").reset_index(drop=True)
            n_samples = len(group)
            if n_samples < total_window:
                continue

            country_id = self.country_mapping[str(geo)]
            accom_id = self.accom_mapping[str(accom)]
            country_ratio = self.country_base_ratios.get(str(geo), 2.5)

            past_vals = group[past_feature_cols].values.astype(np.float32)
            future_vals = group[future_feature_cols].values.astype(np.float32)
            y_arr_vals = group[self.target_arrivals].values.astype(np.float32)
            y_nig_vals = group[self.target_nights].values.astype(np.float32)

            for i in range(n_samples - total_window + 1):
                past_idx = slice(i, i + self.lookback_len)
                future_idx = slice(i + self.lookback_len, i + total_window)

                X_past_list.append(past_vals[past_idx])
                X_future_list.append(future_vals[future_idx])
                X_static_list.append([country_id, accom_id])
                Y_arr_list.append(y_arr_vals[future_idx])
                Y_nig_list.append(y_nig_vals[future_idx])
                base_ratio_list.append(country_ratio)

        return {
            "X_past": np.array(X_past_list, dtype=np.float32),
            "X_future": np.array(X_future_list, dtype=np.float32),
            "X_static": np.array(X_static_list, dtype=np.int64),
            "Y_arr": np.array(Y_arr_list, dtype=np.float32),
            "Y_nig": np.array(Y_nig_list, dtype=np.float32),
            "base_ratio": np.array(base_ratio_list, dtype=np.float32),
        }
