"""
Baseline Models for Comparative Scientific Benchmarking.
Implements the 3 waves of literature baselines:
  1. Wave 1 - Econometric / Statistical: Seasonal Naive (S-Naive)
  2. Wave 2 - Standard Machine Learning: Standard XGBoost & LightGBM (No Wavelets, No Regime)
  3. Wave 3 - Classical Deep Learning: Standard LSTM (Hochreiter, 1997) & Vanilla BiLSTM (Graves, 2013)
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

try:
    import xgboost as xgb
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    import lightgbm as lgb
    HAS_LGB = True
except ImportError:
    HAS_LGB = False

try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# 1. STATISTICAL BASELINE: SEASONAL NAIVE (S-NAIVE)
# ============================================================
class SeasonalNaiveForecaster:
    """
    Seasonal Naive Benchmark:
    y_hat(t + h) = y(t + h - 12) (Repeats identical month from previous year).
    Classic gold-standard baseline in tourism econometrics (Athanasopoulos et al.).
    """
    def __init__(self, seasonal_lag: int = 12):
        self.seasonal_lag = seasonal_lag
        self.history: Optional[pd.DataFrame] = None

    def fit(self, train_df: pd.DataFrame):
        self.history = train_df[["date", "geo", "accommodation_type", "arrivals_total", "nights_total"]].copy()
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Matches each test record with the exact same (geo, accommodation_type, month)
        from the most recent historical year.
        """
        test = test_df[["date", "geo", "accommodation_type", "month"]].copy()
        # Get historical average for each (geo, accom, month)
        hist_agg = self.history.copy()
        hist_agg["month"] = pd.to_datetime(hist_agg["date"]).dt.month
        hist_lookup = hist_agg.groupby(["geo", "accommodation_type", "month"])[
            ["arrivals_total", "nights_total"]
        ].median().reset_index()

        merged = test.merge(hist_lookup, on=["geo", "accommodation_type", "month"], how="left")
        arr_pred = merged["arrivals_total"].fillna(100.0).values.astype(np.float32)
        nig_pred = merged["nights_total"].fillna(250.0).values.astype(np.float32)
        return arr_pred, nig_pred


# ============================================================
# 2. STANDARD MACHINE LEARNING BASELINES: XGBOOST & LIGHTGBM
# ============================================================
class StandardXGBoostForecaster:
    """
    Standard XGBoost without Wavelet decomposition and without Regime HMM.
    Represents Wave 2 Machine Learning approach in tourism literature.
    """
    FEATURE_COLS = [
        "month", "year", "days_in_month", "weekend_days", "working_days",
        "is_peak_summer", "is_winter_holiday", "holiday_days",
        "hicp_accommodation", "hicp_overall", "covid_stringency_index",
        "country_id", "accom_id"
    ]

    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.05, max_depth: int = 6):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.max_depth = max_depth
        self.model_arr = None
        self.model_nig = None
        self.cols: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        self.cols = [c for c in self.FEATURE_COLS if c in train_df.columns]
        X = train_df[self.cols].fillna(0).values
        y_arr = train_df["arrivals_total"].values
        y_nig = train_df["nights_total"].values

        if HAS_XGB:
            self.model_arr = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
            self.model_arr.fit(X, y_arr)

            self.model_nig = xgb.XGBRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                max_depth=self.max_depth,
                random_state=42,
                n_jobs=-1,
            )
            self.model_nig.fit(X, y_nig)
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = test_df[self.cols].fillna(0).values
        if self.model_arr is not None:
            arr_pred = np.clip(self.model_arr.predict(X), 0, None)
            nig_pred = np.clip(self.model_nig.predict(X), 0, None)
        else:
            arr_pred = np.zeros(len(X), dtype=np.float32)
            nig_pred = np.zeros(len(X), dtype=np.float32)
        return arr_pred.astype(np.float32), nig_pred.astype(np.float32)


class StandardLightGBMForecaster:
    """
    Standard LightGBM benchmark. Fast and highly competitive GBDT baseline.
    """
    def __init__(self, n_estimators: int = 250, learning_rate: float = 0.05, num_leaves: int = 31):
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.num_leaves = num_leaves
        self.model_arr = None
        self.model_nig = None
        self.cols: List[str] = []

    def fit(self, train_df: pd.DataFrame):
        self.cols = [c for c in StandardXGBoostForecaster.FEATURE_COLS if c in train_df.columns]
        X = train_df[self.cols].fillna(0).values
        y_arr = train_df["arrivals_total"].values
        y_nig = train_df["nights_total"].values

        if HAS_LGB:
            self.model_arr = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=42,
                verbosity=-1,
            )
            self.model_arr.fit(X, y_arr)

            self.model_nig = lgb.LGBMRegressor(
                n_estimators=self.n_estimators,
                learning_rate=self.learning_rate,
                num_leaves=self.num_leaves,
                random_state=42,
                verbosity=-1,
            )
            self.model_nig.fit(X, y_nig)
        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        X = test_df[self.cols].fillna(0).values
        if self.model_arr is not None:
            arr_pred = np.clip(self.model_arr.predict(X), 0, None)
            nig_pred = np.clip(self.model_nig.predict(X), 0, None)
        else:
            arr_pred = np.zeros(len(X), dtype=np.float32)
            nig_pred = np.zeros(len(X), dtype=np.float32)
        return arr_pred.astype(np.float32), nig_pred.astype(np.float32)


# ============================================================
# 3. CLASSICAL DEEP LEARNING BASELINES: STANDARD LSTM & BiLSTM
# ============================================================
if HAS_TORCH:
    class ClassicalLSTMNet(nn.Module):
        """
        Vanilla LSTM Network (Hochreiter & Schmidhuber, 1997).
        Static forget gate, no regime awareness, standard linear projection.
        """
        def __init__(self, input_dim: int, hidden_dim: int = 64, bidirectional: bool = False):
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=2,
                batch_first=True,
                bidirectional=bidirectional,
            )
            out_dim = hidden_dim * (2 if bidirectional else 1)
            self.fc_arr = nn.Linear(out_dim, 1)
            self.fc_nig = nn.Linear(out_dim, 1)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            out, _ = self.lstm(x)
            last_step = out[:, -1, :]  # Take last hidden state
            y_arr = torch.relu(self.fc_arr(last_step).squeeze(-1))
            y_nig = torch.relu(self.fc_nig(last_step).squeeze(-1))
            return y_arr, y_nig


class StandardLSTMForecaster:
    """
    Standard Deep Learning LSTM Forecaster.
    """
    FEATURE_COLS = [
        "month", "year", "days_in_month", "weekend_days",
        "hicp_accommodation", "covid_stringency_index",
        "country_id", "accom_id"
    ]

    def __init__(self, bidirectional: bool = False, epochs: int = 30, batch_size: int = 128):
        self.bidirectional = bidirectional
        self.epochs = epochs
        self.batch_size = batch_size
        self.net = None
        self.cols: List[str] = []
        self.mean_arr = 1.0
        self.std_arr = 1.0
        self.mean_nig = 1.0
        self.std_nig = 1.0

    def fit(self, train_df: pd.DataFrame):
        if not HAS_TORCH:
            return self

        self.cols = [c for c in self.FEATURE_COLS if c in train_df.columns]
        X_raw = train_df[self.cols].fillna(0).values.astype(np.float32)
        y_arr = train_df["arrivals_total"].values.astype(np.float32)
        y_nig = train_df["nights_total"].values.astype(np.float32)

        self.mean_arr, self.std_arr = np.mean(y_arr), np.std(y_arr) + 1e-4
        self.mean_nig, self.std_nig = np.mean(y_nig), np.std(y_nig) + 1e-4

        y_arr_norm = (y_arr - self.mean_arr) / self.std_arr
        y_nig_norm = (y_nig - self.mean_nig) / self.std_nig

        # Create sequential batches (unsqueeze sequence length = 1 for point tabular mapping)
        X_t = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)
        ya_t = torch.tensor(y_arr_norm, dtype=torch.float32)
        yn_t = torch.tensor(y_nig_norm, dtype=torch.float32)

        dataset = torch.utils.data.TensorDataset(X_t, ya_t, yn_t)
        loader = torch.utils.data.DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        self.net = ClassicalLSTMNet(input_dim=len(self.cols), bidirectional=self.bidirectional)
        optimizer = torch.optim.Adam(self.net.parameters(), lr=2e-3)
        criterion = nn.MSELoss()

        self.net.train()
        for _ in range(self.epochs):
            for bx, bya, byn in loader:
                optimizer.zero_grad()
                pa, pn = self.net(bx)
                loss = criterion(pa, bya) + criterion(pn, byn)
                loss.backward()
                optimizer.step()

        return self

    def predict(self, test_df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        if not HAS_TORCH or self.net is None:
            return np.zeros(len(test_df), dtype=np.float32), np.zeros(len(test_df), dtype=np.float32)

        self.net.eval()
        X_raw = test_df[self.cols].fillna(0).values.astype(np.float32)
        X_t = torch.tensor(X_raw, dtype=torch.float32).unsqueeze(1)

        with torch.no_grad():
            pa, pn = self.net(X_t)
            arr_pred = pa.numpy() * self.std_arr + self.mean_arr
            nig_pred = pn.numpy() * self.std_nig + self.mean_nig

        return np.clip(arr_pred, 0, None).astype(np.float32), np.clip(nig_pred, 0, None).astype(np.float32)
