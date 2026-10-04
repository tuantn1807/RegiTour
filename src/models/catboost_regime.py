"""
CatBoost-Regime Model (GBDT Tabular Macro Forecaster).
Engine 1 in the RA-HE-BiLSTM Architecture.
Learns from:
  - Slow Wave A (Approximation from Wavelet decomposition)
  - HICP accommodation price inflation
  - Oxford stringency index
  - Regime posterior probabilities from HMM
  - Entity identifiers (geo, accommodation_type)
Outputs: Macroeconomic Trend Forecast (Y_trend).
"""

from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    from catboost import CatBoostRegressor
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False
    try:
        from sklearn.ensemble import HistGradientBoostingRegressor
        HAS_SKLEARN_GB = True
    except ImportError:
        HAS_SKLEARN_GB = False


class CatBoostRegimeForecaster:
    """
    CatBoost-Regime: Specializes in tabular macroeconomic relationships, price elasticity,
    and structural trend forecasting (Y_trend) conditioned on market regimes.
    """

    DEFAULT_FEATURE_COLS = [
        "month",
        "year",
        "hicp_accommodation",
        "hicp_overall",
        "hicp_acc_mom_pct",
        "hicp_acc_yoy_pct",
        "covid_stringency_index",
        "covid_gov_response_index",
        "p_normal",
        "p_disruption",
        "p_recovery",
        "country_id",
        "accom_id",
    ]

    def __init__(
        self,
        iterations: int = 400,
        learning_rate: float = 0.05,
        depth: int = 6,
        l2_leaf_reg: float = 3.0,
        random_seed: int = 42,
    ):
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.depth = depth
        self.l2_leaf_reg = l2_leaf_reg
        self.random_seed = random_seed

        self.model_arr = None
        self.model_nig = None
        self.feature_cols: List[str] = []

    def fit(
        self,
        train_df: pd.DataFrame,
        feature_cols: Optional[List[str]] = None,
        target_arr_col: str = "arrivals_total_wave_a",
        target_nig_col: str = "nights_total_wave_a",
    ) -> "CatBoostRegimeForecaster":
        """
        Trains GBDT models on Wave A targets to predict macro baseline trends.
        """
        if feature_cols is None:
            self.feature_cols = [c for c in self.DEFAULT_FEATURE_COLS if c in train_df.columns]
        else:
            self.feature_cols = feature_cols

        X_train = train_df[self.feature_cols].copy().fillna(0)
        y_train_arr = train_df[target_arr_col].values
        y_train_nig = train_df[target_nig_col].values

        cat_features = [i for i, c in enumerate(self.feature_cols) if c in ("country_id", "accom_id")]

        if HAS_CATBOOST:
            self.model_arr = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                cat_features=cat_features if cat_features else None,
                verbose=False,
            )
            self.model_arr.fit(X_train, y_train_arr)

            self.model_nig = CatBoostRegressor(
                iterations=self.iterations,
                learning_rate=self.learning_rate,
                depth=self.depth,
                l2_leaf_reg=self.l2_leaf_reg,
                random_seed=self.random_seed,
                cat_features=cat_features if cat_features else None,
                verbose=False,
            )
            self.model_nig.fit(X_train, y_train_nig)
        elif HAS_SKLEARN_GB:
            self.model_arr = HistGradientBoostingRegressor(
                max_iter=self.iterations,
                learning_rate=self.learning_rate,
                max_depth=self.depth,
                random_state=self.random_seed,
            )
            self.model_arr.fit(X_train, y_train_arr)

            self.model_nig = HistGradientBoostingRegressor(
                max_iter=self.iterations,
                learning_rate=self.learning_rate,
                max_depth=self.depth,
                random_state=self.random_seed,
            )
            self.model_nig.fit(X_train, y_train_nig)
        else:
            # Fallback linear ridge
            class DummyRegressor:
                def __init__(self):
                    self.weights = None
                def fit(self, X, y):
                    X_mat = np.c_[np.ones(len(X)), X.values]
                    self.weights = np.linalg.pinv(X_mat.T @ X_mat + 1e-1 * np.eye(X_mat.shape[1])) @ X_mat.T @ y
                def predict(self, X):
                    X_mat = np.c_[np.ones(len(X)), X.values]
                    return X_mat @ self.weights

            self.model_arr = DummyRegressor()
            self.model_arr.fit(X_train, y_train_arr)
            self.model_nig = DummyRegressor()
            self.model_nig.fit(X_train, y_train_nig)

        return self

    def predict(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predicts macro trend: (Y_trend_arr, Y_trend_nig).
        """
        X = df[self.feature_cols].copy().fillna(0)
        y_trend_arr = np.clip(self.model_arr.predict(X), a_min=0, a_max=None)
        y_trend_nig = np.clip(self.model_nig.predict(X), a_min=0, a_max=None)
        return y_trend_arr.astype(np.float32), y_trend_nig.astype(np.float32)
