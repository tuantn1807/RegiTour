"""
RA-Exo-DLinear: Regime-Aware Exogenous Decomposition Linear Architecture.
Designed for Multi-Horizon Long-Term Tourism & Hospitality Demand Forecasting (AAAI 2023 Extension).

Mathematical Lineage from Vanilla DLinear (Zeng et al., AAAI 2023):
  1. DLinear Backbone:
     RA-Exo-DLinear decomposes target history X into trend X_trend via Moving Average (MA)
     and seasonal remainder X_seasonal = X - X_trend, then projects each to horizon H:
       Y_DLinear = W_T * X_trend + W_S * X_seasonal.
  2. Theoretical Limitation in Tourism Econometrics:
     Vanilla DLinear is endogenous-only, unaware of deterministic future calendars (holidays, bridge days),
     fragile under macroeconomic regime breaks (COVID-19), and uncoupled across co-dependent targets.
  3. Forecast equation (identical additive and relative-deviation forms):
       B = gamma(e_r) * (W_T X_trend) + beta(e_r) + W_S X_seasonal
       S = max(S_empirical, 1.0); delta_trend = (B - S) / S
       Y_raw = ReLU(B + S * (delta_fut + delta_past))
             = ReLU(S * (1.0 + delta_trend + delta_fut + delta_past))
     With FiLM and exogenous streams disabled, this reduces exactly to
     nonnegative DLinear. No seasonal-baseline ensemble or event-year rule is used.
  4. Regime-Aware FiLM Modulation:
     delta_regime = alpha_trend * tanh(W_gamma * e_r + b_gamma), where e_r is the HMM regime embedding.
     alpha_trend is a learnable scaling parameter initialized to 0.35.
  5. Dual-Stream Exogenous Projections:
     delta_fut = alpha_fut * tanh(W_F * X_fut + b_F), alpha_fut initialized to 0.25.
     delta_past = alpha_past * tanh(W_P * X_past + b_P), alpha_past initialized to 0.20.
     All alpha parameters are trainable via AdamW gradient descent with bounded tanh activation.
  6. Domain Business Consistency Constraint:
     Train on raw forecasts so the consistency penalty has a learning signal.
     Apply the physical projection once, after temporal/spatial fusion in the pipeline.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


# ============================================================
# 1. CORE DECOMPOSITION BLOCK (MOVING AVERAGE FILTER)
# ============================================================
if HAS_TORCH:
    class SeriesDecomp(nn.Module):
        """
        Series decomposition block via Moving Average (Zeng et al., AAAI 2023).
        Extracts long-term trend-cyclical component and seasonal remainder.
        Default kernel_size=13 covers a centered 12-month annual cycle for monthly series.
        """
        def __init__(self, kernel_size: int = 13):
            super().__init__()
            if kernel_size < 1 or kernel_size % 2 == 0:
                raise ValueError("kernel_size must be a positive odd integer.")
            self.kernel_size = kernel_size
            self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=1, padding=0)

        def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
            """
            x: (B, L, C)
            Returns: (seasonal, trend) of shape (B, L, C)
            """
            front_pad = (self.kernel_size - 1) // 2
            back_pad = (self.kernel_size - 1) // 2
            front = x[:, 0:1, :].repeat(1, front_pad, 1)
            end = x[:, -1:, :].repeat(1, back_pad, 1)
            x_padded = torch.cat([front, x, end], dim=1)

            trend = self.avg(x_padded.permute(0, 2, 1)).permute(0, 2, 1)
            seasonal = x - trend
            return seasonal, trend


    # ============================================================
    # 2. CLASSICAL BASELINE: VANILLA DLINEAR (AAAI 2023)
    # ============================================================
    class VanillaDLinearNet(nn.Module):
        """
        Vanilla DLinear (Zeng et al., AAAI 2023).
        Channel-independent linear mapping from lookback L to horizon H.
        """
        def __init__(self, lookback_len: int = 24, forecast_horizon: int = 12, num_channels: int = 2, kernel_size: int = 13):
            super().__init__()
            self.decomp = SeriesDecomp(kernel_size=kernel_size)
            self.linear_trend = nn.Linear(lookback_len, forecast_horizon)
            self.linear_seas = nn.Linear(lookback_len, forecast_horizon)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            seasonal, trend = self.decomp(x)
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seas_out = self.linear_seas(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            return trend_out + seas_out


    # ============================================================
    # 3. PROPOSED CORE ARCHITECTURE: RA-Exo-DLinear
    # ============================================================
    class RAExoDLinearNet(nn.Module):
        """
        Regime-Aware Exogenous Decomposition Linear Network (RA-Exo-DLinear).

        Extends Zeng et al. (AAAI 2023) DLinear with:
          - Moving-average decomposition and linear projections from lookback to horizon
          - FiLM modulation conditioned on regime state (Perez et al., 2018)
          - Dual-stream bounded linear exogenous projections (GAM formulation)
          - Learnable scaling parameters (alpha) for relative deviations
          - Raw nonnegative outputs for consistency-regularized training
        """
        def __init__(
            self,
            lookback_len: int = 24,
            forecast_horizon: int = 12,
            target_dim: int = 2,
            past_exo_dim: int = 7,
            future_exo_dim: int = 14,
            num_countries: int = 40,
            num_accom_types: int = 5,
            num_regimes: int = 3,
            regime_emb_dim: int = 8,
            exo_hidden_dim: int = 16,
            kernel_size: int = 13,
            use_film: bool = True,
            use_exo: bool = True,
            use_regime_bias: bool = False,
            use_soft_gating: bool = False,
        ):
            super().__init__()
            self.L = lookback_len
            self.H = forecast_horizon
            self.target_dim = target_dim
            self.use_film = use_film
            self.use_exo = use_exo
            self.use_regime_bias = use_regime_bias
            self.use_soft_gating = use_soft_gating
            if target_dim != 2:
                raise ValueError("RAExoDLinearNet expects arrivals and nights (target_dim=2).")

            # DLinear backbone: decompose each target history, then project both
            # components independently from lookback length L to horizon H.
            self.decomp = SeriesDecomp(kernel_size=kernel_size)
            self.linear_trend = nn.Linear(lookback_len, forecast_horizon)
            self.linear_seas = nn.Linear(lookback_len, forecast_horizon)

            # 1. Regime Embedding
            self.regime_embedding = nn.Embedding(num_regimes, regime_emb_dim)

            # 2. Linear Delta Modulation Projections (Arrivals)
            self.film_trend_a = nn.Linear(regime_emb_dim, 1)
            self.film_bias_a = nn.Linear(regime_emb_dim, 1)
            self.linear_fut_a = nn.Linear(future_exo_dim, 1)
            self.linear_past_a = nn.Linear(lookback_len * past_exo_dim, forecast_horizon)

            # 3. Linear Delta Modulation Projections (Nights)
            self.film_trend_n = nn.Linear(regime_emb_dim, 1)
            self.film_bias_n = nn.Linear(regime_emb_dim, 1)
            self.linear_fut_n = nn.Linear(future_exo_dim, 1)
            self.linear_past_n = nn.Linear(lookback_len * past_exo_dim, forecast_horizon)

            # 4. Soft Learnable Gating (Method 3: Suppresses noise in normal regime)
            if self.use_film and self.use_soft_gating:
                self.regime_gate_a = nn.Linear(regime_emb_dim, 1)
                self.regime_gate_n = nn.Linear(regime_emb_dim, 1)

            # Start from a stable persistence/seasonal-naive forecast. The trend
            # branch carries the latest smoothed value; the seasonal branch uses
            # the corresponding month from the previous year when it is present.
            with torch.no_grad():
                self.linear_trend.weight.zero_()
                self.linear_trend.bias.zero_()
                self.linear_seas.weight.zero_()
                self.linear_seas.bias.zero_()
                for step in range(forecast_horizon):
                    self.linear_trend.weight[step, lookback_len - 1] = 1.0
                    seasonal_index = lookback_len - 12 + step
                    if 0 <= seasonal_index < lookback_len:
                        self.linear_seas.weight[step, seasonal_index] = 1.0

                # Learn regime/exogenous corrections from neutral initial output.
                for layer in (
                    self.film_trend_a, self.film_bias_a,
                    self.film_trend_n, self.film_bias_n,
                    self.linear_fut_a, self.linear_fut_n,
                    self.linear_past_a, self.linear_past_n,
                ):
                    layer.weight.zero_()
                    if layer.bias is not None:
                        layer.bias.zero_()

                # Soft learnable gate starts suppressed (bias=-3.0 -> sigmoid ~0.047)
                if self.use_film and self.use_soft_gating:
                    for gate in (self.regime_gate_a, self.regime_gate_n):
                        gate.weight.zero_()
                        nn.init.constant_(gate.bias, -3.0)

            # Trainable scaling parameters. These initial values are priors from
            # the model specification; optimization updates them during fit().
            self.alpha_trend_a = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
            self.alpha_fut_a = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.alpha_past_a = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

            self.alpha_trend_n = nn.Parameter(torch.tensor(0.35, dtype=torch.float32))
            self.alpha_fut_n = nn.Parameter(torch.tensor(0.25, dtype=torch.float32))
            self.alpha_past_n = nn.Parameter(torch.tensor(0.20, dtype=torch.float32))

        def forward(
            self,
            x_target_history: torch.Tensor, # (B, lookback_len, target_dim)
            x_future_exo: torch.Tensor,     # (B, forecast_horizon, future_exo_dim)
            x_past_exo: torch.Tensor,       # (B, lookback_len, past_exo_dim)
            regime_id: torch.Tensor,        # (B,) -> [0, 1, 2]
            s_base_a: torch.Tensor,         # (B, forecast_horizon)
            s_base_n: torch.Tensor,         # (B, forecast_horizon)
            return_components: bool = False,
        ) -> Union[Tuple[torch.Tensor, torch.Tensor], Dict[str, torch.Tensor]]:
            # 1. DLinear backbone: X -> (X_seasonal, X_trend) -> horizon forecasts.
            seasonal, trend = self.decomp(x_target_history)
            trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
            seas_out = self.linear_seas(seasonal.permute(0, 2, 1)).permute(0, 2, 1)
            e_r = self.regime_embedding(regime_id)  # (B, regime_emb_dim)
            horizon = trend_out.shape[1]

            # 2. FiLM modulation is applied to the projected trend branch.
            if self.use_film:
                gain_raw_a = self.alpha_trend_a * torch.tanh(self.film_trend_a(e_r).squeeze(-1))
                gain_raw_n = self.alpha_trend_n * torch.tanh(self.film_trend_n(e_r).squeeze(-1))
                b_raw_a = self.film_bias_a(e_r).squeeze(-1)
                b_raw_n = self.film_bias_n(e_r).squeeze(-1)
                if self.use_soft_gating:
                    gate_a = torch.sigmoid(self.regime_gate_a(e_r)).squeeze(-1)
                    gate_n = torch.sigmoid(self.regime_gate_n(e_r)).squeeze(-1)
                    film_gain_a = gate_a * gain_raw_a
                    film_gain_n = gate_n * gain_raw_n
                    beta_a = gate_a * b_raw_a
                    beta_n = gate_n * b_raw_n
                else:
                    film_gain_a = gain_raw_a
                    film_gain_n = gain_raw_n
                    beta_a = b_raw_a
                    beta_n = b_raw_n
            elif self.use_regime_bias:
                film_gain_a = x_target_history.new_zeros(x_target_history.shape[0])
                film_gain_n = torch.zeros_like(film_gain_a)
                beta_a = self.film_bias_a(e_r).squeeze(-1)
                beta_n = self.film_bias_n(e_r).squeeze(-1)
            else:
                film_gain_a = x_target_history.new_zeros(x_target_history.shape[0])
                film_gain_n = torch.zeros_like(film_gain_a)
                beta_a = torch.zeros_like(film_gain_a)
                beta_n = torch.zeros_like(film_gain_n)
            gamma_a = 1.0 + film_gain_a.unsqueeze(1)
            gamma_n = 1.0 + film_gain_n.unsqueeze(1)
            trend_mod_a = trend_out[:, :, 0] * gamma_a + beta_a.unsqueeze(1)
            trend_mod_n = trend_out[:, :, 1] * gamma_n + beta_n.unsqueeze(1)
            base_a = trend_mod_a + seas_out[:, :, 0]
            base_n = trend_mod_n + seas_out[:, :, 1]

            # 3. Exogenous relative adjustments. Future-known features are
            # projected separately at every horizon; past features use the same
            # lookback window as the target history.
            if self.use_exo:
                delta_fut_a = self.alpha_fut_a * torch.tanh(self.linear_fut_a(x_future_exo).squeeze(-1))
                delta_fut_n = self.alpha_fut_n * torch.tanh(self.linear_fut_n(x_future_exo).squeeze(-1))
                past_flat = x_past_exo.flatten(start_dim=1)
                delta_past_a = self.alpha_past_a * torch.tanh(self.linear_past_a(past_flat))
                delta_past_n = self.alpha_past_n * torch.tanh(self.linear_past_n(past_flat))
            else:
                zero_horizon = torch.zeros(
                    x_target_history.shape[0], horizon,
                    dtype=x_target_history.dtype,
                    device=x_target_history.device,
                )
                delta_fut_a = zero_horizon
                delta_fut_n = zero_horizon
                delta_past_a = zero_horizon
                delta_past_n = zero_horizon

            # 4. Exact paper equation, evaluated additively to avoid cancellation.
            # Every quantity below has shape (B,H). S is a positive scale, not
            # a second forecast mixed into the learned DLinear output.
            scale_a = s_base_a.clamp_min(1.0)
            scale_n = s_base_n.clamp_min(1.0)
            delta_trend_a = (base_a - scale_a) / scale_a
            delta_trend_n = (base_n - scale_n) / scale_n
            future_a, future_n = scale_a * delta_fut_a, scale_n * delta_fut_n
            past_a, past_n = scale_a * delta_past_a, scale_n * delta_past_n
            pre_activation_a = base_a + future_a + past_a
            pre_activation_n = base_n + future_n + past_n
            arr_pred = F.relu(pre_activation_a)
            nig_pred = F.relu(pre_activation_n)

            # No hard cross-target clipping here: L_cons must see violations.
            # The pipeline applies Pi_C after temporal/spatial fusion.

            if return_components:
                return {
                    "arrivals": arr_pred,
                    "nights": nig_pred,
                    "arrivals_raw": arr_pred,
                    "nights_raw": nig_pred,
                    "baseline_arrivals": scale_a,
                    "baseline_nights": scale_n,
                    "temporal_arrivals": base_a,
                    "temporal_nights": base_n,
                    "future_contribution_arrivals": future_a,
                    "future_contribution_nights": future_n,
                    "past_contribution_arrivals": past_a,
                    "past_contribution_nights": past_n,
                    "nonnegative_adjustment_arrivals": arr_pred - pre_activation_a,
                    "nonnegative_adjustment_nights": nig_pred - pre_activation_n,
                    "dlinear_trend_arrivals": trend_out[:, :, 0],
                    "dlinear_seasonal_arrivals": seas_out[:, :, 0],
                    "dlinear_trend_nights": trend_out[:, :, 1],
                    "dlinear_seasonal_nights": seas_out[:, :, 1],
                    "delta_dlinear_arrivals": delta_trend_a,
                    "delta_dlinear_nights": delta_trend_n,
                    "film_gamma_arrivals": gamma_a.expand(-1, horizon),
                    "film_gamma_nights": gamma_n.expand(-1, horizon),
                    "film_beta_arrivals": beta_a.unsqueeze(1).expand(-1, horizon),
                    "film_beta_nights": beta_n.unsqueeze(1).expand(-1, horizon),
                    "delta_trend_arrivals": delta_trend_a,
                    "delta_future_arrivals": delta_fut_a,
                    "delta_past_arrivals": delta_past_a,
                    "delta_trend_nights": delta_trend_n,
                    "delta_future_nights": delta_fut_n,
                    "delta_past_nights": delta_past_n,
                }

            return arr_pred, nig_pred


    class CoupledConsistencyLinearLoss(nn.Module):
        """
        Scale-Normalized Multi-Task Loss with Domain Business Consistency Penalty.
        Normalizes by baseline scale to ensure fair gradient contributions across
        both large (e.g. France, Spain) and small (e.g. Malta, Luxembourg) destinations.
        """
        def __init__(self, lambda_cons: float = 0.2, delta: float = 1.0):
            super().__init__()
            self.lambda_cons = lambda_cons
            self.delta = delta

        def forward(
            self,
            arr_pred: torch.Tensor,
            arr_true: torch.Tensor,
            nig_pred: torch.Tensor,
            nig_true: torch.Tensor,
            s_base_a: torch.Tensor,
            s_base_n: torch.Tensor,
        ) -> torch.Tensor:
            scale_a = s_base_a.clamp_min(1.0)
            scale_n = s_base_n.clamp_min(1.0)

            l_arr = F.mse_loss(arr_pred / scale_a, arr_true / scale_a)
            l_nig = F.mse_loss(nig_pred / scale_n, nig_true / scale_n)

            # Economic penalty if nights < arrivals
            violation = F.relu(arr_pred - nig_pred)
            penalty = torch.mean((violation / scale_a) ** 2)

            return l_arr + l_nig + self.lambda_cons * penalty

else:
    # Dummy fallback placeholders if torch is not installed
    class SeriesDecomp:
        pass
    class VanillaDLinearNet:
        pass
    class RAExoDLinearNet:
        pass
    class CoupledConsistencyLinearLoss:
        pass


# ============================================================
# 4. HIGH-LEVEL FORECASTER WRAPPERS (FIT / PREDICT API)
# ============================================================
class RAExoDLinearForecaster:
    """
    Scikit-learn compatible multi-horizon RA-Exo-DLinear forecaster.

    Training examples are built per entity from a target lookback window,
    known future calendar features, a past-exogenous lookback window, and H
    consecutive target values. Prediction aggregates overlapping H-step
    forecasts back to one prediction per requested entity/date row. All outputs
    are raw (before Pi_C). use_consistency controls the training penalty only;
    the enclosing pipeline enforces the physical constraint after fusion.
    """
    FUTURE_COLS = [
        "month", "days_in_month", "weekend_days", "working_days",
        "is_peak_summer", "is_shoulder_season", "is_winter_holiday",
        "holiday_days", "holiday_religious_days", "holiday_national_days",
        "holiday_long_weekends_count", "holiday_bridge_days_count",
        "holiday_has_easter", "holiday_has_christmas",
    ]

    PAST_EXO_COLS = [
        "hicp_accommodation", "hicp_overall", "hicp_acc_mom_pct",
        "hicp_acc_yoy_pct", "covid_stringency_index",
        "covid_gov_response_index", "covid_containment_index",
    ]

    TARGET_COLS = ("arrivals_total", "nights_total")
    ENTITY_COLS = ("geo", "accommodation_type")

    def __init__(
        self,
        lookback_len: int = 24,
        forecast_horizon: int = 12,
        epochs: int = 5,
        batch_size: int = 256,
        learning_rate: float = 0.0001,
        kernel_size: int = 13,
        random_state: int = 42,
        history_len: Optional[int] = None,
        num_epochs: Optional[int] = None,
        future_cols: Optional[List[str]] = None,
        past_exo_cols: Optional[List[str]] = None,
        lambda_cons: float = 0.2,
        short_history_policy: str = "edge",
        freeze_backbone: bool = True,
        **kwargs,
    ):
        if history_len is not None:
            lookback_len = history_len
        if num_epochs is not None:
            epochs = num_epochs

        self.future_cols_config = list(future_cols) if future_cols is not None else list(self.FUTURE_COLS)
        self.past_exo_cols_config = list(past_exo_cols) if past_exo_cols is not None else list(self.PAST_EXO_COLS)
        self.lookback_len = int(lookback_len)
        self.forecast_horizon = int(forecast_horizon)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.learning_rate = float(learning_rate)
        self.kernel_size = int(kernel_size)
        self.random_state = int(random_state)
        self.use_film = kwargs.get("use_film", True)
        self.use_exo = kwargs.get("use_exo", True)
        self.use_regime_bias = kwargs.get("use_regime_bias", False)
        self.use_soft_gating = kwargs.get("use_soft_gating", True)
        self.observed_target_col = kwargs.get("observed_target_col")
        self.use_consistency = kwargs.get("use_consistency", True)
        self.lambda_cons = float(lambda_cons) if self.use_consistency else 0.0
        if short_history_policy not in {"edge", "error"}:
            raise ValueError("short_history_policy must be 'edge' or 'error'.")
        self.short_history_policy = short_history_policy
        self.freeze_backbone = bool(freeze_backbone)
        self.last_prediction_diagnostics = pd.DataFrame()

        self.net: Optional[RAExoDLinearNet] = None
        self.device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu") if HAS_TORCH else "cpu"
        self.country_mapping: Dict[str, int] = {}
        self.accom_mapping: Dict[str, int] = {}
        self.seas_arr_map: Dict[Tuple[str, str, int], float] = {}
        self.seas_nig_map: Dict[Tuple[str, str, int], float] = {}
        self.entity_arr_map: Dict[Tuple[str, str], float] = {}
        self.entity_nig_map: Dict[Tuple[str, str], float] = {}
        self.future_exo_cols: List[str] = []
        self.past_exo_cols: List[str] = []
        self.future_exo_mean = np.zeros(0, dtype=np.float32)
        self.future_exo_scale = np.ones(0, dtype=np.float32)
        self.past_exo_mean = np.zeros(0, dtype=np.float32)
        self.past_exo_scale = np.ones(0, dtype=np.float32)
        self.has_monthly_seasonality = False
        self._fit_frame: Optional[pd.DataFrame] = None

    @staticmethod
    def _feature_matrix(
        frame: pd.DataFrame,
        cols: List[str],
        mean: Optional[np.ndarray] = None,
        scale: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        if not cols:
            return np.zeros((len(frame), 0), dtype=np.float32)
        values = frame.reindex(columns=cols).apply(pd.to_numeric, errors="coerce").fillna(0.0)
        matrix = values.to_numpy(dtype=np.float32)
        if mean is not None and scale is not None:
            matrix = (matrix - mean) / scale
        return matrix.astype(np.float32, copy=False)

    @staticmethod
    def _encode_regimes(frame: pd.DataFrame, default: int = 2) -> np.ndarray:
        """Prefer the fitted HMM state over calendar labels used for diagnostics."""
        regime_map = {"pre_2020": 0, "2020_2021": 1, "from_2022": 2}
        values = pd.Series(np.nan, index=frame.index)
        for column in ("regime_state", "regime", "period_group"):
            if column in frame.columns:
                raw = frame[column]
                encoded = pd.to_numeric(raw, errors="coerce").fillna(raw.map(regime_map))
                values = values.fillna(encoded)
        values = values.fillna(default).to_numpy(dtype=np.int64)
        return np.clip(values, 0, 2).astype(np.int64, copy=False)

    def _baseline_sequence(
        self,
        geo: str,
        accom: str,
        future_rows: pd.DataFrame,
    ) -> Tuple[np.ndarray, np.ndarray]:
        entity_key = (str(geo), str(accom))
        fallback_a = self.entity_arr_map.get(entity_key, 100.0)
        fallback_n = self.entity_nig_map.get(entity_key, 250.0)

        if self.has_monthly_seasonality and "month" in future_rows.columns:
            base_a = np.array([
                self.seas_arr_map.get((entity_key[0], entity_key[1], int(month)), fallback_a)
                for month in future_rows["month"]
            ], dtype=np.float32)
            base_n = np.array([
                self.seas_nig_map.get((entity_key[0], entity_key[1], int(month)), fallback_n)
                for month in future_rows["month"]
            ], dtype=np.float32)
        else:
            base_a = np.full(len(future_rows), fallback_a, dtype=np.float32)
            base_n = np.full(len(future_rows), fallback_n, dtype=np.float32)

        return np.maximum(base_a, 1.0), np.maximum(base_n, 1.0)

    def fit(self, train_df: pd.DataFrame):
        """Fit the model from per-entity sliding windows and H-step labels."""
        if not HAS_TORCH:
            raise ImportError("Training RAExoDLinearForecaster requires PyTorch.")
        if self.lookback_len < 1 or self.forecast_horizon < 1:
            raise ValueError("lookback_len and forecast_horizon must both be positive.")

        required = ["date", "geo", *self.TARGET_COLS]
        missing = [col for col in required if col not in train_df.columns]
        if missing:
            raise ValueError(f"Training data is missing required columns: {missing}")
        if self.observed_target_col is not None and self.observed_target_col not in train_df:
            raise ValueError(f"Missing observation flag: {self.observed_target_col}")

        frame = train_df.copy()
        if "accommodation_type" not in frame.columns:
            frame["accommodation_type"] = "total"
        frame["date"] = pd.to_datetime(frame["date"])
        frame = frame.sort_values([*self.ENTITY_COLS, "date"]).reset_index(drop=True)
        self._fit_frame = frame.copy()

        self.has_monthly_seasonality = (
            "month" in frame.columns and frame["month"].nunique() > 1
        )
        entity_arr = frame.groupby(list(self.ENTITY_COLS))["arrivals_total"].median().to_dict()
        entity_nig = frame.groupby(list(self.ENTITY_COLS))["nights_total"].median().to_dict()
        self.entity_arr_map = {
            (str(key[0]), str(key[1])): float(value) for key, value in entity_arr.items()
        }
        self.entity_nig_map = {
            (str(key[0]), str(key[1])): float(value) for key, value in entity_nig.items()
        }

        self.seas_arr_map = {}
        self.seas_nig_map = {}
        if self.has_monthly_seasonality:
            monthly_arr = frame.groupby([*self.ENTITY_COLS, "month"])["arrivals_total"].median().to_dict()
            monthly_nig = frame.groupby([*self.ENTITY_COLS, "month"])["nights_total"].median().to_dict()
            self.seas_arr_map = {
                (str(key[0]), str(key[1]), int(key[2])): float(value)
                for key, value in monthly_arr.items()
            }
            self.seas_nig_map = {
                (str(key[0]), str(key[1]), int(key[2])): float(value)
                for key, value in monthly_nig.items()
            }

        self.past_exo_cols = [col for col in self.past_exo_cols_config if col in frame.columns]
        self.future_exo_cols = [col for col in self.future_cols_config if col in frame.columns]
        raw_future = self._feature_matrix(frame, self.future_exo_cols)
        raw_past = self._feature_matrix(frame, self.past_exo_cols)
        self.future_exo_mean = raw_future.mean(axis=0).astype(np.float32)
        self.future_exo_scale = raw_future.std(axis=0).astype(np.float32)
        self.past_exo_mean = raw_past.mean(axis=0).astype(np.float32)
        self.past_exo_scale = raw_past.std(axis=0).astype(np.float32)
        self.future_exo_scale[self.future_exo_scale < 1e-6] = 1.0
        self.past_exo_scale[self.past_exo_scale < 1e-6] = 1.0

        samples_history = []
        samples_future_exo = []
        samples_past_exo = []
        samples_regime = []
        samples_base_a = []
        samples_base_n = []
        samples_target_a = []
        samples_target_n = []

        for (geo, accom), group in frame.groupby(list(self.ENTITY_COLS), sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            n_rows = len(group)
            first_target = self.lookback_len
            last_target_start = n_rows - self.forecast_horizon
            if last_target_start < first_target:
                continue

            targets = group[list(self.TARGET_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            future_exo = self._feature_matrix(
                group, self.future_exo_cols, self.future_exo_mean, self.future_exo_scale
            )
            past_exo = self._feature_matrix(
                group, self.past_exo_cols, self.past_exo_mean, self.past_exo_scale
            )
            regimes = self._encode_regimes(group, default=0)

            for target_start in range(first_target, last_target_start + 1):
                origin = target_start - 1
                future_rows = group.iloc[target_start:target_start + self.forecast_horizon]
                if self.observed_target_col is not None and not future_rows[self.observed_target_col].all():
                    continue
                base_a, base_n = self._baseline_sequence(
                    str(geo), str(accom), future_rows,
                )

                samples_history.append(targets[target_start - self.lookback_len:target_start])
                samples_future_exo.append(future_exo[target_start:target_start + self.forecast_horizon])
                samples_past_exo.append(past_exo[target_start - self.lookback_len:target_start])
                samples_regime.append(regimes[origin])
                samples_base_a.append(base_a)
                samples_base_n.append(base_n)
                labels = targets[target_start:target_start + self.forecast_horizon]
                samples_target_a.append(labels[:, 0])
                samples_target_n.append(labels[:, 1])

        if not samples_history:
            required_rows = self.lookback_len + self.forecast_horizon
            raise ValueError(
                f"No training windows could be formed. Each entity needs at least {required_rows} rows "
                f"(lookback={self.lookback_len}, horizon={self.forecast_horizon})."
            )

        torch.manual_seed(self.random_state)
        np.random.seed(self.random_state)

        dataset = torch.utils.data.TensorDataset(
            torch.tensor(np.stack(samples_history), dtype=torch.float32),
            torch.tensor(np.stack(samples_future_exo), dtype=torch.float32),
            torch.tensor(np.stack(samples_past_exo), dtype=torch.float32),
            torch.tensor(np.asarray(samples_regime), dtype=torch.long),
            torch.tensor(np.stack(samples_base_a), dtype=torch.float32),
            torch.tensor(np.stack(samples_base_n), dtype=torch.float32),
            torch.tensor(np.stack(samples_target_a), dtype=torch.float32),
            torch.tensor(np.stack(samples_target_n), dtype=torch.float32),
        )
        loader = torch.utils.data.DataLoader(
            dataset, batch_size=self.batch_size, shuffle=True
        )

        self.net = RAExoDLinearNet(
            lookback_len=self.lookback_len,
            forecast_horizon=self.forecast_horizon,
            past_exo_dim=len(self.past_exo_cols),
            future_exo_dim=len(self.future_exo_cols),
            kernel_size=self.kernel_size,
            use_film=self.use_film,
            use_exo=self.use_exo,
            use_regime_bias=self.use_regime_bias,
            use_soft_gating=self.use_soft_gating,
        ).to(self.device)

        backbone_params = list(self.net.linear_trend.parameters()) + list(self.net.linear_seas.parameters())
        modulation_params = [
            p for p in self.net.parameters()
            if not any(p is bp for bp in backbone_params)
        ]

        if self.freeze_backbone:
            for p in backbone_params:
                p.requires_grad = False
            optimizer = torch.optim.AdamW(
                modulation_params, lr=self.learning_rate, weight_decay=1e-4
            )
        else:
            optimizer = torch.optim.AdamW([
                {"params": backbone_params, "lr": self.learning_rate * 0.01},
                {"params": modulation_params, "lr": self.learning_rate},
            ], weight_decay=1e-4)

        criterion = CoupledConsistencyLinearLoss(lambda_cons=self.lambda_cons)
        self.net.train()

        trainable_params = [p for p in self.net.parameters() if p.requires_grad]
        if not trainable_params:
            return self

        for _ in range(self.epochs):
            for batch in loader:
                (
                    b_history, b_future, b_past, b_regime,
                    b_base_a, b_base_n, b_target_a, b_target_n,
                ) = (tensor.to(self.device) for tensor in batch)

                optimizer.zero_grad()
                pred_a, pred_n = self.net(
                    b_history, b_future, b_past, b_regime, b_base_a, b_base_n
                )
                loss = criterion(
                    pred_a, b_target_a, pred_n, b_target_n, b_base_a, b_base_n
                )
                if not loss.requires_grad:
                    break
                loss.backward()
                if trainable_params:
                    torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

        return self

    def _make_forecast_windows(
        self,
        test_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Build windows from past observations up to each forecast origin.

        For a nonempty history shorter than L, the 'edge' policy repeats its
        earliest available value on the left. These are boundary values, not
        additional observations. The 'error' policy requires L actual rows.
        No window can be formed without at least one past observation.
        """
        test = test_df.copy().reset_index(drop=True)
        if "accommodation_type" not in test.columns:
            test["accommodation_type"] = "total"
        test["date"] = pd.to_datetime(test["date"])

        target_lookup = {}
        for row_idx, row in test.iterrows():
            key = (
                str(row["geo"]),
                str(row["accommodation_type"]),
                pd.Timestamp(row["date"]),
            )
            target_lookup[key] = row_idx

        history = self._fit_frame if history_df is None else history_df
        pieces = []
        if history is not None and len(history) > 0:
            pieces.append(history.copy())
        pieces.append(test.copy())
        source = pd.concat(pieces, ignore_index=True, sort=False)
        if "accommodation_type" not in source.columns:
            source["accommodation_type"] = "total"
        source["date"] = pd.to_datetime(source["date"])
        source = source.sort_values([*self.ENTITY_COLS, "date"])
        source = source.drop_duplicates([*self.ENTITY_COLS, "date"], keep="last")

        histories = []
        future_features = []
        past_features = []
        regimes = []
        baselines_a = []
        baselines_n = []
        destinations = []
        history_counts = []

        for (geo, accom), group in source.groupby(list(self.ENTITY_COLS), sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            n_rows = len(group)
            last_target_start = n_rows - self.forecast_horizon
            first_target_start = 1 if self.short_history_policy == "edge" else self.lookback_len
            if last_target_start < first_target_start:
                continue

            targets = group[list(self.TARGET_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            future_exo = self._feature_matrix(
                group, self.future_exo_cols, self.future_exo_mean, self.future_exo_scale
            )
            past_exo = self._feature_matrix(
                group, self.past_exo_cols, self.past_exo_mean, self.past_exo_scale
            )
            regime_values = self._encode_regimes(group, default=2)

            for target_start in range(first_target_start, last_target_start + 1):
                future_rows = group.iloc[target_start:target_start + self.forecast_horizon]
                destination = np.full(self.forecast_horizon, -1, dtype=np.int64)
                for horizon_idx, (_, future_row) in enumerate(future_rows.iterrows()):
                    key = (
                        str(geo),
                        str(accom),
                        pd.Timestamp(future_row["date"]),
                    )
                    if key in target_lookup:
                        destination[horizon_idx] = target_lookup[key]
                if not np.any(destination >= 0):
                    continue

                origin = target_start - 1
                history_start = max(0, target_start - self.lookback_len)
                target_history = targets[history_start:target_start]
                past_history = past_exo[history_start:target_start]
                n_observed = len(target_history)
                n_padding = self.lookback_len - n_observed
                if n_padding:
                    target_history = np.concatenate([
                        np.repeat(target_history[:1], n_padding, axis=0), target_history
                    ])
                    past_history = np.concatenate([
                        np.repeat(past_history[:1], n_padding, axis=0), past_history
                    ])
                histories.append(target_history)
                future_features.append(future_exo[target_start:target_start + self.forecast_horizon])
                past_features.append(past_history)
                history_counts.append(n_observed)
                regimes.append(regime_values[origin])
                base_a, base_n = self._baseline_sequence(
                    str(geo), str(accom), future_rows,
                )
                baselines_a.append(base_a)
                baselines_n.append(base_n)
                destinations.append(destination)

        return {
            "history": np.stack(histories).astype(np.float32) if histories else np.zeros((0, self.lookback_len, 2), dtype=np.float32),
            "future_exo": np.stack(future_features).astype(np.float32) if future_features else np.zeros((0, self.forecast_horizon, len(self.future_exo_cols)), dtype=np.float32),
            "past_exo": np.stack(past_features).astype(np.float32) if past_features else np.zeros((0, self.lookback_len, len(self.past_exo_cols)), dtype=np.float32),
            "regime": np.asarray(regimes, dtype=np.int64),
            "baseline_a": np.stack(baselines_a).astype(np.float32) if baselines_a else np.zeros((0, self.forecast_horizon), dtype=np.float32),
            "baseline_n": np.stack(baselines_n).astype(np.float32) if baselines_n else np.zeros((0, self.forecast_horizon), dtype=np.float32),
            "destinations": np.stack(destinations) if destinations else np.zeros((0, self.forecast_horizon), dtype=np.int64),
            "history_observations": np.asarray(history_counts, dtype=np.int64),
        }

    def _infer_windows(self, windows: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """Run batches of windows and return every DLinear and adjustment component."""
        if len(windows["history"]) == 0:
            return {}

        self.net.eval()
        collected: Dict[str, List[np.ndarray]] = {}
        with torch.no_grad():
            for start in range(0, len(windows["history"]), self.batch_size):
                end = start + self.batch_size
                batch_history = torch.tensor(windows["history"][start:end], dtype=torch.float32, device=self.device)
                batch_future = torch.tensor(windows["future_exo"][start:end], dtype=torch.float32, device=self.device)
                batch_past = torch.tensor(windows["past_exo"][start:end], dtype=torch.float32, device=self.device)
                batch_regime = torch.tensor(windows["regime"][start:end], dtype=torch.long, device=self.device)
                batch_base_a = torch.tensor(windows["baseline_a"][start:end], dtype=torch.float32, device=self.device)
                batch_base_n = torch.tensor(windows["baseline_n"][start:end], dtype=torch.float32, device=self.device)
                components = self.net(
                    batch_history, batch_future, batch_past, batch_regime,
                    batch_base_a, batch_base_n,
                    return_components=True,
                )
                for name, values in components.items():
                    collected.setdefault(name, []).append(values.cpu().numpy())

        return {name: np.concatenate(parts, axis=0) for name, parts in collected.items()}

    def predict(
        self,
        test_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Return raw, nonnegative forecasts from the paper equation.

        Average the model's H-step forecasts over available rolling origins.
        Target values observed before each origin form its history. Final
        cross-target projection belongs to the enclosing RA-HE pipeline.
        """
        if not HAS_TORCH or self.net is None:
            raise RuntimeError("Fit the PyTorch forecaster before requesting predictions.")
        if len(test_df) == 0:
            self.last_prediction_diagnostics = pd.DataFrame()
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        test = test_df.copy().reset_index(drop=True)
        if "accommodation_type" not in test.columns:
            test["accommodation_type"] = "total"
        self.last_prediction_diagnostics = pd.DataFrame()
        windows = self._make_forecast_windows(test, history_df=history_df)
        if len(windows["history"]) == 0:
            raise ValueError("No forecast windows: supply enough per-entity history for the configured lookback.")
        components = self._infer_windows(windows)
        sums_a = np.zeros(len(test), dtype=np.float64)
        sums_n = np.zeros(len(test), dtype=np.float64)
        counts = np.zeros(len(test), dtype=np.int64)
        min_history = np.full(len(test), self.lookback_len, dtype=np.int64)

        for window_idx, destination in enumerate(windows["destinations"]):
            for horizon_idx, row_idx in enumerate(destination):
                if row_idx < 0:
                    continue
                sums_a[row_idx] += float(components["arrivals"][window_idx, horizon_idx])
                sums_n[row_idx] += float(components["nights"][window_idx, horizon_idx])
                counts[row_idx] += 1
                min_history[row_idx] = min(min_history[row_idx], windows["history_observations"][window_idx])

        if np.any(counts == 0):
            uncovered = test.loc[counts == 0, ["geo", "accommodation_type", "date"]]
            raise ValueError(
                f"{int(np.sum(counts == 0))} rows lack a complete forecast window; "
                "supply past observations for every entity, or use short_history_policy='edge' "
                "for nonempty short histories. Examples: "
                f"{uncovered.head(5).to_dict('records')}"
            )
        self.last_prediction_diagnostics = test[["date", "geo", "accommodation_type"]].copy()
        self.last_prediction_diagnostics["forecast_origin_count"] = counts
        self.last_prediction_diagnostics["history_observations_min"] = min_history
        self.last_prediction_diagnostics["history_padding_max"] = self.lookback_len - min_history
        n_padded_rows = int(np.sum(min_history < self.lookback_len))
        if n_padded_rows:
            warnings.warn(
                f"{n_padded_rows} forecast rows use short histories: repeat the earliest past "
                "observation on the left to fill the lookback. Padding counts are recorded "
                "in last_prediction_diagnostics; no future targets are used for padding.",
                UserWarning,
                stacklevel=2,
            )
        return (sums_a / counts).astype(np.float32), (sums_n / counts).astype(np.float32)

    def decompose(
        self,
        test_sample_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Return components averaged over the same origins as predict().

        Include the ReLU adjustment so the additive components reconstruct
        averaged raw predictions even when activation clips some horizons.
        """
        if not HAS_TORCH or self.net is None or len(test_sample_df) == 0:
            return {}

        sample = test_sample_df.copy().reset_index(drop=True)
        if "accommodation_type" not in sample.columns:
            sample["accommodation_type"] = "total"
        windows = self._make_forecast_windows(sample, history_df=history_df)
        components = self._infer_windows(windows)
        if not components:
            return {}

        result = {
            name: np.zeros(len(sample), dtype=np.float64)
            for name in components
        }
        counts = np.zeros(len(sample), dtype=np.int64)
        for window_idx, destination in enumerate(windows["destinations"]):
            for horizon_idx, row_idx in enumerate(destination):
                if row_idx < 0:
                    continue
                for name, values in components.items():
                    result[name][row_idx] += float(values[window_idx, horizon_idx])
                counts[row_idx] += 1

        valid = counts > 0
        for name in result:
            result[name][valid] /= counts[valid]
            result[name] = result[name].astype(np.float32)
        return result
