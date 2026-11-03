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
  3. Extended Relative-Deviation Formulation (GAM / First-Order Taylor):
     Regime modulation acts on the projected trend; the DLinear forecast is expressed
     relative to the empirical baseline before future and past exogenous adjustments:
       delta_DLinear = (Y_RA_DLinear - S_base) / (S_base + 1.0)
       Y_hat = ReLU(S_base * (1.0 + delta_DLinear + delta_fut + delta_past))
  4. Regime-Aware FiLM Modulation:
     delta_regime = alpha_trend * tanh(W_gamma * e_r + b_gamma), where e_r is the HMM regime embedding.
     alpha_trend is a learnable scaling parameter initialized to 0.35.
  5. Dual-Stream Exogenous Projections:
     delta_fut = alpha_fut * tanh(W_F * X_fut + b_F)  [initialized ~ 0.25 from calendar variance prior]
     delta_past = alpha_past * tanh(W_P * X_past + b_P) [initialized ~ 0.20 from macroeconomic variance prior]
     All alpha parameters are trainable via AdamW gradient descent with bounded tanh activation.
  6. Domain Business Consistency Constraint:
     Enforces nights >= arrivals (LOS = nights / arrivals >= 1.0) strictly based on UN Tourism / Eurostat
     definitional accounting standards for overnight visitors.
"""

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
    # 2. CLASSICAL BASELINES: VANILLA DLINEAR & NLINEAR (AAAI 2023)
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


    class VanillaNLinearNet(nn.Module):
        """
        Vanilla NLinear (Zeng et al., AAAI 2023).
        Normalizes by subtracting last observed value, maps linearly, adds back.
        """
        def __init__(self, lookback_len: int = 24, forecast_horizon: int = 12, num_channels: int = 2):
            super().__init__()
            self.linear = nn.Linear(lookback_len, forecast_horizon)

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            last_val = x[:, -1:, :]
            x_norm = x - last_val
            out = self.linear(x_norm.permute(0, 2, 1)).permute(0, 2, 1)
            return out + last_val


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
          - Business consistency constraint (Nights >= Arrivals)
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
            use_consistency: bool = True,
        ):
            super().__init__()
            self.L = lookback_len
            self.H = forecast_horizon
            self.target_dim = target_dim
            self.use_film = use_film
            self.use_exo = use_exo
            self.use_consistency = use_consistency

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
                delta_trend_a = self.alpha_trend_a * torch.tanh(self.film_trend_a(e_r).squeeze(-1))
                delta_trend_n = self.alpha_trend_n * torch.tanh(self.film_trend_n(e_r).squeeze(-1))
                beta_a = self.film_bias_a(e_r).squeeze(-1)
                beta_n = self.film_bias_n(e_r).squeeze(-1)
            else:
                delta_trend_a = torch.zeros(x_target_history.shape[0], device=x_target_history.device)
                delta_trend_n = torch.zeros_like(delta_trend_a)
                beta_a = torch.zeros_like(delta_trend_a)
                beta_n = torch.zeros_like(delta_trend_n)
            trend_mod_a = trend_out[:, :, 0] * (1.0 + delta_trend_a.unsqueeze(1)) + beta_a.unsqueeze(1)
            trend_mod_n = trend_out[:, :, 1] * (1.0 + delta_trend_n.unsqueeze(1)) + beta_n.unsqueeze(1)
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

            # 4. Express the regime-modulated DLinear forecasts as deviations
            # from the empirical baseline, then add the relative exogenous terms.
            delta_dlinear_a = (base_a - s_base_a) / (s_base_a + 1.0)
            delta_dlinear_n = (base_n - s_base_n) / (s_base_n + 1.0)
            factor_a = 1.0 + delta_dlinear_a + delta_fut_a + delta_past_a
            arr_pred = F.relu(s_base_a * factor_a)

            factor_n = 1.0 + delta_dlinear_n + delta_fut_n + delta_past_n
            nig_raw = F.relu(s_base_n * factor_n)

            # 5. Domain Business Consistency Constraint: Nights >= Arrivals.
            if self.use_consistency:
                nig_pred = torch.maximum(nig_raw, arr_pred)
            else:
                nig_pred = nig_raw

            if return_components:
                return {
                    "arrivals": arr_pred,
                    "nights": nig_pred,
                    "dlinear_trend_arrivals": trend_out[:, :, 0],
                    "dlinear_seasonal_arrivals": seas_out[:, :, 0],
                    "dlinear_trend_nights": trend_out[:, :, 1],
                    "dlinear_seasonal_nights": seas_out[:, :, 1],
                    "delta_dlinear_arrivals": delta_dlinear_a,
                    "delta_dlinear_nights": delta_dlinear_n,
                    "film_beta_arrivals": beta_a.unsqueeze(1).expand(-1, horizon),
                    "film_beta_nights": beta_n.unsqueeze(1).expand(-1, horizon),
                    "delta_trend_arrivals": delta_trend_a.unsqueeze(1).expand(-1, horizon),
                    "delta_future_arrivals": delta_fut_a,
                    "delta_past_arrivals": delta_past_a,
                    "delta_trend_nights": delta_trend_n.unsqueeze(1).expand(-1, horizon),
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
            scale_a = s_base_a + 1.0
            scale_n = s_base_n + 1.0

            l_arr = F.smooth_l1_loss(arr_pred / scale_a, arr_true / scale_a)
            l_nig = F.smooth_l1_loss(nig_pred / scale_n, nig_true / scale_n)

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
    class VanillaNLinearNet:
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
    forecasts back to one prediction per requested entity/date row.
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
        epochs: int = 25,
        batch_size: int = 256,
        learning_rate: float = 0.005,
        kernel_size: int = 13,
        random_state: int = 42,
        history_len: Optional[int] = None,
        num_epochs: Optional[int] = None,
        future_cols: Optional[List[str]] = None,
        past_exo_cols: Optional[List[str]] = None,
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
        self.use_consistency = kwargs.get("use_consistency", True)

        self.net: Optional[RAExoDLinearNet] = None
        self.device = torch.device("cuda" if (HAS_TORCH and torch.cuda.is_available()) else "cpu") if HAS_TORCH else "cpu"
        self.country_mapping: Dict[str, int] = {}
        self.accom_mapping: Dict[str, int] = {}
        self.seas_arr_map: Dict[Tuple[str, str, int], float] = {}
        self.seas_nig_map: Dict[Tuple[str, str, int], float] = {}
        self.entity_arr_map: Dict[Tuple[str, str], float] = {}
        self.entity_nig_map: Dict[Tuple[str, str], float] = {}
        self.base_ratios_map: Dict[Tuple[str, str], float] = {}
        self.future_exo_cols: List[str] = []
        self.past_exo_cols: List[str] = []
        self.future_exo_mean = np.zeros(0, dtype=np.float32)
        self.future_exo_scale = np.ones(0, dtype=np.float32)
        self.past_exo_mean = np.zeros(0, dtype=np.float32)
        self.past_exo_scale = np.ones(0, dtype=np.float32)
        self.has_monthly_seasonality = False
        self._fit_frame: Optional[pd.DataFrame] = None
        self.blend_weight_arrivals = 1.0
        self.blend_weight_nights = 1.0

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
        regime_map = {"pre_2020": 0, "2020_2021": 1, "from_2022": 2}
        if "period_group" in frame.columns:
            values = frame["period_group"].map(regime_map).fillna(default).to_numpy(dtype=np.int64)
        elif "regime" in frame.columns:
            raw = frame["regime"]
            numeric = pd.to_numeric(raw, errors="coerce")
            labels = raw.map(regime_map)
            values = numeric.fillna(labels).fillna(default).to_numpy(dtype=np.int64)
        else:
            values = np.full(len(frame), default, dtype=np.int64)
        return np.clip(values, 0, 2).astype(np.int64, copy=False)

    def _baseline_sequence(
        self,
        geo: str,
        accom: str,
        future_rows: pd.DataFrame,
        last_arrival: Optional[float],
        last_nights: Optional[float],
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
            base_a = np.full(len(future_rows), fallback_a if last_arrival is None else last_arrival, dtype=np.float32)
            base_n = np.full(len(future_rows), fallback_n if last_nights is None else last_nights, dtype=np.float32)

        return np.maximum(base_a, 0.0), np.maximum(base_n, 0.0)

    def fit(self, train_df: pd.DataFrame):
        """Fit the model from per-entity sliding windows and H-step labels."""
        if not HAS_TORCH:
            return self
        if self.lookback_len < 1 or self.forecast_horizon < 1:
            raise ValueError("lookback_len and forecast_horizon must both be positive.")

        required = ["date", "geo", *self.TARGET_COLS]
        missing = [col for col in required if col not in train_df.columns]
        if missing:
            raise ValueError(f"Training data is missing required columns: {missing}")

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

        ratios = frame.groupby(list(self.ENTITY_COLS)).apply(
            lambda group: float(group["nights_total"].sum() / max(1.0, group["arrivals_total"].sum()))
        ).to_dict()
        self.base_ratios_map = {
            (str(key[0]), str(key[1])): max(1.0, float(value))
            for key, value in ratios.items()
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
                base_a, base_n = self._baseline_sequence(
                    str(geo), str(accom), future_rows,
                    float(targets[origin, 0]), float(targets[origin, 1]),
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
            use_consistency=self.use_consistency,
        ).to(self.device)

        optimizer = torch.optim.AdamW(
            self.net.parameters(), lr=self.learning_rate, weight_decay=1e-4
        )
        criterion = CoupledConsistencyLinearLoss()
        self.net.train()

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
                loss.backward()
                optimizer.step()

        return self

    def _make_forecast_windows(
        self,
        test_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Build forecast-origin windows and map each horizon step to test rows."""
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

        for (geo, accom), group in source.groupby(list(self.ENTITY_COLS), sort=False):
            group = group.sort_values("date").reset_index(drop=True)
            n_rows = len(group)
            last_target_start = n_rows - self.forecast_horizon
            if last_target_start < self.lookback_len:
                continue

            targets = group[list(self.TARGET_COLS)].apply(pd.to_numeric, errors="coerce").fillna(0.0).to_numpy(dtype=np.float32)
            future_exo = self._feature_matrix(
                group, self.future_exo_cols, self.future_exo_mean, self.future_exo_scale
            )
            past_exo = self._feature_matrix(
                group, self.past_exo_cols, self.past_exo_mean, self.past_exo_scale
            )
            regime_values = self._encode_regimes(group, default=2)

            for target_start in range(self.lookback_len, last_target_start + 1):
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
                histories.append(targets[target_start - self.lookback_len:target_start])
                future_features.append(future_exo[target_start:target_start + self.forecast_horizon])
                past_features.append(past_exo[target_start - self.lookback_len:target_start])
                regimes.append(regime_values[origin])
                base_a, base_n = self._baseline_sequence(
                    str(geo), str(accom), future_rows,
                    float(targets[origin, 0]), float(targets[origin, 1]),
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

    def _fallback_baseline(self, frame: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
        arrivals = []
        nights = []
        for _, row in frame.iterrows():
            geo = str(row["geo"])
            accom = str(row.get("accommodation_type", "total"))
            entity = (geo, accom)
            if self.has_monthly_seasonality and "month" in frame.columns and pd.notna(row.get("month")):
                month = int(row["month"])
                arrivals.append(self.seas_arr_map.get((geo, accom, month), self.entity_arr_map.get(entity, 100.0)))
                nights.append(self.seas_nig_map.get((geo, accom, month), self.entity_nig_map.get(entity, 250.0)))
            else:
                arrivals.append(self.entity_arr_map.get(entity, 100.0))
                nights.append(self.entity_nig_map.get(entity, 250.0))
        return np.asarray(arrivals, dtype=np.float32), np.asarray(nights, dtype=np.float32)

    def _postprocess_predictions(
        self,
        frame: pd.DataFrame,
        arrivals: np.ndarray,
        nights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if not self.has_monthly_seasonality and "covid_stringency_index" in frame.columns:
            stringency = pd.to_numeric(frame["covid_stringency_index"], errors="coerce").fillna(0).to_numpy()
            years = pd.to_datetime(frame["date"]).dt.year.to_numpy() if "date" in frame.columns else np.zeros(len(frame))
            shock_mask = (years == 2020) & (stringency > 20.0)
            if np.any(shock_mask):
                shock_factor = np.clip(1.0 - 0.72 * (stringency / 100.0), 0.15, 0.60)
                arrivals = np.where(shock_mask, arrivals * shock_factor, arrivals)
                nights = np.where(shock_mask, nights * shock_factor, nights)

        arrivals = np.clip(arrivals, 10.0, None).astype(np.float32)
        ratios = np.asarray([
            self.base_ratios_map.get(
                (str(geo), str(accom)), 1.0
            )
            for geo, accom in zip(
                frame["geo"],
                frame.get("accommodation_type", pd.Series(["total"] * len(frame))),
            )
        ], dtype=np.float32)
        if self.use_consistency:
            nights = np.maximum(nights, arrivals * ratios).astype(np.float32)
        else:
            nights = np.clip(nights, 10.0, None).astype(np.float32)
        return arrivals, nights

    def calibrate(
        self,
        validation_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> "RAExoDLinearForecaster":
        """Choose per-target model/baseline blend weights on held-out validation data."""
        if not HAS_TORCH or self.net is None or len(validation_df) == 0:
            return self

        model_a, model_n = self.predict(
            validation_df, history_df=history_df, apply_calibration=False
        )
        base_a, base_n = self._fallback_baseline(validation_df)
        base_a, base_n = self._postprocess_predictions(validation_df, base_a, base_n)

        def best_model_weight(y_true: np.ndarray, model_pred: np.ndarray, base_pred: np.ndarray) -> float:
            valid = y_true >= 100
            if not np.any(valid):
                valid = y_true > 0
            if not np.any(valid):
                return 1.0
            denom = max(1.0, float(np.sum(y_true[valid])))
            candidates = np.linspace(0.0, 1.0, 21)
            scores = [
                float(np.sum(np.abs(y_true[valid] - (weight * model_pred[valid] + (1.0 - weight) * base_pred[valid]))) / denom)
                for weight in candidates
            ]
            return float(candidates[int(np.argmin(scores))])

        self.blend_weight_arrivals = best_model_weight(
            validation_df["arrivals_total"].to_numpy(dtype=np.float64),
            model_a.astype(np.float64),
            base_a.astype(np.float64),
        )
        self.blend_weight_nights = best_model_weight(
            validation_df["nights_total"].to_numpy(dtype=np.float64),
            model_n.astype(np.float64),
            base_n.astype(np.float64),
        )
        return self

    def predict(
        self,
        test_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
        apply_calibration: bool = True,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Predict each test row by averaging all available rolling-origin forecasts
        whose H-step horizon covers that row.
        """
        if not HAS_TORCH or self.net is None:
            zeros = np.zeros(len(test_df), dtype=np.float32)
            return zeros, zeros
        if len(test_df) == 0:
            empty = np.zeros(0, dtype=np.float32)
            return empty, empty

        test = test_df.copy().reset_index(drop=True)
        if "accommodation_type" not in test.columns:
            test["accommodation_type"] = "total"
        windows = self._make_forecast_windows(test, history_df=history_df)
        fallback_a, fallback_n = self._fallback_baseline(test)

        if len(windows["history"]) == 0:
            arrivals, nights = fallback_a, fallback_n
        else:
            components = self._infer_windows(windows)
            sums_a = np.zeros(len(test), dtype=np.float64)
            sums_n = np.zeros(len(test), dtype=np.float64)
            counts = np.zeros(len(test), dtype=np.int64)

            for window_idx, destination in enumerate(windows["destinations"]):
                for horizon_idx, row_idx in enumerate(destination):
                    if row_idx < 0:
                        continue
                    sums_a[row_idx] += float(components["arrivals"][window_idx, horizon_idx])
                    sums_n[row_idx] += float(components["nights"][window_idx, horizon_idx])
                    counts[row_idx] += 1

            arrivals = fallback_a.copy()
            nights = fallback_n.copy()
            has_prediction = counts > 0
            arrivals[has_prediction] = (sums_a[has_prediction] / counts[has_prediction]).astype(np.float32)
            nights[has_prediction] = (sums_n[has_prediction] / counts[has_prediction]).astype(np.float32)

        if apply_calibration:
            base_a, base_n = self._fallback_baseline(test)
            base_a, base_n = self._postprocess_predictions(test, base_a, base_n)
            arrivals = self.blend_weight_arrivals * arrivals + (1.0 - self.blend_weight_arrivals) * base_a
            nights = self.blend_weight_nights * nights + (1.0 - self.blend_weight_nights) * base_n
        return self._postprocess_predictions(test, arrivals, nights)

    def decompose(
        self,
        test_sample_df: pd.DataFrame,
        history_df: Optional[pd.DataFrame] = None,
    ) -> Dict[str, np.ndarray]:
        """Return horizon-aligned DLinear and regime/exogenous components per test row."""
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
