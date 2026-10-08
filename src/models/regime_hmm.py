"""
Hidden Markov Model (HMM) Market Navigator.
Unsupervised macro regime detector that classifies time steps into 3 structural regimes:
  - State 0: Normal / Steady Growth (Bình thường)
  - State 1: Disruption Shock / Crisis (Khủng hoảng / Đứt gãy)
  - State 2: Post-Crisis Recovery / Revenge Travel (Phục hồi)
Outputs posterior state probabilities: P(Regime = k).
"""

from typing import Optional, Tuple
import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HAS_HMMLEARN = True
except ImportError:
    HAS_HMMLEARN = False


class RegimeHMMNavigator:
    """
    Unsupervised HMM Market Navigator.
    Fits a 3-state Gaussian HMM on macroeconomic & crisis indicators:
      - Monthly arrivals / nights YoY growth rate
      - HICP accommodation price inflation rate
      - COVID Stringency index
    """

    REGIME_NAMES = {
        0: "Normal (Bình thường)",
        1: "Disruption Shock (Khủng hoảng)",
        2: "Post-Crisis Recovery (Phục hồi)",
    }

    def __init__(
        self,
        n_regimes: int = 3,
        n_iter: int = 200,
        covariance_type: str = "full",
        random_state: int = 42,
    ):
        self.n_regimes = n_regimes
        self.n_iter = n_iter
        self.covariance_type = covariance_type
        self.random_state = random_state
        self.model = None
        self.state_order_map = {}

    def _prepare_macro_signals(self, df: pd.DataFrame) -> np.ndarray:
        """
        Extracts aggregated macro signals to detect macroeconomic / crisis regimes.
        Adapts seamlessly to both monthly Eurostat and annual World Bank panels.
        """
        price_col = "hicp_accommodation" if "hicp_accommodation" in df.columns else (
            "inflation_lag1" if "inflation_lag1" in df.columns else None
        )
        agg_dict = {
            "arrivals_total": "sum",
            "nights_total": "sum",
        }
        if price_col:
            agg_dict[price_col] = "mean"
        if "covid_stringency_index" in df.columns:
            agg_dict["covid_stringency_index"] = "mean"

        macro_monthly = df.groupby("date").agg(agg_dict).reset_index()

        # Compute growth rate (YoY diff) and volatility
        arr_series = macro_monthly["arrivals_total"]
        shift_step = 12 if len(macro_monthly) >= 60 else 1
        macro_monthly["growth_yoy"] = (
            (arr_series - arr_series.shift(shift_step)) / (arr_series.shift(shift_step) + 1e-5)
        ).fillna(0.0)

        # Signal matrix: [growth_yoy, stringency_index, inflation/price]
        feature_cols = ["growth_yoy"]
        if "covid_stringency_index" in macro_monthly.columns:
            feature_cols.append("covid_stringency_index")
        if price_col and price_col in macro_monthly.columns:
            feature_cols.append(price_col)

        signals = macro_monthly[feature_cols].values
        # Standardize signals
        mean = np.nanmean(signals, axis=0)
        std = np.nanstd(signals, axis=0) + 1e-6
        norm_signals = np.nan_to_num((signals - mean) / std)
        return norm_signals, macro_monthly["date"].values

    def fit(self, df: pd.DataFrame) -> "RegimeHMMNavigator":
        """
        Fits the unsupervised HMM on the historical macro signals.
        Sorts hidden states so that:
          - State 1 has highest crisis/stringency / lowest growth (Disruption)
          - State 2 has highest recovery growth after shock (Recovery)
          - State 0 has steady/normal growth (Normal)
        """
        signals, dates = self._prepare_macro_signals(df)

        if HAS_HMMLEARN:
            self.model = GaussianHMM(
                n_components=self.n_regimes,
                covariance_type=self.covariance_type,
                n_iter=self.n_iter,
                random_state=self.random_state,
            )
            self.model.fit(signals)
            raw_states = self.model.predict(signals)
            raw_probs = self.model.predict_proba(signals)
        else:
            # Fallback heuristic regime segmenter if hmmlearn is unavailable
            raw_states = np.zeros(len(signals), dtype=int)
            raw_probs = np.zeros((len(signals), self.n_regimes), dtype=np.float32)
            # stringency > 20 or growth < -0.4 -> Disruption (1)
            # growth > 0.3 after Disruption -> Recovery (2)
            growth = signals[:, 0]
            stringency = signals[:, 1]
            for t in range(len(signals)):
                if stringency[t] > 0.5 or growth[t] < -1.0:
                    raw_states[t] = 1
                    raw_probs[t] = [0.05, 0.90, 0.05]
                elif growth[t] > 0.8:
                    raw_states[t] = 2
                    raw_probs[t] = [0.10, 0.10, 0.80]
                else:
                    raw_states[t] = 0
                    raw_probs[t] = [0.85, 0.08, 0.07]

        # Order states semantically
        # Disruption: state with lowest growth
        mean_growth_per_state = []
        for s in range(self.n_regimes):
            idx = np.where(raw_states == s)[0]
            if len(idx) > 0:
                mean_growth_per_state.append((s, signals[idx, 0].mean()))
            else:
                mean_growth_per_state.append((s, 0.0))

        # Sort by growth: lowest is Disruption (1), middle is Normal (0), highest is Recovery (2)
        sorted_by_growth = sorted(mean_growth_per_state, key=lambda x: x[1])
        disruption_state = sorted_by_growth[0][0]
        recovery_state = sorted_by_growth[-1][0]
        remaining = [s for s in range(self.n_regimes) if s not in (disruption_state, recovery_state)]
        normal_state = remaining[0] if remaining else 0

        self.state_order_map = {
            normal_state: 0,
            disruption_state: 1,
            recovery_state: 2,
        }
        return self

    def predict_regimes(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
        """
        Infers regime states and posterior probabilities:
        Returns:
          - state_labels: (T,) array of regime IDs {0: Normal, 1: Disruption, 2: Recovery}
          - state_probs: (T, 3) matrix of [P(Normal), P(Disruption), P(Recovery)]
          - regime_df: DataFrame mapped by date
        """
        signals, dates = self._prepare_macro_signals(df)

        if HAS_HMMLEARN and self.model is not None:
            raw_states = self.model.predict(signals)
            raw_probs = self.model.predict_proba(signals)
        else:
            raw_states = np.zeros(len(signals), dtype=int)
            raw_probs = np.zeros((len(signals), self.n_regimes), dtype=np.float32)
            growth = signals[:, 0]
            stringency = signals[:, 1]
            for t in range(len(signals)):
                if stringency[t] > 0.5 or growth[t] < -1.0:
                    raw_states[t] = 1
                    raw_probs[t] = [0.05, 0.90, 0.05]
                elif growth[t] > 0.8:
                    raw_states[t] = 2
                    raw_probs[t] = [0.10, 0.10, 0.80]
                else:
                    raw_states[t] = 0
                    raw_probs[t] = [0.85, 0.08, 0.07]

        # Re-map states according to semantic ordering
        ordered_states = np.array([self.state_order_map.get(s, s) for s in raw_states], dtype=int)
        ordered_probs = np.zeros_like(raw_probs)
        for old_s, new_s in self.state_order_map.items():
            ordered_probs[:, new_s] = raw_probs[:, old_s]

        # Build regime summary table
        regime_df = pd.DataFrame({
            "date": dates,
            "regime_state": ordered_states,
            "p_normal": ordered_probs[:, 0],
            "p_disruption": ordered_probs[:, 1],
            "p_recovery": ordered_probs[:, 2],
        })
        return ordered_states, ordered_probs, regime_df
