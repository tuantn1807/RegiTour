"""
Feature Attribution Module for Tourism Demand Forecasting.
Provides:
1. Exact TreeSHAP computation via CatBoost native ShapValues.
2. Permutation Feature Importance for end-to-end multi-target pipelines.
3. Macroeconomic elasticity analysis (price elasticity, stringency impact).
"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd


def compute_tree_shap(
    model,
    X: np.ndarray,
    feature_names: List[str],
) -> pd.DataFrame:
    """
    Computes exact TreeSHAP values using CatBoost native ShapValues interface.

    Parameters
    ----------
    model : CatBoostRegressor or CatBoostRegimeModel
        Trained model instance.
    X : np.ndarray of shape (N, D)
        Evaluation feature matrix.
    feature_names : List[str]
        List of feature column names.

    Returns
    -------
    pd.DataFrame
        DataFrame of mean absolute SHAP values and relative percentage contributions.
    """
    try:
        import catboost as cb
        raw_model = getattr(model, "model_arr", getattr(model, "model", model))
        if hasattr(raw_model, "get_feature_importance"):
            pool = cb.Pool(X)
            # ShapValues returns (N, D + 1) where the last column is base value E[f(x)]
            shap_matrix = raw_model.get_feature_importance(type="ShapValues", data=pool)
            shap_features = shap_matrix[:, :-1]
            mean_abs_shap = np.mean(np.abs(shap_features), axis=0)
            total_shap = np.sum(mean_abs_shap) + 1e-12

            df_shap = pd.DataFrame({
                "feature": feature_names[:len(mean_abs_shap)],
                "mean_abs_shap": mean_abs_shap,
                "importance_pct": (mean_abs_shap / total_shap) * 100.0,
            }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
            return df_shap
    except Exception as e:
        print(f"[XAI Attribution] TreeSHAP calculation fallback: {e}")

    # Fallback to feature importances if ShapValues is unavailable
    raw_model = getattr(model, "model_arr", getattr(model, "model", model))
    if hasattr(raw_model, "feature_importances_"):
        fi = raw_model.feature_importances_
        df_fi = pd.DataFrame({
            "feature": feature_names[:len(fi)],
            "mean_abs_shap": fi,
            "importance_pct": (fi / (np.sum(fi) + 1e-12)) * 100.0,
        }).sort_values(by="mean_abs_shap", ascending=False).reset_index(drop=True)
        return df_fi

    return pd.DataFrame({"feature": feature_names, "mean_abs_shap": 0.0, "importance_pct": 0.0})


def compute_permutation_importance(
    predict_fn,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: List[str],
    n_repeats: int = 5,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Computes model-agnostic Permutation Feature Importance evaluated via WAPE increase.

    Parameters
    ----------
    predict_fn : callable
        Function f(X) -> y_pred.
    X : np.ndarray of shape (N, D)
        Feature matrix.
    y : np.ndarray of shape (N,)
        True target values.
    feature_names : List[str]
        Names of the features.
    n_repeats : int
        Number of permutation shuffles per feature.

    Returns
    -------
    pd.DataFrame
        Ranking of features based on WAPE increase percentage.
    """
    rng = np.random.default_rng(random_state)
    y_baseline = predict_fn(X)
    denom = np.sum(np.abs(y)) + 1e-12
    base_wape = np.sum(np.abs(y - y_baseline)) / denom

    results = []
    num_features = min(X.shape[1], len(feature_names))

    for col_idx in range(num_features):
        wape_deltas = []
        for _ in range(n_repeats):
            X_permuted = X.copy()
            perm_indices = rng.permutation(X.shape[0])
            X_permuted[:, col_idx] = X_permuted[perm_indices, col_idx]

            y_perm = predict_fn(X_permuted)
            wape_perm = np.sum(np.abs(y - y_perm)) / denom
            wape_deltas.append(max(0.0, wape_perm - base_wape))

        mean_delta = np.mean(wape_deltas)
        std_delta = np.std(wape_deltas)
        results.append({
            "feature": feature_names[col_idx],
            "wape_increase_mean": mean_delta * 100.0,
            "wape_increase_std": std_delta * 100.0,
        })

    df_perm = pd.DataFrame(results).sort_values(by="wape_increase_mean", ascending=False).reset_index(drop=True)
    return df_perm


class FeatureAttributionAnalyzer:
    """
    Unified Manager for Feature Attribution and Elasticity Quantification.
    """
    def __init__(self, feature_names: List[str]):
        self.feature_names = feature_names

    def analyze_macro_elasticity(
        self,
        df_panel: pd.DataFrame,
        price_col: str = "hicp_accommodation",
        target_col: str = "arrivals",
        geo_col: str = "geo",
    ) -> pd.DataFrame:
        """
        Estimates the empirical price elasticity of demand (log-log slope):
        epsilon = d(ln Y) / d(ln P) across countries.
        """
        elasticity_records = []
        for geo, group in df_panel.groupby(geo_col):
            sub = group[[price_col, target_col]].dropna()
            sub = sub[(sub[price_col] > 0) & (sub[target_col] > 0)]
            if len(sub) >= 24:
                ln_p = np.log(sub[price_col].values)
                ln_y = np.log(sub[target_col].values)
                # First differences to eliminate stationarity bias
                d_ln_p = np.diff(ln_p)
                d_ln_y = np.diff(ln_y)
                var_p = np.var(d_ln_p)
                if var_p > 1e-8:
                    beta = np.cov(d_ln_y, d_ln_p)[0, 1] / var_p
                    elasticity_records.append({
                        "geo": geo,
                        "price_elasticity": float(beta),
                        "num_obs": len(sub),
                    })

        df_elasticity = pd.DataFrame(elasticity_records).sort_values(by="price_elasticity")
        return df_elasticity
