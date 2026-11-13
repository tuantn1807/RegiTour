"""
Operational Decision Support Engine for Destination Management & Smart Tourism Governance.

Implements the four operational decision layers defined in Section VI of paper_en.tex
(Tourism Management Implications and Discussion):
  1. Coordinated Arrival Throughput & Stay Demand Planning (Arrivals, Nights, LOS + Pi_C)
  2. Accommodation Capacity & Seasonal Staffing Allocation across NACE I551, I552, I553
  3. Crisis Regime Detection & Stress Testing (HMM + Shock Folds + Macro Stress Simulator)
  4. Benchmarking Post-Shock Recovery Trajectories & 90% Conformal Risk Bands (I_0.90)
"""

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd

from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.regime_hmm import RegimeHMMNavigator


NACE_LABELS: Dict[str, Tuple[str, str]] = {
    "hotel": ("I551", "I551 — Khách sạn & tương đương (Hotels & Similar)"),
    "short_stay": ("I552", "I552 — Lưu trú ngắn ngày (Holiday & Short-Stay)"),
    "camping": ("I553", "I553 — Khu cắm trại (Camping Grounds)"),
    "total": ("I551–I553", "I551–I553 — Tổng hợp toàn ngành lưu trú (Total)"),
}

NACE_OPERATIONAL_CATEGORIES: Tuple[str, ...] = ("hotel", "short_stay", "camping")

PAPER_CASE_STUDIES: Dict[str, str] = {
    "CY": "Incomplete Recovery (Chậm phục hồi — Du lịch biển dài ngày)",
    "SE": "Balanced Recovery (Phục hồi cân bằng — Đô thị & công vụ)",
    "DK": "Accelerated Rebound (Bứt phá tăng trưởng — Vượt mức 2019)",
}


@dataclass
class DSSDataBundle:
    """Container holding all empirical datasets and experiment artifacts for the DSS."""

    forecasts_df: pd.DataFrame
    history_df: Optional[pd.DataFrame] = None
    regime_metrics_df: Optional[pd.DataFrame] = None
    country_los_df: Optional[pd.DataFrame] = None
    cases_df: Optional[pd.DataFrame] = None
    benchmark_df: Optional[pd.DataFrame] = None
    coverage_df: Optional[pd.DataFrame] = None
    elasticity_df: Optional[pd.DataFrame] = None
    fold_summaries: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    country_names: Dict[str, str] = field(default_factory=dict)
    source_mode: str = "reviewer_experiments"


def _safe_read_csv(path: Path) -> Optional[pd.DataFrame]:
    if path.exists():
        return pd.read_csv(path)
    return None


def _safe_read_json(path: Path) -> Optional[Dict[str, Any]]:
    if path.exists():
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    return None


def _load_standardized_forecasts(root_dir: Path) -> Tuple[pd.DataFrame, str]:
    """
    Loads forecast results and standardizes schema across reviewer_experiments and pipeline CSVs.
    Returns (df, source_mode).
    """
    recent_dir = root_dir / "results" / "reviewer_experiments" / "recent"
    intervals_path = recent_dir / "intervals.csv"
    raw_path = recent_dir / "full_raw.csv"
    dlinear_path = recent_dir / "dlinear.csv"

    if intervals_path.exists():
        df = pd.read_csv(intervals_path)
        df["date"] = pd.to_datetime(df["date"])
        if "origin" in df.columns:
            df["origin_label"] = pd.to_datetime(df["origin"]).dt.strftime("%Y-%m-%d")
        else:
            df["origin_label"] = "2023-12-31"

        df = df.rename(
            columns={
                "observed_arrivals_total": "actual_arrivals",
                "observed_nights_total": "actual_nights",
            }
        )

        # Attach unconstrained raw predictions (before Pi_C) for audit comparison
        if raw_path.exists():
            raw_df = pd.read_csv(raw_path)
            raw_df["date"] = pd.to_datetime(raw_df["date"])
            merge_cols = ["geo", "accommodation_type", "date"]
            if "origin" in raw_df.columns and "origin" in df.columns:
                merge_cols.append("origin")
            raw_subset = raw_df[merge_cols + ["pred_arrivals", "pred_nights"]].rename(
                columns={
                    "pred_arrivals": "pred_arrivals_raw",
                    "pred_nights": "pred_nights_raw",
                }
            )
            df = df.merge(raw_subset, on=merge_cols, how="left")
        else:
            df["pred_arrivals_raw"] = df["pred_arrivals"]
            df["pred_nights_raw"] = df["pred_nights"]

        # Attach Vanilla DLinear baseline for direct operational comparison
        if dlinear_path.exists():
            dl_df = pd.read_csv(dlinear_path)
            dl_df["date"] = pd.to_datetime(dl_df["date"])
            merge_cols = ["geo", "accommodation_type", "date"]
            if "origin" in dl_df.columns and "origin" in df.columns:
                merge_cols.append("origin")
            dl_subset = dl_df[merge_cols + ["pred_arrivals", "pred_nights"]].rename(
                columns={
                    "pred_arrivals": "dlinear_pred_arrivals",
                    "pred_nights": "dlinear_pred_nights",
                }
            )
            df = df.merge(dl_subset, on=merge_cols, how="left")

        return df, "reviewer_experiments"

    # Fallback to pipeline CSV output
    fallback_path = root_dir / "results" / "forecast_results_ra_he_dlinear_eurostat.csv"
    if fallback_path.exists():
        df = pd.read_csv(fallback_path)
        df["date"] = pd.to_datetime(df["date"])
        df["origin_label"] = "2023-12-31"
        df["horizon"] = df.groupby(["geo", "accommodation_type"]).cumcount() + 1
        df = df.rename(
            columns={
                "arrivals_total": "actual_arrivals",
                "nights_total": "actual_nights",
                "pred_arrivals_point": "pred_arrivals",
                "pred_nights_point": "pred_nights",
                "pred_arrivals_lower_90": "lower_arrivals",
                "pred_arrivals_upper_90": "upper_arrivals",
                "pred_nights_lower_90": "lower_nights",
                "pred_nights_upper_90": "upper_nights",
            }
        )
        df["country"] = df["geo"]
        df["observed_pair"] = df["actual_arrivals"].notna() & df["actual_nights"].notna()
        return df, "pipeline_csv"

    raise FileNotFoundError(
        "No forecast results found under results/reviewer_experiments/recent/intervals.csv "
        "or results/forecast_results_ra_he_dlinear_eurostat.csv."
    )


def load_dss_bundle(root_dir: Optional[Path] = None) -> DSSDataBundle:
    """Loads all empirical forecasts, historical Eurostat features, and evaluation tables."""
    resolved_root = Path(root_dir) if root_dir is not None else Path(__file__).resolve().parents[2]
    forecasts_df, source_mode = _load_standardized_forecasts(resolved_root)

    parquet_path = resolved_root / "data" / "features" / "tourism_europe_features.parquet"
    history_df = pd.read_parquet(parquet_path) if parquet_path.exists() else None
    if history_df is not None:
        history_df["date"] = pd.to_datetime(history_df["date"])

    # Build clean geo -> display country name lookup
    country_names: Dict[str, str] = {}
    if history_df is not None and {"geo", "country"}.issubset(history_df.columns):
        pairs = history_df[["geo", "country"]].dropna().drop_duplicates()
        country_names.update(dict(zip(pairs["geo"], pairs["country"])))
    if "country" in forecasts_df.columns:
        pairs = forecasts_df[["geo", "country"]].dropna().drop_duplicates()
        for g, c in zip(pairs["geo"], pairs["country"]):
            country_names.setdefault(g, c)

    if "country" not in forecasts_df.columns:
        forecasts_df["country"] = forecasts_df["geo"].map(lambda g: country_names.get(g, g))

    rev_dir = resolved_root / "results" / "reviewer_experiments"
    fold_summaries: Dict[str, Dict[str, Any]] = {}
    for fold in ("tranquil", "shock", "recovery", "recent"):
        fold_summaries[fold] = {}
        for model_key in ("dlinear", "full", "full_core", "full_raw"):
            payload = _safe_read_json(rev_dir / fold / f"{model_key}.json")
            if payload is not None:
                fold_summaries[fold][model_key] = payload

    return DSSDataBundle(
        forecasts_df=forecasts_df,
        history_df=history_df,
        regime_metrics_df=_safe_read_csv(rev_dir / "regime_metrics.csv"),
        country_los_df=_safe_read_csv(rev_dir / "country_los.csv"),
        cases_df=_safe_read_csv(rev_dir / "cases.csv"),
        benchmark_df=_safe_read_csv(rev_dir / "benchmark.csv"),
        coverage_df=_safe_read_csv(rev_dir / "coverage.csv"),
        elasticity_df=_safe_read_csv(resolved_root / "results" / "tables" / "xai_price_elasticity.csv"),
        fold_summaries=fold_summaries,
        country_names=country_names,
        source_mode=source_mode,
    )


def _filter_forecast_window(
    forecasts_df: pd.DataFrame,
    geo: str,
    accom_type: str,
    horizon_months: int = 12,
    origin_label: Optional[str] = None,
) -> pd.DataFrame:
    """Slices a destination-accommodation forecast window up to horizon_months."""
    sub = forecasts_df[
        (forecasts_df["geo"] == geo) & (forecasts_df["accommodation_type"] == accom_type)
    ].copy()
    if sub.empty:
        return sub

    if origin_label is not None and "origin_label" in sub.columns:
        sub = sub[sub["origin_label"] == origin_label].copy()
    elif "origin_label" in sub.columns:
        first_origin = sorted(sub["origin_label"].unique())[0]
        sub = sub[sub["origin_label"] == first_origin].copy()

    sub = sub.sort_values("date").head(horizon_months).reset_index(drop=True)
    return sub


def compute_coordinated_throughput_plan(
    forecasts_df: pd.DataFrame,
    geo: str,
    accom_type: str = "total",
    horizon_months: int = 12,
    origin_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Layer 1: Coordinated Arrival Throughput & Stay Demand Planning.

    Produces joint border/airport/check-in arrival throughput and cumulative
    room-night/utility/catering stay demand, enforcing Pi_C (Nights >= Arrivals)
    and auditing any raw violations corrected before budget allocation.
    """
    sub = _filter_forecast_window(forecasts_df, geo, accom_type, horizon_months, origin_label)
    if sub.empty:
        return sub, {
            "total_arrivals": 0.0,
            "total_nights": 0.0,
            "avg_los": 0.0,
            "peak_daily_arrivals": 0.0,
            "peak_daily_nights": 0.0,
            "raw_violations": 0,
            "post_violations": 0,
            "dlinear_violations": 0,
            "total_nights_corrected": 0.0,
        }

    days_in_month = sub["date"].dt.days_in_month.astype(float)
    arr_raw = sub["pred_arrivals_raw"].fillna(sub["pred_arrivals"]).to_numpy(dtype=np.float64)
    nig_raw = sub["pred_nights_raw"].fillna(sub["pred_nights"]).to_numpy(dtype=np.float64)

    # Apply official Pi_C verification using project's CoupledConsistencyOptimizer
    arr_proj, nig_proj, proj_flags = CoupledConsistencyOptimizer.project_physical_pair(
        sub["pred_arrivals"].to_numpy(dtype=np.float64),
        sub["pred_nights"].to_numpy(dtype=np.float64),
    )

    sub["pred_arrivals_safe"] = arr_proj
    sub["pred_nights_safe"] = nig_proj
    sub["pred_los"] = np.where(arr_proj > 0.0, nig_proj / np.maximum(arr_proj, 1.0), 0.0)
    sub["daily_arrivals_throughput"] = arr_proj / days_in_month
    sub["daily_nights_demand"] = nig_proj / days_in_month
    sub["raw_violation"] = nig_raw < arr_raw
    sub["post_violation"] = proj_flags
    sub["pi_c_adjustment_nights"] = np.maximum(0.0, nig_proj - nig_raw)

    dlinear_violations = 0
    if {"dlinear_pred_arrivals", "dlinear_pred_nights"}.issubset(sub.columns):
        dl_arr = sub["dlinear_pred_arrivals"].to_numpy(dtype=np.float64)
        dl_nig = sub["dlinear_pred_nights"].to_numpy(dtype=np.float64)
        sub["dlinear_violation"] = dl_nig < dl_arr
        dlinear_violations = int(np.sum(sub["dlinear_violation"]))

    total_arr = float(np.sum(arr_proj))
    total_nig = float(np.sum(nig_proj))
    summary = {
        "total_arrivals": total_arr,
        "total_nights": total_nig,
        "avg_los": total_nig / max(total_arr, 1.0),
        "peak_daily_arrivals": float(sub["daily_arrivals_throughput"].max()),
        "peak_daily_nights": float(sub["daily_nights_demand"].max()),
        "raw_violations": int(np.sum(sub["raw_violation"])),
        "post_violations": int(np.sum(sub["post_violation"])),
        "dlinear_violations": dlinear_violations,
        "total_nights_corrected": float(np.sum(sub["pi_c_adjustment_nights"])),
    }
    return sub, summary


def compute_nace_capacity_staffing(
    forecasts_df: pd.DataFrame,
    geo: str,
    horizon_months: int = 12,
    guests_per_unit: Optional[Dict[str, float]] = None,
    available_units: Optional[Dict[str, float]] = None,
    hours_per_arrival: float = 0.50,
    hours_per_night: float = 0.35,
    hours_per_fte_month: float = 160.0,
    maintenance_occ_threshold_pct: float = 35.0,
    origin_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    """
    Layer 2: Accommodation Capacity & Seasonal Staffing Allocation across NACE I551/I552/I553.

    Decomposes arrivals and nights across hotels (I551), short-stay rentals (I552),
    and camping grounds (I553) and translates forecasts into room/unit occupancy %,
    housekeeping & reception FTE requirements, peak strain alerts, and maintenance windows.
    """
    default_guests = {"hotel": 1.8, "short_stay": 2.6, "camping": 2.8}
    if guests_per_unit:
        default_guests.update(guests_per_unit)

    frames: List[pd.DataFrame] = []
    category_summaries: Dict[str, Dict[str, Any]] = {}

    for accom in NACE_OPERATIONAL_CATEGORIES:
        sub = _filter_forecast_window(forecasts_df, geo, accom, horizon_months, origin_label)
        if sub.empty:
            continue

        days = sub["date"].dt.days_in_month.astype(float).to_numpy()
        arr = np.maximum(0.0, sub["pred_arrivals"].to_numpy(dtype=np.float64))
        nig = np.maximum(arr, sub["pred_nights"].to_numpy(dtype=np.float64))

        gpu = max(1.0, float(default_guests.get(accom, 2.0)))
        occupied_units_day = nig / (days * gpu)

        # If operator does not supply explicit room/unit capacity, calibrate baseline
        # capacity at 115% of observed peak occupied units to show relative seasonal load.
        if available_units and accom in available_units and available_units[accom] > 0:
            cap_units = float(available_units[accom])
            capacity_calibrated = False
        else:
            cap_units = max(1.0, float(np.max(occupied_units_day) * 1.15))
            capacity_calibrated = True

        occ_pct = (occupied_units_day / cap_units) * 100.0
        workload_hours = (arr * hours_per_arrival) + (nig * hours_per_night)
        required_fte = workload_hours / max(1.0, hours_per_fte_month)

        status = np.where(
            occ_pct >= 85.0,
            "Peak Strain (>=85%)",
            np.where(occ_pct < maintenance_occ_threshold_pct, "Maintenance Window", "Normal Operations"),
        )

        nace_code, nace_name = NACE_LABELS[accom]
        cat_df = pd.DataFrame(
            {
                "date": sub["date"],
                "month_label": sub["date"].dt.strftime("%Y-%m"),
                "geo": geo,
                "accommodation_type": accom,
                "nace_code": nace_code,
                "nace_name": nace_name,
                "pred_arrivals": arr,
                "pred_nights": nig,
                "occupied_units_per_day": occupied_units_day,
                "available_units_capacity": cap_units,
                "occupancy_pct": occ_pct,
                "required_fte": required_fte,
                "operational_status": status,
            }
        )
        frames.append(cat_df)

        peak_idx = int(np.argmax(required_fte))
        trough_idx = int(np.argmin(required_fte))
        maint_months = cat_df.loc[cat_df["operational_status"] == "Maintenance Window", "month_label"].tolist()
        peak_months = cat_df.loc[cat_df["operational_status"] == "Peak Strain (>=85%)", "month_label"].tolist()

        category_summaries[accom] = {
            "nace_code": nace_code,
            "nace_name": nace_name,
            "capacity_units": cap_units,
            "capacity_calibrated": capacity_calibrated,
            "peak_month": str(cat_df.iloc[peak_idx]["month_label"]),
            "peak_fte": float(cat_df.iloc[peak_idx]["required_fte"]),
            "trough_month": str(cat_df.iloc[trough_idx]["month_label"]),
            "trough_fte": float(cat_df.iloc[trough_idx]["required_fte"]),
            "seasonal_fte_swing": float(cat_df.iloc[peak_idx]["required_fte"] - cat_df.iloc[trough_idx]["required_fte"]),
            "peak_strain_months": peak_months,
            "maintenance_window_months": maint_months,
        }

    if not frames:
        return pd.DataFrame(), {"categories": {}, "total_peak_fte": 0.0, "total_baseline_fte": 0.0}

    combined = pd.concat(frames, ignore_index=True)
    monthly_totals = combined.groupby("month_label")["required_fte"].sum()
    return combined, {
        "categories": category_summaries,
        "total_peak_fte": float(monthly_totals.max()),
        "total_baseline_fte": float(monthly_totals.min()),
        "total_seasonal_hiring_need": float(monthly_totals.max() - monthly_totals.min()),
    }


def compute_hmm_regime_timeline(history_df: pd.DataFrame) -> pd.DataFrame:
    """Fits RegimeHMMNavigator on historical Eurostat aggregate macro signals and returns regime probabilities."""
    if history_df is None or history_df.empty:
        return pd.DataFrame()

    total_df = history_df[history_df["accommodation_type"] == "total"].copy()
    if total_df.empty:
        total_df = history_df.copy()

    nav = RegimeHMMNavigator(n_regimes=3)
    nav.fit(total_df)
    _, _, regime_df = nav.predict_regimes(total_df)
    regime_df["date"] = pd.to_datetime(regime_df["date"])
    regime_df["regime_name"] = regime_df["regime_state"].map(RegimeHMMNavigator.REGIME_NAMES)
    return regime_df.sort_values("date").reset_index(drop=True)


def compute_crisis_regime_diagnostics(bundle: DSSDataBundle, geo: str) -> Dict[str, Any]:
    """
    Layer 3: Crisis Regime Detection & Stress Testing diagnostics from empirical folds and LOS stability.
    """
    fold_rows: List[Dict[str, Any]] = []
    fold_labels = {
        "tranquil": "Tranquil Pre-Shock (2018–2019)",
        "shock": "Pandemic Crisis Shock (2020–2021)",
        "recovery": "Post-Crisis Rebound (2022–2023)",
        "recent": "Recent Cycle (2024–2026)",
    }

    for fold_key, label in fold_labels.items():
        models = bundle.fold_summaries.get(fold_key, {})
        dl = models.get("dlinear", {}).get("test_metrics", {})
        full = models.get("full", {}).get("test_metrics", {})
        if dl and full:
            dl_arr = float(dl["arrivals_total"]["wape"])
            full_arr = float(full["arrivals_total"]["wape"])
            dl_nig = float(dl["nights_total"]["wape"])
            full_nig = float(full["nights_total"]["wape"])
            fold_rows.append(
                {
                    "fold": fold_key,
                    "regime_phase": label,
                    "dlinear_arr_wape": dl_arr,
                    "ra_he_arr_wape": full_arr,
                    "arr_gain_pp": dl_arr - full_arr,
                    "dlinear_nig_wape": dl_nig,
                    "ra_he_nig_wape": full_nig,
                    "nig_gain_pp": dl_nig - full_nig,
                }
            )

    geo_los_rows = pd.DataFrame()
    if bundle.country_los_df is not None and not bundle.country_los_df.empty:
        geo_los_rows = bundle.country_los_df[bundle.country_los_df["geo"] == geo].copy()

    elasticity = -0.52
    if bundle.elasticity_df is not None and not bundle.elasticity_df.empty:
        match = bundle.elasticity_df[bundle.elasticity_df["geo"] == geo]
        if not match.empty:
            raw_e = float(match.iloc[0]["price_elasticity"])
            # Bound extreme small-sample elasticities to plausible economic stress range [-2.5, -0.05]
            elasticity = float(np.clip(raw_e, -2.5, -0.05))
        else:
            elasticity = float(np.clip(bundle.elasticity_df["price_elasticity"].median(), -2.5, -0.05))

    return {
        "fold_comparison_df": pd.DataFrame(fold_rows),
        "geo_los_df": geo_los_rows,
        "country_los_df": bundle.country_los_df,
        "price_elasticity": elasticity,
    }


def simulate_macro_stress(
    plan_df: pd.DataFrame,
    price_elasticity: float,
    hicp_shock_pct: float = 0.0,
    stringency_delta: float = 0.0,
) -> pd.DataFrame:
    """
    Simulates macroeconomic HICP price inflation and policy stringency shocks on forecast demand,
    enforcing Pi_C physical consistency on stressed outputs.
    """
    if plan_df.empty:
        return plan_df.copy()

    out = plan_df.copy()
    arr_base = out["pred_arrivals"].to_numpy(dtype=np.float64)
    nig_base = out["pred_nights"].to_numpy(dtype=np.float64)

    # Price shock impact (nights typically react slightly stronger via shorter LOS)
    arr_price_mult = max(0.05, 1.0 + (hicp_shock_pct / 100.0) * price_elasticity)
    nig_price_mult = max(0.05, 1.0 + (hicp_shock_pct / 100.0) * (price_elasticity * 1.12))

    # Stringency index shock (0-100 scale): empirical collapse sensitivity ~0.85% per stringency point
    str_mult = max(0.05, 1.0 - (stringency_delta / 100.0) * 0.85)

    arr_stressed_raw = arr_base * arr_price_mult * str_mult
    nig_stressed_raw = nig_base * nig_price_mult * str_mult

    arr_safe, nig_safe, corrected = CoupledConsistencyOptimizer.project_physical_pair(
        arr_stressed_raw, nig_stressed_raw
    )

    out["stressed_arrivals"] = arr_safe
    out["stressed_nights"] = nig_safe
    out["stressed_los"] = np.where(arr_safe > 0, nig_safe / np.maximum(arr_safe, 1.0), 0.0)
    out["stress_corrected_pi_c"] = corrected
    out["arrivals_delta_pct"] = np.where(arr_base > 0, (arr_safe - arr_base) / arr_base * 100.0, 0.0)
    out["nights_delta_pct"] = np.where(nig_base > 0, (nig_safe - nig_base) / nig_base * 100.0, 0.0)
    return out


def compute_recovery_benchmarking(
    forecasts_df: pd.DataFrame,
    history_df: Optional[pd.DataFrame],
    geo: str,
    accom_type: str = "total",
    horizon_months: int = 12,
    origin_label: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Layer 4: Benchmarking Post-Shock Recovery Trajectories & Conformal Risk Bands (I_0.90).

    Returns:
      - traj_df: Monthly recovery index against 2019 same-month baseline alongside
                 Upper Capacity-Strain and Lower Revenue-at-Risk conformal bounds.
      - cross_country_df: 38-country recovery & risk-spread benchmarking table.
    """
    sub = _filter_forecast_window(forecasts_df, geo, accom_type, horizon_months, origin_label)
    if sub.empty:
        return pd.DataFrame(), pd.DataFrame()

    sub["month"] = sub["date"].dt.month

    if history_df is not None and not history_df.empty:
        hist_2019 = history_df[
            (history_df["year"] == 2019) & (history_df["accommodation_type"] == accom_type)
        ].copy()
        monthly_2019 = (
            hist_2019.groupby(["geo", "month"])[["arrivals_total", "nights_total"]]
            .mean()
            .reset_index()
            .rename(columns={"arrivals_total": "base_2019_arrivals", "nights_total": "base_2019_nights"})
        )
        sub = sub.merge(monthly_2019[monthly_2019["geo"] == geo], on=["geo", "month"], how="left")
    else:
        sub["base_2019_arrivals"] = np.nan
        sub["base_2019_nights"] = np.nan

    sub["recovery_index_arr_pct"] = np.where(
        sub["base_2019_arrivals"] > 0,
        sub["pred_arrivals"] / sub["base_2019_arrivals"] * 100.0,
        np.nan,
    )
    sub["capacity_strain_index_pct"] = np.where(
        sub["base_2019_arrivals"] > 0,
        sub["upper_arrivals"] / sub["base_2019_arrivals"] * 100.0,
        np.nan,
    )
    sub["revenue_risk_index_pct"] = np.where(
        sub["base_2019_arrivals"] > 0,
        sub["lower_arrivals"] / sub["base_2019_arrivals"] * 100.0,
        np.nan,
    )
    sub["conformal_uncertainty_width_arr"] = sub["upper_arrivals"] - sub["lower_arrivals"]
    sub["conformal_uncertainty_width_nig"] = sub["upper_nights"] - sub["lower_nights"]

    # Cross-country 38-destination recovery benchmark
    fc_total = forecasts_df[forecasts_df["accommodation_type"] == accom_type].copy()
    if origin_label is not None and "origin_label" in fc_total.columns:
        fc_total = fc_total[fc_total["origin_label"] == origin_label]
    elif "origin_label" in fc_total.columns:
        first_origin = sorted(fc_total["origin_label"].unique())[0]
        fc_total = fc_total[fc_total["origin_label"] == first_origin]

    fc_horizon = (
        fc_total.sort_values("date")
        .assign(month=lambda d: d["date"].dt.month)
        .groupby(["geo", "country"], as_index=False, group_keys=False)
        .head(horizon_months)
    )
    target_months = sorted(fc_horizon["month"].unique().tolist())

    fc_agg = (
        fc_horizon.groupby(["geo", "country"], as_index=False)[
            ["pred_arrivals", "lower_arrivals", "upper_arrivals", "pred_nights", "lower_nights", "upper_nights"]
        ]
        .sum()
    )

    if history_df is not None and not history_df.empty and target_months:
        hist19_geo = (
            history_df[
                (history_df["year"] == 2019)
                & (history_df["accommodation_type"] == accom_type)
                & (history_df["month"].isin(target_months))
            ]
            .groupby("geo", as_index=False)[["arrivals_total", "nights_total"]]
            .sum()
            .rename(columns={"arrivals_total": "scaled_2019_arrivals", "nights_total": "scaled_2019_nights"})
        )
        fc_agg = fc_agg.merge(hist19_geo, on="geo", how="left")
    else:
        fc_agg["scaled_2019_arrivals"] = np.nan
        fc_agg["scaled_2019_nights"] = np.nan

    fc_agg["recovery_pct"] = np.where(
        fc_agg["scaled_2019_arrivals"] > 0,
        fc_agg["pred_arrivals"] / fc_agg["scaled_2019_arrivals"] * 100.0,
        np.nan,
    )
    fc_agg["pred_los"] = np.where(
        fc_agg["pred_arrivals"] > 0,
        fc_agg["pred_nights"] / np.maximum(fc_agg["pred_arrivals"], 1.0),
        0.0,
    )
    fc_agg["recovery_tier"] = np.where(
        fc_agg["recovery_pct"].isna(),
        "No 2019 Baseline",
        np.where(
            fc_agg["recovery_pct"] < 100.0,
            "Incomplete Recovery (<100%)",
            np.where(
                fc_agg["recovery_pct"] <= 110.0,
                "Balanced Recovery (100–110%)",
                "Accelerated Rebound (>110%)",
            ),
        ),
    )
    fc_agg["paper_case_study"] = fc_agg["geo"].map(lambda g: PAPER_CASE_STUDIES.get(g, ""))

    return sub, fc_agg.sort_values("recovery_pct", ascending=False).reset_index(drop=True)
