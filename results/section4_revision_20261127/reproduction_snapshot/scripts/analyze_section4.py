"""Verify keyed Section 4 experiments and generate evidence for all six sections."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from catboost import CatBoostRegressor, Pool
from sklearn.metrics import r2_score

from scripts.section4_experiments import (OUTPUT, TARGETS, KEYS, CALENDAR, MACRO, CONFIGS,
    get_scores, project, predict_core, build_core_inputs, write_json, file_hash, save_predictions)

TABLES = ROOT / "paper/tables"
FIGURES = ROOT / "paper/figures"
NAMES = {"snaive": "Seasonal Naive", "ets": "ETS", "sarima": "SARIMA", "sarimax": "SARIMAX",
         "lightgbm": "LightGBM", "dlinear": "Frozen DLinear", "learned_dlinear": "Learned DLinear", "exo": "DLinear + Exogenous",
         "hmm": "DLinear + HMM bias", "film_static": "DLinear + static FiLM",
         "hmm_film": "DLinear + HMM + ungated FiLM", "exo_hmm_film": "Exogenous + HMM + ungated FiLM",
         "soft_no_consistency": "Soft FiLM, no coupled loss", "full_core": "Full temporal core",
         "static": "Static blend (0.7), raw", "static_projected": "Static 0.7 + projection", "adaptive_raw": "Adaptive before projection",
         "adaptive": "RA-HE-DLinear", "trmf": "TRMF-only", "patchtst": "PatchTST",
         "itransformer": "iTransformer", "timesnet": "TimesNet", "persistence": "Persistence"}


def read_predictions(directory: Path, name: str) -> pd.DataFrame:
    """Reject duplicate keys, corrupt predictions, and mismatched saved metrics."""
    path = directory / f"{name}.csv"
    frame = pd.read_csv(path, parse_dates=["date", "origin"])
    if frame.duplicated(KEYS).any() or not (frame.origin < frame.date).all():
        raise ValueError(f"Invalid keys/origins: {path}")
    prediction = frame[["pred_arrivals", "pred_nights"]].to_numpy()
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite forecasts: {path}")
    details = json.loads((directory / f"{name}.json").read_text(encoding="utf-8"))
    if details.get("forecast_sha256") != file_hash(path):
        raise ValueError(f"Forecast checksum differs: {path}")
    for target, actual in get_scores(frame, prediction).items():
        expected = details["test_metrics"][target]
        if actual["n"] != expected["n"] or not all(np.isclose(actual[k], expected[k], atol=1e-6) for k in ["mae", "wape"]):
            raise ValueError(f"Metric discrepancy: {path}/{target}")
    return frame


def verify_alignment(frames: dict[str, pd.DataFrame]) -> None:
    """Require identical outcomes, origins, horizons, and original masks for comparisons."""
    columns = KEYS + ["origin", "horizon", "observed_pair", "observed_arrivals_total", "observed_nights_total"]
    reference = next(iter(frames.values()))[columns]
    for name, frame in frames.items():
        pd.testing.assert_frame_equal(reference, frame[columns], obj=name)


def verify_source_outcomes(frame: pd.DataFrame, source: Path) -> None:
    """Independently join original outcomes rather than trusting agreement between models."""
    raw = pd.read_csv(source, parse_dates=["date"])
    paired = frame[KEYS].merge(raw[KEYS + TARGETS], on=KEYS, how="left", validate="one_to_one")
    eligible = paired[TARGETS].notna().all(axis=1) & paired[TARGETS].ge(0).all(axis=1)
    if not np.array_equal(eligible, frame.observed_pair):
        raise ValueError(f"Original observation mask mismatch: {source}")
    observed = frame[["observed_" + target for target in TARGETS]].to_numpy()
    if not np.allclose(observed, paired[TARGETS].to_numpy(), equal_nan=True, rtol=1e-12, atol=1e-5):
        raise ValueError(f"Original target values mismatch: {source}")


def describe_scores(frame: pd.DataFrame) -> dict[str, Any]:
    """Compute accuracy plus physical constraint violations on all forecast rows."""
    result = {"grid_rows": len(frame), "scored_pairs": int(frame.observed_pair.sum())}
    for target, values in get_scores(frame, frame[["pred_arrivals", "pred_nights"]].to_numpy()).items():
        result[target] = values
    result["physical_violation_pct"] = float(100 * frame.pred_nights.lt(frame.pred_arrivals).mean())
    return result


def describe_los(frame: pd.DataFrame) -> dict[str, Any]:
    """Score LOS only where observed and forecast arrivals denominators are positive."""
    eligible = frame.observed_pair & frame.observed_arrivals_total.gt(0)
    valid = eligible & frame.pred_arrivals.gt(0)
    actual = frame.loc[valid, "observed_nights_total"] / frame.loc[valid, "observed_arrivals_total"]
    forecast = frame.loc[valid, "pred_nights"] / frame.loc[valid, "pred_arrivals"]
    error = forecast - actual
    return {"n": int(valid.sum()), "undefined_predictions": int((eligible & ~valid).sum()),
            "mae": float(error.abs().mean()), "rmse": float(np.sqrt(np.square(error).mean()))}


def get_active_parameters(model: Any, inputs: list[torch.Tensor] | torch.Tensor) -> int:
    """Count registered scalar parameters that participate in the forecast graph."""
    parameters = list(model.net.parameters())
    flags = [p.requires_grad for p in parameters]
    for parameter in parameters:
        parameter.requires_grad_(True)
    model.net.eval()
    if isinstance(inputs, list):
        output = model.net(*[tensor[:1] for tensor in inputs])
        objective = sum(tensor.sum() for tensor in output)
    else:
        objective = model.net(inputs[:1]).sum()
    gradients = torch.autograd.grad(objective, parameters, allow_unused=True)
    count = sum(p.numel() for p, gradient in zip(parameters, gradients) if gradient is not None)
    for parameter, flag in zip(parameters, flags):
        parameter.requires_grad_(flag)
    return count


def build_surrogate_features(frame: pd.DataFrame, rows: pd.DataFrame) -> pd.DataFrame:
    """Summarize the same pre-origin histories and future calendar inputs used by the core."""
    result = pd.DataFrame(index=rows.index)
    groups = {(g, a): group for (g, a), group in frame.groupby(["geo", "accommodation_type"])}
    for (geo, accom, origin), block in rows.groupby(["geo", "accommodation_type", "origin"], sort=False):
        history = groups[(geo, accom)].loc[lambda f: f.date.le(origin)].tail(24)
        for index, row in block.iterrows():
            features = {"geo": str(geo), "category": str(accom), "horizon": row.horizon,
                        "last_arrivals": history.arrivals_total.iloc[-1], "last_nights": history.nights_total.iloc[-1],
                        "seasonal_arrivals": history.arrivals_total.iloc[-12 + int(row.horizon) - 1],
                        "seasonal_nights": history.nights_total.iloc[-12 + int(row.horizon) - 1]}
            features.update({column: row[column] for column in CALENDAR})
            features.update({"past_" + column: history[column].mean() for column in MACRO})
            for key, value in features.items():
                result.loc[index, key] = value
    return result


def run_diagnostics() -> None:
    """Evaluate direct sensitivities and held-out TreeSHAP surrogate fidelity."""
    directory = OUTPUT / "monthly/recent"
    frame = pd.read_pickle(directory / "completed_panel.pkl")
    rows = pd.read_pickle(directory / "forecast_rows.pkl")
    model = torch.load(directory / "full_core.pt", weights_only=False, map_location="cpu")
    weight = json.loads((directory / "adaptive.json").read_text())["w_temporal"]
    temporal = pd.read_csv(directory / "full_core_all.csv")[["pred_arrivals", "pred_nights"]].to_numpy()
    spatial = pd.read_csv(directory / "trmf_all.csv")[["pred_arrivals", "pred_nights"]].to_numpy()
    reference = project(weight * temporal + (1 - weight) * spatial)
    selected = rows.year.ge(2024) & rows.geo.isin(["FR", "ES", "IT", "DE"]) & rows.accommodation_type.eq("hotel") & rows.observed_pair
    sensitivities = []
    for change in [-0.10, -0.05, 0.05, 0.10]:
        perturbed = frame.copy()
        mask = perturbed.geo.isin(["FR", "ES", "IT", "DE"]) & perturbed.accommodation_type.eq("hotel")
        perturbed.loc[mask, "hicp_accommodation"] *= 1 + change
        changed = project(weight * predict_core(model, perturbed, rows) + (1 - weight) * spatial)
        for geo in ["FR", "ES", "IT", "DE", "pooled"]:
            eligible = selected & (rows.geo.eq(geo) if geo != "pooled" else True)
            sensitivities.append({"geo": geo, "hicp_change_pct": change * 100,
                "arrivals_change_pct": float(100 * (changed[eligible, 0].sum() / reference[eligible, 0].sum() - 1)),
                "nights_change_pct": float(100 * (changed[eligible, 1].sum() / reference[eligible, 1].sum() - 1)),
                "n": int(eligible.sum())})
    pd.DataFrame(sensitivities).to_csv(OUTPUT / "input_sensitivity.csv", index=False)
    features = build_surrogate_features(frame, rows)
    historical = rows.year.le(2023).to_numpy()
    testing = rows.year.ge(2024).to_numpy() & rows.observed_pair.to_numpy()
    numeric = [column for column in features if column not in ["geo", "category"]]
    features[numeric] = features[numeric].astype(float)
    shap_rows, fidelity = [], []
    for channel, target in enumerate(TARGETS):
        surrogate = CatBoostRegressor(iterations=300, depth=6, learning_rate=0.05,
                                      random_seed=42, thread_count=1, verbose=False, allow_writing_files=False)
        surrogate.fit(features.loc[historical], np.log1p(reference[historical, channel]), cat_features=["geo", "category"])
        predicted_log = surrogate.predict(features.loc[testing])
        predicted = np.maximum(0, np.expm1(predicted_log))
        truth = reference[testing, channel]
        fidelity.append({"target": target, "r2_counts": float(r2_score(truth, predicted)),
                         "r2_log": float(r2_score(np.log1p(truth), predicted_log)),
                         "mae_counts": float(np.abs(truth - predicted).mean()),
                         "training_prediction_years": [2022, 2023], "test_prediction_years": [2024, 2025, 2026],
                         "target": target})
        values = surrogate.get_feature_importance(Pool(features.loc[testing], cat_features=["geo", "category"]), type="ShapValues")[:, :-1]
        mean = np.abs(values).mean(axis=0)
        shap_rows.extend({"target": target, "feature": column, "mean_abs_shap_log1p": float(value),
                          "share_pct": float(100 * value / mean.sum())} for column, value in zip(features.columns, mean))
        surrogate.save_model(str(OUTPUT / f"surrogate_{target}.cbm"))
    pd.DataFrame(shap_rows).to_csv(OUTPUT / "tree_shap.csv", index=False)
    write_json(fidelity, OUTPUT / "surrogate_fidelity.json")
    write_json({"script_sha256": file_hash(Path(__file__)), "core_checkpoint_sha256": file_hash(directory / "full_core.pt"),
                "fused_forecast_sha256": file_hash(directory / "adaptive.csv"),
                "surrogate_features": list(features.columns), "surrogate_response": "log1p of actual fused projected forecasts",
                "direct_sensitivity": "Uniform HICP accommodation-level perturbation of historical inputs; percentage changes, calendar, regimes and gate held fixed"}, OUTPUT / "diagnostics_manifest.json")
    print("DIAGNOSTICS", json.dumps(fidelity), flush=True)


def make_table(path: Path, caption: str, label: str, headers: list[str], rows: list[list[str]],
               columns: str, note: str = "", wide: bool = False) -> None:
    """Write a compact LaTeX table from computed, traceable cells."""
    env = "table*" if wide else "table"
    lines = [rf"\begin{{{env}}}[!t]", r"\centering", rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.08}",
             rf"\begin{{tabular}}{{{columns}}}", r"\toprule", " & ".join(r"\textbf{" + value + "}" for value in headers) + " " + chr(92) * 2, r"\midrule"]
    lines += [" & ".join(row) + r" \\" for row in rows]
    lines += [r"\bottomrule", r"\end{tabular}"]
    if note:
        lines += [r"\par\vspace{3pt}", r"\begin{minipage}{" + (r"\textwidth" if wide else r"\columnwidth") + "}",
                  r"\scriptsize\raggedright " + note, r"\end{minipage}"]
    lines += [rf"\end{{{env}}}"]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_tables() -> None:
    """Generate accuracy, regime, ablation, global, consistency, and uncertainty evidence."""
    summary: dict[str, Any] = {"audit_date": "2026-11-27", "monthly": {}, "annual": {}}
    models = ["snaive", "ets", "sarima", "sarimax", "lightgbm", "timesnet", "patchtst", "itransformer", "dlinear", "learned_dlinear", "exo", "adaptive"]
    recent = OUTPUT / "monthly/recent"
    frames = {name: read_predictions(recent, name) for name in models}
    verify_alignment(frames)
    verify_source_outcomes(frames["adaptive"], ROOT / "src/data/tourism_europe_enriched.csv")
    panel = pd.read_pickle(recent / "completed_panel.pkl")
    benchmark, counts = [], []
    for name in models:
        scores = describe_scores(frames[name])
        metadata_name = "full_core" if name == "adaptive" else name
        details = json.loads((recent / f"{metadata_name}.json").read_text())
        params = None
        if (recent / f"{metadata_name}.pt").exists():
            model = torch.load(recent / f"{metadata_name}.pt", weights_only=False, map_location="cpu")
            if metadata_name in CONFIGS:
                work = panel.copy()
                if not CONFIGS[metadata_name]["hmm"]:
                    work["regime_state"] = 0
                rows = pd.read_pickle(recent / "forecast_rows.pkl")
                inputs, _ = build_core_inputs(model, work, rows)
                params = get_active_parameters(model, inputs)
            else:
                params = get_active_parameters(model, torch.ones((1, 24, 2)))
            counts.append({"model": name, "active_parameters": params, "registered_parameters": details["registered_parameters"]})
        elapsed = details.get("fit_and_forecast_seconds", details.get("runtime_seconds", details.get("fit_seconds", 0) + details.get("forecast_seconds_median3", 0)))
        if name == "adaptive":
            elapsed += json.loads((recent / "trmf.json").read_text())["fit_and_forecast_seconds"]
        scores.update(cpu_seconds=elapsed, active_parameters=params)
        summary["monthly"][name] = scores
        benchmark.append([NAMES[name], f"{scores[TARGETS[0]]['wape']:.2f}", f"{scores[TARGETS[0]]['mae']:,.0f}",
                          f"{scores[TARGETS[1]]['wape']:.2f}", f"{scores[TARGETS[1]]['mae']:,.0f}",
                          f"{params:,}" if params is not None else "--", f"{elapsed:.2f}"])
    pd.DataFrame(counts).to_csv(OUTPUT / "parameter_counts.csv", index=False)
    make_table(TABLES / "section4_benchmark.tex", "Matched Eurostat accuracy and CPU cost, January 2024--July 2026 (4,522 observed pairs).", "tab:section4_benchmark",
        ["Model", r"Arr. WAPE (\%)", "Arr. MAE", r"Nig. WAPE (\%)", "Nig. MAE", "Active params", "CPU (s)"], benchmark, "@{}lrrrrrr@{}",
        r"MAE is in arrivals or nights. CPU cost covers instrumented fitting and one test forecast including feature construction, excluding disk I/O, panel completion and shared HMM state preparation; neural forecast time is the median of three runs. RA-HE also includes spatial preparation. One PyTorch CPU thread is used. Active parameters participate in the forecast graph; disabled branches in the shared wrapper are excluded. Learned DLinear and Transformers use up to 30 validation-selected epochs; linear correction cores use five.", True)
    regime_rows, legacy, consistency, coverage, los, los_paired = [], {}, [], [], [], []
    for fold, label in [("tranquil", "Pre-pandemic 2018--2019"), ("shock", "Pandemic 2020--2021"),
                        ("recovery", "Recovery 2022--2023"), ("recent", "Recent 2024--July 2026")]:
        directory = OUTPUT / "monthly" / fold
        static = read_predictions(directory, "static")
        save_predictions(static, project(static[["pred_arrivals", "pred_nights"]].to_numpy()), "static_projected", directory,
                         {"w_temporal": 0.7, "parent_forecast_sha256": file_hash(directory / "static.csv"),
                          "analysis_script_sha256": file_hash(Path(__file__))})
        names = ["snaive", "dlinear", "learned_dlinear", "exo", "trmf", "static_projected", "adaptive"]
        fold_frames = {name: read_predictions(directory, name) for name in names}
        verify_alignment(fold_frames)
        verify_source_outcomes(fold_frames["adaptive"], ROOT / "src/data/tourism_europe_enriched.csv")
        regime_rows.append([rf"\multicolumn{{5}}{{@{{}}l}}{{\textbf{{{label}}} ($n={int(fold_frames['adaptive'].observed_pair.sum()):,}$)}}", "", "", "", ""])
        for name in names:
            scores = describe_scores(fold_frames[name])
            w = json.loads((directory / f"{name}.json").read_text()).get("w_temporal")
            if name in ["dlinear", "learned_dlinear", "exo"]:
                w = 1.0
            elif name == "trmf":
                w = 0.0
            regime_rows.append([NAMES[name], f"{scores[TARGETS[0]]['wape']:.2f}", f"{scores[TARGETS[1]]['wape']:.2f}", f"{w:.3f}" if w is not None else "--", "current"])
        old_dir = ROOT / "results/reviewer_experiments" / fold
        old = pd.read_csv(old_dir / "full.csv", parse_dates=["date", "origin"])
        old_record = json.loads((old_dir / "full.json").read_text())
        old_scores = describe_scores(old)
        for target in TARGETS:
            for statistic in ["mae", "wape"]:
                if not np.isclose(old_scores[target][statistic], old_record["test_metrics"][target][statistic]):
                    raise ValueError(f"Archived legacy score mismatch: {fold}/{target}")
        legacy[fold] = {"metrics": old_scores, "w_temporal": old_record["w_temporal"],
                        "fusion": "validation-selected regularized sigmoid (lambda=0.005), ungated FiLM",
                        "source_hashes": old_record["source_hashes"]}
        if fold == "shock":
            regime_rows.append(["Legacy sigmoid fusion", f"{old_scores[TARGETS[0]]['wape']:.2f}", f"{old_scores[TARGETS[1]]['wape']:.2f}", f"{old_record['w_temporal']:.3f}", "archived"])
        for name in ["dlinear", "adaptive_raw", "adaptive"]:
            frame = read_predictions(directory, name)
            total = frame.loc[frame.accommodation_type.eq("total")]
            los.append({"fold": fold, "model": name, **describe_los(total)})
            consistency.append({"fold": fold, "model": name, **describe_scores(frame)})
        pair_frames = {name: read_predictions(directory, name) for name in ["dlinear", "adaptive_raw", "adaptive"]}
        verify_alignment(pair_frames)
        shared = pair_frames["adaptive"].accommodation_type.eq("total") & pair_frames["adaptive"].observed_pair & pair_frames["adaptive"].observed_arrivals_total.gt(0)
        for frame in pair_frames.values():
            shared &= frame.pred_arrivals.gt(0)
        for name, frame in pair_frames.items():
            los_paired.append({"fold": fold, "model": name, **describe_los(frame.loc[shared])})
        interval = pd.read_csv(directory / "intervals.csv")
        eligible = interval.observed_pair
        for target, short in zip(TARGETS, ["arrivals", "nights"]):
            truth = interval.loc[eligible, "observed_" + target]
            lower, upper = interval.loc[eligible, f"lower_{short}"], interval.loc[eligible, f"upper_{short}"]
            coverage.append({"fold": fold, "target": target, "n": len(truth),
                             "coverage_pct": float(100 * truth.between(lower, upper).mean()),
                             "npiw": float((upper - lower).mean() / truth.mean())})
    # Group rows are already multicolumn and therefore have no additional cells.
    regime_text = []
    for row in regime_rows:
        regime_text.append(row[:1] if row[0].startswith(r"\multicolumn") else row)
    make_table(TABLES / "section4_regimes.tex", "Regime-dependent accuracy with common keys and forecast origins within each fold.", "tab:section4_regimes",
        ["Configuration", r"Arr. WAPE (\%)", r"Nig. WAPE (\%)", r"$w_{\rm time}$", "Source"], regime_text, "@{}lrrrr@{}",
        r"Static fusion fixes $w_{\rm time}=0.7$ before validation. Both current fusion rows apply the same $\Pi_{\mathcal C}$ after blending. Adaptive fusion selects one weight per fold on validation. The archived 66.69\% row uses the earlier regularized sigmoid gate with ungated FiLM, not the fixed static blend or the current adaptive model.")
    summary["legacy"] = legacy
    pd.DataFrame(los).to_csv(OUTPUT / "los_metrics.csv", index=False)
    pd.DataFrame(los_paired).to_csv(OUTPUT / "los_paired.csv", index=False)
    pd.DataFrame(coverage).to_csv(OUTPUT / "coverage.csv", index=False)
    write_json(consistency, OUTPUT / "consistency_metrics.json")
    component_names = ["dlinear", "exo", "hmm", "film_static", "hmm_film", "exo_hmm_film", "soft_no_consistency", "full_core", "static", "adaptive_raw", "adaptive"]
    ablation = {}
    for name in component_names:
        ablation[name] = describe_scores(read_predictions(recent, name))
    summary["ablation"] = ablation
    annual_dir = OUTPUT / "annual"
    annual_frames = {name: read_predictions(annual_dir, name) for name in ["persistence", "dlinear", "exo", "trmf", "adaptive"]}
    verify_alignment(annual_frames)
    verify_source_outcomes(annual_frames["adaptive"], ROOT / "data/processed/tourism_un_tourism_enriched.csv")
    for name, frame in annual_frames.items():
        summary["annual"][name] = {"all": describe_scores(frame)}
        for label, years in [("pre", [2019]), ("shock", [2020, 2021]), ("recovery", [2022, 2023, 2024])]:
            summary["annual"][name][label] = describe_scores(frame.loc[frame.date.dt.year.isin(years)])
        summary["annual"][name]["oecd_nights"] = describe_scores(frame.loc[frame.nights_source.eq("oecd_observed")])
        summary["annual"][name]["constructed_nights"] = describe_scores(frame.loc[frame.is_synthetic_nights.eq(1)])
    summary["coverage"] = coverage
    summary["los"] = los
    summary["los_paired"] = los_paired
    # Paired bootstrap resamples entire destinations, retaining category/month dependence.
    full, baseline = frames["adaptive"], frames["dlinear"]
    differences = []
    for target, prediction in zip(TARGETS, ["pred_arrivals", "pred_nights"]):
        eligible = full.observed_pair
        blocks = pd.DataFrame({"geo": full.loc[eligible, "geo"], "truth": full.loc[eligible, "observed_" + target],
            "full_error": (full.loc[eligible, "observed_" + target] - full.loc[eligible, prediction]).abs(),
            "baseline_error": (baseline.loc[eligible, "observed_" + target] - baseline.loc[eligible, prediction]).abs()}).groupby("geo").sum()
        rng = np.random.default_rng(42)
        selections = rng.integers(0, len(blocks), size=(1000, len(blocks)))
        denominator = blocks.truth.to_numpy()[selections].sum(axis=1)
        delta = 100 * (blocks.full_error.to_numpy()[selections].sum(axis=1) - blocks.baseline_error.to_numpy()[selections].sum(axis=1)) / denominator
        differences.append({"target": target, "delta_wape_pp": summary["monthly"]["adaptive"][target]["wape"] - summary["monthly"]["dlinear"][target]["wape"],
                            "lower_95": float(np.quantile(delta, .025)), "upper_95": float(np.quantile(delta, .975)), "clusters": len(blocks), "replicates": 1000})
    summary["paired_country_bootstrap"] = differences
    pd.DataFrame(differences).to_csv(OUTPUT / "paired_country_bootstrap.csv", index=False)
    raw, projected = read_predictions(recent, "adaptive_raw"), frames["adaptive"]
    arrival_difference = float(np.max(np.abs(raw.pred_arrivals - projected.pred_arrivals)))
    summary["projection"] = {"arrivals_max_abs_change": arrival_difference,
        "raw_violations": int(raw.pred_nights.lt(raw.pred_arrivals).sum()),
        "projected_violations": int(projected.pred_nights.lt(projected.pred_arrivals).sum()),
        "intervention_pct": float(100 * raw.pred_nights.lt(raw.pred_arrivals).mean())}
    summary["cases"] = {geo: {"full": describe_scores(projected.loc[projected.geo.eq(geo) & projected.accommodation_type.eq("total")]),
                              "dlinear": describe_scores(frames["dlinear"].loc[frames["dlinear"].geo.eq(geo) & frames["dlinear"].accommodation_type.eq("total")])}
                        for geo in ["CY", "SE", "DK"]}
    exo, core = read_predictions(recent, "exo"), read_predictions(recent, "full_core")
    summary["exo_full_equality"] = {"max_abs_arrivals_difference": float(np.max(np.abs(exo.pred_arrivals - projected.pred_arrivals))),
        "mean_abs_arrivals_difference": float(np.mean(np.abs(exo.pred_arrivals - projected.pred_arrivals))),
        "mean_abs_core_vs_exo": float(np.mean(np.abs(exo.pred_arrivals - core.pred_arrivals)))}
    write_json(summary, OUTPUT / "summary.json")
    create_combined_table(summary)
    create_number_macros(summary)
    create_figure(frames["adaptive"], frames["dlinear"], pd.read_csv(recent / "intervals.csv", parse_dates=["date"]))
    create_architecture()
    print("VERIFIED", json.dumps({"monthly_full": summary["monthly"]["adaptive"], "projection": summary["projection"], "equality": summary["exo_full_equality"]}), flush=True)


def create_combined_table(summary: dict[str, Any]) -> None:
    """Keep distinct component and global panels within Table 4."""
    def cells(scores: dict[str, Any]) -> list[str]:
        return [rf"\shortstack{{{scores[target]['wape']:.2f}\%\\{scores[target]['mae']:,.0f}}}" for target in TARGETS]
    lines = [r"\begin{table*}[!t]", r"\centering", r"\caption{Component contributions on Eurostat (Panel A) and re-estimated annual global benchmark (Panel B). Each cell reports WAPE and MAE in original counts.}",
             r"\label{tab:section4_components_global}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}", r"\renewcommand{\arraystretch}{1.10}"]
    for panel in ["A", "B"]:
        lines += [r"\begin{minipage}[t]{0.49\textwidth}", r"\centering", r"\begin{tabular}[t]{@{}lrr@{}}", r"\toprule",
                  r"\textbf{Panel " + panel + (r": Eurostat components}" if panel == "A" else r": Annual global benchmark}") + r" & \textbf{Arrivals} & \textbf{Nights} " + chr(92) * 2, r"\midrule"]
        if panel == "A":
            for name, scores in summary["ablation"].items():
                lines.append(" & ".join([NAMES[name], *cells(scores)]) + r" \\")
        else:
            for name in ["persistence", "dlinear", "exo", "trmf", "adaptive"]:
                lines.append(" & ".join([NAMES[name] + " (all)", *cells(summary["annual"][name]["all"])]) + r" \\")
            lines.append(r"\midrule")
            for label, title in [("pre", "Full: pre-COVID 2019"), ("shock", "Full: shock 2020--2021"), ("recovery", "Full: recovery 2022--2024")]:
                lines.append(" & ".join([title, *cells(summary["annual"]["adaptive"][label])]) + r" \\")
            lines.append(" & ".join(["Persistence: recovery", *cells(summary["annual"]["persistence"]["recovery"])]) + r" \\")
        lines += [r"\bottomrule", r"\end{tabular}", r"\end{minipage}" + (r"\hfill" if panel == "A" else "")]
    lines += [r"\par\vspace{3pt}", r"\begin{minipage}{\textwidth}",
              r"\scriptsize\raggedright Panel A uses 4,522 observed pairs and a single seed. Ungated and soft FiLM are separate controls; static fusion uses $w_{\rm time}=0.7$. Panel B scores 786 reported arrivals paired with mixed-source nights, of which 678 nights targets are constructed and 108 are OECD observations. Annual models are retrained with $L=4$, $H=1$ on a calendar-year grid; this is cross-dataset re-estimation rather than zero-shot transfer.", r"\end{minipage}", r"\end{table*}"]
    (TABLES / "section4_components_global.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_number_macros(summary: dict[str, Any]) -> None:
    """Bind manuscript claims to the same numeric records that generate its tables."""
    values = {}
    for model, prefix in [("adaptive", "Full"), ("dlinear", "Frozen"), ("learned_dlinear", "Learned"),
                           ("exo", "Exo"), ("snaive", "Naive")]:
        for target, short in zip(TARGETS, ["Arr", "Nig"]):
            metric = summary["monthly"][model][target]
            values[prefix + short + "WAPE"] = f"{metric['wape']:.2f}"
            values[prefix + short + "MAE"] = f"{metric['mae']:,.0f}"
    for model, prefix in [("full_core", "Core"), ("static", "Static"), ("soft_no_consistency", "NoCons")]:
        for target, short in zip(TARGETS, ["Arr", "Nig"]):
            values[prefix + short + "WAPE"] = f"{summary['ablation'][model][target]['wape']:.2f}"
    for model, prefix in [("adaptive", "AnnualFull"), ("persistence", "AnnualNaive")]:
        for period, period_name in [("all", "All"), ("pre", "Pre"), ("shock", "Shock"), ("recovery", "Recovery")]:
            for target, short in zip(TARGETS, ["Arr", "Nig"]):
                values[prefix + period_name + short] = f"{summary['annual'][model][period][target]['wape']:.2f}"
    values["AnnualObservedNig"] = f"{summary['annual']['adaptive']['oecd_nights'][TARGETS[1]]['wape']:.2f}"
    values["AnnualConstructedNig"] = f"{summary['annual']['adaptive']['constructed_nights'][TARGETS[1]]['wape']:.2f}"
    coverage_folds = {"tranquil": "Tranquil", "shock": "Shock", "recovery": "Recovery", "recent": "Recent"}
    for row in summary["coverage"]:
        short = "Arr" if row["target"] == TARGETS[0] else "Nig"
        values["Coverage" + coverage_folds[row["fold"]] + short] = f"{row['coverage_pct']:.2f}"
        if row["fold"] == "recent":
            values["Coverage" + short] = f"{row['coverage_pct']:.2f}"
            values["Width" + short] = f"{row['npiw']:.3f}"
    los_prefixes = {"dlinear": "Frozen", "adaptive_raw": "Raw", "adaptive": "Full"}
    for row in summary["los_paired"]:
        if row["fold"] == "recent" and row["model"] in los_prefixes:
            prefix = los_prefixes[row["model"]]
            values[prefix + "LOSMAE"] = f"{row['mae']:.4f}"
            values[prefix + "LOSRMSE"] = f"{row['rmse']:.4f}"
            values["LOSSharedCount"] = str(row["n"])
    for row in summary["los"]:
        if row["fold"] == "recent" and row["model"] in ["dlinear", "adaptive"]:
            prefix = "Full" if row["model"] == "adaptive" else "Frozen"
            values[prefix + "LOSUndefined"] = str(row["undefined_predictions"])
    for row in summary["paired_country_bootstrap"]:
        short = "Arr" if row["target"] == TARGETS[0] else "Nig"
        values["Delta" + short] = f"{row['delta_wape_pp']:+.2f}"
        values["Lower" + short] = f"{row['lower_95']:+.2f}"
        values["Upper" + short] = f"{row['upper_95']:+.2f}"
    values["ProjectionCount"] = str(summary["projection"]["raw_violations"])
    values["ProjectionRate"] = f"{summary['projection']['intervention_pct']:.2f}"
    values["ProjectionRounding"] = f"{summary['projection']['arrivals_max_abs_change']:.2f}"
    values["FullParameters"] = f"{summary['monthly']['adaptive']['active_parameters']:,}"
    sensitivity = pd.read_csv(OUTPUT / "input_sensitivity.csv")
    scenario = sensitivity.loc[sensitivity.geo.eq("pooled") & sensitivity.hicp_change_pct.eq(10)].iloc[0]
    values["PriceArr"] = f"{scenario.arrivals_change_pct:+.3f}"
    values["PriceNig"] = f"{scenario.nights_change_pct:+.3f}"
    for row in json.loads((OUTPUT / "surrogate_fidelity.json").read_text()):
        short = "Arr" if row["target"] == TARGETS[0] else "Nig"
        values["FidelityCounts" + short] = f"{row['r2_counts']:.3f}"
        values["FidelityLog" + short] = f"{row['r2_log']:.3f}"
    for fold, title in [("tranquil", "Tranquil"), ("shock", "Shock"), ("recovery", "Recovery"), ("recent", "Recent")]:
        folder = OUTPUT / "monthly" / fold
        details = json.loads((folder / "adaptive.json").read_text())
        values["Weight" + title] = f"{details['w_temporal']:.3f}"
        for name, prefix in [("adaptive", "Adaptive"), ("static", "Static"), ("dlinear", "Frozen")]:
            record = json.loads((folder / f"{name}.json").read_text())
            values[prefix + title + "Arr"] = f"{record['test_metrics'][TARGETS[0]]['wape']:.2f}"
    values["AnnualHash"] = hashlib.sha256((ROOT / "data/processed/tourism_un_tourism_enriched.csv").read_bytes().replace(b"\r\n", b"\n")).hexdigest()[:12]
    lines = ["% Generated by scripts/analyze_section4.py from verified keyed forecasts."]
    lines += [r"\newcommand{\Result" + name + "}{" + value + "}" for name, value in values.items()]
    (TABLES / "section4_numbers.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def create_figure(full: pd.DataFrame, baseline: pd.DataFrame, intervals: pd.DataFrame) -> None:
    """Plot actual keyed forecasts and intervals for the preselected destination cases."""
    import matplotlib.dates as dates
    fig, axes = plt.subplots(2, 3, figsize=(9.4, 4.6), sharex="col")
    for column, (geo, name) in enumerate([("CY", "Cyprus"), ("SE", "Sweden"), ("DK", "Denmark")]):
        mask = full.geo.eq(geo) & full.accommodation_type.eq("total")
        f, b, ci = full.loc[mask].sort_values("date"), baseline.loc[mask].sort_values("date"), intervals.loc[mask].sort_values("date")
        for channel, (target, short) in enumerate(zip(TARGETS, ["arrivals", "nights"])):
            ax = axes[channel, column]
            observed = f.loc[f.observed_pair]
            ax.fill_between(ci.date, ci[f"lower_{short}"] / 1e6, ci[f"upper_{short}"] / 1e6, color="#156e90", alpha=.15, label="90% nominal interval")
            ax.plot(observed.date, observed["observed_" + target] / 1e6, color="#202b36", lw=1.2, label="Observed")
            ax.plot(b.date, b["pred_" + short] / 1e6, color="#c47724", lw=1, ls="--", label="Frozen DLinear")
            ax.plot(f.date, f["pred_" + short] / 1e6, color="#156e90", lw=1.1, label="Adaptive full")
            ax.set_title(name if channel == 0 else "", fontsize=11)
            ax.set_ylabel(("Arrivals" if channel == 0 else "Nights") + " (million)", fontsize=10)
            ax.tick_params(axis="both", labelsize=9)
            ax.grid(alpha=.18)
            if channel == 1:
                ax.xaxis.set_major_locator(dates.YearLocator())
                ax.xaxis.set_major_formatter(dates.DateFormatter("%Y"))
                ax.tick_params(axis="x", rotation=0)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=9, frameon=False)
    fig.tight_layout(rect=(0, .06, 1, 1))
    fig.savefig(FIGURES / "section4_country_cases.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def create_architecture() -> None:
    """Render a vector diagram matching the evaluated gate and 90% calibration."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    fig, ax = plt.subplots(figsize=(9.4, 3.2))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.axis("off")
    boxes = {
        "history": (.01, .63, .19, .30, "Pre-origin histories\nArrivals and nights\nHistorical macro inputs"),
        "calendar": (.01, .13, .19, .27, "Known future\ncalendar inputs"),
        "temporal": (.26, .52, .23, .39, "Temporal core\nFrozen decomposition maps\nLearned exogenous / FiLM terms\nForward-filtered HMM state"),
        "spatial": (.26, .08, .23, .27, "TRMF-inspired factors\nTraining-only fit\nAR extrapolation"),
        "fusion": (.55, .36, .17, .31, "Validation-selected\nfusion weight\nFixed within fold"),
        "projection": (.78, .61, .20, .29, "Point projection\nNights >= arrivals\nArrivals anchored"),
        "interval": (.78, .09, .20, .30, "Held-out residual\ncalibration\n90% nominal intervals"),
    }
    for x, y, width, height, label in boxes.values():
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.008", linewidth=.9,
                                   edgecolor="#243140", facecolor="white"))
        ax.text(x + width / 2, y + height / 2, label, ha="center", va="center", fontsize=8.8)
    for start, end in [((.20, .78), (.26, .75)), ((.20, .27), (.26, .60)),
                       ((.49, .72), (.55, .57)), ((.49, .22), (.55, .45)),
                       ((.72, .53), (.78, .75)), ((.88, .61), (.88, .39))]:
        ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=10,
                                     linewidth=.9, color="#243140"))
    ax.text(.62, .10, "Calibration follows\nvalidation chronologically", ha="center", va="center", fontsize=8)
    fig.tight_layout(pad=.1)
    fig.savefig(FIGURES / "section4_architecture.pdf", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    """Run diagnostics or regenerate verified manuscript tables."""
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["diagnostics", "tables", "all"], default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    OUTPUT = args.output
    torch.set_num_threads(1)
    if args.stage in ["diagnostics", "all"]:
        run_diagnostics()
    if args.stage in ["tables", "all"]:
        run_tables()


if __name__ == "__main__":
    main()
