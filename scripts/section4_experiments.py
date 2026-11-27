"""Run the matched monthly and annual experiments for manuscript Section 4.

Outputs are isolated from earlier runs. Forecast rows retain origins, original
observation masks, source labels, and model-specific provenance. Reference dates
are not publication dates; source vintages remain a documented limitation.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path
import platform
import shutil
import sys
import time
from typing import Any, Callable

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from hmmlearn.hmm import GaussianHMM

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

from scripts import reviewer_experiments as base
from src.models.ra_exo_dlinear import RAExoDLinearForecaster
from src.models.transformer_baselines import PatchTSTForecaster, iTransformerForecaster, TimesNetForecaster
from src.models.trmf_spatial import TRMFSpatialSpillover
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.dcqf_conformal import ResidualConformalForecaster

OUTPUT = ROOT / "results/section4_revision_20261127"
KEYS = base.KEYS
TARGETS = base.TARGETS
CALENDAR = base.CALENDAR
MACRO = base.MACRO
ANNUAL_FUTURE = ["annual_holidays_count", "is_mega_event"]
ANNUAL_PAST = ["gdp_lag1", "gdp_growth_lag1", "inflation_lag1", "unemployment_lag1",
               "exchange_rate_lag1", "covid_stringency_index"]
CONFIGS = {
    "dlinear": dict(use_exo=False, use_film=False, use_consistency=False, hmm=False),
    "exo": dict(use_exo=True, use_film=False, use_consistency=False, hmm=False),
    "hmm": dict(use_exo=False, use_film=False, use_regime_bias=True, use_consistency=False, hmm=True),
    "film_static": dict(use_exo=False, use_film=True, use_soft_gating=False, use_consistency=False, hmm=False),
    "hmm_film": dict(use_exo=False, use_film=True, use_soft_gating=False, use_consistency=False, hmm=True),
    "exo_hmm_film": dict(use_exo=True, use_film=True, use_soft_gating=False, use_consistency=False, hmm=True),
    "soft_no_consistency": dict(use_exo=True, use_film=True, use_soft_gating=True, use_consistency=False, hmm=True),
    "full_core": dict(use_exo=True, use_film=True, use_soft_gating=True, use_consistency=True, hmm=True),
}


def write_json(value: Any, path: Path) -> None:
    """Write readable provenance, converting NumPy scalar values."""
    path.write_text(json.dumps(value, indent=2, default=lambda v: v.item() if isinstance(v, np.generic)
                              else str(v), ensure_ascii=False), encoding="utf-8")


def file_hash(path: Path) -> str:
    """Hash exact file bytes."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def get_scores(rows: pd.DataFrame, prediction: np.ndarray) -> dict[str, Any]:
    """Score both targets on one originally observed, nonnegative pair mask."""
    valid = rows.observed_pair.to_numpy(dtype=bool)
    return {target: base.metric(rows.loc[valid, "observed_" + target], prediction[valid, channel])
            for channel, target in enumerate(TARGETS)}


def save_predictions(rows: pd.DataFrame, prediction: np.ndarray, name: str,
                     directory: Path, metadata: dict[str, Any]) -> None:
    """Save keyed forecasts and independently computable scores."""
    if prediction.shape != (len(rows), 2) or not np.isfinite(prediction).all():
        raise ValueError(f"Invalid prediction array: {name}")
    frame = base.prediction_frame(rows, prediction)
    for column in ["nights_source", "is_synthetic_nights"]:
        if column in rows:
            frame[column] = rows[column].to_numpy()
    frame["model"] = name
    frame.to_csv(directory / f"{name}.csv", index=False)
    details = {**metadata, "model": name, "test_metrics": get_scores(rows, prediction),
               "forecast_sha256": file_hash(directory / f"{name}.csv")}
    write_json(details, directory / f"{name}.json")
    print(name, json.dumps(details["test_metrics"]), flush=True)


def project(prediction: np.ndarray) -> np.ndarray:
    """Preserve nonnegative arrivals and lift nights to the stated lower bound."""
    arrivals, nights, _ = CoupledConsistencyOptimizer.project_physical_pair(prediction[:, 0], prediction[:, 1])
    return np.column_stack([arrivals, nights])


def time_call(function: Callable[[], Any], repeats: int = 3) -> tuple[Any, float]:
    """Return the last result and median wall seconds, including feature assembly."""
    timings = []
    result = None
    for _ in range(repeats):
        started = time.perf_counter()
        result = function()
        timings.append(time.perf_counter() - started)
    return result, float(np.median(timings))


def get_manifest(source: Path, **settings: Any) -> dict[str, Any]:
    """Capture the input, implementation revision, environment, and protocol."""
    import importlib.metadata as packages
    paths = [Path(__file__), ROOT / "scripts/reviewer_experiments.py",
             ROOT / "src/models/ra_exo_dlinear.py", ROOT / "src/models/trmf_spatial.py",
             ROOT / "src/models/transformer_baselines.py", ROOT / "src/models/dcqf_conformal.py",
             ROOT / "src/models/coupled_optimizer.py", ROOT / "src/data/dataset_loader.py"]
    return {"source": str(source.relative_to(ROOT)), "data_sha256": file_hash(source),
            "data_sha256_lf": hashlib.sha256(source.read_bytes().replace(b"\r\n", b"\n")).hexdigest(),
            "source_hashes": {str(p.relative_to(ROOT)): file_hash(p) for p in paths},
            "environment": {p: packages.version(p) for p in ["numpy", "pandas", "torch", "scipy", "hmmlearn", "statsmodels", "lightgbm", "catboost"]},
            "python": platform.python_version(), "device": "cpu", "torch_threads": 1,
            "seed": 42, "hardware": platform.processor(), "protocol": "Retrospective January origins with prefix-only target histories; parameters fixed after training",
            **settings}


def prepare_directory(directory: Path, manifest: dict[str, Any]) -> None:
    """Resume only when input, configuration, and all code hashes still match."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / "manifest.json"
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != manifest:
        raise ValueError(f"Run identity changed; use a fresh output directory: {directory}")
    write_json(manifest, path)
    archive = directory / "source_snapshot"
    for relative in manifest["source_hashes"]:
        target = archive / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(ROOT / relative, target)


def build_core_inputs(model: RAExoDLinearForecaster, frame: pd.DataFrame,
                      rows: pd.DataFrame) -> tuple[list[torch.Tensor], list[tuple[np.ndarray, int]]]:
    """Build tensors of shapes (B,L,2), (B,H,F), (B,L,P), (B,), (B,H), (B,H)."""
    samples, positions = [], []
    groups = {(str(g), str(a)): group.sort_values("date")
              for (g, a), group in frame.groupby(["geo", "accommodation_type"])}
    for (geo, accom, origin), block in rows.groupby(["geo", "accommodation_type", "origin"], sort=False):
        group = groups[(str(geo), str(accom))]
        history = group.loc[group.date.le(origin)].tail(model.lookback_len)
        future = block.sort_values("date").reset_index(drop=True)
        count = len(future)
        if count < model.forecast_horizon:
            future = pd.concat([future] + [future.tail(1)] * (model.forecast_horizon - count), ignore_index=True)
        if len(history) != model.lookback_len:
            raise ValueError(f"Insufficient history at {geo}/{accom}/{origin}")
        base_a, base_n = model._baseline_sequence(str(geo), str(accom), future)
        samples.append((history[TARGETS].to_numpy(dtype=np.float32),
                        model._feature_matrix(future, model.future_exo_cols, model.future_exo_mean, model.future_exo_scale),
                        model._feature_matrix(history, model.past_exo_cols, model.past_exo_mean, model.past_exo_scale),
                        int(history.regime_state.iloc[-1]), base_a, base_n))
        positions.append((block.sort_values("date").index.to_numpy(), count))
    tensors = [torch.tensor(np.asarray([sample[j] for sample in samples]), device=model.device) for j in range(6)]
    tensors[3] = tensors[3].long()
    return tensors, positions


def predict_core(model: RAExoDLinearForecaster, frame: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Forecast direct blocks using only histories strictly before each origin."""
    tensors, positions = build_core_inputs(model, frame, rows)
    model.net.eval()
    with torch.no_grad():
        arrivals, nights = model.net(*tensors)
    values = np.stack([arrivals.cpu().numpy(), nights.cpu().numpy()], axis=-1)
    prediction = np.empty((len(rows), 2))
    for index, (destination, count) in enumerate(positions):
        prediction[destination] = values[index, :count]
    return prediction


def fit_transformer(model: Any, train: pd.DataFrame, frame: pd.DataFrame,
                    validation: pd.DataFrame) -> dict[str, Any]:
    """Fit local Transformer implementations with the same observed output windows.

    Training inputs have shape (N,24,2), labels (N,12,2). Validation is a
    January-origin 12-month block with an original-observation loss mask.
    """
    torch.manual_seed(42)
    inputs, targets = [], []
    for key, group in train.groupby(["geo", "accommodation_type"]):
        values = model._log_targets(group.sort_values("date"))
        mean, scale = values.mean(axis=0), values.std(axis=0) + 1e-5
        model.scalers[tuple(map(str, key))] = (mean, scale)
        scaled = (values - mean) / scale
        mask = group.sort_values("date").observed_pair.to_numpy()
        for start in range(24, len(group) - 11):
            if mask[start:start + 12].all():
                inputs.append(scaled[start - 24:start])
                targets.append(scaled[start:start + 12])
    data = TensorDataset(torch.tensor(np.asarray(inputs)), torch.tensor(np.asarray(targets)))
    loader = DataLoader(data, batch_size=128, shuffle=True)
    model.net = model.MODEL_CLASS(24, 12).to("cpu")
    optimizer = torch.optim.AdamW(model.net.parameters(), lr=1e-3, weight_decay=1e-5)
    vx, vy, vm = [], [], []
    for key, block in validation.groupby(["geo", "accommodation_type"]):
        history = frame.loc[frame.geo.eq(key[0]) & frame.accommodation_type.eq(key[1]) & frame.date.le(block.origin.iloc[0])].tail(24)
        mean, scale = model.scalers[tuple(map(str, key))]
        vx.append((model._log_targets(history) - mean) / scale)
        vy.append((model._log_targets(block.sort_values("date")) - mean) / scale)
        vm.append(block.sort_values("date").observed_pair.to_numpy())
    vx, vy, vm = torch.tensor(np.asarray(vx)), torch.tensor(np.asarray(vy)), torch.tensor(np.asarray(vm))[..., None]
    best, state, stale = float("inf"), None, 0
    for epoch in range(30):
        model.net.train()
        for bx, by in loader:
            optimizer.zero_grad(set_to_none=True)
            loss = torch.nn.functional.mse_loss(model.net(bx), by)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.net.parameters(), 1.0)
            optimizer.step()
        model.net.eval()
        with torch.no_grad():
            value = float((((model.net(vx) - vy) ** 2) * vm).sum() / (2 * vm.sum()).clamp_min(1))
        if value < best - 1e-6:
            best, state, stale = value, copy.deepcopy(model.net.state_dict()), 0
        else:
            stale += 1
        print(f"  {model.MODEL_NAME} epoch {epoch + 1}: validation MSE={value:.5f}", flush=True)
        if stale >= 6:
            break
    model.net.load_state_dict(state)
    return {"epochs_run": epoch + 1, "best_validation_mse": best, "training_windows": len(data),
            "learning_rate": 1e-3, "batch_size": 128, "patience": 6}


def predict_transformer(model: Any, frame: pd.DataFrame, rows: pd.DataFrame) -> np.ndarray:
    """Use newly observed histories at each January origin, without recursive test feedback."""
    inputs, keys, scales, counts = [], [], [], []
    groups = {(g, a): group for (g, a), group in frame.groupby(["geo", "accommodation_type"])}
    for (geo, accom, origin), block in rows.groupby(["geo", "accommodation_type", "origin"], sort=False):
        history = groups[(geo, accom)].loc[lambda f: f.date.le(origin)].tail(24)
        mean, scale = model.scalers[(str(geo), str(accom))]
        inputs.append((model._log_targets(history) - mean) / scale)
        keys.append(block.sort_values("date").index.to_numpy())
        scales.append((mean, scale))
        counts.append(len(block))
    model.net.eval()
    with torch.no_grad():
        scaled = model.net(torch.tensor(np.asarray(inputs))).numpy()
    prediction = np.empty((len(rows), 2))
    for i, (indices, (mean, scale), count) in enumerate(zip(keys, scales, counts)):
        prediction[indices] = np.maximum(0, np.expm1(scaled[i, :count] * scale + mean))
    return prediction


def save_fusions(frame: pd.DataFrame, rows: pd.DataFrame, temporal: np.ndarray, spatial: np.ndarray,
                 directory: Path, validation_year: int, calibration_year: int) -> None:
    """Select a single validation weight and calibrate after projection on a later year."""
    val = rows.year.eq(validation_year).to_numpy()
    cal = rows.year.eq(calibration_year).to_numpy()
    test = rows.year.gt(calibration_year).to_numpy()
    weight = base.choose_weight(rows.loc[val], temporal[val], spatial[val])
    for name, w in [("static", 0.7), ("adaptive_raw", weight)]:
        prediction = w * temporal + (1 - w) * spatial
        save_predictions(rows.loc[test].reset_index(drop=True), prediction[test], name, directory,
                         {"w_temporal": w, "selection": "fixed prior" if name == "static" else "validation mean of two target WAPEs"})
    mixed = weight * temporal + (1 - weight) * spatial
    projected = project(mixed)
    eligible = cal & rows.observed_pair.to_numpy()
    conformal = ResidualConformalForecaster(0.90).calibrate_residuals(
        rows.loc[eligible, "observed_arrivals_total"], projected[eligible, 0],
        rows.loc[eligible, "observed_nights_total"], projected[eligible, 1])
    save_predictions(rows.loc[test].reset_index(drop=True), projected[test], "adaptive", directory,
                     {"w_temporal": weight, "calibration_pairs": int(eligible.sum()),
                      "calibration_quantiles": [conformal.conformal_q_arr, conformal.conformal_q_nig]})
    intervals = base.prediction_frame(rows.loc[test].reset_index(drop=True), projected[test])
    for target, triple in conformal.predict_conformal_intervals(projected[test, 0], projected[test, 1]).items():
        intervals[f"lower_{target}"] = triple[0]
        intervals[f"upper_{target}"] = triple[2]
    intervals.to_csv(directory / "intervals.csv", index=False)


def run_monthly(folds: list[str], transformers: bool = True, statistics: bool = True) -> None:
    """Run monthly controls, current-code ablations, and matched baseline forecasts."""
    source = ROOT / "src/data/tourism_europe_enriched.csv"
    for fold in folds:
        end = base.FOLDS[fold]
        directory = OUTPUT / "monthly" / fold
        manifest = get_manifest(source, dataset="eurostat", fold=fold, train_end=end,
                                validation_year=end + 1, calibration_year=end + 2, lookback=24, horizon=12,
                                core_epochs=5, core_learning_rate=1e-4, static_temporal_weight=0.7)
        prepare_directory(directory, manifest)
        frame = base.load_panel(source, end)
        frame["regime_state"] = frame.date.map(base.filtered_regimes(frame, end, 42)).astype(int)
        frame.to_pickle(directory / "completed_panel.pkl")
        train = frame.loc[frame.year.le(end)].copy()
        rows = base.block_rows(frame, end + 1, 2026 if fold == "recent" else end + 4)
        test_mask = rows.year.gt(end + 2).to_numpy()
        test = rows.loc[test_mask].reset_index(drop=True)
        rows.to_pickle(directory / "forecast_rows.pkl")
        if not (directory / "trmf.json").exists():
            started = time.perf_counter()
            spatial = base.spatial_blocks(train, rows)
            elapsed = time.perf_counter() - started
            save_predictions(rows, spatial, "trmf_all", directory, {"fit_and_forecast_seconds": elapsed})
            save_predictions(test, spatial[test_mask], "trmf", directory, {"fit_and_forecast_seconds": elapsed})
        else:
            spatial = pd.read_csv(directory / "trmf_all.csv")[["pred_arrivals", "pred_nights"]].to_numpy()
        names = list(CONFIGS) if fold == "recent" else ["dlinear", "exo", "full_core"]
        for name in names:
            if (directory / f"{name}.json").exists():
                continue
            options = CONFIGS[name].copy()
            use_hmm = options.pop("hmm")
            work = frame.copy()
            if not use_hmm:
                work["regime_state"] = 0
            model = RAExoDLinearForecaster(epochs=5, batch_size=256, learning_rate=1e-4,
                observed_target_col="observed_pair", future_cols=CALENDAR, past_exo_cols=MACRO, **options)
            model.device = torch.device("cpu")
            started = time.perf_counter()
            model.fit(work.loc[work.year.le(end)])
            fit_seconds = time.perf_counter() - started
            prediction = predict_core(model, work, rows)
            _, forecast_seconds = time_call(lambda: predict_core(model, work, test))
            metadata = {"fit_seconds": fit_seconds, "forecast_seconds_median3": forecast_seconds,
                        "registered_parameters": sum(p.numel() for p in model.net.parameters()),
                        "trainable_parameters": sum(p.numel() for p in model.net.parameters() if p.requires_grad),
                        "components": CONFIGS[name]}
            torch.save(model, directory / f"{name}.pt")
            save_predictions(rows, prediction, name + "_all", directory, metadata)
            save_predictions(test, prediction[test_mask], name, directory, metadata)
        temporal = pd.read_csv(directory / "full_core_all.csv")[["pred_arrivals", "pred_nights"]].to_numpy()
        if not (directory / "adaptive.json").exists():
            save_fusions(frame, rows, temporal, spatial, directory, end + 1, end + 2)
        baseline_names = ["snaive"] + (["ets", "sarima", "sarimax", "lightgbm"] if fold == "recent" and statistics else [])
        for name in baseline_names:
            if (directory / f"{name}.json").exists():
                continue
            started = time.perf_counter()
            if name == "lightgbm":
                prediction = base.lightgbm_blocks(train, frame, test, 42)
                diagnostics = []
            else:
                prediction, diagnostics = base.statistical_blocks(name, train, frame, test)
            save_predictions(test, prediction, name, directory, {"fit_and_forecast_seconds": time.perf_counter() - started,
                             "fit_diagnostics": diagnostics, "seasonal_fallback_fits": sum(d["fit_error"] is not None for d in diagnostics)})
        if fold == "recent" and transformers:
            val = rows.loc[rows.year.eq(end + 1)].reset_index(drop=True)
            for name, klass in [("patchtst", PatchTSTForecaster), ("itransformer", iTransformerForecaster), ("timesnet", TimesNetForecaster)]:
                if (directory / f"{name}.json").exists():
                    continue
                model = klass(device="cpu")
                started = time.perf_counter()
                metadata = fit_transformer(model, train, frame, val)
                metadata["fit_seconds"] = time.perf_counter() - started
                prediction, elapsed = time_call(lambda: predict_transformer(model, frame, test))
                metadata.update(forecast_seconds_median3=elapsed,
                                registered_parameters=sum(p.numel() for p in model.net.parameters()),
                                trainable_parameters=sum(p.numel() for p in model.net.parameters()))
                torch.save(model, directory / f"{name}.pt")
                save_predictions(test, prediction, name, directory, metadata)


def load_annual() -> pd.DataFrame:
    """Reindex annual histories and reconstruct macro lags from unfilled WDI data."""
    source = pd.read_csv(ROOT / "data/processed/tourism_un_tourism_enriched.csv", parse_dates=["date"])
    source = source.sort_values(KEYS)
    if source.duplicated(KEYS).any():
        raise ValueError("Duplicate annual keys")
    source["observed_pair"] = source[TARGETS].notna().all(axis=1) & source[TARGETS].ge(0).all(axis=1)
    for target in TARGETS:
        source["observed_" + target] = source[target]
    pieces = []
    raw = pd.read_excel(ROOT / "data_oced/tourism_economics_worldbank-3.xlsx", sheet_name="All Data")
    mappings = {"GDP": "gdp_lag1", "GDP_Growth": "gdp_growth_lag1", "Inflation": "inflation_lag1",
                "Unemployment": "unemployment_lag1", "Exchange_Rate": "exchange_rate_lag1"}
    for geo, group in source.groupby("geo"):
        group = group.set_index("year").reindex(range(1995, 2025)).copy()
        group["year"] = group.index
        group["date"] = pd.to_datetime(group.index.astype(str) + "-01-01")
        group["geo"], group["accommodation_type"] = geo, "total"
        group["country"] = str(source.loc[source.geo.eq(geo), "Country"].iloc[0])
        group["observed_pair"] = group.observed_pair.fillna(False).astype(bool)
        macro = raw.loc[raw["Country Code"].eq(geo)].set_index("Year")
        for original, target in mappings.items():
            group[target] = macro[original].reindex(range(1994, 2025)).shift(1).reindex(group.index).to_numpy()
        pieces.append(group.reset_index(drop=True))
    frame = pd.concat(pieces, ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    columns = TARGETS + ANNUAL_PAST + ANNUAL_FUTURE
    train = frame.loc[frame.year.le(2016)]
    medians = train.groupby("geo")[columns].median()
    pooled = train[columns].median().fillna(0)
    for geo, indices in frame.groupby("geo").groups.items():
        frame.loc[indices, columns] = frame.loc[indices, columns].replace([np.inf, -np.inf], np.nan).ffill().fillna(medians.loc[geo].fillna(pooled)).to_numpy()
    total = frame.groupby("date").agg(arrivals=("arrivals_total", "sum"),
                                     stringency=("covid_stringency_index", "mean"), inflation=("inflation_lag1", "mean"))
    signals = np.column_stack([total.arrivals.pct_change().fillna(0), total.stringency, total.inflation])
    training = total.index.year <= 2016
    signals = (signals - signals[training].mean(axis=0)) / np.maximum(signals[training].std(axis=0), 1e-6)
    hmm = GaussianHMM(n_components=3, n_iter=200, random_state=42, transmat_prior=1.1).fit(signals[training])
    order = np.argsort(hmm.means_[:, 0])
    semantic = {int(order[0]): 1, int(order[1]): 0, int(order[2]): 2}
    emissions = np.column_stack([multivariate_normal.logpdf(signals, mean=hmm.means_[k], cov=hmm.covars_[k], allow_singular=True) for k in range(3)])
    belief, states = np.log(np.maximum(hmm.startprob_, 1e-300)), []
    for index, emission in enumerate(emissions):
        if index:
            belief = logsumexp(belief[:, None] + np.log(np.maximum(hmm.transmat_, 1e-300)), axis=0)
        belief += emission
        belief -= logsumexp(belief)
        states.append(semantic[int(np.argmax(belief))])
    frame["regime_state"] = frame.date.map(pd.Series(states, index=total.index)).astype(int)
    return frame


def run_annual() -> None:
    """Evaluate re-estimated annual temporal and fused models on retained targets."""
    source = ROOT / "data/processed/tourism_un_tourism_enriched.csv"
    directory = OUTPUT / "annual"
    manifest = get_manifest(source, dataset="un_tourism", train_end=2016, validation_year=2017,
                            calibration_year=2018, lookback=4, horizon=1, core_epochs=30,
                            core_learning_rate=0.005, macro_source_sha256=file_hash(ROOT / "data_oced/tourism_economics_worldbank-3.xlsx"),
                            preprocessing="Calendar-year grid; lag1 reconstructed from raw WDI, forward-fill and train medians only")
    prepare_directory(directory, manifest)
    frame = load_annual()
    frame.to_pickle(directory / "completed_panel.pkl")
    rows = frame.loc[frame.year.ge(2017)].copy().reset_index(drop=True)
    rows["origin"] = pd.to_datetime((rows.year - 1).astype(str) + "-12-31") + pd.Timedelta(hours=23, minutes=59, seconds=59)
    rows["horizon"] = 1
    rows.to_pickle(directory / "forecast_rows.pkl")
    train = frame.loc[frame.year.le(2016)].copy()
    test_mask = rows.year.gt(2018).to_numpy()
    test = rows.loc[test_mask].reset_index(drop=True)
    started = time.perf_counter()
    spatial_model = TRMFSpatialSpillover(rank=4, ar_lags=(1, 2)).fit_dual(train)
    arr, nig = spatial_model.forecast_dual(8)
    spatial = np.asarray([[arr[row.geo][row.year - 2017], nig[row.geo][row.year - 2017]] for row in rows.itertuples()])
    save_predictions(rows, spatial, "trmf_all", directory, {})
    save_predictions(test, spatial[test_mask], "trmf", directory, {"fit_and_forecast_seconds": time.perf_counter() - started})
    for name in ["dlinear", "exo", "full_core"]:
        if (directory / f"{name}.json").exists():
            continue
        options = CONFIGS[name].copy()
        use_hmm = options.pop("hmm")
        work = frame.copy()
        if not use_hmm:
            work["regime_state"] = 0
        model = RAExoDLinearForecaster(lookback_len=4, forecast_horizon=1, epochs=30,
            learning_rate=0.005, future_cols=ANNUAL_FUTURE, past_exo_cols=ANNUAL_PAST,
            observed_target_col="observed_pair", **options)
        model.device = torch.device("cpu")
        started = time.perf_counter()
        model.fit(work.loc[work.year.le(2016)])
        fit_seconds = time.perf_counter() - started
        prediction = predict_core(model, work, rows)
        _, elapsed = time_call(lambda: predict_core(model, work, test))
        torch.save(model, directory / f"{name}.pt")
        details = {"fit_seconds": fit_seconds, "forecast_seconds_median3": elapsed,
                   "components": CONFIGS[name], "registered_parameters": sum(p.numel() for p in model.net.parameters())}
        save_predictions(rows, prediction, name + "_all", directory, details)
        save_predictions(test, prediction[test_mask], name, directory, details)
    temporal = pd.read_csv(directory / "full_core_all.csv")[["pred_arrivals", "pred_nights"]].to_numpy()
    save_fusions(frame, rows, temporal, spatial, directory, 2017, 2018)
    lag = frame.groupby("geo")[TARGETS].shift(1)
    naive = frame.assign(pred_arrivals=lag.arrivals_total, pred_nights=lag.nights_total)
    prediction = naive.loc[naive.year.gt(2018), ["pred_arrivals", "pred_nights"]].to_numpy()
    save_predictions(test, prediction, "persistence", directory, {})


def main() -> None:
    """Run selected stages without replacing previous research outputs."""
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["monthly", "annual"], required=True)
    parser.add_argument("--fold", choices=["all", *base.FOLDS], default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-transformers", action="store_true")
    parser.add_argument("--skip-statistics", action="store_true")
    args = parser.parse_args()
    OUTPUT = args.output
    if not HAS_TORCH:
        raise ImportError("PyTorch is required for these experiments")
    torch.set_num_threads(1)
    np.random.seed(42)
    if args.stage == "annual":
        run_annual()
    else:
        run_monthly(list(base.FOLDS) if args.fold == "all" else [args.fold],
                    not args.skip_transformers, not args.skip_statistics)


if __name__ == "__main__":
    main()
