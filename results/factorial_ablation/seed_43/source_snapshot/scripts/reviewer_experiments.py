"""Reproduce the additional Eurostat benchmarks and component ablations.

Runs use the manuscript CSV snapshot, fixed January forecast origins, and
observed scoring outcomes. Release vintages are unavailable, so this is a
retrospective reference-period experiment. Each model writes its own keyed
predictions and provenance, allowing interrupted runs to resume safely.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from hmmlearn.hmm import GaussianHMM
from lightgbm import LGBMRegressor
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp
from scipy.stats import multivariate_normal
from statsmodels.tsa.exponential_smoothing.ets import ETSModel
from statsmodels.tsa.statespace.sarimax import SARIMAX

from src.data.dataset_loader import TourismDataLoader
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.dcqf_conformal import ResidualConformalForecaster
from src.models.ra_exo_dlinear import RAExoDLinearForecaster
from src.models.trmf_spatial import TRMFSpatialSpillover


OUTPUT = ROOT / "results" / "reviewer_experiments"
KEYS = ["geo", "accommodation_type", "date"]
TARGETS = ["arrivals_total", "nights_total"]
CALENDAR = TourismDataLoader.EUROSTAT_FUTURE_COLS
MACRO = TourismDataLoader.EUROSTAT_MACRO_COLS
FOLDS = {"tranquil": 2015, "shock": 2017, "recovery": 2019, "recent": 2021}
CORES = {
    "dlinear": dict(use_exo=False, use_film=False, hmm=False),
    "exo": dict(use_exo=True, use_film=False, hmm=False),
    "hmm": dict(use_exo=False, use_film=False, use_regime_bias=True, hmm=True),
    "film_static": dict(use_exo=False, use_film=True, hmm=False),
    "hmm_film": dict(use_exo=False, use_film=True, hmm=True),
    "exo_hmm_film": dict(use_exo=True, use_film=True, hmm=True),
    "full_core": dict(use_exo=True, use_film=True, hmm=True),
}
FUSIONS = {
    "hmm_film": "hmm_film_trmf",
    "exo_hmm_film": "full_no_consistency",
    "full_core": "full",
}


def save_csv(frame, path):
    temporary = path.with_suffix(".tmp")
    frame.to_csv(temporary, index=False)
    temporary.replace(path)


def save_json(value, path):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def metric(truth, prediction):
    truth = np.asarray(truth, dtype=float)
    prediction = np.asarray(prediction, dtype=float)
    valid = np.isfinite(truth) & np.isfinite(prediction) & (truth >= 0)
    errors = np.abs(truth[valid] - prediction[valid])
    return {
        "n": int(valid.sum()),
        "mae": float(errors.mean()) if len(errors) else None,
        "wape": float(100 * errors.sum() / max(1.0, truth[valid].sum())),
    }


def load_panel(source, train_end):
    frame = pd.read_csv(source, parse_dates=["date"])
    frame = frame.loc[frame.date <= pd.Timestamp("2026-07-01")].copy()
    frame = frame.sort_values(KEYS).reset_index(drop=True)
    if frame.duplicated(KEYS).any():
        raise ValueError("The experiment requires unique destination-category-month keys.")
    frame["observed_pair"] = frame[TARGETS].notna().all(axis=1) & frame[TARGETS].ge(0).all(axis=1)
    for column in TARGETS:
        frame[f"observed_{column}"] = frame[column]
    train = frame.loc[frame.year <= train_end]
    columns = TARGETS + CALENDAR + MACRO
    medians = train.groupby(["geo", "accommodation_type"])[columns].median()
    pooled = train[columns].median().fillna(0)
    for _, indices in frame.groupby(["geo", "accommodation_type"]).groups.items():
        key = tuple(frame.loc[indices[0], ["geo", "accommodation_type"]])
        fallback = medians.loc[key].fillna(pooled)
        values = frame.loc[indices, columns].replace([np.inf, -np.inf], np.nan)
        frame.loc[indices, columns] = values.ffill().fillna(fallback).to_numpy()
    frame[TARGETS] = frame[TARGETS].clip(lower=0)
    return frame


def filtered_regimes(frame, train_end, seed):
    total = frame.loc[frame.accommodation_type.eq("total")]
    aggregate = total.groupby("date").agg(
        arrivals=("arrivals_total", "sum"),
        stringency=("covid_stringency_index", "mean"),
        price=("hicp_accommodation", "mean"),
    )
    growth = aggregate.arrivals.pct_change(12).fillna(0)
    signals = np.column_stack([growth, aggregate.stringency, aggregate.price])
    training = aggregate.index.year <= train_end
    center = signals[training].mean(axis=0)
    scale = np.maximum(signals[training].std(axis=0), 1e-6)
    signals = np.nan_to_num((signals - center) / scale)
    hmm = GaussianHMM(n_components=3, n_iter=200, random_state=seed, transmat_prior=1.1)
    hmm.fit(signals[training])
    order = np.argsort(hmm.means_[:, 0])
    semantics = {int(order[0]): 1, int(order[1]): 0, int(order[2]): 2}
    emissions = np.column_stack([
        multivariate_normal.logpdf(signals, mean=hmm.means_[k], cov=hmm.covars_[k], allow_singular=True)
        for k in range(3)
    ])
    transition = np.log(np.maximum(hmm.transmat_, 1e-300))
    belief = np.log(np.maximum(hmm.startprob_, 1e-300))
    states = []
    for i, emission in enumerate(emissions):
        if i:
            belief = logsumexp(belief[:, None] + transition, axis=0)
        belief = belief + emission
        belief = belief - logsumexp(belief)
        states.append(semantics[int(np.argmax(belief))])
    return pd.Series(states, index=aggregate.index)


def block_rows(frame, start_year, end_year):
    blocks = []
    for year in range(start_year, end_year + 1):
        block = frame.loc[frame.year.eq(year)].copy()
        if not block.empty:
            block["origin"] = pd.Timestamp(year, 1, 1) - pd.Timedelta(seconds=1)
            block["horizon"] = block.month
            blocks.append(block)
    return pd.concat(blocks, ignore_index=True).sort_values(KEYS).reset_index(drop=True)


def prediction_frame(rows, prediction):
    keep = KEYS + ["country", "origin", "horizon", "observed_pair"]
    keep += [f"observed_{column}" for column in TARGETS]
    output = rows[keep].copy()
    output[["pred_arrivals", "pred_nights"]] = prediction
    return output


def neural_blocks(model, frame, rows):
    samples = []
    positions = []
    for (geo, accom, origin), block in rows.groupby(["geo", "accommodation_type", "origin"], sort=False):
        history = frame.loc[frame.geo.eq(geo) & frame.accommodation_type.eq(accom) & frame.date.le(origin)].tail(24)
        future = block.sort_values("date")
        count = len(future)
        padded = future.reset_index(drop=True)
        if count < 12:
            padded = pd.concat([padded, pd.concat([padded.tail(1)] * (12 - count), ignore_index=True)],
                               ignore_index=True)
        base_a, base_n = model._baseline_sequence(geo, accom, padded)
        samples.append((
            history[TARGETS].to_numpy(dtype=np.float32),
            model._feature_matrix(padded, model.future_exo_cols, model.future_exo_mean, model.future_exo_scale),
            model._feature_matrix(history, model.past_exo_cols, model.past_exo_mean, model.past_exo_scale),
            int(history.regime_state.iloc[-1]), base_a, base_n,
        ))
        positions.append((future.index.to_numpy(), count))
    tensors = [torch.tensor(np.asarray([sample[j] for sample in samples]), device=model.device) for j in range(6)]
    tensors[3] = tensors[3].long()
    model.net.eval()
    with torch.no_grad():
        arrivals, nights = model.net(*tensors)
    predictions = np.empty((len(rows), 2))
    for i, (indices, count) in enumerate(positions):
        predictions[indices, 0] = arrivals[i, :count].cpu().numpy()
        predictions[indices, 1] = nights[i, :count].cpu().numpy()
    return predictions


def spatial_blocks(train, rows):
    model = TRMFSpatialSpillover(rank=4, ar_lags=(1, 2, 12))
    model.fit_dual(train, accom_type="total")
    dates = pd.date_range(train.date.max() + pd.offsets.MonthBegin(), rows.date.max(), freq="MS")
    arrivals, nights = model.forecast_dual(len(dates))
    date_index = {date: i for i, date in enumerate(dates)}
    totals = train.loc[train.accommodation_type.eq("total")].groupby("geo")[TARGETS].sum()
    categories = train.groupby(["geo", "accommodation_type"])[TARGETS].sum()
    result = np.empty((len(rows), 2))
    for i, row in enumerate(rows.itertuples()):
        shares = categories.loc[(row.geo, row.accommodation_type)] / totals.loc[row.geo].clip(lower=1)
        step = date_index[row.date]
        result[i] = arrivals[row.geo][step] * shares.iloc[0], nights[row.geo][step] * shares.iloc[1]
    return result


def choose_weight(rows, temporal, spatial, reg_lambda=0.005):
    valid = rows.observed_pair.to_numpy()
    truth = rows[[f"observed_{c}" for c in TARGETS]].to_numpy()[valid]
    y_temp = temporal[valid]
    y_spat = spatial[valid]

    # Method B: Sigmoid Gating (Mixture of Experts / MoE)
    # w = sigmoid(theta) = 1 / (1 + exp(-theta)), theta in (-inf, +inf)
    # Unconstrained optimization over R with L2 regularization
    denom_arr = max(1.0, float(np.sum(truth[:, 0])))
    denom_nig = max(1.0, float(np.sum(truth[:, 1])))

    def objective(theta):
        w = 1.0 / (1.0 + np.exp(-theta))
        pred = w * y_temp + (1.0 - w) * y_spat
        wape_arr = float(np.sum(np.abs(truth[:, 0] - pred[:, 0])) / denom_arr)
        wape_nig = float(np.sum(np.abs(truth[:, 1] - pred[:, 1])) / denom_nig)
        return 0.5 * (wape_arr + wape_nig) + reg_lambda * (theta ** 2)

    res = minimize_scalar(objective, method="brent")
    theta_star = float(res.x)
    weight = float(1.0 / (1.0 + np.exp(-theta_star)))
    return weight


def tabular_features(history, future, geo_id, accom_id):
    target_lags = np.log1p(history[TARGETS].to_numpy(dtype=float)).reshape(-1)
    past = history[MACRO].tail(1).to_numpy(dtype=float).reshape(-1)
    return np.concatenate([target_lags, past, future[CALENDAR].to_numpy(dtype=float),
                           [future.month, geo_id, accom_id]])


def lightgbm_blocks(train, frame, rows, seed):
    countries = {geo: i for i, geo in enumerate(sorted(frame.geo.unique()))}
    categories = {accom: i for i, accom in enumerate(sorted(frame.accommodation_type.unique()))}
    features, targets = [], []
    for (geo, accom), group in train.groupby(["geo", "accommodation_type"]):
        group = group.sort_values("date").reset_index(drop=True)
        for start in range(24, len(group) - 11):
            future = group.iloc[start:start + 12]
            if not future.observed_pair.all():
                continue
            history = group.iloc[start - 24:start]
            for horizon, (_, row) in enumerate(future.iterrows(), 1):
                features.append(np.r_[tabular_features(history, row, countries[geo], categories[accom]), horizon])
                targets.append(row[TARGETS].to_numpy(dtype=float))
    features = np.asarray(features)
    targets = np.asarray(targets)
    future_features = np.empty((len(rows), features.shape[1]))
    for (geo, accom, origin), block in rows.groupby(["geo", "accommodation_type", "origin"]):
        history = frame.loc[frame.geo.eq(geo) & frame.accommodation_type.eq(accom) & frame.date.le(origin)].tail(24)
        for index, row in block.iterrows():
            future_features[index] = np.r_[tabular_features(history, row, countries[geo], categories[accom]), row.horizon]
    results = []
    for channel in range(2):
        model = LGBMRegressor(n_estimators=200, learning_rate=0.05, num_leaves=31,
                              random_state=seed, n_jobs=1, verbosity=-1)
        model.fit(features, np.log1p(targets[:, channel]))
        results.append(np.maximum(0, np.expm1(model.predict(future_features))))
    return np.column_stack(results)


def classical_exog(frame):
    macro = frame[["hicp_accommodation", "hicp_overall", "covid_stringency_index"]].to_numpy(dtype=float)
    return np.column_stack([np.sin(2 * np.pi * frame.month / 12),
                            np.cos(2 * np.pi * frame.month / 12),
                            frame.holiday_days / 31, macro])


def statistical_blocks(name, train, frame, rows):
    output = np.empty((len(rows), 2))
    diagnostics = []
    groups = list(train.groupby(["geo", "accommodation_type"]))
    for group_index, ((geo, accom), training) in enumerate(groups):
        all_history = frame.loc[frame.geo.eq(geo) & frame.accommodation_type.eq(accom)].sort_values("date")
        requested = rows.loc[rows.geo.eq(geo) & rows.accommodation_type.eq(accom)]
        for channel, target in enumerate(TARGETS):
            fitted = None
            fit_error = None
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                try:
                    values = np.log1p(training[target].to_numpy(dtype=float))
                    if name != "snaive" and np.std(values) < 1e-8:
                        raise ValueError("Constant training series")
                    if name == "ets":
                        fitted = ETSModel(values, error="add", trend="add", damped_trend=True,
                                          seasonal="add", seasonal_periods=12).fit(disp=False, maxiter=300)
                    elif name in {"sarima", "sarimax"}:
                        exog = classical_exog(training) if name == "sarimax" else None
                        fitted = SARIMAX(values, exog=exog, order=(1, 0, 0), seasonal_order=(1, 0, 0, 12),
                                         trend="c", enforce_stationarity=True,
                                         enforce_invertibility=True).fit(disp=False, maxiter=100)
                    if fitted is not None and not fitted.mle_retvals.get("converged", True):
                        raise ValueError("Optimizer did not converge")
                except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                    fit_error = str(error)
                    fitted = None
            if caught or fit_error:
                diagnostics.append({"geo": geo, "accommodation_type": accom, "target": target,
                                    "fit_error": fit_error, "warnings": sorted({str(w.message) for w in caught})})
            for origin, block in requested.groupby("origin"):
                history = all_history.loc[all_history.date.le(origin)]
                forecast = history[target].tail(12).to_numpy()[:len(block)]
                if fitted is not None:
                    observed = np.log1p(history[target].to_numpy(dtype=float))
                    if name == "ets":
                        updated = ETSModel(observed, error="add", trend="add", damped_trend=True,
                                           seasonal="add", seasonal_periods=12).smooth(fitted.params)
                        forecast = np.expm1(updated.forecast(len(block)))
                    else:
                        exog = classical_exog(history) if name == "sarimax" else None
                        updated = fitted.apply(observed, exog=exog, refit=False)
                        future_exog = None
                        if name == "sarimax":
                            future = block.copy()
                            for macro in ["hicp_accommodation", "hicp_overall", "covid_stringency_index"]:
                                future[macro] = history[macro].iloc[-1]
                            future_exog = classical_exog(future)
                        forecast = np.expm1(updated.forecast(len(block), exog=future_exog))
                    if not np.isfinite(forecast).all():
                        raise ValueError(f"Nonfinite {name} forecast for {geo}/{accom}/{target}")
                output[block.index, channel] = np.maximum(0, forecast)
        if (group_index + 1) % 25 == 0:
            print(f"  {name}: {group_index + 1}/{len(groups)} entities", flush=True)
    return output, diagnostics


def record(rows, prediction, name, fold_dir, metadata, extra=None):
    if not np.isfinite(prediction).all():
        raise ValueError(f"Nonfinite predictions in {name}")
    output = prediction_frame(rows, prediction)
    output["model"] = name
    save_csv(output, fold_dir / f"{name}.csv")
    details = {**metadata, **(extra or {}), "model": name}
    valid = output.observed_pair
    details["test_metrics"] = {
        target: metric(output.loc[valid, f"observed_{target}"], output.loc[valid, pred])
        for target, pred in zip(TARGETS, ["pred_arrivals", "pred_nights"])
    }
    save_json(details, fold_dir / f"{name}.json")
    print(name, details["test_metrics"], flush=True)


def run_fold(args, fold):
    train_end = FOLDS[fold]
    frame = load_panel(args.data, train_end)
    states = filtered_regimes(frame, train_end, args.seed)
    frame["regime_state"] = frame.date.map(states).astype(int)
    train = frame.loc[frame.year.le(train_end)].copy()
    val = block_rows(frame, train_end + 1, train_end + 1)
    cal = block_rows(frame, train_end + 2, train_end + 2)
    test = block_rows(frame, train_end + 3, 2026 if fold == "recent" else train_end + 4)
    future = pd.concat([val, cal, test], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    fold_dir = OUTPUT / fold
    fold_dir.mkdir(parents=True, exist_ok=True)
    source_hashes = {
        str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in [Path(__file__), ROOT / "src/models/ra_exo_dlinear.py", ROOT / "src/models/trmf_spatial.py"]
    }
    metadata = {"fold": fold, "train_end": train_end, "validation_year": train_end + 1,
                "calibration_year": train_end + 2, "test_start": str(test.date.min()),
                "test_end": str(test.date.max()), "test_grid_rows": len(test),
                "test_observed_pairs": int(test.observed_pair.sum()), "seed": args.seed,
                "epochs": args.epochs, "lookback": 24, "horizon": 12,
                "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
                "source_hashes": source_hashes, "torch_version": torch.__version__,
                "protocol": "Retrospective January-origin blocks, training-fitted parameters, prefix-only state updates"}
    manifest = fold_dir / "manifest.json"
    save_json(metadata, manifest)
    print(f"FOLD {fold}: train <= {train_end}, test {test.date.min()} .. {test.date.max()}", flush=True)
    spatial = spatial_blocks(train, future)
    val_mask = future.year.eq(train_end + 1).to_numpy()
    cal_mask = future.year.eq(train_end + 2).to_numpy()
    test_mask = future.year.ge(train_end + 3).to_numpy()
    test_rows = future.loc[test_mask].reset_index(drop=True)
    core_names = list(CORES) if fold == "recent" else ["dlinear", "full_core"]
    requested = set(args.models) if args.models else None
    for name in core_names:
        if requested is not None and name not in requested:
            continue
        final_name = FUSIONS.get(name, name)
        if (fold_dir / f"{final_name}.json").exists():
            print(f"Resume: {fold}/{final_name}", flush=True)
            continue
        started = time.perf_counter()
        options = CORES[name].copy()
        use_hmm = options.pop("hmm")
        work = frame.copy()
        if not use_hmm:
            work["regime_state"] = 0
        model = RAExoDLinearForecaster(epochs=args.epochs, batch_size=256, random_state=args.seed,
                                       observed_target_col="observed_pair", future_cols=CALENDAR,
                                       past_exo_cols=MACRO, use_consistency=name == "full_core", **options)
        model.fit(work.loc[work.year.le(train_end)])
        temporal = neural_blocks(model, work, future)
        record(test_rows, temporal[test_mask], name, fold_dir, metadata,
               {"runtime_seconds": time.perf_counter() - started, "components": CORES[name]})
        if name in FUSIONS:
            weight = choose_weight(future.loc[val_mask], temporal[val_mask], spatial[val_mask])
            mixed = weight * temporal + (1 - weight) * spatial
            record(test_rows, mixed[test_mask], final_name + "_raw", fold_dir, metadata,
                   {"w_temporal": weight})
            if final_name == "full":
                a, n, violations = CoupledConsistencyOptimizer.project_physical_pair(mixed[:, 0], mixed[:, 1])
                projected = np.column_stack([a, n])
                calibration = future.loc[cal_mask & future.observed_pair.to_numpy()]
                cal_prediction = projected[cal_mask & future.observed_pair.to_numpy()]
                conformal = ResidualConformalForecaster(0.90).calibrate_residuals(
                    calibration.observed_arrivals_total, cal_prediction[:, 0],
                    calibration.observed_nights_total, cal_prediction[:, 1])
                intervals = conformal.predict_conformal_intervals(projected[test_mask, 0], projected[test_mask, 1])
                record(test_rows, projected[test_mask], final_name, fold_dir, metadata,
                       {"w_temporal": weight, "runtime_seconds": time.perf_counter() - started,
                        "calibration_pairs": len(calibration), "calibration_quantiles":
                        [conformal.conformal_q_arr, conformal.conformal_q_nig],
                        "raw_violations": int(violations[test_mask].sum())})
                interval_frame = prediction_frame(test_rows, projected[test_mask])
                for target, triple in intervals.items():
                    interval_frame[f"lower_{target}"] = triple[0]
                    interval_frame[f"upper_{target}"] = triple[2]
                save_csv(interval_frame, fold_dir / "intervals.csv")
            else:
                record(test_rows, mixed[test_mask], final_name, fold_dir, metadata, {"w_temporal": weight})
    if fold != "recent":
        return
    for name in ["snaive", "ets", "sarima", "sarimax", "lightgbm"]:
        if requested is not None and name not in requested:
            continue
        if (fold_dir / f"{name}.json").exists():
            print(f"Resume: {fold}/{name}", flush=True)
            continue
        started = time.perf_counter()
        if name == "lightgbm":
            prediction = lightgbm_blocks(train, frame, test_rows, args.seed)
            diagnostics = []
        else:
            prediction, diagnostics = statistical_blocks(name, train, frame, test_rows)
        record(test_rows, prediction, name, fold_dir, metadata,
               {"runtime_seconds": time.perf_counter() - started, "fit_diagnostics": diagnostics,
                "seasonal_fallback_fits": sum(d["fit_error"] is not None for d in diagnostics)})


def main():
    global OUTPUT
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/processed/tourism_europe_enriched.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--fold", choices=["all", *FOLDS], default="all")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--models", nargs="+", choices=[*CORES, "snaive", "ets", "sarima", "sarimax", "lightgbm"])
    args = parser.parse_args()
    OUTPUT = args.output
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    for fold in (["recent", "tranquil", "shock", "recovery"] if args.fold == "all" else [args.fold]):
        run_fold(args, fold)


if __name__ == "__main__":
    main()
