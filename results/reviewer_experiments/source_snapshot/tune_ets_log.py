"""Select seasonal ETS trend specifications using the 2022 validation block."""

import hashlib
import json
from pathlib import Path
import shutil
import sys
import time
import warnings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
from statsmodels.tsa.exponential_smoothing.ets import ETSModel

from scripts.reviewer_experiments import (
    TARGETS, block_rows, load_panel, record, save_json,
)


def fit_ets(values, trend):
    return ETSModel(values, error="add", trend="add" if trend else None,
                    damped_trend=trend, seasonal="add", seasonal_periods=12)


def main():
    directory = ROOT / "results/reviewer_experiments/recent"
    metadata = json.loads((directory / "manifest.json").read_text(encoding="utf-8"))
    data = ROOT / "src/data/tourism_europe_enriched.csv"
    if hashlib.sha256(data.read_bytes()).hexdigest() != metadata["data_sha256"]:
        raise ValueError("The validation-tuning data differs from the matched benchmark.")
    frame = load_panel(data, metadata["train_end"])
    training = frame.loc[frame.year.le(metadata["train_end"])]
    validation = block_rows(frame, 2022, 2022)
    test = block_rows(frame, 2024, 2026)
    predictions = np.empty((len(test), 2))
    selections = []
    started = time.perf_counter()
    for number, ((geo, accom), train) in enumerate(training.groupby(["geo", "accommodation_type"]), 1):
        group = frame.loc[frame.geo.eq(geo) & frame.accommodation_type.eq(accom)].sort_values("date")
        val = validation.loc[validation.geo.eq(geo) & validation.accommodation_type.eq(accom)]
        requested = test.loc[test.geo.eq(geo) & test.accommodation_type.eq(accom)]
        for channel, target in enumerate(TARGETS):
            values = np.log1p(train[target].to_numpy(dtype=float))
            candidates = []
            failures = []
            for trend in [False, True]:
                try:
                    if np.std(values) < 1e-8:
                        raise ValueError("Constant training series")
                    with warnings.catch_warnings():
                        warnings.simplefilter("ignore")
                        fitted = fit_ets(values, trend).fit(disp=False, maxiter=300)
                    if not fitted.mle_retvals.get("converged", True):
                        raise ValueError("Optimizer did not converge")
                    forecast = np.expm1(fitted.forecast(len(val)))
                    observed = val.observed_pair.to_numpy()
                    if not np.isfinite(forecast).all():
                        raise ValueError("Nonfinite validation prediction")
                    error = float(np.abs(forecast[observed] - val.loc[observed, f"observed_{target}"].to_numpy()).sum())
                    candidates.append((error, trend, fitted))
                except (ValueError, np.linalg.LinAlgError, OverflowError) as error:
                    failures.append({"damped_trend": trend, "reason": str(error)})
            selected = min(candidates, key=lambda item: item[0]) if candidates else None
            selections.append({"geo": geo, "accommodation_type": accom, "target": target,
                               "selected_damped_trend": selected[1] if selected else None,
                               "validation_abs_error": selected[0] if selected else None,
                               "fallback": selected is None, "failures": failures})
            for origin, block in requested.groupby("origin"):
                history = group.loc[group.date.le(origin)]
                forecast = history[target].tail(12).to_numpy()[:len(block)]
                if selected is not None:
                    error, trend, fitted = selected
                    updated = fit_ets(np.log1p(history[target].to_numpy(dtype=float)), trend).smooth(fitted.params)
                    forecast = np.expm1(updated.forecast(len(block)))
                if not np.isfinite(forecast).all():
                    raise ValueError(f"Nonfinite selected ETS prediction for {geo}/{accom}/{target}")
                predictions[block.index, channel] = np.maximum(0, forecast)
        if number % 25 == 0:
            print(f"ETS validation selection: {number}/152 entities", flush=True)
    for suffix in ["csv", "json"]:
        source = directory / f"ets.{suffix}"
        archive = directory / f"ets_fixed_damped.{suffix}"
        if source.exists() and not archive.exists():
            shutil.copyfile(source, archive)
    extra = {"runtime_seconds": time.perf_counter() - started,
             "selection": "Per entity and target, validation absolute error over no-trend and damped-trend additive seasonal ETS on log1p counts",
             "tuning_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
             "seasonal_fallback_fits": sum(row["fallback"] for row in selections)}
    record(test, predictions, "ets", directory, metadata, extra)
    save_json(selections, directory / "ets_validation_selection.json")


if __name__ == "__main__":
    main()
