"""Compare a committed Eurostat pipeline with the working-tree implementation.

Runs both versions on the same current data, split boundaries, epoch budget,
seed, and CPU thread count. Keeps each version's own forecasting equation,
postprocessing, baseline calibration, and fusion-weight search behavior.
Writes separate results; never replaces the main pipeline's prediction CSV.
"""

import argparse
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import types

import numpy as np
import pandas as pd
import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.pipeline.linear_pipeline import TourismLinearForecastingPipeline, calculate_metrics


def git_text(ref: str, path: str) -> str:
    return subprocess.run(
        ["git", "show", f"{ref}:{path}"], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout


def load_module(name: str, source: str, path: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__file__ = str(ROOT / path)
    sys.modules[name] = module
    exec(compile(source, f"git:{path}", "exec"), module.__dict__)
    return module


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def metrics_row(version: str, stage: str, frame: pd.DataFrame, arrival: str, night: str) -> dict:
    ma = calculate_metrics(frame.arrivals_total.to_numpy(), frame[arrival].to_numpy())
    mn = calculate_metrics(frame.nights_total.to_numpy(), frame[night].to_numpy())
    raw_a = "pred_arrivals_raw" if stage == "framework" else arrival
    raw_n = "pred_nights_raw" if stage == "framework" else night
    raw_violations = int((frame[raw_n] < frame[raw_a]).sum())
    return {
        "version": version, "stage": stage, "test_rows": len(frame),
        "arrivals_evaluated_rows": ma["n"], "nights_evaluated_rows": mn["n"],
        "arrivals_wape_pct": ma["WAPE"], "nights_wape_pct": mn["WAPE"],
        "arrivals_mae": ma["MAE"], "nights_mae": mn["MAE"],
        "arrivals_rmse": ma["RMSE"], "nights_rmse": mn["RMSE"],
        "raw_physical_violations": raw_violations,
        "raw_physical_violation_pct": raw_violations / len(frame) * 100,
        "final_physical_violations": int((frame[night] < frame[arrival]).sum()),
        "w_temporal": float(frame.w_temporal.iloc[0]) if stage == "framework" else None,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-ref", default="HEAD", help="Git commit containing the old DLinear implementation")
    parser.add_argument("--train-end-year", type=int, default=2022)
    parser.add_argument("--val-end-year", type=int, default=2023)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results" / "eurostat_version_comparison")
    args = parser.parse_args()
    torch.set_num_threads(args.threads)
    old_commit = subprocess.run(
        ["git", "rev-parse", args.old_ref], cwd=ROOT, check=True,
        capture_output=True, text=True, encoding="utf-8",
    ).stdout.strip()
    paths = {
        "model": "src/models/ra_exo_dlinear.py",
        "projection": "src/models/coupled_optimizer.py",
        "pipeline": "src/pipeline/linear_pipeline.py",
    }
    old_sources = {name: git_text(old_commit, path) for name, path in paths.items()}
    old_model = load_module("comparison_old_model", old_sources["model"], paths["model"])
    old_projection = load_module("comparison_old_projection", old_sources["projection"], paths["projection"])
    old_pipeline = load_module("comparison_old_pipeline", old_sources["pipeline"], paths["pipeline"])
    if not hasattr(old_model.RAExoDLinearForecaster, "calibrate"):
        raise ValueError("Select an older commit with the original baseline-calibration implementation.")
    old_pipeline.RAExoDLinearForecaster = old_model.RAExoDLinearForecaster
    old_pipeline.CoupledConsistencyOptimizer = old_projection.CoupledConsistencyOptimizer

    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    rows = []
    frames = {}
    configurations = {}
    for version, pipeline_class in (
        ("old", old_pipeline.TourismLinearForecastingPipeline),
        ("new", TourismLinearForecastingPipeline),
    ):
        np.random.seed(42)
        torch.manual_seed(42)
        print(f"RUN {version}: Eurostat, train <= {args.train_end_year}, validation <= {args.val_end_year}, epochs={args.epochs}", flush=True)
        started = time.perf_counter()
        with (out / f"{version}.log").open("w", encoding="utf-8", buffering=1) as log, redirect_stdout(log), redirect_stderr(log):
            pipeline = pipeline_class(
                dataset="eurostat", train_end_year=args.train_end_year,
                val_end_year=args.val_end_year, epochs=args.epochs,
            )
            result = pipeline.run_pipeline()
            history = pd.concat([pipeline.train_df, pipeline.val_df], ignore_index=True)
            temporal_a, temporal_n = pipeline.linear_engine.predict(pipeline.test_df, history_df=history)
        elapsed = time.perf_counter() - started
        frame = result["results_df"].copy()
        frame["pred_temporal_arrivals"] = temporal_a
        frame["pred_temporal_nights"] = temporal_n
        if not np.isfinite(frame[["pred_arrivals_point", "pred_nights_point", "pred_temporal_arrivals", "pred_temporal_nights"]].to_numpy()).all():
            raise ValueError(f"{version} produced nonfinite predictions.")
        frame.to_csv(out / f"{version}_predictions.csv", index=False)
        frames[version] = frame
        rows.append(metrics_row(version, "temporal", frame, "pred_temporal_arrivals", "pred_temporal_nights"))
        rows.append(metrics_row(version, "framework", frame, "pred_arrivals_point", "pred_nights_point"))
        configurations[version] = {
            "train_rows": len(pipeline.train_df), "validation_rows": len(pipeline.val_df),
            "test_rows": len(frame), "test_start": str(frame.date.min()), "test_end": str(frame.date.max()),
            "lookback": pipeline.lookback_len, "horizon": pipeline.forecast_horizon,
            "kernel_size": pipeline.linear_engine.kernel_size, "learning_rate": pipeline.linear_engine.learning_rate,
            "network_parameters": sum(p.numel() for p in pipeline.linear_engine.net.parameters() if p.requires_grad),
            "w_temporal": result["w_time"], "runtime_seconds": elapsed,
            "arrivals_baseline_blend": getattr(pipeline.linear_engine, "blend_weight_arrivals", None),
            "nights_baseline_blend": getattr(pipeline.linear_engine, "blend_weight_nights", None),
        }
        print(f"DONE {version}: arrivals WAPE={rows[-1]['arrivals_wape_pct']:.2f}%, nights WAPE={rows[-1]['nights_wape_pct']:.2f}%, w_temporal={result['w_time']:.4f}, seconds={elapsed:.1f}", flush=True)

    keys = ["date", "geo", "accommodation_type"]
    aligned = frames["old"].merge(frames["new"], on=keys, how="outer", validate="one_to_one", indicator=True, suffixes=("_old", "_new"))
    if not aligned["_merge"].eq("both").all():
        raise ValueError("The two runs have different test rows; a paired comparison is invalid.")
    for target in ("arrivals_total", "nights_total"):
        np.testing.assert_array_equal(aligned[f"{target}_old"], aligned[f"{target}_new"])
    summary = pd.DataFrame(rows)
    summary.to_csv(out / "metrics.csv", index=False)
    metadata = {
        "old_commit": old_commit, "epochs": args.epochs, "threads": args.threads, "seed": 42,
        "train_end_year": args.train_end_year, "val_end_year": args.val_end_year,
        "torch_version": torch.__version__, "configurations": configurations,
        "data_sha256": digest(pipeline.data_loader.parquet_path.read_bytes()),
        "sources_sha256": {
            name: {"old": digest(old_sources[name].encode("utf-8")), "new": digest((ROOT / path).read_bytes())}
            for name, path in paths.items()
        },
        "metric_filter": "Compute each target's metrics where observed target >= 100, as in calculate_metrics().",
        "comparison_scope": "Full implementation versions; includes changes in loss, baseline calibration, projection and fusion search. Not an isolated equation ablation.",
    }
    (out / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    report = [
        "# Eurostat: old versus new implementation", "",
        f"Old commit: `{old_commit}`. New: working tree identified by source hashes in metadata.json.", "",
        f"Both runs: train through {args.train_end_year}; validation through {args.val_end_year}; {len(frames['new']):,} identical test rows. L=24, H=12, epochs={args.epochs}, seed=42.", "",
        "| Version | Stage | Arrivals WAPE (%) | Nights WAPE (%) | Arrivals MAE | Nights MAE | Temporal weight |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        weight = "--" if row["w_temporal"] is None else f"{row['w_temporal']:.4f}"
        report.append(f"| {row['version']} | {row['stage']} | {row['arrivals_wape_pct']:.2f} | {row['nights_wape_pct']:.2f} | {row['arrivals_mae']:,.2f} | {row['nights_mae']:,.2f} | {weight} |")
    report.extend([
        "", "These are point estimates from single seeded runs; no bootstrap uncertainty or significance test was computed.",
        "The comparison measures the complete implementation change, including each version's calibration, projection and fusion search. It does not isolate the causal effect of one component.",
        "The old temporal stage includes its baseline blend and hard postprocessing; the new temporal stage returns unprojected neural forecasts.",
        "The existing preprocessing and HMM protocol is reused; this comparison alone does not validate that protocol.",
        "Historical manuscript values (14.59% / 25.00%, stated n=4,712) are a separate reference and are not substituted for the old version's measured results above.", "",
    ])
    (out / "comparison.md").write_text("\n".join(report), encoding="utf-8")
    print(summary.to_string(index=False), flush=True)
    print(f"Saved comparison to: {out}", flush=True)


if __name__ == "__main__":
    main()
