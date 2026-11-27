"""Add validation-selected ETS and a genuinely trainable DLinear comparator."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import torch

from scripts.section4_experiments import (OUTPUT, file_hash, write_json, fit_transformer,
    predict_transformer, time_call, save_predictions)
from scripts.tune_ets_baseline import main as tune_ets
from src.models.ra_exo_dlinear import VanillaDLinearNet
from src.models.transformer_baselines import PatchTSTForecaster


def run_extra_baselines(output: Path, skip_ets: bool = False) -> None:
    """Use the same completed frames, observed windows, validation, and January origins."""
    for fold in ["tranquil", "shock", "recovery", "recent"]:
        directory = output / "monthly" / fold
        manifest = json.loads((directory / "manifest.json").read_text())
        if not (directory / "learned_dlinear.json").exists():
            frame = pd.read_pickle(directory / "completed_panel.pkl")
            rows = pd.read_pickle(directory / "forecast_rows.pkl")
            train = frame.loc[frame.year.le(manifest["train_end"])]
            validation = rows.loc[rows.year.eq(manifest["validation_year"])].reset_index(drop=True)
            testing = rows.loc[rows.year.gt(manifest["calibration_year"])].reset_index(drop=True)
            model = PatchTSTForecaster(device="cpu")
            model.MODEL_CLASS = VanillaDLinearNet
            model.MODEL_NAME = "Learned DLinear"
            started = time.perf_counter()
            metadata = fit_transformer(model, train, frame, validation)
            metadata["fit_seconds"] = time.perf_counter() - started
            prediction, elapsed = time_call(lambda: predict_transformer(model, frame, testing))
            metadata.update(forecast_seconds_median3=elapsed,
                            registered_parameters=sum(p.numel() for p in model.net.parameters()),
                            trainable_parameters=sum(p.numel() for p in model.net.parameters()),
                            extra_baseline_script_sha256=file_hash(Path(__file__)))
            torch.save(model, directory / "learned_dlinear.pt")
            save_predictions(testing, prediction, "learned_dlinear", directory, metadata)
    directory = output / "monthly/recent"
    if not skip_ets:
        tune_ets(directory)
        path = directory / "ets.json"
        details = json.loads(path.read_text())
        details["forecast_sha256"] = file_hash(directory / "ets.csv")
        write_json(details, path)
    write_json({"script_sha256": file_hash(Path(__file__)),
                "ets_tuner_sha256": file_hash(ROOT / "scripts/tune_ets_baseline.py"),
                "dlinear": "Trainable VanillaDLinearNet; same log-standardized inputs, complete observed training windows, and 30-epoch/6-patience validation protocol as Transformer comparators",
                "ets": "Validation-selected additive seasonal ETS on training-median-scaled counts; fixed log-ETS retained only as a sensitivity archive"}, output / "extra_baselines_manifest.json")


def main() -> None:
    """Add controls to an existing matched run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--skip-ets", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(1)
    run_extra_baselines(args.output, args.skip_ets)


if __name__ == "__main__":
    main()
