"""Run a 2x2x2 ablation of temporal components on the frozen Eurostat panel.

E toggles both exogenous streams. H toggles a train-fitted, forward-filtered
HMM state against a constant state ID. F toggles the multiplicative FiLM gain;
the additive state head remains enabled in every factorial cell. Thus H0F0
has a learned constant intercept. The eight factorial cells share the full
model's coupled loss, spatial forecasts, validation gate, and projection.
Each fit produces a raw temporal and a projected fused forecast.
"""

import argparse
import hashlib
import itertools
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.reviewer_experiments import (
    CALENDAR, KEYS, MACRO, TARGETS, block_rows, choose_weight, filtered_regimes,
    load_panel, metric, neural_blocks, prediction_frame, save_csv, save_json, spatial_blocks,
)
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.models.ra_exo_dlinear import RAExoDLinearForecaster

OUTPUT = ROOT / "results/factorial_ablation"
LEARNING_RATE = 0.0001
PREDICTION_COLUMNS = {
    "temporal": ("pred_arrivals", "pred_nights"),
    "fused": ("fused_arrivals", "fused_nights"),
}
CELLS = {f"E{e}H{h}F{f}": (e, h, f)
         for e, h, f in itertools.product((0, 1), repeat=3)}


def digest(path: Path) -> str:
    """Return the SHA-256 of a run input."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run(args: argparse.Namespace) -> None:
    """Train and score the matched ablation for one seed."""
    torch.set_num_threads(args.threads)
    np.random.seed(args.seed)
    frame = load_panel(args.data, 2021)
    states = filtered_regimes(frame, 2021, args.seed)
    validation = block_rows(frame, 2022, 2022)
    calibration = block_rows(frame, 2023, 2023)
    test = block_rows(frame, 2024, 2026)
    future = pd.concat([validation, calibration, test], ignore_index=True).sort_values(KEYS).reset_index(drop=True)
    validation_mask = future.year.eq(2022).to_numpy()
    test_mask = future.year.ge(2024).to_numpy()
    test = future.loc[test_mask].reset_index(drop=True)
    training_mask = frame.year.le(2021)
    spatial = spatial_blocks(frame.loc[training_mask], future)
    directory = args.output / f"seed_{args.seed}"
    directory.mkdir(parents=True, exist_ok=True)
    source_paths = [
        Path(__file__), ROOT / "scripts/reviewer_experiments.py",
        ROOT / "src/models/ra_exo_dlinear.py", ROOT / "src/data/dataset_loader.py",
        ROOT / "src/models/trmf_spatial.py", ROOT / "src/models/coupled_optimizer.py",
    ]
    metadata = {
        "seed": args.seed, "epochs": args.epochs, "batch_size": 256,
        "learning_rate": LEARNING_RATE, "freeze_backbone": True, "threads": args.threads,
        "lookback": 24, "horizon": 12,
        "kernel_size": 13, "train_years": [2010, 2021],
        "validation_year": 2022, "calibration_year": 2023,
        "test_reference_periods": ["2024-01", "2026-07"],
        "test_grid_rows": len(test), "test_observed_pairs": int(test.observed_pair.sum()),
        "data_sha256": digest(args.data),
        "source_hashes": {p.relative_to(ROOT).as_posix(): digest(p) for p in source_paths},
        "torch_version": torch.__version__, "device": "cuda" if torch.cuda.is_available() else "cpu",
        "coupled_penalty_in_all_factorial_cells": True,
        "additive_state_head_enabled_in_all_factorial_cells": True,
        "selection": "The full cell selects its sigmoid weight on 2022 validation. The fused cells share that weight, TRMF predictions, and physical projection.",
        "availability": "Retrospective reference-period inputs, not PIT.",
        "hmm_origin_states": {str(origin): int(states.loc[states.index <= origin].iloc[-1])
                              for origin in sorted(test.origin.unique())},
        "hmm_training_state_counts": {str(k): int(v) for k, v in
                                      states.loc[states.index.year <= 2021].value_counts().items()},
    }
    manifest = directory / "manifest.json"
    if manifest.exists() and json.loads(manifest.read_text("utf-8")) != metadata:
        raise ValueError(f"Run settings changed; choose another output directory: {directory}")
    save_json(metadata, manifest)
    for source_path in source_paths:
        snapshot = directory / "source_snapshot" / source_path.relative_to(ROOT)
        snapshot.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source_path, snapshot)
    full_cell = "E1H1F1"
    configurations = {full_cell: CELLS[full_cell], **CELLS, "plain_dlinear": (0, 0, 0)}
    for name, (e, h, f) in configurations.items():
        path = directory / f"{name}.json"
        if path.exists() and (directory / f"{name}.csv").exists():
            if name == full_cell:
                weight = json.loads(path.read_text("utf-8"))["fused_temporal_weight"]
            print(f"Resume {args.seed}/{name}", flush=True)
            continue
        work = frame.copy()
        work["regime_state"] = work.date.map(states).astype(int) if h else 0
        started = time.perf_counter()
        is_factorial = name != "plain_dlinear"
        model = RAExoDLinearForecaster(
            epochs=args.epochs, batch_size=256, learning_rate=LEARNING_RATE, freeze_backbone=True,
            random_state=args.seed, observed_target_col="observed_pair",
            future_cols=CALENDAR, past_exo_cols=MACRO,
            use_consistency=is_factorial, use_exo=bool(e), use_film=bool(f),
            use_regime_bias=is_factorial,
        )
        model.fit(work.loc[training_mask])
        if name == full_cell:
            prediction = neural_blocks(model, work, future)
            weight = choose_weight(future.loc[validation_mask], prediction[validation_mask], spatial[validation_mask])
            prediction = prediction[test_mask]
        else:
            prediction = neural_blocks(model, work, test)
        mixed = weight * prediction + (1 - weight) * spatial[test_mask]
        arrivals, nights, _ = CoupledConsistencyOptimizer.project_physical_pair(mixed[:, 0], mixed[:, 1])
        if not all(np.isfinite(values).all() for values in (prediction, arrivals, nights)):
            raise ValueError(f"Nonfinite predictions: {args.seed}/{name}")
        result = prediction_frame(test, prediction)
        result["fused_arrivals"] = arrivals
        result["fused_nights"] = nights
        save_csv(result, directory / f"{name}.csv")
        valid = result.observed_pair
        scores = {
            mode: {target: metric(result.loc[valid, f"observed_{target}"], result.loc[valid, column])
                   for target, column in zip(TARGETS, columns)}
            for mode, columns in PREDICTION_COLUMNS.items()
        }
        details = {**metadata, "configuration": name, "E": e, "H": h, "F": f,
                   "additive_state_head": is_factorial, "coupled_penalty": is_factorial,
                   "fused_temporal_weight": weight,
                   "runtime_seconds": time.perf_counter() - started, "test_metrics": scores}
        save_json(details, path)
        print(f"DONE seed={args.seed} {name}: fused arrivals={scores['fused'][TARGETS[0]]['wape']:.3f}% "
              f"nights={scores['fused'][TARGETS[1]]['wape']:.3f}% "
              f"seconds={details['runtime_seconds']:.1f}", flush=True)


def latex_table(path: Path, caption: str, label: str, rows: list[list[str]], note: str) -> None:
    """Write an IEEE-style ablation table."""
    measures = r"$\Delta$WAPE (pp) / $\Delta$MAE" if label == "tab:factorial_effects" else r"WAPE (\%) / MAE"
    header = (r"Control & \shortstack{Tourist Arrivals\\" + measures
              + r"} & \shortstack{Overnight Stays\\" + measures + r"} \\")
    text = [r"\begin{table}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
            rf"\label{{{label}}}", r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
            r"\renewcommand{\arraystretch}{1.08}",
            r"\begin{tabular}{>{\raggedright\arraybackslash}p{1.5cm}"
            r">{\centering\arraybackslash}p{3.1cm}>{\centering\arraybackslash}p{3.1cm}}",
            r"\toprule", header, r"\midrule"]
    text.extend(" & ".join(row) + r" \\" for row in rows)
    text.extend([r"\bottomrule", r"\end{tabular}", r"\par\vspace{3pt}",
                 r"\begin{minipage}{\columnwidth}", r"\scriptsize\raggedright " + note,
                 r"\end{minipage}", r"\end{table}"])
    path.write_text("\n".join(text) + "\n", encoding="utf-8")


def summarize(args: argparse.Namespace) -> None:
    """Verify keyed forecasts and summarize paired effects across seeds."""
    records, reference = [], None
    for seed in args.seeds:
        directory = args.output / f"seed_{seed}"
        full_info = json.loads((directory / "E1H1F1.json").read_text("utf-8"))
        for name in ["plain_dlinear", *CELLS]:
            info = json.loads((directory / f"{name}.json").read_text("utf-8"))
            frame = pd.read_csv(directory / f"{name}.csv")
            check = frame[KEYS + ["origin", "horizon", "observed_pair",
                                  "observed_arrivals_total", "observed_nights_total"]]
            if reference is None:
                reference = check
            else:
                pd.testing.assert_frame_equal(reference, check)
            if frame.duplicated(KEYS).any():
                raise ValueError(f"Duplicate keys: {name}/{seed}")
            valid = frame.observed_pair
            if info["fused_temporal_weight"] != full_info["fused_temporal_weight"]:
                raise ValueError(f"Inconsistent fusion weight: {seed}/{name}")
            for mode, columns in PREDICTION_COLUMNS.items():
                for target, prediction in zip(TARGETS, columns):
                    score = metric(frame.loc[valid, f"observed_{target}"], frame.loc[valid, prediction])
                    expected = info["test_metrics"][mode][target]
                    if score["n"] != expected["n"] or not all(
                        np.isclose(score[measure], expected[measure]) for measure in ("wape", "mae")
                    ):
                        raise ValueError(f"Score mismatch: {seed}/{name}/{mode}/{target}")
                    records.append({"seed": seed, "configuration": name, "target": target, "mode": mode,
                                    **score, "E": info["E"], "H": info["H"], "F": info["F"]})
    metrics = pd.DataFrame(records)
    save_csv(metrics, args.output / "seed_metrics.csv")
    summary = metrics.groupby(["mode", "configuration", "target"], sort=False).agg(
        wape_mean=("wape", "mean"), wape_sd=("wape", "std"),
        mae_mean=("mae", "mean"), mae_sd=("mae", "std"), seeds=("seed", "nunique"))
    save_csv(summary.reset_index(), args.output / "summary.csv")
    contrasts = []
    lookup = metrics.set_index(["seed", "configuration", "target", "mode"])
    for seed, target, factor, mode in itertools.product(args.seeds, TARGETS, ["E", "H", "F"], PREDICTION_COLUMNS):
        index = ["E", "H", "F"].index(factor)
        for name, flags in CELLS.items():
            if flags[index]:
                continue
            switched = list(flags)
            switched[index] = 1
            on = f"E{switched[0]}H{switched[1]}F{switched[2]}"
            off_score, on_score = lookup.loc[(seed, name, target, mode)], lookup.loc[(seed, on, target, mode)]
            contrasts.append({"seed": seed, "target": target, "factor": factor, "mode": mode,
                              "off": name, "on": on,
                              "delta_wape_pp": on_score.wape - off_score.wape,
                              "delta_mae": on_score.mae - off_score.mae})
    conditional = pd.DataFrame(contrasts)
    save_csv(conditional, args.output / "conditional_contrasts.csv")
    # Average the four fixed-background contrasts within each seed first.
    averaged = conditional.groupby(["seed", "target", "factor", "mode"], as_index=False)[
        ["delta_wape_pp", "delta_mae"]].mean()
    save_csv(averaged, args.output / "seed_average_effects.csv")
    effects = averaged.groupby(["mode", "factor", "target"]).agg(
        delta_wape_mean=("delta_wape_pp", "mean"), delta_wape_sd=("delta_wape_pp", "std"),
        delta_mae_mean=("delta_mae", "mean"), delta_mae_sd=("delta_mae", "std"),
        seeds_improved=("delta_wape_pp", lambda values: int(values.lt(0).sum())))
    save_csv(effects.reset_index(), args.output / "average_effects.csv")
    tables = args.tables
    tables.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ["plain_dlinear", *CELLS]:
        cells = []
        for target in TARGETS:
            score = summary.loc[("temporal", name, target)]
            cells.append(r"\shortstack{" + f"{score.wape_mean:.2f} $\\pm$ {score.wape_sd:.2f}\\%"
                         + r"\\{\scriptsize " + f"{score.mae_mean:,.0f} $\\pm$ {score.mae_sd:,.0f}" + "}}")
        label = "Plain DLinear" if name == "plain_dlinear" else f"{CELLS[name][0]}/{CELLS[name][1]}/{CELLS[name][2]}"
        rows.append([label, *cells])
    note = (r"Factorial rows show E/H/F: E enables both exogenous streams; H uses a train-fitted, "
            r"forward-filtered HMM state instead of constant state 0; F enables multiplicative FiLM gains. "
            r"The additive state head is enabled in all eight cells, including H=0. Plain DLinear "
            r"disables that head and the coupled loss and is outside the factorial. "
            r"The eight factorial cells use the same coupled loss. The table scores raw temporal forecasts. "
            r"Cells show WAPE (\%) above MAE (arrivals or nights), as mean $\pm$ sample SD across seeds "
            + ", ".join(map(str, args.seeds)) + r". All runs score the same 4,522 observed pairs. "
            r"SD describes seed variability, not sampling uncertainty. Inputs remain retrospective.")
    latex_table(tables / "factorial_ablation.tex",
                "Bias-controlled factorial temporal ablation with three training seeds",
                "tab:factorial_ablation", rows, note)
    rows = []
    for mode, factor in itertools.product(PREDICTION_COLUMNS, "EHF"):
        label = f"{mode.title()} {factor}"
        cells = []
        for target in TARGETS:
            effect = effects.loc[(mode, factor, target)]
            cells.append(r"\shortstack{" + f"{effect.delta_wape_mean:+.2f} $\\pm$ {effect.delta_wape_sd:.2f}"
                         + r"\\{\scriptsize " + f"{effect.delta_mae_mean:+,.0f} $\\pm$ {effect.delta_mae_sd:,.0f}" + "}}")
        rows.append([label, *cells])
    latex_table(tables / "factorial_effects.tex",
                "Average conditional effects of enabling each temporal component",
                "tab:factorial_effects", rows,
                r"Each seed averages four on-minus-off contrasts, holding the other two flags fixed. "
                r"Cells report $\Delta$WAPE in percentage points above $\Delta$MAE in arrivals or nights; "
                r"negative values indicate lower error. Values are mean $\pm$ sample SD over the three seeds. "
                r"E denotes exogenous projections, H denotes HMM states, and F denotes FiLM gains. "
                r"The fused forecasts share the full cell's validation weight, spatial forecast, and projection within each seed. "
                r"The SD describes seed variability.")
    print(summary.to_string())
    print("\nAVERAGE CONDITIONAL EFFECTS\n", effects.to_string())


def main() -> None:
    """Run one seed or regenerate tables for completed seeds."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "src/data/tourism_europe_enriched.csv")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 43, 44])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--tables", type=Path, default=ROOT / "paper/tables")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--summarize", action="store_true")
    args = parser.parse_args()
    summarize(args) if args.summarize else run(args)


if __name__ == "__main__":
    main()
