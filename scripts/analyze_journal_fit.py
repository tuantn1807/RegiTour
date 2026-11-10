"""Income, latent-state, and LOS diagnostics from retained keyed forecasts.

No forecasting model is retrained. HMM states are reconstructed with the same
training-only filter and seed as the reference experiment, for an issuance-state
audit. Income groups use a frozen FY2024 crosswalk and never enter the models.
"""

from datetime import date
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.reviewer_experiments import FOLDS, KEYS, TARGETS, filtered_regimes, load_panel, metric
from scripts.analyze_reviewer_experiments import check_alignment, table

OUT = ROOT / "results/journal_fit"
RUNS = ROOT / "results/reviewer_experiments"
TABLES = ROOT / "paper/tables"
PERIODS = {"tranquil": "Pre-pandemic 2018--2019", "shock": "Pandemic 2020--2021",
           "recovery": "Recovery 2022--2023", "recent": "Recent 2024--July 2026"}

# Income-column transcription for the study destinations; no region labels used.
# Evidence: FY24 classification table, pages 1--4, hosted by OTHM and identifying
# World Bank Country and Lending Groups as its source. The official FY24 release
# independently fixes the classification period and the July 2023 effective date.
HIGH = "AT BE CH CY CZ DE DK EE EL ES FI FR HR HU IE IS IT LI LT LU LV MT NL NO PL PT RO SE SI SK UK".split()
UPPER = "AL BG ME MK RS TR XK".split()
SOURCE_PAGES = {
    **dict.fromkeys("AL AT BE BG CH CY CZ DE DK".split(), 1),
    **dict.fromkeys("EE EL ES FI FR HR HU IE IS IT LI LT LU LV UK".split(), 2),
    **dict.fromkeys("ME MK MT NL NO PL PT RO RS".split(), 3),
    **dict.fromkeys("SE SI SK TR XK".split(), 4),
}
OFFICIAL = "https://blogs.worldbank.org/en/opendata/new-world-bank-group-country-classifications-income-level-fy24"
REPRINT = "https://othm.org.uk/doc/march-23/Current-region-classification-2023-24.pdf"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(fold, name):
    frame = pd.read_csv(RUNS / fold / f"{name}.csv", parse_dates=["date", "origin"])
    if frame.duplicated(KEYS).any():
        raise ValueError(f"Duplicate keys: {fold}/{name}")
    return frame


def count_scores(frame: pd.DataFrame) -> dict[str, int | float | None]:
    """Score observed count pairs and leave empty samples unscored."""
    valid = frame.observed_pair
    out = {"grid_rows": len(frame), "observed_pairs": int(valid.sum())}
    for target, pred in zip(TARGETS, ["pred_arrivals", "pred_nights"]):
        result = metric(frame.loc[valid, f"observed_{target}"], frame.loc[valid, pred])
        if result["n"] == 0:
            result["wape"] = None
        out.update({f"{target}_{key}": value for key, value in result.items()})
    return out


def los_scores(frame):
    eligible = frame.observed_pair & frame.observed_arrivals_total.gt(0)
    defined = eligible & frame.pred_arrivals.gt(0)
    truth = frame.loc[defined, "observed_nights_total"] / frame.loc[defined, "observed_arrivals_total"]
    pred = frame.loc[defined, "pred_nights"] / frame.loc[defined, "pred_arrivals"]
    error = pred - truth
    return {"eligible_pairs": int(eligible.sum()), "defined_pairs": int(defined.sum()),
            "undefined_pairs": int((eligible & ~defined).sum()),
            "los_mae": float(error.abs().mean()), "los_rmse": float(np.sqrt(error.pow(2).mean())),
            "los_bias": float(error.mean()), "actual_los_mean": float(truth.mean()),
            "predicted_los_mean": float(pred.mean())}


def main():
    torch.set_num_threads(1)
    OUT.mkdir(parents=True, exist_ok=True)
    source = ROOT / "src/data/tourism_europe_enriched.csv"
    raw = pd.read_csv(source, parse_dates=["date"])
    geos = set(raw.geo.unique())
    assert set(HIGH).isdisjoint(UPPER) and set(HIGH) | set(UPPER) == geos
    assert geos == set(SOURCE_PAGES)
    crosswalk = pd.DataFrame({"geo": sorted(geos)})
    crosswalk["country"] = crosswalk.geo.map(raw.groupby("geo").country.first())
    crosswalk["income_group"] = crosswalk.geo.map({**dict.fromkeys(HIGH, "High income"),
                                                   **dict.fromkeys(UPPER, "Upper middle income")})
    crosswalk["classification_fy"] = 2024
    crosswalk["effective_from"] = "2023-07-01"
    crosswalk["source_table_page"] = crosswalk.geo.map(SOURCE_PAGES)
    crosswalk.to_csv(OUT / "income_groups_fy2024.csv", index=False)
    groups = crosswalk.set_index("geo").income_group
    frames = {name: read("recent", name) for name in ["snaive", "dlinear", "full", "full_raw"]}
    check_alignment(frames)
    income = []
    country_metrics = []
    for name, frame in frames.items():
        total = frame.loc[frame.accommodation_type.eq("total")].copy()
        total["income_group"] = total.geo.map(groups)
        assert total.income_group.notna().all()
        for group, subset in total.groupby("income_group"):
            income.append({"income_group": group, "model": name, "destinations": subset.geo.nunique(),
                           **count_scores(subset), **los_scores(subset)})
        for geo, subset in total.groupby("geo"):
            country_metrics.append({"geo": geo, "income_group": groups.loc[geo], "model": name,
                                    **count_scores(subset), **los_scores(subset)})
    country_metrics = pd.DataFrame(country_metrics)
    country_metrics.to_csv(OUT / "income_country_metrics.csv", index=False)
    group_keys = ["income_group", "model"]
    scored = country_metrics.loc[country_metrics.observed_pairs.gt(0)]
    country_summary = scored.groupby(group_keys, as_index=False).agg(
        scored_destinations=("geo", "nunique"),
        country_mean_arrivals_wape=("arrivals_total_wape", "mean"),
        country_mean_nights_wape=("nights_total_wape", "mean"),
    )
    income = pd.DataFrame(income).merge(country_summary, on=group_keys, how="left", validate="one_to_one")
    income["scored_destinations"] = income["scored_destinations"].fillna(0).astype(int)
    income.to_csv(OUT / "income_metrics.csv", index=False)
    leave_one_out = []
    full_total = frames["full"].loc[frames["full"].accommodation_type.eq("total")].copy()
    full_total["income_group"] = full_total.geo.map(groups)
    for group, subset in full_total.groupby("income_group"):
        all_scores = los_scores(subset)
        group_abs_error = all_scores["los_mae"] * all_scores["defined_pairs"]
        for geo, country in subset.groupby("geo"):
            country_scores = los_scores(country)
            remainder = los_scores(subset.loc[subset.geo.ne(geo)])
            leave_one_out.append({"income_group": group, "omitted_geo": geo,
                                  "group_los_mae": all_scores["los_mae"],
                                  "omitted_abs_los_error_share_pct": 100 * country_scores["los_mae"] * country_scores["defined_pairs"] / group_abs_error,
                                  **{f"remaining_{k}": v for k, v in remainder.items()}})
    pd.DataFrame(leave_one_out).to_csv(OUT / "income_los_leave_one_out.csv", index=False)
    full_income = income.loc[income.model.eq("full")].set_index("income_group")
    assert full_income["observed_pairs"].sum() == 1129
    high_scored = full_income.loc["High income", "scored_destinations"]
    upper_scored = full_income.loc["Upper middle income", "scored_destinations"]
    rows = []
    for group, short in [("High income", "High"), ("Upper middle income", "Upper-middle")]:
        for name, label in [("snaive", "S. naive"), ("dlinear", "DLinear"), ("full", "Full")]:
            row = income.loc[income.income_group.eq(group) & income.model.eq(name)].iloc[0]
            cells = [r"\shortstack{" + f"{row[target + '_wape']:.2f}\\%"
                     + r"\\{\scriptsize " + f"{row[target + '_mae']:,.0f}" + "}}" for target in TARGETS]
            rows.append([f"{short}: {label}", str(row.observed_pairs), *cells])
    table(TABLES / "journal_income.tex", "Tourism forecast accuracy by fixed World Bank FY2024 income group, January 2024--July 2026.",
          "tab:income_groups", ["Income Group and Model", r"\shortstack{Pairs\\(n)}",
                                 r"\shortstack{Tourist Arrivals\\WAPE (\%) / MAE}",
                                 r"\shortstack{Overnight Stays\\WAPE (\%) / MAE}"], rows,
          r">{\raggedright\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{0.8cm}"
          r">{\centering\arraybackslash}p{2.3cm}>{\centering\arraybackslash}p{2.3cm}",
          notes=r"The FY2024 panel contains 31 high-income and seven upper-middle-income destinations. "
          f"The scores cover {high_scored} and {upper_scored} destinations, respectively. "
          r"All models use the same observed country-month pairs in the aggregate accommodation category. "
          r"Each target cell gives pooled WAPE above MAE. S. naive denotes seasonal naive. Full denotes RA-HE-DLinear.")

    # Common-denominator and model-complete LOS diagnostics, kept separate.
    los = []
    stability = []
    labels = pd.read_csv(RUNS / "country_groups.csv").set_index("geo").los_stability
    for fold in FOLDS:
        pair = {name: read(fold, name) for name in ["dlinear", "full"]}
        check_alignment(pair)
        shared = pair["full"].accommodation_type.eq("total") & pair["full"].observed_pair
        shared &= pair["full"].observed_arrivals_total.gt(0)
        shared &= pair["full"].pred_arrivals.gt(0) & pair["dlinear"].pred_arrivals.gt(0)
        for name, frame in pair.items():
            total = frame.loc[frame.accommodation_type.eq("total")]
            los.append({"fold": fold, "model": name, "sample": "all eligible", **los_scores(total)})
            los.append({"fold": fold, "model": name, "sample": "common defined", **los_scores(frame.loc[shared])})
            if fold == "recent":
                for group in ["low", "middle", "high"]:
                    membership = frame.geo.map(labels).eq(group)
                    stability.append({"los_variation_tercile": group, "model": name,
                                      "sample": "all eligible", **los_scores(frame.loc[membership & frame.accommodation_type.eq("total")])})
                    stability.append({"los_variation_tercile": group, "model": name,
                                      "sample": "common defined", **los_scores(frame.loc[membership & shared])})
    recent_snaive = frames["snaive"].loc[frames["snaive"].accommodation_type.eq("total")]
    los.append({"fold": "recent", "model": "snaive", "sample": "all eligible", **los_scores(recent_snaive)})
    pd.DataFrame(los).to_csv(OUT / "los_samples.csv", index=False)
    pd.DataFrame(stability).to_csv(OUT / "los_stability_metrics.csv", index=False)
    a = frames["full"].loc[frames["full"].accommodation_type.eq("total") & frames["full"].observed_pair]
    b = frames["full_raw"].loc[a.index]
    projection_change = float(np.max(np.abs(a.pred_nights / a.pred_arrivals - b.pred_nights / b.pred_arrivals)))
    pooled_bias = {target: float(100 * (a[pred] - a[f"observed_{target}"]).sum() / a[f"observed_{target}"].sum())
                   for target, pred in zip(TARGETS, ["pred_arrivals", "pred_nights"])}

    # Decode no new neural forecasts; audit the conditioning state at each origin.
    origin_rows, profiles = [], []
    for fold, train_end in FOLDS.items():
        run_manifest = json.loads((RUNS / fold / "manifest.json").read_text("utf-8"))
        assert run_manifest["data_sha256"] == sha(source)
        hashes = {key.replace("\\", "/"): value for key, value in run_manifest["source_hashes"].items()}
        assert hashes["scripts/reviewer_experiments.py"] == sha(ROOT / "scripts/reviewer_experiments.py")
        frame = load_panel(source, train_end)
        states = filtered_regimes(frame, train_end, 42)
        full = read(fold, "full")
        details = json.loads((RUNS / fold / "full.json").read_text("utf-8"))
        for origin, subset in full.groupby("origin"):
            state = int(states.loc[states.index <= origin].iloc[-1])
            origin_rows.append({"fold": fold, "calendar_period": PERIODS[fold], "train_end": train_end,
                                "origin": origin.isoformat(), "hmm_state": state,
                                "state_rank": {1: "low-growth centroid", 0: "middle-growth centroid", 2: "high-growth centroid"}[state],
                                "grid_rows": len(subset), "scored_pairs": int(subset.observed_pair.sum()),
                                "temporal_weight": details["w_temporal"]})
        total = frame.loc[frame.accommodation_type.eq("total")].groupby("date").agg(
            arrivals=("arrivals_total", "sum"), stringency=("covid_stringency_index", "mean"),
            price=("hicp_accommodation", "mean"))
        total["growth"] = total.arrivals.pct_change(12).fillna(0)
        total["hmm_state"] = states
        for state, subset in total.loc[total.index.year <= train_end].groupby("hmm_state"):
            profiles.append({"fold": fold, "hmm_state": state, "months": len(subset),
                             "arrival_yoy_mean": subset.growth.mean(), "mean_stringency": subset.stringency.mean(),
                             "mean_hicp": subset.price.mean()})
    origin_rows = pd.DataFrame(origin_rows)
    origin_rows.to_csv(OUT / "hmm_origin_audit.csv", index=False)
    pd.DataFrame(profiles).to_csv(OUT / "hmm_training_profiles.csv", index=False)
    rows = []
    for fold in FOLDS:
        subset = origin_rows.loc[origin_rows.fold.eq(fold)].sort_values("origin")
        rows.append([PERIODS[fold], ", ".join(str(v) for v in subset.hmm_state), f"{subset.temporal_weight.iloc[0]:.2f}"])
    table(TABLES / "journal_hmm_origins.tex", "Calendar evaluation periods and actual HMM conditioning states at January forecast origins.",
          "tab:hmm_calendar", ["Calendar Test Period", r"\shortstack{HMM State IDs\\at January Origins}", r"\shortstack{Temporal\\Fusion\\Weight}"], rows,
          r">{\raggedright\arraybackslash}p{3.2cm}>{\centering\arraybackslash}p{2.6cm}>{\centering\arraybackslash}p{1.8cm}",
          notes=r"IDs are ordered by fitted training growth centroids: 1 low, 0 middle, 2 high. "
          r"They are statistical states, not calendar event labels, and each fold fits its own HMM. "
          r"States are listed in chronological origin order, one per January forecast block. "
          r"The weight measures the share of the HMM-conditioned temporal branch in the blend. This is a retrospective reference-period audit.")
    metadata = {"analysis_date": date.today().isoformat(), "data_sha256": sha(source), "analysis_code_sha256": sha(Path(__file__)),
                "income_classification": {"fiscal_year": 2024, "effective_from": "2023-07-01",
                    "official_release": OFFICIAL, "row_evidence_reprint": REPRINT,
                    "accessed_on": "2026-11-08", "evidence_method": "Income column transcribed for the 38 study geographies; added region classifications ignored. The historical XLSX was not downloaded.",
                    "crosswalk_sha256": sha(OUT / "income_groups_fy2024.csv"), "counts": {"High income": 31, "Upper middle income": 7}},
                "recent_aggregate_observed_pairs": len(a), "recent_max_los_projection_change": projection_change,
                "recent_pooled_count_bias_pct": pooled_bias,
                "forecast_input_hashes": {f"{fold}/{name}.csv": sha(RUNS / fold / f"{name}.csv")
                                          for fold in FOLDS for name in ["dlinear", "full"]}}
    metadata["forecast_input_hashes"].update({f"recent/{name}.csv": sha(RUNS / "recent" / f"{name}.csv")
                                             for name in ["snaive", "full_raw"]})
    metadata["los_stability_groups_sha256"] = sha(RUNS / "country_groups.csv")
    (OUT / "manifest.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print("INCOME\n", income.to_string(index=False))
    print("HMM ORIGINS\n", origin_rows.to_string(index=False))
    print("LOS SAMPLES\n", pd.DataFrame(los).to_string(index=False))
    print("LOS STABILITY\n", pd.DataFrame(stability).to_string(index=False))
    print("MAX LOS PROJECTION CHANGE", projection_change)


if __name__ == "__main__":
    main()
