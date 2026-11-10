"""Create tourism diagnostics and manuscript tables from the keyed experiments."""

import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd

from scripts.reviewer_experiments import metric, save_csv, save_json, TARGETS


RESULTS = ROOT / "results/reviewer_experiments"
TABLES = ROOT / "paper/tables"
FIGURES = ROOT / "paper/figures"
PREDICTIONS = ["pred_arrivals", "pred_nights"]
ARRIVALS_ERROR_HEADER = r"\shortstack{Tourist Arrivals\\WAPE (\%)\\MAE (arrivals)}"
NIGHTS_ERROR_HEADER = r"\shortstack{Overnight Stays\\WAPE (\%)\\MAE (nights)}"
PHYSICAL_HEADER = r"\shortstack{Physical\\Violation\\Rate (\%)}"
COUNT_ERROR_NOTE = (
    r"WAPE is weighted absolute percentage error; MAE is mean absolute error. "
    r"Each target cell reports WAPE (\%) on its first line and MAE on its second line. "
    r"MAE is measured in accommodation arrivals for Tourist Arrivals and nights for Overnight Stays. "
    r"Physical violation rate is the percentage of all 4,712 forecast pairs with "
    r"$\hat{Y}^{\mathrm{nig}}<\hat{Y}^{\mathrm{arr}}$."
)
MODEL_NAMES = {
    "snaive": "Seasonal naive", "ets": "ETS / Holt--Winters", "sarima": "SARIMA",
    "sarimax": "SARIMAX", "lightgbm": "LightGBM", "dlinear": "DLinear",
    "exo": "DLinear + Exogenous", "hmm": "DLinear + HMM bias",
    "film_static": "DLinear + static FiLM", "hmm_film": "DLinear + HMM + FiLM",
    "hmm_film_trmf": "DLinear + HMM + FiLM + TRMF",
    "exo_hmm_film": "DLinear + Exogenous + HMM + FiLM",
    "full_no_consistency": "Full without coupled loss",
    "full_core": "Full temporal core", "full_raw": "Full before projection",
    "full": "Full RA-HE-DLinear",
}


def tex(value):
    return str(value).replace("&", r"\&").replace("_", r"\_").replace("%", r"\%")


def table(path, caption, label, headers, rows, columns, notes=None, wide=False):
    environment = "table*" if wide else "table"
    width = r"\textwidth" if wide else r"\columnwidth"
    lines = [rf"\begin{{{environment}}}[!htbp]", r"\centering", rf"\caption{{{caption}}}",
             rf"\label{{{label}}}", r"\small" if wide else r"\footnotesize", r"\setlength{\tabcolsep}{3pt}",
             r"\renewcommand{\arraystretch}{1.08}", rf"\begin{{tabular}}{{{columns}}}",
             r"\toprule", " & ".join(r"\textbf{" + header + "}" for header in headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(row) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if notes:
        lines.extend([r"\par\vspace{3pt}", rf"\begin{{minipage}}{{{width}}}",
                      r"\scriptsize\raggedright " + notes, r"\end{minipage}"])
    lines.append(rf"\end{{{environment}}}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_model(fold, model):
    frame = pd.read_csv(RESULTS / fold / f"{model}.csv", parse_dates=["date", "origin"])
    if frame.duplicated(["geo", "accommodation_type", "date"]).any():
        raise ValueError(f"Duplicate evaluation keys: {fold}/{model}")
    if not np.isfinite(frame[PREDICTIONS]).all().all():
        raise ValueError(f"Nonfinite evaluation predictions: {fold}/{model}")
    return frame


def scores(frame):
    valid = frame.observed_pair
    result = {"rows": len(frame), "observed_pairs": int(valid.sum())}
    for target, prediction in zip(TARGETS, PREDICTIONS):
        for key, value in metric(frame.loc[valid, f"observed_{target}"], frame.loc[valid, prediction]).items():
            result[f"{target}_{key}"] = value
    result["physical_violation_pct"] = float(100 * (frame.pred_nights < frame.pred_arrivals).mean())
    return result


def error_cell(row, target):
    wape = row[f"{target}_wape"]
    mae = row[f"{target}_mae"]
    return "\\shortstack{" + f"{wape:.2f}\\%" + r"\\{\scriptsize " + f"{mae:,.0f}" + "}}"


def country_labels(raw):
    total = raw.loc[raw.accommodation_type.eq("total")].copy()
    train = total.loc[total.year.le(2021) & total[TARGETS].notna().all(axis=1)]
    labels = pd.DataFrame(index=sorted(total.geo.unique()))
    labels["country"] = total.groupby("geo").country.first()
    labels["scale_value"] = train.groupby("geo").arrivals_total.mean()
    profile = train.groupby(["geo", "month"]).arrivals_total.mean()
    labels["seasonality_value"] = profile.groupby("geo").std() / profile.groupby("geo").mean()
    valid_los = train.loc[train.arrivals_total.gt(0)].copy()
    valid_los["los"] = valid_los.nights_total / valid_los.arrivals_total
    labels["los_stability_value"] = valid_los.groupby("geo").los.std() / valid_los.groupby("geo").los.mean()
    annual = total.loc[total.year.isin([2019, 2023])].groupby(["geo", "year"]).arrivals_total.agg(["sum", "count"])
    annual = annual.loc[annual["count"].eq(12)]["sum"].unstack("year")
    labels["recovery_value"] = annual[2023] / annual[2019].replace(0, np.nan)
    for field in ["scale", "seasonality", "los_stability", "recovery"]:
        values = labels[f"{field}_value"]
        labels[field] = pd.qcut(values.rank(method="first"), 3, labels=["low", "middle", "high"])
    return labels


def ratio_errors(frame):
    eligible = frame.observed_pair & frame.observed_arrivals_total.gt(0)
    valid = eligible & frame.pred_arrivals.gt(0)
    actual = frame.loc[valid, "observed_nights_total"] / frame.loc[valid, "observed_arrivals_total"]
    predicted = frame.loc[valid, "pred_nights"] / frame.loc[valid, "pred_arrivals"]
    errors = predicted - actual
    return {"n": int(valid.sum()), "undefined_predictions": int((eligible & ~valid).sum()),
            "los_mae": float(errors.abs().mean()) if len(errors) else None,
            "los_rmse": float(np.sqrt(np.mean(errors ** 2))) if len(errors) else None,
            "observed_los_mean": float(actual.mean()) if len(errors) else None,
            "predicted_los_mean": float(predicted.mean()) if len(errors) else None}


def main():
    TABLES.mkdir(exist_ok=True)
    raw = pd.read_csv(ROOT / "src/data/tourism_europe_enriched.csv", parse_dates=["date"])
    labels = country_labels(raw)
    save_csv(labels.reset_index(names="geo"), RESULTS / "country_groups.csv")
    models = ["snaive", "ets", "sarima", "sarimax", "lightgbm", *[
        "dlinear", "exo", "hmm", "film_static", "hmm_film", "hmm_film_trmf",
        "exo_hmm_film", "full_no_consistency", "full_core", "full_raw", "full"]]
    frames = {name: read_model("recent", name) for name in models}
    reference_keys = frames["full"][["geo", "accommodation_type", "date"]]
    for name, frame in frames.items():
        if not reference_keys.equals(frame[["geo", "accommodation_type", "date"]]):
            raise ValueError(f"Evaluation keys differ for {name}")
    benchmark = pd.DataFrame([{"model": name, **scores(frame)} for name, frame in frames.items()])
    save_csv(benchmark, RESULTS / "benchmark.csv")
    by_model = benchmark.set_index("model")
    primary = ["snaive", "ets", "sarima", "sarimax", "lightgbm", "dlinear", "full"]
    table(TABLES / "reviewer_benchmark.tex", "Eurostat benchmark performance for January 2024--July 2026 with common January forecast origins and a maximum 12-month horizon.",
          "tab:reviewer_benchmark", ["Benchmark Model", ARRIVALS_ERROR_HEADER, NIGHTS_ERROR_HEADER, PHYSICAL_HEADER],
          [[MODEL_NAMES[n], error_cell(by_model.loc[n], TARGETS[0]), error_cell(by_model.loc[n], TARGETS[1]),
            f"{by_model.loc[n, 'physical_violation_pct']:.3f}"] for n in primary],
          r">{\raggedright\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{1.4cm}",
          notes=COUNT_ERROR_NOTE + " Error metrics use 4,522 originally observed target pairs.")
    fallbacks = []
    for name in ["ets", "sarima", "sarimax"]:
        details = json.loads((RESULTS / "recent" / f"{name}.json").read_text(encoding="utf-8"))
        fallbacks.append(f"{MODEL_NAMES[name]} {details['seasonal_fallback_fits']}/304")
    (TABLES / "reviewer_fit_discussion.tex").write_text(
        "The number of destination--category--target fits using the predeclared seasonal fallback is "
        + ", ".join(fallbacks) + ". These counts include constant or unsupported training series. "
        "Per-fit diagnostics are retained so the contribution of fallback forecasts can be examined.\n", encoding="utf-8")
    ablations = ["dlinear", "exo", "hmm", "film_static", "hmm_film", "hmm_film_trmf",
                 "exo_hmm_film", "full_no_consistency", "full_raw", "full"]
    table(TABLES / "reviewer_ablation.tex", "Single-seed component configuration ladder on the matched Eurostat test set.",
          "tab:reviewer_ablation", ["Ablation Configuration", ARRIVALS_ERROR_HEADER, NIGHTS_ERROR_HEADER, PHYSICAL_HEADER],
          [[MODEL_NAMES[n], error_cell(by_model.loc[n], TARGETS[0]), error_cell(by_model.loc[n], TARGETS[1]),
            f"{by_model.loc[n, 'physical_violation_pct']:.3f}"] for n in ablations],
          r">{\raggedright\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{1.4cm}",
          notes=COUNT_ERROR_NOTE + " All configurations use the same 4,522 observed scoring pairs and seed 42. "
          "Fusion weights are tuned separately; all TRMF-equipped rows select zero temporal weight. "
          "This ladder does not isolate individual temporal contributions to the fused forecast.")

    stress_rows, dynamics, fold_frames = [], [], {}
    phase_labels = {"tranquil": "Pre-pandemic 2018--2019", "shock": "Pandemic 2020--2021",
                    "recovery": "Recovery 2022--2023", "recent": "Recent 2024--2026"}
    for fold in ["tranquil", "shock", "recovery", "recent"]:
        fold_frames[fold] = {model: read_model(fold, model) for model in ["dlinear", "full"]}
        for model, frame in fold_frames[fold].items():
            stress_rows.append({"phase": fold, "model": model, **scores(frame)})
        frame = fold_frames[fold]["full"]
        country_total = frame.loc[frame.accommodation_type.eq("total")].copy()
        country_total.loc[~country_total.observed_pair, [f"observed_{c}" for c in TARGETS]] = np.nan
        growth = []
        for geo, country in country_total.groupby("geo"):
            country = country.sort_values("date")
            delta = np.log1p(country[[f"observed_{c}" for c in TARGETS]]).diff().dropna()
            growth.append([delta.iloc[:, 0].std(), delta.iloc[:, 1].std(), delta.iloc[:, 0].corr(delta.iloc[:, 1])])
        values = np.nanmean(growth, axis=0)
        dynamics.append({"phase": fold, "arrivals_log_growth_sd": values[0],
                         "nights_log_growth_sd": values[1], "growth_correlation": values[2]})
    stress = pd.DataFrame(stress_rows)
    save_csv(stress, RESULTS / "regime_metrics.csv")
    save_csv(pd.DataFrame(dynamics), RESULTS / "demand_dynamics.csv")
    table(TABLES / "reviewer_regimes.tex", "Forecast accuracy across chronological tourism stress-test periods. Each test period follows its own training, validation, and calibration partitions.",
          "tab:reviewer_regimes", ["Calendar Test Period", "Benchmark Model", ARRIVALS_ERROR_HEADER, NIGHTS_ERROR_HEADER],
          [[phase_labels[r.phase], "DLinear" if r.model == "dlinear" else "RA-HE-DLinear",
            error_cell(r, TARGETS[0]), error_cell(r, TARGETS[1])] for _, r in stress.iterrows()],
          r">{\raggedright\arraybackslash}p{1.8cm}>{\raggedright\arraybackslash}p{1.7cm}>{\centering\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{2.2cm}",
          notes=r"Each target cell reports weighted absolute percentage error (WAPE, \%) above mean absolute error (MAE). MAE units are arrivals and nights, respectively. The recent period ends in July 2026. Period names are calendar labels, not decoded HMM states. Scores use originally observed target pairs in each fold.")
    table(TABLES / "reviewer_dynamics.tex", "Monthly tourism-demand volatility and co-movement of tourist arrivals and overnight stays, averaged across destinations.",
          "tab:reviewer_dynamics", ["Calendar Period", r"\shortstack{Tourist Arrivals\\Log-growth SD\\(unitless)}",
                                    r"\shortstack{Overnight Stays\\Log-growth SD\\(unitless)}", r"\shortstack{Growth\\Correlation\\(Pearson $r$)}"],
          [[phase_labels[r["phase"]], f"{r['arrivals_log_growth_sd']:.3f}",
            f"{r['nights_log_growth_sd']:.3f}", f"{r['growth_correlation']:.3f}"] for r in dynamics],
          r">{\raggedright\arraybackslash}p{2.0cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{1.7cm}",
          notes=r"SD is the standard deviation of consecutive monthly changes in $\log(1+Y)$. Pearson $r$ measures the correlation between arrivals and overnight-stays log growth. Both statistics are unitless and are calculated per destination before averaging. Only aggregate accommodation series are used.")

    transitions = []
    for fold, months, years in [("shock", [3, 4, 5], [2020, 2021]),
                               ("recovery", [1, 2, 3], [2022, 2023])]:
        for model, frame in fold_frames[fold].items():
            for year in years:
                subset = frame.loc[frame.accommodation_type.eq("total") & frame.date.dt.year.eq(year)
                                   & frame.date.dt.month.isin(months)]
                transitions.append({"fold": fold, "year": year, "model": model, **scores(subset)})
    save_csv(pd.DataFrame(transitions), RESULTS / "transition_metrics.csv")

    group_metrics = []
    for field in ["scale", "seasonality", "recovery", "los_stability"]:
        for group in ["low", "middle", "high"]:
            geos = labels.index[labels[field].eq(group)]
            for model in ["dlinear", "full"]:
                frame = frames[model]
                subset = frame.loc[frame.accommodation_type.eq("total") & frame.geo.isin(geos)]
                group_metrics.append({"grouping": field, "group": group, "countries": len(geos),
                                      "model": model, **scores(subset)})
    group_metrics = pd.DataFrame(group_metrics)
    save_csv(group_metrics, RESULTS / "group_metrics.csv")
    rows = []
    for field in ["scale", "seasonality", "recovery"]:
        for group in ["low", "middle", "high"]:
            selected = group_metrics.loc[group_metrics.grouping.eq(field) & group_metrics.group.eq(group)].set_index("model")
            description = {"scale": "Tourism demand scale", "seasonality": "Seasonality strength",
                           "recovery": "Pre-test recovery index"}[field]
            rows.append([f"{description}, {group}", str(int(selected.loc['full', 'countries'])),
                         f"{selected.loc['dlinear', 'arrivals_total_wape']:.2f}",
                         f"{selected.loc['dlinear', 'nights_total_wape']:.2f}",
                         f"{selected.loc['full', 'arrivals_total_wape']:.2f}",
                         f"{selected.loc['full', 'nights_total_wape']:.2f}"])
    table(TABLES / "reviewer_groups.tex", "Destination-group forecasting performance on aggregate accommodation series in the recent Eurostat test period.",
          "tab:reviewer_groups", ["Destination Group", r"\shortstack{Destinations\\(n)}",
              r"\shortstack{DLinear\\Tourist Arrivals\\WAPE (\%)}", r"\shortstack{DLinear\\Overnight Stays\\WAPE (\%)}",
              r"\shortstack{RA-HE-DLinear\\Tourist Arrivals\\WAPE (\%)}", r"\shortstack{RA-HE-DLinear\\Overnight Stays\\WAPE (\%)}"], rows,
          r">{\raggedright\arraybackslash}p{3.4cm}>{\centering\arraybackslash}p{2.0cm}*{4}{>{\centering\arraybackslash}p{2.6cm}}",
          notes=r"Low, middle, and high groups are terciles. Demand scale and seasonality use 2010--2021 training data. The pre-test recovery index is the ratio of 2023 to 2019 annual arrivals, requiring 12 observed months in both years. Destination counts vary with data eligibility. Every error column reports WAPE in percent, with the forecast target and benchmark model identified explicitly.",
          wide=True)
    group_text = []
    for field in ["scale", "seasonality", "recovery", "los_stability"]:
        subset = group_metrics.loc[group_metrics.grouping.eq(field) & group_metrics.model.eq("full")]
        low = subset.loc[subset.group.eq("low")].iloc[0]
        high = subset.loc[subset.group.eq("high")].iloc[0]
        name = {"scale": "demand scale", "seasonality": "seasonality strength",
                "recovery": "recovery ratios", "los_stability": "training LOS variation"}[field]
        group_text.append(
            f"For low versus high {name}, full-model arrivals WAPE is "
            f"{low.arrivals_total_wape:.2f}\\% versus {high.arrivals_total_wape:.2f}\\%, "
            f"and nights WAPE is {low.nights_total_wape:.2f}\\% versus {high.nights_total_wape:.2f}\\%."
        )
    (TABLES / "reviewer_group_discussion.tex").write_text("\n\n".join(group_text) + "\n", encoding="utf-8")

    los_rows, country_los, paired_los = [], [], []
    for fold, forecasts in fold_frames.items():
        for model, frame in forecasts.items():
            total = frame.loc[frame.accommodation_type.eq("total")]
            los_rows.append({"fold": fold, "model": model, **ratio_errors(total)})
            if fold == "recent":
                for geo, country in total.groupby("geo"):
                    country_los.append({"geo": geo, "country": labels.loc[geo, "country"],
                                        "model": model, **ratio_errors(country)})
        baseline = forecasts["dlinear"]
        full = forecasts["full"]
        shared = baseline.accommodation_type.eq("total") & baseline.observed_pair & baseline.observed_arrivals_total.gt(0)
        shared &= baseline.pred_arrivals.gt(0) & full.pred_arrivals.gt(0)
        for model, frame in forecasts.items():
            result = ratio_errors(frame.loc[shared])
            total_result = ratio_errors(frame.loc[frame.accommodation_type.eq("total")])
            result["undefined_predictions"] = total_result["undefined_predictions"]
            paired_los.append({"fold": fold, "model": model, **result})
    raw_total = frames["full_raw"].loc[frames["full_raw"].accommodation_type.eq("total")]
    los_rows.append({"fold": "recent", "model": "full_raw", **ratio_errors(raw_total)})
    recent_baseline = frames["dlinear"]
    shared_raw = raw_total.observed_pair & raw_total.observed_arrivals_total.gt(0)
    shared_raw &= raw_total.pred_arrivals.gt(0) & recent_baseline.loc[raw_total.index, "pred_arrivals"].gt(0)
    paired_los.append({"fold": "recent", "model": "full_raw", **ratio_errors(raw_total.loc[shared_raw])})
    save_csv(pd.DataFrame(los_rows), RESULTS / "los_metrics.csv")
    save_csv(pd.DataFrame(paired_los), RESULTS / "los_paired_metrics.csv")
    save_csv(pd.DataFrame(country_los), RESULTS / "country_los.csv")
    table(TABLES / "reviewer_los.tex", "Accuracy of the directly predicted length-of-stay ratio on common defined denominators.",
          "tab:reviewer_los", ["Evaluation Period and Model", r"\shortstack{LOS\\MAE\\(nights per\\arrival)}",
                               r"\shortstack{LOS\\RMSE\\(nights per\\arrival)}", r"\shortstack{Paired\\Sample\\Size\\(n)}",
                               r"\shortstack{Undefined\\LOS\\Predictions\\(n)}"],
          [[phase_labels[r["fold"]] + (" Raw" if r["model"] == "full_raw" else " Full" if r["model"] == "full" else " DLinear"),
            f"{r['los_mae']:.3f}", f"{r['los_rmse']:.3f}", str(r["n"]), str(r["undefined_predictions"])] for r in paired_los],
          r">{\raggedright\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{1.45cm}>{\centering\arraybackslash}p{1.45cm}>{\centering\arraybackslash}p{1.1cm}>{\centering\arraybackslash}p{1.6cm}",
          notes=r"Length of stay (LOS) is overnight stays divided by tourist arrivals for matching destination, accommodation category, and month. Mean absolute error (MAE) and root mean squared error (RMSE) are expressed in nights per arrival. Paired sample size counts outcomes with a defined predicted LOS for both DLinear and the full model. Undefined counts refer to all eligible observed outcomes before this intersection. Full denotes RA-HE-DLinear after projection, and Raw denotes the full model before projection.")

    projection = []
    for model in ["full_raw", "full"]:
        row = by_model.loc[model]
        projection.append(["Before" if model == "full_raw" else "After", error_cell(row, TARGETS[0]),
                           error_cell(row, TARGETS[1]), f"{row.physical_violation_pct:.3f}"])
    table(TABLES / "reviewer_projection.tex", "Effect of physical-consistency projection on count errors and the rate of physically inconsistent forecast pairs.",
          "tab:reviewer_projection", ["Projection Stage", ARRIVALS_ERROR_HEADER, NIGHTS_ERROR_HEADER, PHYSICAL_HEADER], projection,
          r">{\raggedright\arraybackslash}p{2.2cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{1.4cm}",
          notes=COUNT_ERROR_NOTE + r" Before denotes the raw fused forecast. After denotes application of $\Pi_{\mathcal{C}}$ to the same forecast pairs.")

    intervals = pd.read_csv(RESULTS / "recent/intervals.csv", parse_dates=["date", "origin"])
    coverage = []
    for target, suffix in zip(TARGETS, ["arrivals", "nights"]):
        valid = intervals.observed_pair
        truth = intervals.loc[valid, f"observed_{target}"]
        lower, upper = intervals.loc[valid, f"lower_{suffix}"], intervals.loc[valid, f"upper_{suffix}"]
        coverage.append({"target": target, "n": int(valid.sum()), "coverage_pct": float(100 * ((truth >= lower) & (truth <= upper)).mean()),
                         "mean_width": float((upper - lower).mean())})
    save_csv(pd.DataFrame(coverage), RESULTS / "coverage.csv")
    table(TABLES / "reviewer_coverage.tex", "Residual-conformal prediction-interval coverage and mean interval width after physical projection, calibrated on held-out 2023 outcomes.",
          "tab:reviewer_coverage", ["Forecast Target", r"\shortstack{Nominal\\Coverage\\(\%)}", r"\shortstack{Empirical\\Coverage\\(\%)}",
                                    r"\shortstack{Mean Interval\\Width\\(count units)}", r"\shortstack{Scored\\Outcomes\\(n)}"],
          [["Tourist Arrivals" if r["target"] == TARGETS[0] else "Overnight Stays", "90.00", f"{r['coverage_pct']:.2f}", f"{r['mean_width']:,.0f}", f"{r['n']:,}"] for r in coverage],
          r">{\raggedright\arraybackslash}p{1.8cm}>{\centering\arraybackslash}p{1.3cm}>{\centering\arraybackslash}p{1.4cm}>{\centering\arraybackslash}p{1.9cm}>{\centering\arraybackslash}p{1.3cm}",
          notes=r"Empirical coverage is the percentage of originally observed outcomes lying inside their prediction intervals. Mean interval width is measured in accommodation arrivals for Tourist Arrivals and nights for Overnight Stays. Scored outcomes is the number of observed target pairs used for each coverage estimate.")

    cases = []
    for group in ["low", "middle", "high"]:
        candidates = labels.loc[labels.recovery.eq(group)].sort_values("recovery_value")
        geo = candidates.index[len(candidates) // 2]
        cases.append(geo)
    case_rows = []
    for geo in cases:
        full = frames["full"].loc[frames["full"].geo.eq(geo) & frames["full"].accommodation_type.eq("total")]
        baseline = frames["dlinear"].loc[frames["dlinear"].geo.eq(geo) & frames["dlinear"].accommodation_type.eq("total")]
        f, b = scores(full), scores(baseline)
        case_rows.append({"geo": geo, "country": labels.loc[geo, "country"],
                          "recovery_ratio": float(labels.loc[geo, "recovery_value"]),
                          "dlinear_arrivals_wape": b["arrivals_total_wape"], "full_arrivals_wape": f["arrivals_total_wape"],
                          "dlinear_nights_wape": b["nights_total_wape"], "full_nights_wape": f["nights_total_wape"],
                          "full_los_mae": ratio_errors(full)["los_mae"]})
    save_csv(pd.DataFrame(case_rows), RESULTS / "cases.csv")
    table(TABLES / "reviewer_cases.tex", "Country-case forecast accuracy and pre-test tourism recovery, with cases selected by the median recovery ratio within each tercile.",
          "tab:reviewer_cases", ["Destination", r"\shortstack{Tourist Arrivals\\Recovery Index\\2023 / 2019\\(unitless)}",
              r"\shortstack{DLinear\\Tourist Arrivals\\WAPE (\%)}", r"\shortstack{DLinear\\Overnight Stays\\WAPE (\%)}",
              r"\shortstack{RA-HE-DLinear\\Tourist Arrivals\\WAPE (\%)}", r"\shortstack{RA-HE-DLinear\\Overnight Stays\\WAPE (\%)}"],
          [[tex(r["country"]), f"{r['recovery_ratio']:.2f}",
            f"{r['dlinear_arrivals_wape']:.2f}", f"{r['dlinear_nights_wape']:.2f}",
            f"{r['full_arrivals_wape']:.2f}", f"{r['full_nights_wape']:.2f}"] for r in case_rows],
          r">{\raggedright\arraybackslash}p{2.1cm}>{\centering\arraybackslash}p{2.8cm}*{4}{>{\centering\arraybackslash}p{2.6cm}}",
          notes=r"The recovery index is annual tourist arrivals in 2023 divided by annual arrivals in 2019. An index of 1 indicates the same annual arrivals level as in 2019. Forecast errors cover January 2024--July 2026 and use only the aggregate accommodation category with originally observed target pairs.",
          wide=True)
    case_text = []
    for row in case_rows:
        case_text.append(
            tex(row["country"]) + f" has a 2023/2019 arrivals ratio of {row['recovery_ratio']:.2f}. "
            f"In the recent test, DLinear and the full model record arrivals WAPE of "
            f"{row['dlinear_arrivals_wape']:.2f}\\% and {row['full_arrivals_wape']:.2f}\\%, "
            f"and nights WAPE of {row['dlinear_nights_wape']:.2f}\\% and {row['full_nights_wape']:.2f}\\%, respectively. "
            f"The full-model LOS MAE is {row['full_los_mae']:.3f} nights per arrival."
        )
    (TABLES / "reviewer_case_discussion.tex").write_text("\n\n".join(case_text) + "\n", encoding="utf-8")

    plt.rcParams.update({"font.size": 9, "axes.spines.top": False, "axes.spines.right": False})
    figure, axes = plt.subplots(3, 2, figsize=(8.2, 7.2), sharex=True)
    for i, geo in enumerate(cases):
        truth = raw.loc[raw.geo.eq(geo) & raw.accommodation_type.eq("total") & raw.year.ge(2018)]
        for j, (target, prediction) in enumerate(zip(TARGETS, PREDICTIONS)):
            axis = axes[i, j]
            axis.plot(truth.date, truth[target] / 1e6, color="black", lw=1.2, label="Observed")
            for model, color in [("dlinear", "#d97924"), ("full", "#176797")]:
                history = pd.concat([f[model] for f in fold_frames.values()])
                history = history.loc[history.geo.eq(geo) & history.accommodation_type.eq("total")].sort_values("date")
                axis.plot(history.date, history[prediction] / 1e6, color=color, lw=1,
                          label="DLinear" if model == "dlinear" else "Full")
            interval = intervals.loc[intervals.geo.eq(geo) & intervals.accommodation_type.eq("total")]
            suffix = "arrivals" if j == 0 else "nights"
            axis.fill_between(interval.date, interval[f"lower_{suffix}"] / 1e6,
                              interval[f"upper_{suffix}"] / 1e6, color="#176797", alpha=0.15, label="90% interval")
            axis.set_title(f"{labels.loc[geo, 'country']} — {'arrivals' if j == 0 else 'nights'}")
            axis.set_ylabel("Millions")
            axis.axvspan(pd.Timestamp("2020-01-01"), pd.Timestamp("2021-12-31"), color="grey", alpha=0.08)
            axis.grid(alpha=0.15)
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, legend_labels, loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    figure.savefig(FIGURES / "reviewer_country_cases.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    figure, axes = plt.subplots(1, 3, figsize=(8.2, 2.7), sharex=True)
    for axis, geo in zip(axes, cases):
        for model, color, label in [("dlinear", "#d97924", "DLinear"), ("full_raw", "#888888", "Full raw"), ("full", "#176797", "Full projected")]:
            frame = frames[model]
            subset = frame.loc[frame.geo.eq(geo) & frame.accommodation_type.eq("total")]
            ratio = subset.pred_nights / subset.pred_arrivals.replace(0, np.nan)
            axis.plot(subset.date, ratio, color=color, lw=1, label=label)
        observed = frames["full"]
        observed = observed.loc[observed.geo.eq(geo) & observed.accommodation_type.eq("total") & observed.observed_pair]
        axis.plot(observed.date, observed.observed_nights_total / observed.observed_arrivals_total.replace(0, np.nan),
                  color="black", lw=1.2, label="Observed")
        axis.set_title(labels.loc[geo, "country"])
        axis.set_ylabel("Nights per arrival")
        axis.xaxis.set_major_locator(mdates.YearLocator())
        axis.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        axis.grid(alpha=0.15)
    figure.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(FIGURES / "reviewer_los_cases.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    raw_predictions = frames["full_raw"]
    projected = frames["full"]
    figure, axis = plt.subplots(figsize=(4.1, 3.1))
    axis.scatter(raw_predictions.pred_arrivals / 1e6, raw_predictions.pred_nights / 1e6,
                 s=5, alpha=0.25, color="#176797", label="Before projection")
    changed = raw_predictions.pred_nights.lt(raw_predictions.pred_arrivals)
    axis.scatter(projected.loc[changed, "pred_arrivals"] / 1e6,
                 projected.loc[changed, "pred_nights"] / 1e6,
                 s=12, marker="x", color="#b43d35", label=f"Adjusted points (n={int(changed.sum())})")
    bound = float(projected.pred_arrivals.max()) / 1e6
    axis.plot([0, bound], [0, bound], color="grey", lw=0.8, label="Nights = arrivals")
    axis.set_xlabel("Predicted arrivals, millions")
    axis.set_ylabel("Predicted nights, millions")
    axis.legend(frameon=False, fontsize=7)
    figure.tight_layout()
    figure.savefig(FIGURES / "reviewer_projection.png", dpi=220, bbox_inches="tight")
    plt.close(figure)

    full = by_model.loc["full"]
    baseline = by_model.loc["dlinear"]
    raw_scores = by_model.loc["full_raw"]
    best_arrivals = by_model.loc[primary, "arrivals_total_wape"].idxmin()
    best_nights = by_model.loc[primary, "nights_total_wape"].idxmin()
    numbers = {
        "ReviewerArrivalWAPE": f"{full.arrivals_total_wape:.2f}",
        "ReviewerNightsWAPE": f"{full.nights_total_wape:.2f}",
        "ReviewerArrivalMAE": f"{full.arrivals_total_mae:,.0f}",
        "ReviewerNightsMAE": f"{full.nights_total_mae:,.0f}",
        "ReviewerDLinearArrivalWAPE": f"{baseline.arrivals_total_wape:.2f}",
        "ReviewerDLinearNightsWAPE": f"{baseline.nights_total_wape:.2f}",
        "ReviewerObservedPairs": f"{int(full.observed_pairs):,}",
        "ReviewerRawViolation": f"{raw_scores.physical_violation_pct:.3f}",
        "ReviewerBestArrivalModel": MODEL_NAMES[best_arrivals],
        "ReviewerBestNightsModel": MODEL_NAMES[best_nights],
        "ReviewerArrivalCoverage": f"{coverage[0]['coverage_pct']:.2f}",
        "ReviewerNightsCoverage": f"{coverage[1]['coverage_pct']:.2f}",
    }
    (TABLES / "reviewer_numbers.tex").write_text("\n".join(
        rf"\newcommand{{\{key}}}{{{value}}}" for key, value in numbers.items()) + "\n", encoding="utf-8")

    summary = {"recent": by_model.to_dict(orient="index"), "cases": case_rows, "dynamics": dynamics,
               "coverage": coverage, "los": los_rows, "transitions": transitions,
               "best_arrivals_model": best_arrivals, "best_nights_model": best_nights}
    save_json(summary, RESULTS / "summary.json")
    print(benchmark[["model", "observed_pairs", "arrivals_total_wape", "nights_total_wape"]].to_string(index=False))
    print(pd.DataFrame(case_rows).to_string(index=False))


if __name__ == "__main__":
    main()
