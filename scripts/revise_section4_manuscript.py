"""Install the six requested result sections using generated numeric macros."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

SECTION = r"""\section{Experimental Results and Discussion}
\label{sec:results}

\subsection{Datasets and Evaluation Protocol}
\label{sec:experimental_setup}
We evaluate two frozen panels with different statistical populations (Table~\ref{tab:datasets}). Eurostat arrivals count registrations of resident and non-resident guests at accommodation establishments, and nights count their overnight stays; these are not counts of distinct people or same-day visits \cite{eurostat_tourism_release,eurostat_tourism_scope}. The global panel combines UN Tourism inbound arrivals, retrieved through Our World in Data, with historical WDI arrivals \cite{un_tourism_database,owid_tourist_arrivals,worldbank_arrivals_metadata}. The two panels are evaluated separately.

\begin{table}[!t]
\centering
\caption{Dataset snapshots and chronological evaluation folds. Train starts in 2010 for monthly data and 1995 for annual data.}
\label{tab:datasets}
\label{tab:protocol_folds}
\scriptsize
\setlength{\tabcolsep}{2.5pt}
\renewcommand{\arraystretch}{1.06}
\begin{tabular}{@{}p{0.22\columnwidth}p{0.35\columnwidth}p{0.35\columnwidth}@{}}
\toprule
\textbf{Panel A: Data} & \textbf{Eurostat monthly} & \textbf{Global annual} \\
\midrule
Coverage & 38 destinations, 152 series & 203 economies, 203 series \\
Reference span & Jan.\ 2010--Jul.\ 2026 & 1995--2024 \\
Source rows & 30,248 & 5,444 (unbalanced) \\
Lookback / horizon & 24 / 12 months & 4 / 1 calendar years \\
Exogenous inputs & 14 calendar, 7 historical & 2 calendar/event, 6 historical \\
Recent test & 4,522 of 4,712 grid pairs, 37 observed destinations & 786 of 1,218 grid pairs, 186 observed economies \\
\bottomrule
\end{tabular}
\par\vspace{4pt}
\begin{tabular}{@{}lcccc@{}}
\toprule
\textbf{Panel B: Fold} & \textbf{Train end} & \textbf{Val.} & \textbf{Cal.} & \textbf{Test} \\
\midrule
Monthly pre-pandemic & 2015 & 2016 & 2017 & 2018--2019 \\
Monthly pandemic & 2017 & 2018 & 2019 & 2020--2021 \\
Monthly recovery & 2019 & 2020 & 2021 & 2022--2023 \\
Monthly recent & 2021 & 2022 & 2023 & 2024--Jul.\ 2026 \\
Annual global & 2016 & 2017 & 2018 & 2019--2024 \\
\bottomrule
\end{tabular}
\end{table}

\textbf{Targets and eligibility.} Monthly series are keyed by destination, category, and reference month. Hotels, short-stay accommodation, and camping overlap their aggregate category \texttt{total}; pooled benchmarks include all four, while destination and LOS diagnostics use only \texttt{total}. The original observation mask retains pairs with both targets reported and nonnegative before any filling. The recent test includes 166 pairs with at least one zero and excludes 190 missing pairs. The monthly train/validation/calibration partitions contain 21,888/1,824/1,824 grid rows and 18,519/1,740/1,735 eligible pairs. Missing historical inputs are forward-filled within series, then filled with series or pooled training medians; zero is the final fallback. All neural and LightGBM output windows must have originally observed labels throughout their horizon. Imputed targets can enter histories but are never promoted to observed evaluation outcomes.

Annual nights are mixed-source targets: 315 of 5,444 rows use eligible OECD observations and 5,129 are constructed from arrivals and income-adjusted regional LOS assumptions. The test contains 108 OECD and 678 constructed nights targets. Thus nights accuracy and the lower-bound constraint are partly evaluated against a construction rule. Annual training, validation, and calibration have 4,274, 194, and 190 observed source rows. Test counts in 2019--2024 are 186, 177, 159, 132, 67, and 65, so cross-period comparisons also reflect changing country coverage.

Annual histories are reindexed to a complete calendar-year grid, and the five macro lags are reconstructed by calendar-year shifting the unfilled WDI workbook. Missing and non-finite values in target histories and annual covariates are filled within each country by forward-filling in chronological order, followed by country-specific training medians, pooled training medians, and zero if both medians are unavailable. All medians use only training years through 2016; no backward filling is applied. Original targets, the pre-imputation eligibility mask, and nights-source flags are preserved separately. Filled target values can enter histories, while evaluation uses only pairs with both source targets present and nonnegative before filling; constructed nights retain their source labels.

\textbf{Forecast origins and information sets.} At the end of December before each monthly forecast year, models receive 24 historical months and issue a January--December block; the 2026 block is scored through July. Annual models use four preceding calendar-year rows for one-year forecasts. Observations from earlier test years can enter later origins, but parameters remain fixed after training. Statistical states are updated using pre-origin histories without parameter refitting. The spatial component extrapolates from the training endpoint. Future inputs contain only calendar/event features; macro inputs and HMM states use the historical prefix. Target-derived LOS ratios and observation/source masks are excluded from model covariates. HMM parameters and standardization are fitted on training data and states are forward-filtered. SARIMAX holds future macro inputs at their last pre-origin values. The retained monthly calendar defines \texttt{weekend\_days} as Friday--Sunday and \texttt{working\_days} as the remaining days; public holidays are encoded separately.

These are retrospective reference-period experiments. Reference dates are not publication dates, and historical release vintages are unavailable. The monthly source enrichment code uses pre-2022 HICP medians, potentially extending beyond early folds' training boundaries; this upstream uncertainty is not removed by fold-specific filling. Historical event calendars can also reflect later schedule changes. Accordingly, we do not claim a publication-aware real-time backtest. The annual macro reconstruction removes backward filling from the new macro preprocessing but cannot recover unavailable historical vintages or validate the constructed nights targets.

\textbf{Configuration and reproducibility.} Linear correction models use a 13-step decomposition kernel, AdamW with learning rate $10^{-4}$ and weight decay $10^{-4}$, batch size 256, five epochs, seed 42, and coupled-loss weight 0.2 when enabled. Their backbone projections are frozen at their initialization; the \emph{Frozen DLinear} reference therefore differs from a trained Vanilla DLinear. The separate learned DLinear and local Transformer implementations use per-series training-fitted log standardization, AdamW ($10^{-3}$, weight decay $10^{-5}$), batch size 128, at most 30 epochs, and validation early stopping with patience six. The annual correction models use 30 epochs and learning rate 0.005, with rank-four spatial factors and annual lags $\{1,2\}$. HMMs have three Gaussian states and 200 fitting iterations. ETS selects no-trend versus damped-trend seasonal specifications on validation; SARIMA/SARIMAX use $(1,0,0)(1,0,0)_{12}$; LightGBM uses 200 trees, learning rate 0.05, and 31 leaves. Unsupported statistical fits use recorded seasonal-naive fallbacks.

For each target on the common eligible set $\mathcal E$, metrics in original counts are
\begin{equation}
\mathrm{WAPE}=100\frac{\sum_{j\in\mathcal E}|Y_j-\hat Y_j|}{\max(1,\sum_{j\in\mathcal E}Y_j)},\qquad
\mathrm{MAE}=\frac{\sum_{j\in\mathcal E}|Y_j-\hat Y_j|}{|\mathcal E|}.
\label{eq:evaluation_metrics}
\end{equation}
Pooled WAPE is not the mean of country percentage errors. All principal results use seed 42. The reproducibility package retains exact input and source hashes, source snapshots, checkpoints, origin-keyed predictions, masks, timings, and independently recalculated metrics. The monthly snapshot uses LF-normalized SHA-256 prefix \texttt{a2f70a0c9667}. The current annual prefix is \texttt{\ResultAnnualHash}. Byte hashes also record line-ending differences. Computation uses an Intel Core i7-10750H CPU with one PyTorch thread, Python 3.14.8, PyTorch 2.15.0, NumPy 2.5.4, pandas 3.0.6, statsmodels 0.15.0, LightGBM 4.7.0, hmmlearn 0.3.3, and SciPy 1.18.1. No historical result is silently substituted for a new configuration.

\subsection{Forecasting Accuracy and Efficiency}
\label{sec:accuracy_efficiency}
Seasonal Naive achieves the lowest pooled errors in Table~\ref{tab:section4_benchmark} (\ResultNaiveArrWAPE\% arrivals and \ResultNaiveNigWAPE\% nights). All models share identical test keys, outcomes, observation masks, and January origins. The full framework obtains \ResultFullArrWAPE\% and \ResultFullNigWAPE\%, compared with \ResultFrozenArrWAPE\% and \ResultFrozenNigWAPE\% for the frozen reference. Exogenous-only correction reaches \ResultExoArrWAPE\% and \ResultExoNigWAPE\%; the full framework therefore does not improve both targets over this simpler configuration. Learned DLinear obtains \ResultLearnedArrWAPE\%/\ResultLearnedNigWAPE\% under the separate trainable-backbone protocol. These comparisons concern the evaluated implementations and tuning budgets, not universal rankings of model families.

\input{tables/section4_benchmark.tex}

The full temporal computation graph has \ResultFullParameters{} active scalar parameters. CPU costs cover instrumented fitting and prediction calls, including feature construction but excluding common panel completion and shared HMM state preparation. Disabled wrapper branches are excluded from active parameter counts. These measurements concern this CPU and implementation; they are not universal latency or model-family rankings.

Unstable statistical fits are retained in the aggregate scores rather than selectively removed; their convergence diagnostics and fallback decisions are archived. In particular, a numerically finite inverse-log SARIMAX forecast can still be extremely large, so its aggregate error does not establish that all econometric forecasting is unreliable. The primary ETS row uses validation-selected seasonal models on scaled counts; fixed log-ETS results are retained only as a sensitivity archive. Across 1,000 paired destination-block bootstrap resamples, full-minus-frozen WAPE differences are \ResultDeltaArr{} percentage points for arrivals (95\% interval [\ResultLowerArr, \ResultUpperArr]) and \ResultDeltaNig{} for nights ([\ResultLowerNig, \ResultUpperNig]). These descriptive intervals preserve within-destination dependence and do not establish robustness to all cross-country dependencies.

\subsection{Regime-Dependent Forecasting Performance}
\label{sec:regime_performance}
Forecasting performance varies across market regimes: the fixed static blend improves pandemic arrivals accuracy but increases error in tranquil periods, while validation-selected fusion can deteriorate during recovery (Table~\ref{tab:section4_regimes}). The comparison separates temporal-only references, TRMF-only extrapolation, fixed static blending, and the current adaptive framework. Seasonal Naive and learned DLinear are included in every fold. All configurations within a fold share reference periods and scoring eligibility; no test outcomes select fusion weights.

\input{tables/section4_regimes.tex}

In the pandemic fold, the frozen temporal reference obtains \ResultFrozenShockArr\% arrivals WAPE. The current adaptive framework reduces this to \ResultAdaptiveShockArr\%, while the predeclared static blend ($w_{\rm time}=0.7$) gives \ResultStaticShockArr\%. Both current fusion rows use the same final point projection. The previously reported \textbf{66.69\%} is traced separately to the archived 2020--2021 experiment: training ends in 2017, validation is 2018, calibration is 2019, and the earlier regularized sigmoid gate selects $w_{\rm time}=0.7380328366$. That run uses ungated FiLM and an $L_2$ penalty of 0.005 on the gate logit. It is neither the fixed 0.7 blend nor the current bounded adaptive configuration. Its archived row is included for provenance, not as a replacement for the new full-model result.

Current adaptive temporal weights are \ResultWeightTranquil, \ResultWeightShock, \ResultWeightRecovery, and \ResultWeightRecent{} for the pre-pandemic, pandemic, recovery, and recent folds. A single validation-selected weight is held fixed throughout each fold's test interval; ``adaptive'' here denotes validation-based selection, not online updating at every forecast month. The recovery fold illustrates its limitation: validation during 2020 selects substantial spatial weight, but the resulting 2022--2023 arrivals WAPE is \ResultAdaptiveRecoveryArr\%, compared with \ResultFrozenRecoveryArr\% for the frozen reference. Validation improvements therefore need not survive a change in market regime.

\subsection{Component Contributions and Adaptive Fusion}
\label{sec:component_contributions}
Exogenous correction yields lower arrivals and nights errors than the full framework on the recent fold, while ungated FiLM and fixed spatial blending can increase error (Table~\ref{tab:section4_components_global}, Panel A). The component ladder reports single-seed configurations, while Table~\ref{tab:section4_regimes} evaluates temporal robustness across folds. The two tables have different purposes; the component ladder is not interpreted as a multi-seed factorial estimate of causal main effects.

\input{tables/section4_components_global.tex}

Exogenous correction lowers arrivals WAPE from \ResultFrozenArrWAPE\% to \ResultExoArrWAPE\% and nights WAPE from \ResultFrozenNigWAPE\% to \ResultExoNigWAPE\%. HMM-only intercept conditioning has little effect under this setup, and ungated FiLM can degrade both targets. Soft gates and the coupled loss are reported as separate configurations so their effects are not conflated with HMM state conditioning. The full temporal core yields \ResultCoreArrWAPE\%/\ResultCoreNigWAPE\%, compared with \ResultNoConsArrWAPE\%/\ResultNoConsNigWAPE\% without its coupled loss. These modest, target-dependent differences do not support a claim that every added component improves accuracy.

The unprojected static 0.7 blend yields \ResultStaticArrWAPE\%/\ResultStaticNigWAPE\%, whereas validation-selected fusion reduces its spatial contribution and gives \ResultFullArrWAPE\%/\ResultFullNigWAPE\% after projection. However, it still underperforms the temporal core on this test set. The gate objective averages the two targets' WAPE fractions and applies $\epsilon=10^{-5}$ on that scale. The fallback in Equation~\ref{eq:adaptive_fallback} protects validation performance only. The spatial implementation is TRMF-inspired: rank-four factors, clipped ridge AR extrapolation, and damping approximate cross-country pooling; it is not an exact optimizer of the full TRMF objective. Its forecasts demonstrate shared low-rank structure, not identified causal cross-border spillovers.

On the recent fold, exogenous-only correction achieves arrivals WAPE of \ResultExoArrWAPE\% and MAE of \ResultExoArrMAE, compared with \ResultFullArrWAPE\% and \ResultFullArrMAE{} for the full framework. The full framework additionally includes regime-conditioned modulation, the coupled training loss, and spatial fusion with a validation-selected temporal weight of \ResultWeightRecent.

\subsection{Global Benchmark and Generalization}
\label{sec:global_generalization}
Persistence achieves lower pooled arrivals and nights WAPE than the re-estimated full model on the annual global panel (Table~\ref{tab:section4_components_global}, Panel B). The annual benchmark uses the calendar-grid completion and missing-data protocol in Section~\ref{sec:experimental_setup}, preserving the pre-imputation scoring mask. The model is retrained with annual histories, calendar/event inputs, forward-filtered annual regimes, and annual spatial lags. This is a cross-dataset benchmark, not zero-shot transfer of monthly parameters. A chronological calibration year remains separate from validation, and the new package includes predictions and source flags for every annual evaluation key.

Across 2019--2024, full-model arrivals WAPE is \ResultAnnualFullAllArr\%, compared with \ResultAnnualNaiveAllArr\% for Persistence; corresponding mixed-source nights errors are \ResultAnnualFullAllNig\% and \ResultAnnualNaiveAllNig\%. Full-model arrivals WAPE is \ResultAnnualFullPreArr\% in 2019, \ResultAnnualFullShockArr\% in 2020--2021, and \ResultAnnualFullRecoveryArr\% in 2022--2024. Persistence yields \ResultAnnualNaivePreArr\%, \ResultAnnualNaiveShockArr\%, and \ResultAnnualNaiveRecoveryArr\% in those periods, respectively. It has lower arrivals error in all three groups under this configuration. These comparisons are interpreted alongside the unequal country counts, rather than attributed solely to regime changes.

Annual nights results require particular caution. Separately evaluating the 108 OECD nights observations gives \ResultAnnualObservedNig\% full-model nights WAPE, versus \ResultAnnualConstructedNig\% on the 678 constructed targets. The pooled figure cannot establish independent validation of global overnight-stay forecasting. Cross-source definitions, constructed LOS assumptions, sparse later-year coverage, and unavailable historical releases limit the strength of generalization claims. The evidence supports testing the framework in a second data setting, with substantial remaining limitations, rather than asserting universal superiority across global tourism markets.

\subsection{Forecast Consistency, Uncertainty and Explainability}
\label{sec:results_xai}
\textbf{Point consistency and LOS.} Projection intervenes on \ResultProjectionCount{} of the 4,712 recent forecast pairs (\ResultProjectionRate\%) and removes all violations of the specified point constraint $\hat N\geq\hat A$. Arrivals change only through float32 rounding (maximum \ResultProjectionRounding{} arrivals). The source itself contains 14 reported pairs with $N<A$, including two recent test pairs; these remain in the common scoring mask. The constraint is therefore an imposed operational rule, not a statement that every source observation already satisfies it. The operator does not impose an upper LOS bound or a historical economic band.

LOS is compared on a shared subset of \ResultLOSSharedCount{} observed aggregate-category pairs with positive observed arrivals and positive forecast arrivals for all three forecasts. LOS MAE/RMSE are \ResultFrozenLOSMAE/\ResultFrozenLOSRMSE{} nights per arrival for Frozen DLinear, \ResultRawLOSMAE/\ResultRawLOSRMSE{} for adaptive fusion before projection, and \ResultFullLOSMAE/\ResultFullLOSRMSE{} after projection. Most of the improvement over the frozen reference is already present before projection; the operator provides a smaller additional reduction in LOS error while enforcing the point constraint. Undefined forecast ratios over the eligible aggregate-category observations are counted separately (full: \ResultFullLOSUndefined; frozen: \ResultFrozenLOSUndefined); projection does not make a zero arrivals denominator positive.

\begin{figure*}[!t]
\centering
\includegraphics[width=0.94\textwidth]{figures/section4_country_cases.png}
\caption{Observed aggregate-category arrivals and nights, frozen DLinear forecasts, and current adaptive full-model forecasts with marginal 90\% nominal intervals. Cyprus, Sweden, and Denmark are the destination cases selected before this rerun using pre-test recovery profiles. Curves and intervals are generated from the same keyed forecasts used for the tables.}
\Description{Six panels show actual arrivals and nights and two forecasts for Cyprus, Sweden, and Denmark, with shaded conformal intervals.}
\label{fig:tourism_cases}
\end{figure*}

\textbf{Uncertainty.} For each fold, separate absolute-residual conformal quantiles are estimated after point projection on its held-out calibration year (Table~\ref{tab:protocol_folds}), using finite-sample rank $\lceil0.9(n+1)\rceil$. At 90\% nominal coverage, empirical arrivals/nights coverage is \ResultCoverageTranquilArr\%/\ResultCoverageTranquilNig\% before the pandemic, \ResultCoverageShockArr\%/\ResultCoverageShockNig\% during the pandemic, \ResultCoverageRecoveryArr\%/\ResultCoverageRecoveryNig\% during recovery, and \ResultCoverageRecentArr\%/\ResultCoverageRecentNig\% in the recent fold. The substantial pandemic undercoverage shows that this fixed calibration rule does not maintain its nominal level across market shifts.

The recent fold uses 1,735 observed 2023 calibration pairs. Its normalized mean interval widths are \ResultWidthArr{} for arrivals and \ResultWidthNig{} for nights, with each mean width divided by that target's mean observed count. Figure~\ref{fig:tourism_cases} illustrates the recent destination forecasts; it does not replace pooled coverage evaluation. Nominal 90\% coverage is an exchangeability-based marginal property, not an unconditional guarantee under temporal distribution shift. Separate marginal intervals also do not guarantee simultaneous 90\% coverage or joint cross-target feasibility.

\textbf{Surrogate fidelity and direct sensitivity.} TreeSHAP \cite{lundberg2020treeshap} is recomputed on CatBoost surrogates fitted to the actual fused, projected model forecasts for 2022--2023, with 2024--2026 reserved for fidelity evaluation. Surrogate $R^2$ is \ResultFidelityCountsArr{} for arrivals and \ResultFidelityCountsNig{} for nights in count space, and \ResultFidelityLogArr/\ResultFidelityLogNig{} in log space. The largest mean absolute attributions concern prior-year seasonal arrivals/nights; accommodation category and recent target levels also contribute. SHAP values explain the log-output surrogate and retain its approximation error, particularly for nights; they are not exact causal contributions of the forecasting network.

Direct sensitivity is measured separately by passing perturbed histories through the stored full model. A uniform $+10\%$ accommodation HICP shift for France, Spain, Italy, and Germany hotels changes pooled forecast arrivals by \ResultPriceArr\% and nights by \ResultPriceNig\%, over 124 eligible test pairs. Percentage price-change inputs, other variables, HMM states, spatial forecasts, and the gate are held fixed. These small responses describe this trained model under the stated intervention; they do not estimate a causal demand elasticity or justify a pricing policy. Operational consistency, interval calibration, surrogate approximation, and input sensitivity thus remain distinct evaluation claims.

"""

ABSTRACT = r"""Multi-country tourism demand forecasting requires models that accommodate seasonal patterns, regime changes, and related targets. We evaluate RA-HE-DLinear, which combines an exogenous linear temporal core, forward-filtered HMM conditioning, spatial pooling inspired by TRMF, fusion selected on validation, projection of point forecasts, and residual conformal calibration. Matched retrospective experiments cover 38 European destinations at monthly frequency and 203 economies at annual frequency. On the recent Eurostat test, arrivals WAPE is \ResultFullArrWAPE\% and nights WAPE is \ResultFullNigWAPE\%. Seasonal Naive obtains \ResultNaiveArrWAPE\%/\ResultNaiveNigWAPE\%, and exogenous-only correction obtains \ResultExoArrWAPE\%/\ResultExoNigWAPE\%. During the pandemic fold, adaptive fusion gives \ResultAdaptiveShockArr\% arrivals WAPE versus \ResultFrozenShockArr\% for the frozen temporal reference; a predeclared static blend gives \ResultStaticShockArr\%. The archived 66.69\% result is identified as a different regularized-gate configuration. Recent empirical interval coverage is \ResultCoverageArr\%/\ResultCoverageNig\% at 90\% nominal coverage. Annual re-estimation highlights limitations from regime shifts, unequal coverage, and constructed nights targets. The results support selective use of model components while showing that added complexity and validation gains do not ensure better test accuracy.
"""

CONCLUSION = r"""The matched experiments show that the contribution of exogenous correction, regime conditioning, and cross-country pooling depends on the target and evaluation period. The full framework obtains \ResultFullArrWAPE\%/\ResultFullNigWAPE\% arrivals/nights WAPE on the recent monthly panel, while Seasonal Naive and the simpler exogenous-only configuration perform better in this evaluation. Adaptive fusion improves the frozen temporal reference during the pandemic fold, but a validation-selected weight can transfer poorly to recovery periods. The archived 66.69\% pandemic result belongs to the earlier regularized sigmoid configuration and is distinguished from the current adaptive implementation.

Point projection enforces the specified arrivals--nights lower bound, while conformal intervals achieve empirical coverage of \ResultCoverageArr\%/\ResultCoverageNig\% relative to 90\% nominal. Neither result establishes supply feasibility or distribution-free real-time operational performance. Annual global re-estimation does not outperform Persistence across the complete test window, and its nights results are limited by target construction and cross-source coverage. Future work should recover publication-aware vintages, rebuild monthly source-stage imputation by fold, evaluate broader tuning budgets and multiple seeds, and develop calibration and fusion updates that respond to distribution shift. The retained forecasts, source hashes, checkpoints, and verification scripts make the present comparisons independently auditable.

"""

XAI_METHOD = r"""\subsection{Feature Attribution and Explainability (XAI)}
To examine forecast associations, we fit CatBoost surrogates to log-transformed outputs of the fused, projected forecasting model. Surrogate training uses forecasts for 2022--2023 and fidelity is evaluated on 2024--2026 forecasts. Inputs summarize the same pre-origin target histories and historical covariates, together with known calendar variables. TreeSHAP \cite{lundberg2020treeshap} decomposes each surrogate prediction as $g(\mathbf x)=\phi_0+\sum_j\phi_j$; mean absolute values quantify surrogate associations in log-output units. They approximate the forecaster only to the extent supported by held-out fidelity.

We separately measure direct input sensitivity by perturbing historical accommodation HICP levels by $\pm5\%$ and $\pm10\%$, rerunning the stored forecasting model, and holding other inputs, states, spatial forecasts, and fusion weights fixed. Uniform level shifts leave percentage-change features unchanged. This model intervention is not an identified economic elasticity; its measured responses are reported in Section~\ref{sec:results_xai}.

"""


def main() -> None:
    """Replace Section 4 and update dependent claims without changing acknowledgments."""
    path = ROOT / "paper/paper_en.tex"
    source = path.read_text(encoding="utf-8")
    source = source.replace(r"\input{tables/reviewer_numbers.tex}", r"\input{tables/section4_numbers.tex}")
    start = source.index(r"\section{Experimental Results")
    end = source.index(r"\section{Conclusion}", start)
    source = source[:start] + SECTION + source[end:]
    start = source.index(r"\begin{abstract}") + len(r"\begin{abstract}")
    end = source.index(r"\end{abstract}", start)
    abstract = ABSTRACT.replace(r"The archived 66.69\% result is identified as a different regularized-gate configuration. ", "")
    abstract = abstract.replace("Annual re-estimation highlights limitations from regime shifts, unequal coverage, and constructed nights targets.",
        r"Annual arrivals WAPE is \ResultAnnualFullAllArr\%, versus \ResultAnnualNaiveAllArr\% for Persistence, with additional limitations from unequal coverage and constructed nights targets.")
    source = source[:start] + "\n" + abstract + source[end:]
    start = source.index(r"\section{Conclusion}")
    body = source.index(r"\label{sec:conclusion}", start) + len(r"\label{sec:conclusion}")
    end = source.index(r"\section*{Acknowledgments}", body)
    conclusion = CONCLUSION.replace("The archived 66.69\\% pandemic result belongs to the earlier regularized sigmoid configuration and is distinguished from the current adaptive implementation.", "")
    conclusion = conclusion.replace("The retained forecasts, source hashes, checkpoints, and verification scripts make the present comparisons independently auditable.", "")
    source = source[:body] + "\n" + conclusion + source[end:]
    start = source.index(r"\subsection{Feature Attribution and Explainability (XAI)}")
    end = source.index(r"\section{Experimental Results", start)
    source = source[:start] + XAI_METHOD + source[end:]
    source = source.replace("distribution-free conformal calibration, and TreeSHAP sensitivity analysis", "residual conformal calibration, surrogate TreeSHAP analysis, and direct input sensitivity")
    source = source.replace("% Tighten float spacing for 5-page 2-column conference format", "% Float spacing for the two-column conference format")
    marker = "Extrapolated latent factors are rescaled"
    if "The implemented spatial solver is an approximation" not in source:
        position = source.index(marker)
        source = source[:position] + "The implemented spatial solver is an approximation to this reference objective: it alternates ridge loading updates and factor reconstruction steps, fits clipped ridge AR coefficients, and applies forecast damping of 0.96. It does not jointly optimize the full AR regularizer; we therefore describe the evaluated component as TRMF-inspired.\n\n" + source[position:]
    source = source.replace("figures/ra_he_dlinear_architecture_en.png", "figures/section4_architecture.pdf")
    source = source.replace(r"\includegraphics[width=0.68\textwidth]{figures/section4_architecture.pdf}", r"\includegraphics[width=0.94\textwidth]{figures/section4_architecture.pdf}")
    path.write_text("\n".join(line.rstrip() for line in source.splitlines()) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
