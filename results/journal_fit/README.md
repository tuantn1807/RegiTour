# Income groups latent states and LOS diagnostics

Run `python scripts/analyze_journal_fit.py` from the repository root. It reads
retained keyed forecasts and reconstructs their training-fitted HMM states with
the same seed and filter. It does not refit neural models or change predictions.
The manifest records hashes, classification evidence, and the scoring inputs.

The FY2024 income crosswalk fixes membership at 1 July 2023 for the January
2024--July 2026 recent test. The official World Bank announcement establishes
the fiscal-year version and effective date. The income column for the 38 study
destinations is transcribed from the four-page FY2024 table reproduced by OTHM,
which identifies World Bank Country and Lending Groups as its source. The
additional region-classification column is not used. Source URLs and table page
numbers accompany the crosswalk and manifest. The historical XLSX was not
downloaded. Membership is diagnostic metadata and never enters model fitting.

There are 31 high-income destinations and seven upper-middle-income destinations;
the sample contains neither low- nor lower-middle-income economies. Income
metrics use only aggregate accommodation series, with 921 and 208 observed
country-month pairs. Seasonal naive, DLinear, and full forecasts use matching
keys and masks. Country-level outputs and leave-one-destination-out LOS checks
show how individual destinations influence pooled comparisons. These are
descriptive checks, without income-effect or significance claims.

HMM IDs refer to ranked fitted growth centroids, not calendar events. IDs 1, 0,
and 2 correspond to low, middle, and high fitted growth coordinates. Each fold
has its own estimated HMM. `hmm_origin_audit.csv` logs the state actually used
at issuance and the temporal fusion weight, while `hmm_training_profiles.csv`
summarizes decoded training-month observations. Calendar labels are fixed test
windows. No equality between a calendar phase and an inferred state is assumed.

LOS is the monthly ratio of matching nights to accommodation arrivals. It does
not measure individual trip duration. `los_samples.csv` distinguishes the
common defined-denominator comparison from each model's complete valid sample.
The recent common comparison uses 347 of 1,129 eligible pairs because DLinear
predicts zero arrivals in 782 rows. Zero denominators are counted as undefined,
not repaired with an arbitrary epsilon. The full model defines LOS in all
1,129 rows. `los_stability_metrics.csv` uses training-defined LOS-variation
terciles. The leave-one-out analysis retains every country in the primary
reported result and supplies sensitivity diagnostics only.
Seasonal naive is also evaluated on all 1,129 recent aggregate pairs, without
undefined denominators, and retained as a separate LOS baseline in the sample
diagnostics. Its all-sample LOS MAE/RMSE is 0.095/0.245 nights per arrival,
compared with 0.532/0.915 for the full model.

All forecasts remain retrospective reference-period experiments. These
diagnostics do not resolve the missing Original Availability versus PIT test.
