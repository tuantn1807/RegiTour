# Annual World Bank tourism forecast evaluation

- Model: World Bank Ridge Linear
- Scope: all 186 country/area codes represented in the World Bank test data
- Test years: 2018–2020
- Test records: 497

## Overall test metrics

```text
Split                         Model   n           MAE           RMSE  WAPE_pct  MAPE_pct
 test       World Bank Ridge Linear 497 9,491,852.877 36,696,303.656    92.992   139.012
 test One-year persistence baseline 496 3,059,234.577 11,975,313.337    29.911    97.313
```

## Output files

- Forecast rows: `forecast_results_ra_he_dlinear.csv`
- Metrics by country: `tables/metrics_by_country.csv`
- Metrics by year: `tables/metrics_by_year.csv`
- Actual vs predicted chart: `figures/worldbank_ridge_actual_vs_predicted_by_year.png`

The workbook provides annual arrivals only. It has no monthly seasonality, accommodation categories,
or accommodation nights, so this report does not include those Eurostat-only breakdowns or interval coverage.
Any existing `metrics_by_accommodation.csv` from an earlier Eurostat run is left untouched and is not a current World Bank result.
