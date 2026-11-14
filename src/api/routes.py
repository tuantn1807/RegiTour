"""
FastAPI route handlers for RegiTour Decision Support & Analytics.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status
import numpy as np
import pandas as pd

from src.api.schemas import (
    DestinationInfo,
    DestinationsResponse,
    HealthResponse,
    MonthlyPlanItem,
    MonthlyStaffingItem,
    MonthlyStressItem,
    NACEStaffingCategory,
    PiCOptimizerRequest,
    PiCOptimizerResponse,
    PlanRequest,
    PlanResponse,
    PlanSummary,
    RecoveryBenchmarkItem,
    RecoveryBenchmarkResponse,
    RegimeTimelineItem,
    RegimeTimelineResponse,
    ReportGenerateRequest,
    ReportGenerateResponse,
    StaffingRequest,
    StaffingResponse,
    StaffingSummary,
    StressTestRequest,
    StressTestResponse,
)
from src.decision_support.dmo_engine import (
    NACE_LABELS,
    PAPER_CASE_STUDIES,
    DSSDataBundle,
    compute_coordinated_throughput_plan,
    compute_crisis_regime_diagnostics,
    compute_hmm_regime_timeline,
    compute_nace_capacity_staffing,
    compute_recovery_benchmarking,
    load_dss_bundle,
    simulate_macro_stress,
)
from src.models.coupled_optimizer import CoupledConsistencyOptimizer
from src.services.report_generator import ExecutiveReportGenerator

router = APIRouter()

# Global cache for the data bundle
_BUNDLE_CACHE: Optional[DSSDataBundle] = None
_ROOT_DIR = Path(__file__).resolve().parents[2]


def get_bundle() -> DSSDataBundle:
    global _BUNDLE_CACHE
    if _BUNDLE_CACHE is None:
        _BUNDLE_CACHE = load_dss_bundle(_ROOT_DIR)
    return _BUNDLE_CACHE


@router.get(
    "/health",
    response_model=HealthResponse,
    summary="System health & metadata check",
    tags=["System"],
)
def get_health(bundle: DSSDataBundle = Depends(get_bundle)) -> HealthResponse:
    origins = sorted(bundle.forecasts_df["origin_label"].unique().tolist())
    dest_count = int(bundle.forecasts_df["geo"].nunique())
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        total_destinations=dest_count,
        available_origins=origins,
        source_mode=bundle.source_mode,
    )


@router.get(
    "/destinations",
    response_model=DestinationsResponse,
    summary="List all monitored destinations and categories",
    tags=["Metadata"],
)
def get_destinations(bundle: DSSDataBundle = Depends(get_bundle)) -> DestinationsResponse:
    geo_list = sorted(bundle.forecasts_df["geo"].unique().tolist())
    dests = [
        DestinationInfo(geo=g, name=bundle.country_names.get(g, g))
        for g in geo_list
    ]
    origins = sorted(bundle.forecasts_df["origin_label"].unique().tolist())
    nace_cats = {k: v[1] for k, v in NACE_LABELS.items()}
    return DestinationsResponse(
        total=len(dests),
        destinations=dests,
        nace_categories=nace_cats,
        available_origins=origins,
    )


@router.post(
    "/forecast/plan",
    response_model=PlanResponse,
    summary="Generate coordinated arrival & night throughput plan (Pi_C safe)",
    tags=["Decision Support"],
)
def post_forecast_plan(
    req: PlanRequest, bundle: DSSDataBundle = Depends(get_bundle)
) -> PlanResponse:
    plan_df, summary = compute_coordinated_throughput_plan(
        bundle.forecasts_df,
        geo=req.geo,
        accom_type=req.accommodation_type,
        horizon_months=req.horizon_months,
        origin_label=req.origin_label,
    )
    if plan_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No forecast data available for geo='{req.geo}', accom='{req.accommodation_type}', origin='{req.origin_label}'.",
        )

    monthly_items = []
    for _, row in plan_df.iterrows():
        monthly_items.append(
            MonthlyPlanItem(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                pred_arrivals_safe=float(row["pred_arrivals_safe"]),
                pred_nights_safe=float(row["pred_nights_safe"]),
                pred_los=float(row["pred_los"]),
                daily_arrivals_throughput=float(row["daily_arrivals_throughput"]),
                daily_nights_demand=float(row["daily_nights_demand"]),
                lower_arrivals=float(row["lower_arrivals"]) if pd.notna(row.get("lower_arrivals")) else None,
                upper_arrivals=float(row["upper_arrivals"]) if pd.notna(row.get("upper_arrivals")) else None,
                lower_nights=float(row["lower_nights"]) if pd.notna(row.get("lower_nights")) else None,
                upper_nights=float(row["upper_nights"]) if pd.notna(row.get("upper_nights")) else None,
                actual_arrivals=float(row["actual_arrivals"]) if pd.notna(row.get("actual_arrivals")) else None,
                actual_nights=float(row["actual_nights"]) if pd.notna(row.get("actual_nights")) else None,
            )
        )

    plan_summary = PlanSummary(
        total_arrivals=float(summary["total_arrivals"]),
        total_nights=float(summary["total_nights"]),
        avg_los=float(summary["avg_los"]),
        peak_daily_arrivals=float(summary["peak_daily_arrivals"]),
        peak_daily_nights=float(summary["peak_daily_nights"]),
        raw_violations=int(summary["raw_violations"]),
        dlinear_violations=int(summary["dlinear_violations"]),
        post_violations=int(summary["post_violations"]),
        total_nights_corrected=float(summary["total_nights_corrected"]),
    )

    return PlanResponse(
        geo=req.geo,
        country=bundle.country_names.get(req.geo, req.geo),
        accommodation_type=req.accommodation_type,
        horizon_months=req.horizon_months,
        origin_label=req.origin_label or "default",
        summary=plan_summary,
        monthly_plan=monthly_items,
    )


@router.post(
    "/capacity/staffing",
    response_model=StaffingResponse,
    summary="Compute NACE capacity & seasonal staffing breakdown (I551–I553)",
    tags=["Decision Support"],
)
def post_capacity_staffing(
    req: StaffingRequest, bundle: DSSDataBundle = Depends(get_bundle)
) -> StaffingResponse:
    cap_df, cap_summary = compute_nace_capacity_staffing(
        bundle.forecasts_df,
        geo=req.geo,
        horizon_months=req.horizon_months,
        guests_per_unit=req.guests_per_unit,
        hours_per_arrival=req.hours_per_arrival,
        hours_per_night=req.hours_per_night,
        hours_per_fte_month=req.hours_per_fte_month,
        maintenance_occ_threshold_pct=req.maintenance_occ_threshold_pct,
        origin_label=req.origin_label,
    )
    if cap_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No staffing data available for geo='{req.geo}'.",
        )

    categories = {}
    for cat_name, cat_data in cap_summary["categories"].items():
        categories[cat_name] = NACEStaffingCategory(
            nace_code=cat_data["nace_code"],
            nace_name=cat_data["nace_name"],
            peak_month=str(cat_data["peak_month"]),
            peak_fte=float(cat_data["peak_fte"]),
            trough_month=str(cat_data["trough_month"]),
            trough_fte=float(cat_data["trough_fte"]),
            seasonal_fte_swing=float(cat_data["seasonal_fte_swing"]),
            maintenance_window_months=list(cat_data["maintenance_window_months"]),
            peak_strain_months=list(cat_data["peak_strain_months"]),
        )

    summary_obj = StaffingSummary(
        total_peak_fte=float(cap_summary["total_peak_fte"]),
        total_baseline_fte=float(cap_summary["total_baseline_fte"]),
        total_seasonal_hiring_need=float(cap_summary["total_seasonal_hiring_need"]),
        categories=categories,
    )

    monthly_items = []
    for _, row in cap_df.iterrows():
        is_maint = row["operational_status"] == "Maintenance Window"
        is_strain = "Peak Strain" in str(row["operational_status"])
        monthly_items.append(
            MonthlyStaffingItem(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                nace_code=str(row["nace_code"]),
                pred_arrivals=float(row["pred_arrivals"]),
                pred_nights=float(row["pred_nights"]),
                required_hours=float(row["required_fte"] * req.hours_per_fte_month),
                required_fte=float(row["required_fte"]),
                occupancy_pct=float(row["occupancy_pct"]),
                is_maintenance_window=bool(is_maint),
                is_peak_strain=bool(is_strain),
            )
        )

    return StaffingResponse(
        geo=req.geo,
        country=bundle.country_names.get(req.geo, req.geo),
        horizon_months=req.horizon_months,
        origin_label=req.origin_label or "default",
        summary=summary_obj,
        monthly_breakdown=monthly_items,
    )


@router.post(
    "/simulation/stress-test",
    response_model=StressTestResponse,
    summary="Simulate macroeconomic price shock & border stringency stress",
    tags=["Simulation & Resilience"],
)
def post_stress_test(
    req: StressTestRequest, bundle: DSSDataBundle = Depends(get_bundle)
) -> StressTestResponse:
    plan_df, _ = compute_coordinated_throughput_plan(
        bundle.forecasts_df,
        geo=req.geo,
        accom_type=req.accommodation_type,
        horizon_months=req.horizon_months,
        origin_label=req.origin_label,
    )
    if plan_df.empty:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No baseline data available for geo='{req.geo}'.",
        )

    diag = compute_crisis_regime_diagnostics(bundle, req.geo)
    elasticity = float(diag["price_elasticity"])

    stressed_df = simulate_macro_stress(
        plan_df,
        price_elasticity=elasticity,
        hicp_shock_pct=req.hicp_shock_pct,
        stringency_delta=req.stringency_delta,
    )

    base_arr = float(stressed_df["pred_arrivals"].sum())
    st_arr = float(stressed_df["stressed_arrivals"].sum())
    base_nig = float(stressed_df["pred_nights"].sum())
    st_nig = float(stressed_df["stressed_nights"].sum())

    monthly_items = []
    for _, row in stressed_df.iterrows():
        monthly_items.append(
            MonthlyStressItem(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                baseline_arrivals=float(row["pred_arrivals"]),
                stressed_arrivals=float(row["stressed_arrivals"]),
                baseline_nights=float(row["pred_nights"]),
                stressed_nights=float(row["stressed_nights"]),
                arrivals_delta_pct=float(row["arrivals_delta_pct"]),
                nights_delta_pct=float(row["nights_delta_pct"]),
                stress_corrected_pi_c=bool(row["stress_corrected_pi_c"]),
            )
        )

    return StressTestResponse(
        geo=req.geo,
        country=bundle.country_names.get(req.geo, req.geo),
        price_elasticity=elasticity,
        hicp_shock_pct=req.hicp_shock_pct,
        stringency_delta=req.stringency_delta,
        total_baseline_arrivals=base_arr,
        total_stressed_arrivals=st_arr,
        arrivals_change_pct=float((st_arr - base_arr) / max(base_arr, 1.0) * 100),
        total_baseline_nights=base_nig,
        total_stressed_nights=st_nig,
        nights_change_pct=float((st_nig - base_nig) / max(base_nig, 1.0) * 100),
        post_pi_c_violations=int((stressed_df["stressed_nights"] < stressed_df["stressed_arrivals"]).sum()),
        pi_c_protected_months=int(stressed_df["stress_corrected_pi_c"].sum()),
        monthly_impact=monthly_items,
    )


@router.get(
    "/market-regime/timeline",
    response_model=RegimeTimelineResponse,
    summary="Get Gaussian HMM market regime timeline & probabilities",
    tags=["Market Regimes"],
)
def get_market_regime_timeline(bundle: DSSDataBundle = Depends(get_bundle)) -> RegimeTimelineResponse:
    if bundle.history_df is None:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Historical features dataset not loaded.",
        )

    hmm_df = compute_hmm_regime_timeline(bundle.history_df)
    if hmm_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="HMM timeline is empty.")

    latest_row = hmm_df.iloc[-1]
    items = []
    for _, row in hmm_df.iterrows():
        items.append(
            RegimeTimelineItem(
                date=pd.to_datetime(row["date"]).strftime("%Y-%m-%d"),
                regime_state=int(row["regime_state"]),
                regime_name=str(row["regime_name"]),
                p_normal=float(row["p_normal"]),
                p_disruption=float(row["p_disruption"]),
                p_recovery=float(row["p_recovery"]),
            )
        )

    return RegimeTimelineResponse(
        total_observations=len(items),
        latest_state=int(latest_row["regime_state"]),
        latest_regime=str(latest_row["regime_name"]),
        timeline=items,
    )


@router.get(
    "/recovery/benchmark",
    response_model=RecoveryBenchmarkResponse,
    summary="Get cross-destination recovery benchmarks vs 2019 pre-crisis level",
    tags=["Decision Support"],
)
def get_recovery_benchmark(
    horizon_months: int = Query(12, ge=1, le=36),
    origin_label: Optional[str] = Query("2023-12-31"),
    bundle: DSSDataBundle = Depends(get_bundle),
) -> RecoveryBenchmarkResponse:
    _, cross_df = compute_recovery_benchmarking(
        bundle.forecasts_df,
        bundle.history_df,
        geo="FR",
        accom_type="total",
        horizon_months=horizon_months,
        origin_label=origin_label,
    )
    if cross_df.empty:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No recovery benchmarking available.")

    benchmarks = []
    for _, row in cross_df.iterrows():
        benchmarks.append(
            RecoveryBenchmarkItem(
                geo=str(row["geo"]),
                country=str(row["country"]),
                recovery_pct=float(row["recovery_pct"]) if pd.notna(row.get("recovery_pct")) else None,
                recovery_tier=str(row["recovery_tier"]) if pd.notna(row.get("recovery_tier")) else None,
                pred_los=float(row["pred_los"]) if pd.notna(row.get("pred_los")) else None,
                paper_case_study=str(row["paper_case_study"]) if pd.notna(row.get("paper_case_study")) else None,
            )
        )

    return RecoveryBenchmarkResponse(
        total_countries=len(benchmarks),
        horizon_months=horizon_months,
        benchmarks=benchmarks,
    )


@router.post(
    "/optimizer/pi-c",
    response_model=PiCOptimizerResponse,
    summary="Standalone Asymmetric Anchor Consistency Projection (Pi_C)",
    tags=["Optimizer & Algorithms"],
)
def post_pi_c_optimizer(req: PiCOptimizerRequest) -> PiCOptimizerResponse:
    if len(req.arrivals) != len(req.nights):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Arrays 'arrivals' and 'nights' must have the same length.",
        )

    arr_raw = np.array(req.arrivals, dtype=np.float32)
    nig_raw = np.array(req.nights, dtype=np.float32)

    arr_safe, nig_safe, raw_violations = CoupledConsistencyOptimizer.project_physical_pair(
        arr_raw, nig_raw
    )

    los = np.where(arr_safe > 0, nig_safe / np.maximum(arr_safe, 1e-6), 1.0)
    adjusted_nights = float(np.sum(nig_safe - nig_raw))

    return PiCOptimizerResponse(
        raw_arrivals=[float(x) for x in arr_raw],
        raw_nights=[float(x) for x in nig_raw],
        projected_arrivals=[float(x) for x in arr_safe],
        projected_nights=[float(x) for x in nig_safe],
        projected_los=[float(x) for x in los],
        violations_detected=int(np.sum(raw_violations)),
        total_nights_adjusted=adjusted_nights,
        all_constraints_satisfied=bool(np.all(nig_safe >= arr_safe) and np.all(arr_safe >= 0)),
    )


@router.post(
    "/reports/generate",
    response_model=ReportGenerateResponse,
    summary="Generate executive DMO governance briefing in Markdown or JSON",
    tags=["Reporting & Export"],
)
def post_generate_report(
    req: ReportGenerateRequest, bundle: DSSDataBundle = Depends(get_bundle)
) -> ReportGenerateResponse:
    generator = ExecutiveReportGenerator(bundle)
    geo_name = bundle.country_names.get(req.geo, req.geo)

    from datetime import datetime, timezone
    import json

    if req.format.lower() == "json":
        report_dict = generator.generate_report_dict(
            geo=req.geo,
            accom_type=req.accommodation_type,
            horizon_months=req.horizon_months,
            origin_label=req.origin_label,
        )
        content_str = json.dumps(report_dict, indent=2, default=str)
    else:
        content_str = generator.generate_markdown_report(
            geo=req.geo,
            accom_type=req.accommodation_type,
            horizon_months=req.horizon_months,
            origin_label=req.origin_label,
        )

    return ReportGenerateResponse(
        geo=req.geo,
        country=geo_name,
        format=req.format.lower(),
        generated_at=datetime.now(timezone.utc).isoformat(),
        content=content_str,
    )
