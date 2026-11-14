"""
Pydantic data models and schemas for the RegiTour REST API.
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    status: str = Field(..., example="healthy")
    version: str = Field(..., example="1.0.0")
    total_destinations: int = Field(..., example=38)
    available_origins: List[str] = Field(..., example=["2023-12-31", "2024-12-31", "2025-12-31"])
    source_mode: str = Field(..., example="reviewer_experiments")


class DestinationInfo(BaseModel):
    geo: str = Field(..., example="FR")
    name: str = Field(..., example="France")


class DestinationsResponse(BaseModel):
    total: int = Field(..., example=38)
    destinations: List[DestinationInfo]
    nace_categories: Dict[str, str]
    available_origins: List[str]


class PlanRequest(BaseModel):
    geo: str = Field(default="FR", description="2-letter Eurostat country code (e.g. FR, CY, SE, DK)")
    accommodation_type: str = Field(default="total", description="hotel, short_stay, camping, or total")
    horizon_months: int = Field(default=12, ge=1, le=36, description="Forecast horizon in months")
    origin_label: Optional[str] = Field(default="2023-12-31", description="Origin date for the forecast")


class MonthlyPlanItem(BaseModel):
    date: str
    pred_arrivals_safe: float
    pred_nights_safe: float
    pred_los: float
    daily_arrivals_throughput: float
    daily_nights_demand: float
    lower_arrivals: Optional[float] = None
    upper_arrivals: Optional[float] = None
    lower_nights: Optional[float] = None
    upper_nights: Optional[float] = None
    actual_arrivals: Optional[float] = None
    actual_nights: Optional[float] = None


class PlanSummary(BaseModel):
    total_arrivals: float
    total_nights: float
    avg_los: float
    peak_daily_arrivals: float
    peak_daily_nights: float
    raw_violations: int
    dlinear_violations: int
    post_violations: int
    total_nights_corrected: float


class PlanResponse(BaseModel):
    geo: str
    country: str
    accommodation_type: str
    horizon_months: int
    origin_label: str
    summary: PlanSummary
    monthly_plan: List[MonthlyPlanItem]


class StaffingRequest(BaseModel):
    geo: str = Field(default="FR")
    horizon_months: int = Field(default=12, ge=1, le=36)
    origin_label: Optional[str] = Field(default="2023-12-31")
    hours_per_arrival: float = Field(default=0.50, gt=0)
    hours_per_night: float = Field(default=0.35, gt=0)
    hours_per_fte_month: float = Field(default=160.0, gt=0)
    maintenance_occ_threshold_pct: float = Field(default=35.0, ge=5.0, le=90.0)
    guests_per_unit: Optional[Dict[str, float]] = Field(
        default_factory=lambda: {"hotel": 1.8, "short_stay": 2.6, "camping": 2.8}
    )


class NACEStaffingCategory(BaseModel):
    nace_code: str
    nace_name: str
    peak_month: str
    peak_fte: float
    trough_month: str
    trough_fte: float
    seasonal_fte_swing: float
    maintenance_window_months: List[str]
    peak_strain_months: List[str]


class StaffingSummary(BaseModel):
    total_peak_fte: float
    total_baseline_fte: float
    total_seasonal_hiring_need: float
    categories: Dict[str, NACEStaffingCategory]


class MonthlyStaffingItem(BaseModel):
    date: str
    nace_code: str
    pred_arrivals: float
    pred_nights: float
    required_hours: float
    required_fte: float
    occupancy_pct: float
    is_maintenance_window: bool
    is_peak_strain: bool


class StaffingResponse(BaseModel):
    geo: str
    country: str
    horizon_months: int
    origin_label: str
    summary: StaffingSummary
    monthly_breakdown: List[MonthlyStaffingItem]


class StressTestRequest(BaseModel):
    geo: str = Field(default="CY")
    accommodation_type: str = Field(default="total")
    horizon_months: int = Field(default=12, ge=1, le=36)
    origin_label: Optional[str] = Field(default="2023-12-31")
    hicp_shock_pct: float = Field(default=15.0, description="HICP price shock percentage (+15%)")
    stringency_delta: float = Field(default=20.0, description="Oxford stringency delta (+20 points)")


class MonthlyStressItem(BaseModel):
    date: str
    baseline_arrivals: float
    stressed_arrivals: float
    baseline_nights: float
    stressed_nights: float
    arrivals_delta_pct: float
    nights_delta_pct: float
    stress_corrected_pi_c: bool


class StressTestResponse(BaseModel):
    geo: str
    country: str
    price_elasticity: float
    hicp_shock_pct: float
    stringency_delta: float
    total_baseline_arrivals: float
    total_stressed_arrivals: float
    arrivals_change_pct: float
    total_baseline_nights: float
    total_stressed_nights: float
    nights_change_pct: float
    post_pi_c_violations: int
    pi_c_protected_months: int
    monthly_impact: List[MonthlyStressItem]


class RegimeTimelineItem(BaseModel):
    date: str
    regime_state: int
    regime_name: str
    p_normal: float
    p_disruption: float
    p_recovery: float


class RegimeTimelineResponse(BaseModel):
    total_observations: int
    latest_state: int
    latest_regime: str
    timeline: List[RegimeTimelineItem]


class RecoveryBenchmarkItem(BaseModel):
    geo: str
    country: str
    recovery_pct: Optional[float] = None
    recovery_tier: Optional[str] = None
    pred_los: Optional[float] = None
    paper_case_study: Optional[str] = None


class RecoveryBenchmarkResponse(BaseModel):
    total_countries: int
    horizon_months: int
    benchmarks: List[RecoveryBenchmarkItem]


class PiCOptimizerRequest(BaseModel):
    arrivals: List[float] = Field(..., description="Array of raw predicted arrival counts")
    nights: List[float] = Field(..., description="Array of raw predicted room-night counts")
    c_los: float = Field(default=1.0, ge=0.0, description="Minimum physical length of stay constraint (nights/arrival)")


class PiCOptimizerResponse(BaseModel):
    raw_arrivals: List[float]
    raw_nights: List[float]
    projected_arrivals: List[float]
    projected_nights: List[float]
    projected_los: List[float]
    violations_detected: int
    total_nights_adjusted: float
    all_constraints_satisfied: bool


class ReportGenerateRequest(BaseModel):
    geo: str = Field(default="FR")
    accommodation_type: str = Field(default="total")
    horizon_months: int = Field(default=12, ge=1, le=36)
    origin_label: Optional[str] = Field(default="2023-12-31")
    format: str = Field(default="markdown", description="'markdown' or 'json'")


class ReportGenerateResponse(BaseModel):
    geo: str
    country: str
    format: str
    generated_at: str
    content: str

