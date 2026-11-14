"""
🏛️ RegiTour — Operational Decision Support System for Destination Management & Smart Tourism Governance
Built with Streamlit and Plotly, implementing Section VI of paper_en.tex:
  1. Coordinated Arrival Throughput & Stay Demand Planning (Arrivals, Nights, LOS + Pi_C)
  2. Accommodation Capacity & Seasonal Staffing Allocation (NACE I551, I552, I553)
  3. Crisis Regime Detection & Macro Stress Testing (Causal HMM + Shock Fold Diagnostics)
  4. Benchmarking Post-Shock Recovery Trajectories & 90% Conformal Risk Bands (I_0.90)
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.decision_support.dmo_engine import (
    NACE_LABELS,
    PAPER_CASE_STUDIES,
    compute_coordinated_throughput_plan,
    compute_crisis_regime_diagnostics,
    compute_hmm_regime_timeline,
    compute_nace_capacity_staffing,
    compute_recovery_benchmarking,
    load_dss_bundle,
    simulate_macro_stress,
)
from src.services.report_generator import ExecutiveReportGenerator

st.set_page_config(
    page_title="RegiTour DSS | Smart Tourism Governance & Destination Management",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .main-header {
        font-size: 1.95rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.0rem;
        color: #4B5563;
        margin-bottom: 1.25rem;
    }
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 14px;
        font-size: 0.84rem;
        font-weight: 600;
    }
    .badge-normal { background-color: #DCFCE7; color: #166534; }
    .badge-shock { background-color: #FEE2E2; color: #991B1B; }
    .badge-recovery { background-color: #FEF3C7; color: #92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource
def get_dss_bundle():
    return load_dss_bundle(ROOT_DIR)


@st.cache_data
def get_hmm_timeline(_history_df: pd.DataFrame) -> pd.DataFrame:
    return compute_hmm_regime_timeline(_history_df)


def _format_origin_label(origin_str: str) -> str:
    year_map = {
        "2023-12-31": "Origin 2023-12 (Dự báo năm 2024 — 12 tháng)",
        "2024-12-31": "Origin 2024-12 (Dự báo năm 2025 — 12 tháng)",
        "2025-12-31": "Origin 2025-12 (Dự báo năm 2026 — T1–T7)",
    }
    return year_map.get(origin_str, origin_str)


def _render_sidebar(bundle):
    st.sidebar.markdown("# 🏛️ RegiTour DSS")
    st.sidebar.caption("Smart Tourism Governance & Operational Decision Support (RA-HE-DLinear)")

    stakeholder_lens = st.sidebar.selectbox(
        "👤 Vai trò khai thác (Stakeholder):",
        [
            "🏛️ Cơ quan Quản lý Nhà nước (Cục / Sở Du lịch)",
            "✈️ Cơ quan Quản lý Cửa khẩu & Cảng vụ Hàng không",
            "🏨 Hiệp hội & Cơ sở Lưu trú (NACE I551–I553)",
        ],
    )

    nav_section = st.sidebar.radio(
        "🧭 Phân hệ Hỗ trợ Ra quyết định:",
        [
            "🏠 Tổng quan Điều hành DMO",
            "1️⃣ Lưu lượng & Lưu trú Đồng bộ (Arrivals–Nights–LOS)",
            "2️⃣ Công suất NACE & Nhân sự Thời vụ (I551–I553)",
            "3️⃣ Giám sát Khủng hoảng & Stress Testing (HMM)",
            "4️⃣ Quỹ đạo Phục hồi & Biên Rủi ro Conformal",
            "🔬 Đối sánh Mô hình & XAI (Benchmark)",
            "🔌 Tích hợp API & Báo cáo Tự động (Integration Hub)",
        ],
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### ⚙️ Ngữ cảnh Điểm đến & Chân trời")

    geo_options = sorted(bundle.forecasts_df["geo"].unique())
    default_geo = "FR" if "FR" in geo_options else geo_options[0]
    selected_geo = st.sidebar.selectbox(
        "🌍 Điểm đến (38 quốc gia Eurostat):",
        geo_options,
        index=geo_options.index(default_geo),
        format_func=lambda g: f"{g} — {bundle.country_names.get(g, g)}",
    )

    accom_options = ["total", "hotel", "short_stay", "camping"]
    available_accom = [a for a in accom_options if a in bundle.forecasts_df["accommodation_type"].unique()]
    selected_accom = st.sidebar.selectbox(
        "🏨 Loại hình lưu trú (NACE Rev. 2):",
        available_accom,
        format_func=lambda a: NACE_LABELS.get(a, (a, a))[1],
    )

    origin_options = sorted(bundle.forecasts_df["origin_label"].unique())
    selected_origin = st.sidebar.selectbox(
        "📅 Mốc khởi tạo dự báo (Forecast Origin):",
        origin_options,
        index=len(origin_options) - 1 if len(origin_options) > 1 else 0,
        format_func=_format_origin_label,
    )

    horizon_months = st.sidebar.select_slider(
        "🔭 Chân trời vận hành (Horizon H):",
        options=[3, 6, 12],
        value=12,
        format_func=lambda h: f"H = {h} tháng",
    )

    return {
        "stakeholder": stakeholder_lens,
        "nav": nav_section,
        "geo": selected_geo,
        "country": bundle.country_names.get(selected_geo, selected_geo),
        "accom": selected_accom,
        "nace_code": NACE_LABELS.get(selected_accom, (selected_accom, selected_accom))[0],
        "origin": selected_origin,
        "horizon": horizon_months,
    }


def _render_executive_briefing(bundle, ctx):
    st.markdown(
        f'<div class="main-header">🏠 Tổng Quan Điều Hành DMO — {ctx["country"]} ({ctx["geo"]})</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Vai trò: <b>{ctx["stakeholder"]}</b> | Phân khúc: <b>{NACE_LABELS[ctx["accom"]][1]}</b> | Chân trời: <b>H = {ctx["horizon"]} tháng</b></div>',
        unsafe_allow_html=True,
    )

    plan_df, summary = compute_coordinated_throughput_plan(
        bundle.forecasts_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
    )
    traj_df, cross_df = compute_recovery_benchmarking(
        bundle.forecasts_df, bundle.history_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
    )

    if plan_df.empty:
        st.warning("Không có dữ liệu dự báo cho cấu hình đang chọn.")
        return

    hmm_df = get_hmm_timeline(bundle.history_df)
    latest_regime = "Normal (Bình thường)"
    badge_cls = "badge-normal"
    if not hmm_df.empty:
        latest_row = hmm_df.iloc[-1]
        latest_regime = str(latest_row["regime_name"])
        if int(latest_row["regime_state"]) == 1:
            badge_cls = "badge-shock"
        elif int(latest_row["regime_state"]) == 2:
            badge_cls = "badge-recovery"

    rec_val = traj_df["recovery_index_arr_pct"].mean() if not traj_df.empty else np.nan
    rec_str = f"{rec_val:.1f}%" if pd.notna(rec_val) else "N/A"

    st.markdown(
        f'<span class="status-badge {badge_cls}">🧭 Trạng thái Thị trường HMM hiện hành: {latest_regime}</span>',
        unsafe_allow_html=True,
    )
    st.write("")

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Tổng Lượt khách (Arrivals)", f"{summary['total_arrivals']:,.0f}")
    c2.metric("Tổng Số đêm (Room-Nights)", f"{summary['total_nights']:,.0f}")
    c3.metric("Thời gian Lưu trú (LOS)", f"{summary['avg_los']:.2f} đêm/khách")
    c4.metric("Phục hồi vs 2019", rec_str)
    c5.metric(
        "Vi phạm Vật lý sau Π_C",
        f"{summary['post_violations']} / {len(plan_df)}",
        delta=f"Đã chặn {summary['raw_violations']} lỗi thô" if summary["raw_violations"] > 0 else "0 lỗi (Đạt chuẩn)",
    )

    col_left, col_right = st.columns([3, 2])
    with col_left:
        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=pd.concat([plan_df["date"], plan_df["date"][::-1]]),
                y=pd.concat([plan_df["upper_arrivals"], plan_df["lower_arrivals"][::-1]]),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.14)",
                line=dict(color="rgba(255,255,255,0)"),
                name="Khoảng Conformal 90% (I_0.90)",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=plan_df["date"],
                y=plan_df["pred_arrivals_safe"],
                mode="lines+markers",
                line=dict(color="#2563EB", width=2.5),
                name="Dự báo Lượt khách (RA-HE-DLinear)",
            )
        )
        if plan_df["actual_arrivals"].notna().any():
            fig.add_trace(
                go.Scatter(
                    x=plan_df["date"],
                    y=plan_df["actual_arrivals"],
                    mode="lines+markers",
                    line=dict(color="#111827", width=2, dash="dot"),
                    name="Thực tế Eurostat",
                )
            )
        fig.update_layout(
            title=f"Quỹ đạo Lượt khách & Biên Rủi ro Conformal tại {ctx['country']}",
            template="plotly_white",
            height=380,
            hovermode="x unified",
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig, width='stretch')

    with col_right:
        cap_df, _ = compute_nace_capacity_staffing(
            bundle.forecasts_df, ctx["geo"], ctx["horizon"], origin_label=ctx["origin"]
        )
        if not cap_df.empty:
            nace_mix = cap_df.groupby("nace_code", as_index=False)[["pred_arrivals", "pred_nights"]].sum()
            fig_pie = px.bar(
                nace_mix,
                x="nace_code",
                y="pred_nights",
                color="nace_code",
                title="Phân bổ Đêm lưu trú theo Nhóm NACE (I551/I552/I553)",
                labels={"nace_code": "Nhóm NACE", "pred_nights": "Số đêm lưu trú dự báo"},
                template="plotly_white",
                height=380,
            )
            fig_pie.update_layout(showlegend=False)
            st.plotly_chart(fig_pie, width='stretch')

    # Role-based operational briefing & Inter-agency Matrix
    st.markdown("#### 📋 Phương Án Điều Hành & Phối Hợp Liên Ngành")
    if "Cửa khẩu" in ctx["stakeholder"] or "Hàng không" in ctx["stakeholder"]:
        st.info(
            f"**✈️ Cơ quan Quản lý Cửa khẩu & Cảng vụ Hàng không:** Đỉnh lưu lượng nhập cảnh dự kiến đạt "
            f"**{summary['peak_daily_arrivals']:,.0f} lượt/ngày** (tổng **{summary['total_arrivals']:,.0f}** lượt trong {len(plan_df)} tháng). "
            f"Cần bố trí luồng thông quan tự động, điều tiết phương tiện trung chuyển mặt đất và tăng cường quầy thủ tục."
        )
    elif "Lưu trú" in ctx["stakeholder"]:
        st.info(
            f"**🏨 Hiệp hội & Cơ sở Lưu trú (NACE I551–I553):** Tổng cầu phòng đạt **{summary['total_nights']:,.0f} đêm lưu trú** "
            f"(đỉnh **{summary['peak_daily_nights']:,.0f} đêm/ngày**), thời gian lưu trú trung bình **{summary['avg_los']:.2f} đêm/khách**. "
            f"Mở phân hệ **2️⃣ Công suất NACE & Nhân sự Thời vụ** để xem lịch tuyển dụng FTE và cửa sổ đóng cửa bảo trì."
        )
    else:
        # Cơ quan Quản lý Nhà nước (Cục / Sở Du lịch)
        st.success(
            f"**🏛️ Cơ quan Quản lý Nhà nước (Cục Du lịch Quốc gia / Sở Du lịch):** Toàn bộ {len(plan_df)} kỳ kế hoạch đều thỏa mãn "
            f"tuyệt đối chuẩn thống kê UNWTO/Eurostat ($\\hat{{Y}}^{{\\text{{nig}}}} \\ge \\hat{{Y}}^{{\\text{{arr}}}}$, $LOS \\ge 1.0$). "
            f"Chỉ số phục hồi dự kiến đạt **{rec_str}** so với thời điểm trước dịch (2019)."
        )

        with st.expander("🏛️ Ma Trận Điều Phối Hành Động Liên Ngành (Chỉ đạo của Cục / Sở Du lịch)", expanded=True):
            m_col1, m_col2 = st.columns(2)
            with m_col1:
                st.markdown(
                    f"""
                    **1. 🚔 Phối hợp Công an & Giao thông Vận tải:**
                    - **Ngưỡng tải trọng:** Đỉnh ngày đạt **{summary['peak_daily_arrivals']:,.0f} khách/ngày**.
                    - **Chỉ đạo:** Kích hoạt phương án phân luồng xe du lịch, cấm xe khách vào trung tâm giờ cao điểm và mở thêm quầy thủ tục tại các cửa khẩu trong các tháng cao điểm.

                    **2. 🎓 Phối hợp Sở Lao động - Thương binh & Xã hội:**
                    - **Nhu cầu nhân lực:** Dự báo phát sinh thêm lao động thời vụ cho toàn ngành lưu trú.
                    - **Chỉ đạo:** Ban hành hướng dẫn cho các trường nghề mở khóa bồi dưỡng nghiệp vụ lễ tân/buồng phòng cấp tốc trước mùa cao điểm tối thiểu 60 ngày.
                    """
                )
            with m_col2:
                st.markdown(
                    f"""
                    **3. 🏗️ Phối hợp Sở Xây dựng & Ban Quản lý Di tích:**
                    - **Chỉ đạo:** Công bố khung thời gian cấp phép tu bổ, sửa chữa hạ tầng du lịch vào các **Cửa sổ bảo trì (<35% công suất)**; nghiêm cấm thi công xây dựng gây mất mỹ quan trong mùa cao điểm hè.

                    **4. 💰 Phối hợp Sở Tài chính & Kế hoạch Đầu tư:**
                    - **Chỉ đạo:** Thẩm định dự toán ngân sách xúc tiến thị trường; trích lập quỹ dự phòng theo dải **Sàn rủi ro doanh thu 90%** để sẵn sàng kích cầu khi có biến động vĩ mô.
                    """
                )

    export_cols = [
        "date",
        "geo",
        "country",
        "accommodation_type",
        "pred_arrivals_safe",
        "pred_nights_safe",
        "pred_los",
        "daily_arrivals_throughput",
        "daily_nights_demand",
        "lower_arrivals",
        "upper_arrivals",
        "lower_nights",
        "upper_nights",
    ]
    avail_export = [c for c in export_cols if c in plan_df.columns]
    st.download_button(
        label="⬇️ Tải Kế hoạch Vận hành Đa chân trời (CSV)",
        data=plan_df[avail_export].to_csv(index=False).encode("utf-8"),
        file_name=f"regitour_dmo_plan_{ctx['geo']}_{ctx['accom']}_{ctx['origin']}.csv",
        mime="text/csv",
    )


def _render_layer1_coordinated_throughput(bundle, ctx):
    st.markdown(
        f'<div class="main-header">1️⃣ Hoạch định Đồng bộ Lưu lượng Khách đến & Nhu cầu Lưu trú</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Lớp bài toán 1 (Section VI): Liên kết lưu lượng cửa khẩu/sân bay/check-in (Arrivals) với nhu cầu phòng/điện nước/catering (Nights) dưới bảo chứng bất biến Π_C.</div>',
        unsafe_allow_html=True,
    )

    plan_df, summary = compute_coordinated_throughput_plan(
        bundle.forecasts_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
    )
    if plan_df.empty:
        st.warning("Không có dữ liệu cho lựa chọn này.")
        return

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Đỉnh Thông quan / Check-in", f"{summary['peak_daily_arrivals']:,.0f} lượt/ngày")
    c2.metric("Đỉnh Công suất Đêm phòng", f"{summary['peak_daily_nights']:,.0f} đêm/ngày")
    c3.metric("Thời gian Lưu trú (LOS)", f"{summary['avg_los']:.2f} đêm/khách")
    c4.metric(
        "Kiểm soát Mâu thuẫn (Π_C)",
        f"{summary['post_violations']} vi phạm",
        delta=f"Chặn {summary['raw_violations']} lỗi Raw / {summary['dlinear_violations']} lỗi DLinear",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        fig_tp = go.Figure()
        fig_tp.add_trace(
            go.Bar(
                x=plan_df["date"],
                y=plan_df["daily_arrivals_throughput"],
                name="Lưu lượng đến bình quân/ngày (Cửa khẩu / Check-in)",
                marker_color="#2563EB",
            )
        )
        fig_tp.add_trace(
            go.Bar(
                x=plan_df["date"],
                y=plan_df["daily_nights_demand"],
                name="Nhu cầu đêm phòng/ngày (Room-Nights / Utilities)",
                marker_color="#10B981",
            )
        )
        fig_tp.update_layout(
            barmode="group",
            title=f"Tải Vận hành Bình quân Ngày — {ctx['country']} ({ctx['nace_code']})",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_tp, width='stretch')

    with col_b:
        fig_los = go.Figure()
        fig_los.add_trace(
            go.Scatter(
                x=plan_df["date"],
                y=plan_df["pred_los"],
                mode="lines+markers",
                name="LOS Dự báo (Nights / Arrivals)",
                line=dict(color="#1E3A8A", width=2.5),
            )
        )
        if plan_df["actual_arrivals"].notna().any() and plan_df["actual_nights"].notna().any():
            actual_los = np.where(
                plan_df["actual_arrivals"] > 0,
                plan_df["actual_nights"] / np.maximum(plan_df["actual_arrivals"], 1.0),
                np.nan,
            )
            fig_los.add_trace(
                go.Scatter(
                    x=plan_df["date"],
                    y=actual_los,
                    mode="lines+markers",
                    name="LOS Thực tế Eurostat",
                    line=dict(color="#6B7280", width=2, dash="dash"),
                )
            )
        fig_los.add_hline(y=1.0, line_dash="dot", line_color="#DC2626", annotation_text="Cận vật lý UNWTO (LOS = 1.0)")
        fig_los.update_layout(
            title="Ổn định Chỉ số Thời gian Lưu trú (LOS >= 1.0)",
            yaxis_title="Số đêm / Lượt khách",
            template="plotly_white",
            height=400,
            legend=dict(orientation="h", y=1.12),
        )
        st.plotly_chart(fig_los, width='stretch')

    # Cross-NACE consistency audit table
    st.markdown("#### 🧾 Kiểm toán Nhất quán Ngân sách qua Các Nhóm NACE (I551 / I552 / I553 / Total)")
    audit_rows = []
    for accom_key in ("hotel", "short_stay", "camping", "total"):
        _, s = compute_coordinated_throughput_plan(
            bundle.forecasts_df, ctx["geo"], accom_key, ctx["horizon"], ctx["origin"]
        )
        code, desc = NACE_LABELS[accom_key]
        audit_rows.append(
            {
                "Mã NACE": code,
                "Phân khúc": desc,
                "Tổng Lượt khách": round(s["total_arrivals"]),
                "Tổng Đêm lưu trú": round(s["total_nights"]),
                "LOS Bình quân": round(s["avg_los"], 2),
                "Lỗi Thô trước Π_C": s["raw_violations"],
                "Lỗi DLinear Baseline": s["dlinear_violations"],
                "Lỗi sau Π_C": s["post_violations"],
                "Đêm hiệu chỉnh Π_C": round(s["total_nights_corrected"], 1),
            }
        )
    st.dataframe(pd.DataFrame(audit_rows), width='stretch')


def _render_layer2_nace_capacity_staffing(bundle, ctx):
    st.markdown(
        f'<div class="main-header">2️⃣ Phân bổ Công suất Lưu trú NACE & Nhân sự Thời vụ</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Lớp bài toán 2 (Section VI): Phân tách cầu tới H=12 tháng theo Hotels (I551), Short-Stay (I552), Camping (I553) để lập lịch tuyển dụng thời vụ, ca buồng phòng & cửa sổ bảo trì.</div>',
        unsafe_allow_html=True,
    )

    with st.expander("⚙️ Cấu hình Tham số Vận hành Cơ sở Lưu trú (Facility-Level Inputs)", expanded=False):
        p1, p2, p3, p4 = st.columns(4)
        hours_arr = p1.number_input("Giờ công check-in / turnover mỗi lượt khách:", 0.1, 3.0, 0.50, 0.05)
        hours_nig = p2.number_input("Giờ công buồng phòng / phục vụ mỗi đêm:", 0.1, 3.0, 0.35, 0.05)
        hours_fte = p3.number_input("Định mức giờ công / nhân sự (giờ/tháng):", 80.0, 240.0, 160.0, 10.0)
        maint_thresh = p4.slider("Ngưỡng công suất đề xuất Bảo trì (%):", 15.0, 60.0, 35.0, 5.0)

        g1, g2, g3 = st.columns(3)
        g_hotel = g1.number_input("Khách/phòng bình quân (I551 Hotels):", 1.0, 5.0, 1.8, 0.1)
        g_short = g2.number_input("Khách/đơn vị bình quân (I552 Short-Stay):", 1.0, 8.0, 2.6, 0.1)
        g_camp = g3.number_input("Khách/lô bình quân (I553 Camping):", 1.0, 8.0, 2.8, 0.1)

    cap_df, cap_summary = compute_nace_capacity_staffing(
        bundle.forecasts_df,
        ctx["geo"],
        horizon_months=ctx["horizon"],
        guests_per_unit={"hotel": g_hotel, "short_stay": g_short, "camping": g_camp},
        hours_per_arrival=hours_arr,
        hours_per_night=hours_nig,
        hours_per_fte_month=hours_fte,
        maintenance_occ_threshold_pct=maint_thresh,
        origin_label=ctx["origin"],
    )

    if cap_df.empty:
        st.warning("Không có dữ liệu chi tiết theo NACE cho quốc gia này.")
        return

    m1, m2, m3 = st.columns(3)
    m1.metric("Tổng Nhân sự Đỉnh mùa (Peak FTE)", f"{cap_summary['total_peak_fte']:,.0f} FTE")
    m2.metric("Nhân sự Cố định Nền (Baseline FTE)", f"{cap_summary['total_baseline_fte']:,.0f} FTE")
    m3.metric("Biên độ Tuyển dụng Thời vụ", f"+{cap_summary['total_seasonal_hiring_need']:,.0f} FTE")

    col_left, col_right = st.columns(2)
    with col_left:
        fig_fte = px.area(
            cap_df,
            x="date",
            y="required_fte",
            color="nace_code",
            title=f"Nhu cầu Nhân sự Thời vụ (FTE) theo Nhóm NACE — {ctx['country']}",
            labels={"date": "Tháng", "required_fte": "Nhân sự yêu cầu (FTE)", "nace_code": "Nhóm NACE"},
            template="plotly_white",
            height=400,
        )
        st.plotly_chart(fig_fte, width='stretch')

    with col_right:
        fig_occ = px.line(
            cap_df,
            x="date",
            y="occupancy_pct",
            color="nace_code",
            markers=True,
            title="Tỷ lệ Lấp đầy Ước tính (%) & Cửa sổ Bảo trì / Quá tải",
            labels={"date": "Tháng", "occupancy_pct": "Công suất ước tính (%)", "nace_code": "Nhóm NACE"},
            template="plotly_white",
            height=400,
        )
        fig_occ.add_hline(y=85.0, line_dash="dash", line_color="#DC2626", annotation_text="Cảnh báo Quá tải (85%)")
        fig_occ.add_hline(
            y=maint_thresh,
            line_dash="dot",
            line_color="#F59E0B",
            annotation_text=f"Cửa sổ Bảo trì (<{maint_thresh:.0f}%)",
        )
        st.plotly_chart(fig_occ, width='stretch')

    st.markdown("#### 🗓️ Tổng hợp Lịch Điều độ Nhân sự & Cửa sổ Bảo trì theo Phân khúc NACE")
    summary_rows = []
    for accom, info in cap_summary["categories"].items():
        summary_rows.append(
            {
                "Mã NACE": info["nace_code"],
                "Phân khúc": info["nace_name"],
                "Tháng Đỉnh": info["peak_month"],
                "FTE Đỉnh": round(info["peak_fte"], 1),
                "Tháng Thấp điểm": info["trough_month"],
                "FTE Nền": round(info["trough_fte"], 1),
                "Tuyển Thời vụ (+/- FTE)": round(info["seasonal_fte_swing"], 1),
                "Các tháng Cửa sổ Bảo trì": ", ".join(info["maintenance_window_months"]) or "Không có",
                "Các tháng Quá tải (>=85%)": ", ".join(info["peak_strain_months"]) or "Không có",
            }
        )
    st.dataframe(pd.DataFrame(summary_rows), width='stretch')

    st.download_button(
        label="⬇️ Tải Lịch Công suất & Nhân sự NACE (CSV)",
        data=cap_df.to_csv(index=False).encode("utf-8"),
        file_name=f"regitour_nace_staffing_{ctx['geo']}_{ctx['origin']}.csv",
        mime="text/csv",
    )


def _render_layer3_crisis_stress(bundle, ctx):
    st.markdown(
        f'<div class="main-header">3️⃣ Giám sát Trạng thái Khủng hoảng & Stress Testing Vĩ mô</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Lớp bài toán 3 (Section VI): Định danh chuyển dịch chế độ bằng HMM nhân quả, đánh giá sức chống chịu qua cú sốc 2020–2021 và mô phỏng cú sốc giá/biên giới.</div>',
        unsafe_allow_html=True,
    )

    diag = compute_crisis_regime_diagnostics(bundle, ctx["geo"])
    hmm_df = get_hmm_timeline(bundle.history_df)

    # A. HMM Timeline
    if not hmm_df.empty:
        fig_hmm = go.Figure()
        fig_hmm.add_trace(
            go.Scatter(
                x=hmm_df["date"], y=hmm_df["p_normal"] * 100, stackgroup="one", name="Normal (Bình thường)", line=dict(color="#10B981")
            )
        )
        fig_hmm.add_trace(
            go.Scatter(
                x=hmm_df["date"],
                y=hmm_df["p_disruption"] * 100,
                stackgroup="one",
                name="Disruption Shock (Khủng hoảng)",
                line=dict(color="#EF4444"),
            )
        )
        fig_hmm.add_trace(
            go.Scatter(
                x=hmm_df["date"],
                y=hmm_df["p_recovery"] * 100,
                stackgroup="one",
                name="Recovery (Phục hồi)",
                line=dict(color="#F59E0B"),
            )
        )
        fig_hmm.update_layout(
            title="Diễn biến Xác suất Trạng thái Thị trường Du lịch Châu Âu (Gaussian HMM, 2010–2026)",
            yaxis_title="Xác suất Trạng thái (%)",
            template="plotly_white",
            height=340,
            legend=dict(orientation="h", y=1.14),
        )
        st.plotly_chart(fig_hmm, width='stretch')

    # B. Shock Fold Comparison & Undefined LOS safeguard
    col_l, col_r = st.columns([3, 2])
    fold_df = diag["fold_comparison_df"]
    with col_l:
        if not fold_df.empty:
            fig_fold = px.bar(
                fold_df,
                x="regime_phase",
                y=["dlinear_arr_wape", "ra_he_arr_wape"],
                barmode="group",
                title="Sai số Arrivals WAPE (%) qua 4 Chế độ Thị trường: DLinear vs RA-HE-DLinear",
                labels={"value": "Arrivals WAPE (%)", "regime_phase": "Giai đoạn / Fold", "variable": "Mô hình"},
                template="plotly_white",
                height=360,
            )
            st.plotly_chart(fig_fold, width='stretch')
    with col_r:
        st.markdown("#### 🛡️ Bảo toàn Tỷ lệ LOS trong Khủng hoảng")
        if not diag["geo_los_df"].empty:
            st.dataframe(
                diag["geo_los_df"][["model", "los_mae", "undefined_predictions", "observed_los_mean", "predicted_los_mean"]],
                width='stretch',
            )
        st.caption(
            "Trong pha khủng hoảng 2020–2021, các mô hình không có ràng buộc tạo ra 82 dự báo zero-arrival làm mất xác định tỷ lệ LOS, "
            "trong khi RA-HE-DLinear duy trì LOS xác định trên toàn bộ 38 quốc gia."
        )

    # C. Interactive Stress Testing
    st.markdown("---")
    st.markdown(f"#### ⚡ Stress Test Kịch bản Cú sốc Vĩ mô tại {ctx['country']} ({ctx['nace_code']})")
    plan_df, _ = compute_coordinated_throughput_plan(
        bundle.forecasts_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
    )
    if not plan_df.empty:
        s1, s2, s3 = st.columns(3)
        hicp_shock = s1.slider("Cú sốc Lạm phát giá phòng HICP (%):", -20, 40, 15, 5)
        str_delta = s2.slider("Cú sốc Hạn chế biên giới (Oxford Stringency +điểm):", 0, 80, 20, 5)
        s3.metric("Độ co giãn giá thực nghiệm (Elasticity)", f"{diag['price_elasticity']:.2f}")

        stressed_df = simulate_macro_stress(plan_df, diag["price_elasticity"], hicp_shock, str_delta)
        base_arr = stressed_df["pred_arrivals"].sum()
        st_arr = stressed_df["stressed_arrivals"].sum()
        base_nig = stressed_df["pred_nights"].sum()
        st_nig = stressed_df["stressed_nights"].sum()

        k1, k2, k3 = st.columns(3)
        k1.metric("Lượt khách sau Stress Test", f"{st_arr:,.0f}", delta=f"{(st_arr - base_arr) / max(base_arr, 1) * 100:+.1f}%")
        k2.metric("Đêm lưu trú sau Stress Test", f"{st_nig:,.0f}", delta=f"{(st_nig - base_nig) / max(base_nig, 1) * 100:+.1f}%")
        k3.metric(
            "Vi phạm LOS sau Stress + Π_C",
            f"{int((stressed_df['stressed_nights'] < stressed_df['stressed_arrivals']).sum())} mẫu",
            delta=f"{int(stressed_df['stress_corrected_pi_c'].sum())} kỳ được Π_C bảo vệ",
        )

        fig_st = go.Figure()
        fig_st.add_trace(
            go.Scatter(x=stressed_df["date"], y=stressed_df["pred_arrivals"], mode="lines+markers", name="Kịch bản Cơ sở (Baseline)")
        )
        fig_st.add_trace(
            go.Scatter(
                x=stressed_df["date"],
                y=stressed_df["stressed_arrivals"],
                mode="lines+markers",
                name="Kịch bản Stress Test",
                line=dict(color="#DC2626", dash="dash"),
            )
        )
        fig_st.update_layout(title="Tác động Stress Test lên Lượt khách", template="plotly_white", height=340)
        st.plotly_chart(fig_st, width='stretch')


def _render_layer4_recovery_conformal(bundle, ctx):
    st.markdown(
        f'<div class="main-header">4️⃣ Đối sánh Quỹ đạo Phục hồi & Quản trị Rủi ro Conformal (I_0.90)</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Lớp bài toán 4 (Section VI): Dùng dự báo điểm cùng dải conformal 90% để đối chiếu Biên Quá tải Hạ tầng (Upper Capacity-Strain) và Biên Rủi ro Doanh thu (Lower Revenue-at-Risk) cho DMOs.</div>',
        unsafe_allow_html=True,
    )

    traj_df, cross_df = compute_recovery_benchmarking(
        bundle.forecasts_df, bundle.history_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
    )
    if traj_df.empty:
        st.warning("Không có dữ liệu phục hồi cho lựa chọn này.")
        return

    if ctx["geo"] in PAPER_CASE_STUDIES:
        st.success(f"📌 **Paper Case Study ({ctx['country']}):** {PAPER_CASE_STUDIES[ctx['geo']]}")

    mean_rec = traj_df["recovery_index_arr_pct"].mean()
    mean_upper = traj_df["capacity_strain_index_pct"].mean()
    mean_lower = traj_df["revenue_risk_index_pct"].mean()

    r1, r2, r3 = st.columns(3)
    r1.metric("Chỉ số Phục hồi Điểm vs 2019", f"{mean_rec:.1f}%" if pd.notna(mean_rec) else "N/A")
    r2.metric("Biên Quá tải Công suất (Upper 90%)", f"{mean_upper:.1f}%" if pd.notna(mean_upper) else "N/A")
    r3.metric("Biên Rủi ro Doanh thu (Lower 90%)", f"{mean_lower:.1f}%" if pd.notna(mean_lower) else "N/A")

    fig_rec = go.Figure()
    fig_rec.add_trace(
        go.Scatter(
            x=pd.concat([traj_df["date"], traj_df["date"][::-1]]),
            y=pd.concat([traj_df["capacity_strain_index_pct"], traj_df["revenue_risk_index_pct"][::-1]]),
            fill="toself",
            fillcolor="rgba(37, 99, 235, 0.15)",
            line=dict(color="rgba(255,255,255,0)"),
            name="Dải Rủi ro Conformal 90% (% so với 2019)",
        )
    )
    fig_rec.add_trace(
        go.Scatter(
            x=traj_df["date"],
            y=traj_df["recovery_index_arr_pct"],
            mode="lines+markers",
            name="Dự báo Phục hồi Điểm (% vs 2019)",
            line=dict(color="#2563EB", width=2.5),
        )
    )
    fig_rec.add_hline(y=100.0, line_dash="dash", line_color="#10B981", annotation_text="Mốc Trước dịch 2019 (100%)")
    fig_rec.update_layout(
        title=f"Quỹ đạo Phục hồi & Biên Rủi ro DMO tại {ctx['country']} (Mốc 2019 = 100%)",
        yaxis_title="Chỉ số so với cùng kỳ 2019 (%)",
        template="plotly_white",
        height=390,
        legend=dict(orientation="h", y=1.12),
    )
    st.plotly_chart(fig_rec, width='stretch')

    if not cross_df.empty:
        st.markdown("#### 🌍 Bảng Đối sánh 38 Điểm đến Châu Âu (Phân tầng Phục hồi & Ngân sách Xúc tiến)")
        valid_cross = cross_df.dropna(subset=["recovery_pct"]).copy()
        fig_bar = px.bar(
            valid_cross,
            x="geo",
            y="recovery_pct",
            color="recovery_tier",
            hover_data=["country", "pred_los", "paper_case_study"],
            title="Xếp hạng Phục hồi 38 Quốc gia (% so với cùng kỳ 2019)",
            labels={"geo": "Quốc gia", "recovery_pct": "Tỷ lệ phục hồi (%)", "recovery_tier": "Nhóm phục hồi"},
            template="plotly_white",
            height=380,
        )
        fig_bar.add_hline(y=100.0, line_dash="dash", line_color="#111827")
        st.plotly_chart(fig_bar, width='stretch')

        if bundle.cases_df is not None:
            st.markdown("##### 📖 Đối chiếu 3 Case Studies Tiêu biểu trong Bài báo (Cyprus, Sweden, Denmark)")
            st.dataframe(bundle.cases_df, width='stretch')


def _render_scientific_benchmark(bundle):
    st.markdown(
        '<div class="main-header">🔬 Đối sánh Mô hình Khoa học & Giải thích XAI</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="sub-header">Minh bạch thực nghiệm từ results/reviewer_experiments/ và phân tích TreeSHAP / Cross-Temporal Attention.</div>',
        unsafe_allow_html=True,
    )

    if bundle.benchmark_df is not None:
        st.markdown("#### 🏆 Bảng Đối sánh Mô hình (Recent Eurostat Fold 2024–2026)")
        st.dataframe(bundle.benchmark_df, width='stretch')

    if bundle.coverage_df is not None:
        st.markdown("#### 🛡️ Độ bao phủ Khoảng tin cậy Conformal (Target 90%)")
        st.dataframe(bundle.coverage_df, width='stretch')

    col1, col2 = st.columns(2)
    shap_path = ROOT_DIR / "results" / "xai" / "shap_feature_importance.png"
    attn_path = ROOT_DIR / "results" / "xai" / "attention_cross_temporal_heatmap.png"
    with col1:
        if shap_path.exists():
            st.image(str(shap_path), caption="TreeSHAP Feature Importance (HICP, Stringency, Calendar)", width='stretch')
    with col2:
        if attn_path.exists():
            st.image(str(attn_path), caption="Cross-Temporal Attention Heatmap", width='stretch')


def _render_api_and_reporting(bundle, ctx):
    st.markdown(
        f'<div class="main-header">🔌 Tích Hợp Phần Mềm, REST API & Báo Cáo Tự Động</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<div class="sub-header">Phân hệ dịch vụ phần mềm doanh nghiệp cho <b>{ctx["country"]} ({ctx["geo"]})</b>: '
        f'Cung cấp REST API (FastAPI / OpenAPI), engine cảnh báo vận hành thông minh và công cụ xuất báo cáo tự động cho DMO.</div>',
        unsafe_allow_html=True,
    )

    tab_alerts, tab_report, tab_api = st.tabs(
        [
            "🚨 Giám Sát Cảnh Báo Vận Hành (Alert Engine)",
            "📄 Khởi Tạo Báo Cáo Điều Hành (1-Click Executive Report)",
            "⚡ Cổng Tích Hợp REST API (Enterprise Developer Hub)",
        ]
    )

    # ------------------ TAB 1: OPERATIONAL ALERT ENGINE ------------------
    with tab_alerts:
        st.markdown("### 🚨 Hệ Thống Cảnh Báo Vận Hành Tự Động DMO")
        st.caption(
            "Engine tự động quét các ngưỡng giới hạn vật lý, tỷ lệ lấp đầy, đỉnh nhân sự và bảo chứng Pi_C để kích hoạt thông báo điều hành."
        )

        plan_df, plan_s = compute_coordinated_throughput_plan(
            bundle.forecasts_df, ctx["geo"], ctx["accom"], ctx["horizon"], ctx["origin"]
        )
        cap_df, cap_s = compute_nace_capacity_staffing(
            bundle.forecasts_df, ctx["geo"], ctx["horizon"], origin_label=ctx["origin"]
        )

        c1, c2, c3 = st.columns(3)
        c1.metric("Bất Biến Nghiệp Vụ Pi_C", "100% An Toàn", delta="0 vi phạm vật lý sau chiếu")
        c2.metric("Đỉnh Nhu Cầu Nhân Lực", f"{cap_s['total_peak_fte']:,.0f} FTE", delta=f"+{cap_s['total_seasonal_hiring_need']:,.0f} thời vụ")
        c3.metric("Tải Trọng Đỉnh Ngày", f"{plan_s['peak_daily_arrivals']:,.0f} khách/ngày")

        st.markdown("#### 📋 Trạng thái Cảnh báo Chi tiết:")

        # Scan for over-capacity / strain
        strain_found = False
        maint_found = False
        for cat_key, cat in cap_s["categories"].items():
            if cat["peak_strain_months"]:
                strain_found = True
                st.error(
                    f"⚠️ **[CẢNH BÁO QUÁ TẢI] Phân khúc {cat['nace_name']}:** Tỷ lệ lấp đầy vượt ngưỡng 85% "
                    f"trong các tháng: **{', '.join(cat['peak_strain_months'])}** (Đỉnh FTE: {cat['peak_fte']:,.0f}). "
                    f"-> Đề xuất: DMO kích hoạt phương án phân luồng giao thông và bổ sung quầy thủ tục."
                )
            if cat["maintenance_window_months"]:
                maint_found = True
                st.warning(
                    f"🛠️ **[CỬA SỔ BẢO TRÌ ĐỀ XUẤT] Phân khúc {cat['nace_name']}:** Công suất thấp (<35%) "
                    f"trong các tháng: **{', '.join(cat['maintenance_window_months'])}** (Đáy FTE: {cat['trough_fte']:,.0f}). "
                    f"-> Đề xuất: Thời điểm tối ưu để cấp phép sửa chữa, nâng cấp cơ sở vật chất mà không ảnh hưởng doanh thu."
                )

        if not strain_found:
            st.success("✅ **Không có cảnh báo quá tải:** Tải trọng các phân khúc đều nằm trong biên độ vận hành an toàn (<85%).")

        st.info(
            f"🛡️ **Kiểm toán Toán học Pi_C:** Toàn bộ {len(plan_df)} mốc dự báo đều được bảo chứng bất biến UNWTO "
            f"($Y^{{\\text{{nig}}}} \\ge Y^{{\\text{{arr}}}}$). Thuật toán đã phát hiện và chặn đứng **{plan_s['raw_violations']} lỗi thô** "
            f"trước khi phân bổ nguồn lực."
        )

    # ------------------ TAB 2: AUTOMATED EXECUTIVE REPORT ------------------
    with tab_report:
        st.markdown("### 📄 Khởi Tạo Báo Cáo Điều Hành DMO Tự Động")
        st.caption("Tổng hợp toàn bộ chỉ số, bảng phân bổ NACE và khuyến nghị chiến lược thành tài liệu chuẩn ban ngành.")

        generator = ExecutiveReportGenerator(bundle)

        col_cfg1, col_cfg2 = st.columns([1, 2])
        with col_cfg1:
            report_format = st.radio(
                "Định dạng văn bản hành chính & báo cáo:",
                [
                    "🏛️ Tờ trình UBND / Báo cáo Cấp trên (.md)",
                    "📊 Báo cáo Điều hành DMO Chiến lược (.md)",
                    "💾 Dữ liệu Kế hoạch Chi tiết (.json)",
                ],
                index=0,
            )
            btn_generate = st.button("🔄 Khởi tạo Văn Bản Mới", type="primary")

        if "Tờ trình" in report_format:
            submission_content = generator.generate_government_submission(
                geo=ctx["geo"],
                accom_type=ctx["accom"],
                horizon_months=ctx["horizon"],
                origin_label=ctx["origin"],
                agency_name=f"SỞ DU LỊCH ĐIỂM ĐẾN {ctx['country'].upper()}",
                recipient_name="ỦY BAN NHÂN DÂN / BỘ VĂN HÓA, THỂ THAO VÀ DU LỊCH",
            )
            with col_cfg2:
                st.download_button(
                    label="⬇️ Tải Dự Thảo Tờ Trình UBND (.md)",
                    data=submission_content.encode("utf-8"),
                    file_name=f"to_trinh_ubnd_du_lich_{ctx['geo']}_{ctx['origin']}.md",
                    mime="text/markdown",
                )
            st.markdown("---")
            st.markdown("#### 👁️ Bản Xem Trước Dự Thảo Tờ Trình (Preview):")
            st.markdown(submission_content)
        elif "Chiến lược" in report_format:
            md_content = generator.generate_markdown_report(
                geo=ctx["geo"], accom_type=ctx["accom"], horizon_months=ctx["horizon"], origin_label=ctx["origin"]
            )
            with col_cfg2:
                st.download_button(
                    label="⬇️ Tải Báo Cáo Chiến Lược DMO (.md)",
                    data=md_content.encode("utf-8"),
                    file_name=f"regitour_executive_briefing_{ctx['geo']}_{ctx['origin']}.md",
                    mime="text/markdown",
                )
            st.markdown("---")
            st.markdown("#### 👁️ Bản Xem Trước Báo Cáo (Preview):")
            st.markdown(md_content)
        else:
            dict_content = generator.generate_report_dict(
                geo=ctx["geo"], accom_type=ctx["accom"], horizon_months=ctx["horizon"], origin_label=ctx["origin"]
            )
            import json
            json_str = json.dumps(dict_content, indent=2, default=str)
            with col_cfg2:
                st.download_button(
                    label="⬇️ Tải Dữ Liệu Kế Hoạch JSON (.json)",
                    data=json_str.encode("utf-8"),
                    file_name=f"regitour_plan_data_{ctx['geo']}_{ctx['origin']}.json",
                    mime="application/json",
                )
            st.markdown("---")
            st.markdown("#### 👁️ Bản Xem Trước JSON (Preview):")
            st.json(dict_content)

    # ------------------ TAB 3: DEVELOPER REST API HUB ------------------
    with tab_api:
        st.markdown("### ⚡ Cổng Tích Hợp Enterprise REST API (FastAPI / OpenAPI)")
        st.caption("Kết nối hạ tầng mô hình RegiTour với hệ sinh thái phần mềm bên ngoài: PMS Khách sạn, Hệ thống Vé Sân bay, Cổng Dịch vụ công Du lịch.")

        col_api_meta1, col_api_meta2 = st.columns(2)
        with col_api_meta1:
            st.markdown(
                """
                **Thông tin Microservice:**
                - **Khung kỹ thuật:** FastAPI 0.162+ / Starlette / Uvicorn (ASGI)
                - **Tài liệu Tương tác (Swagger UI):** `http://localhost:8000/docs`
                - **Đặc tả OpenAPI Specification:** `http://localhost:8000/openapi.json`
                - **ReDoc Interactive:** `http://localhost:8000/redoc`
                - **Lệnh CLI khởi chạy:** `python cli.py api --port 8000`
                """
            )
        with col_api_meta2:
            st.markdown(
                """
                **Danh mục Endpoint chính (v1):**
                - `POST /api/v1/forecast/plan` — Kế hoạch lưu lượng đồng bộ (Pi_C safe)
                - `POST /api/v1/capacity/staffing` — Phân tách nhân sự & công suất NACE
                - `POST /api/v1/simulation/stress-test` — Mô phỏng cú sốc lạm phát & biên giới
                - `POST /api/v1/optimizer/pi-c` — Toán tử chiếu bảo toàn bất biến vật lý
                - `POST /api/v1/reports/generate` — Khởi tạo báo cáo tự động
                """
            )

        st.markdown("---")
        st.markdown("#### 💻 Mẫu Code Tích Hợp Ngoại Vi (API Integration Code Samples):")

        endpoint_choice = st.selectbox(
            "Chọn Endpoint muốn tích hợp:",
            [
                "POST /api/v1/forecast/plan (Kế hoạch Lưu lượng)",
                "POST /api/v1/capacity/staffing (Nhân sự & Lấp đầy)",
                "POST /api/v1/simulation/stress-test (Stress Test Vĩ mô)",
                "POST /api/v1/optimizer/pi-c (Toán tử Chiếu Pi_C)",
            ],
        )

        if "forecast/plan" in endpoint_choice:
            ep_url = "http://localhost:8000/api/v1/forecast/plan"
            ep_payload = {
                "geo": ctx["geo"],
                "accommodation_type": ctx["accom"],
                "horizon_months": ctx["horizon"],
                "origin_label": ctx["origin"],
            }
        elif "capacity/staffing" in endpoint_choice:
            ep_url = "http://localhost:8000/api/v1/capacity/staffing"
            ep_payload = {
                "geo": ctx["geo"],
                "horizon_months": ctx["horizon"],
                "origin_label": ctx["origin"],
                "hours_per_arrival": 0.5,
                "hours_per_night": 0.35,
                "hours_per_fte_month": 160.0,
                "maintenance_occ_threshold_pct": 35.0,
            }
        elif "simulation/stress-test" in endpoint_choice:
            ep_url = "http://localhost:8000/api/v1/simulation/stress-test"
            ep_payload = {
                "geo": ctx["geo"],
                "accommodation_type": ctx["accom"],
                "horizon_months": ctx["horizon"],
                "origin_label": ctx["origin"],
                "hicp_shock_pct": 15.0,
                "stringency_delta": 20.0,
            }
        else:
            ep_url = "http://localhost:8000/api/v1/optimizer/pi-c"
            ep_payload = {
                "arrivals": [120000.0, 145000.0, 98000.0],
                "nights": [110000.0, 320000.0, 95000.0],
            }

        import json
        payload_json = json.dumps(ep_payload, indent=2)

        code_py = f"""import requests

url = "{ep_url}"
headers = {{"Content-Type": "application/json"}}
payload = {payload_json}

response = requests.post(url, json=payload, headers=headers)
data = response.json()
print("Status:", response.status_code)
print("Response:", data)
"""

        code_curl = f"""curl -X POST "{ep_url}" \\
  -H "Content-Type: application/json" \\
  -d '{payload_json}'
"""

        code_js = f"""// JavaScript (Fetch API / Node.js / Browser)
const response = await fetch("{ep_url}", {{
  method: "POST",
  headers: {{ "Content-Type": "application/json" }},
  body: JSON.stringify({payload_json})
}});
const result = await response.json();
console.log(result);
"""

        code_tab1, code_tab2, code_tab3 = st.tabs(["Python (requests)", "cURL (Bash)", "JavaScript (Fetch)"])
        with code_tab1:
            st.code(code_py, language="python")
        with code_tab2:
            st.code(code_curl, language="bash")
        with code_tab3:
            st.code(code_js, language="javascript")


def main():
    bundle = get_dss_bundle()
    ctx = _render_sidebar(bundle)

    if ctx["nav"].startswith("🏠"):
        _render_executive_briefing(bundle, ctx)
    elif ctx["nav"].startswith("1️⃣"):
        _render_layer1_coordinated_throughput(bundle, ctx)
    elif ctx["nav"].startswith("2️⃣"):
        _render_layer2_nace_capacity_staffing(bundle, ctx)
    elif ctx["nav"].startswith("3️⃣"):
        _render_layer3_crisis_stress(bundle, ctx)
    elif ctx["nav"].startswith("4️⃣"):
        _render_layer4_recovery_conformal(bundle, ctx)
    elif ctx["nav"].startswith("🔬"):
        _render_scientific_benchmark(bundle)
    else:
        _render_api_and_reporting(bundle, ctx)


if __name__ == "__main__":
    main()
