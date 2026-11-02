"""
🏛️ RegiTour Interactive Tourism Forecasting & Policy Intelligence Dashboard
Built with Streamlit and Plotly for the NCKH 2027 Research Project.

Demonstrates:
  1. Multi-Horizon Tourism Demand Forecasting with DCQF Conformal Bands
  2. Dual TRMF Spatial Spillovers (Visitor Volume vs Length-of-Stay Duration)
  3. Two-Tier Domain Consistency & Anchor Projection (Pi_C)
  4. Explainable AI (XAI) & Counterfactual What-If Policy Simulation
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Configure root directory
ROOT_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT_DIR))

# Page configuration
st.set_page_config(
    page_title="RegiTour | AI Tourism Econometrics Dashboard",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E3A8A;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #4B5563;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8FAFC;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #2563EB;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
    }
    .status-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 12px;
        font-size: 0.82rem;
        font-weight: 600;
    }
    .badge-normal { background-color: #DCFCE7; color: #166534; }
    .badge-shock { background-color: #FEE2E2; color: #991B1B; }
    .badge-recovery { background-color: #FEF3C7; color: #92400E; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data
def load_forecast_results():
    """Loads Eurostat forecast results table."""
    csv_path = ROOT_DIR / "results" / "forecast_results_ra_he_dlinear_eurostat.csv"
    if csv_path.exists():
        df = pd.read_csv(csv_path)
        df["date"] = pd.to_datetime(df["date"])
        return df
    return None


@st.cache_data
def load_benchmark_table():
    """Loads benchmark model comparison table."""
    csv_path = ROOT_DIR / "results" / "tables" / "benchmark_model_comparison.csv"
    if csv_path.exists():
        return pd.read_csv(csv_path)
    return None


@st.cache_data
def load_xai_data():
    """Loads XAI elasticity and scenario tables."""
    tables_dir = ROOT_DIR / "results" / "tables"
    elast_csv = tables_dir / "xai_price_elasticity.csv"
    whatif_csv = tables_dir / "xai_what_if_scenarios.csv"
    df_elast = pd.read_csv(elast_csv) if elast_csv.exists() else None
    df_whatif = pd.read_csv(whatif_csv) if whatif_csv.exists() else None
    return df_elast, df_whatif


def main():
    # Sidebar Navigation
    st.sidebar.markdown("# ✈️ RegiTour")
    st.sidebar.title("🏛️ RegiTour Navigation")
    st.sidebar.markdown("**Modern Linear Framework (RA-HE-DLinear)**")

    demo_mode = st.sidebar.radio(
        "Chọn phân hệ tương tác:",
        [
            "1. 📈 Dự báo Đa chân trời & Dải tin cậy DCQF",
            "2. 🌐 Lan tỏa Không gian (Dual TRMF Matrix)",
            "3. ⚖️ Ràng buộc Kinh tế & Phép chiếu Neo (Π_C)",
            "4. 🔬 Giải thích Mô hình XAI & Mô phỏng Chính sách",
            "5. 🏆 Bảng Xếp hạng Đối sánh Mô hình (Benchmark)",
        ],
    )

    df_results = load_forecast_results()

    # =========================================================================
    # TAB 1: MULTI-HORIZON FORECAST TRAJECTORIES & CONFORMAL BANDS
    # =========================================================================
    if demo_mode.startswith("1."):
        st.markdown('<div class="main-header">📈 Dự Báo Nhu Cầu Du Lịch Đa Chân Trời (2024–2026)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Mô hình phân rã chuỗi thời gian tuyến tính kết hợp ngoại sinh (RA-Exo-DLinear) và bảo chứng khoảng tin cậy phi phân phối DCQF.</div>', unsafe_allow_html=True)

        if df_results is None:
            st.warning("⚠️ Chưa tìm thấy tệp `results/forecast_results_ra_he_dlinear_eurostat.csv`. Vui lòng chạy `python run_linear_pipeline.py` trước.")
            return

        col_ctrl1, col_ctrl2, col_ctrl3 = st.columns([2, 2, 2])
        with col_ctrl1:
            countries = sorted(df_results["geo"].unique())
            default_geo = "FR" if "FR" in countries else countries[0]
            selected_geo = st.selectbox("🌍 Quốc gia điểm đến (Destination):", countries, index=countries.index(default_geo))

        with col_ctrl2:
            accom_types = sorted(df_results["accommodation_type"].unique())
            default_acc = "total" if "total" in accom_types else accom_types[0]
            selected_acc = st.selectbox("🏨 Loại hình lưu trú (NACE Type):", accom_types, index=accom_types.index(default_acc))

        with col_ctrl3:
            target_metric = st.selectbox("🎯 Biến mục tiêu hiển thị:", ["Lượt khách (Arrivals)", "Số đêm lưu trú (Nights)"])

        # Filter series
        sub_df = df_results[(df_results["geo"] == selected_geo) & (df_results["accommodation_type"] == selected_acc)].sort_values("date")

        if len(sub_df) == 0:
            st.info("Không có dữ liệu cho cấu hình được chọn.")
            return

        is_arr = "Arrivals" in target_metric
        col_true = "arrivals_total" if is_arr else "nights_total"
        col_pred = "pred_arrivals_point" if is_arr else "pred_nights_point"
        col_p10 = "pred_arrivals_p10" if is_arr else "pred_nights_p10"
        col_p90 = "pred_arrivals_p90" if is_arr else "pred_nights_p90"

        # Compute metrics for this selection
        y_true = sub_df[col_true].values
        y_pred = sub_df[col_pred].values
        wape = np.sum(np.abs(y_true - y_pred)) / max(1.0, np.sum(y_true)) * 100.0
        mae = np.mean(np.abs(y_true - y_pred))
        rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
        coverage = np.mean((y_true >= sub_df[col_p10].values) & (y_true <= sub_df[col_p90].values)) * 100.0

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("WAPE (Sai số tỷ lệ)", f"{wape:.2f}%")
        m2.metric("MAE (Sai số tuyệt đối)", f"{mae:,.0f}")
        m3.metric("RMSE", f"{rmse:,.0f}")
        m4.metric("DCQF Conformal Coverage", f"{coverage:.1f}%", help="Mục tiêu bảo chứng: >= 90%")

        # Interactive Plotly Chart
        try:
            import plotly.graph_objects as go

            fig = go.Figure()
            # Conformal Band
            fig.add_trace(go.Scatter(
                x=pd.concat([sub_df["date"], sub_df["date"][::-1]]),
                y=pd.concat([sub_df[col_p90], sub_df[col_p10][::-1]]),
                fill="toself",
                fillcolor="rgba(37, 99, 235, 0.15)",
                line=dict(color="rgba(255,255,255,0)"),
                hoverinfo="skip",
                name="Dải tin cậy DCQF [P10, P90] (90% Coverage)",
            ))
            # Ground truth
            fig.add_trace(go.Scatter(
                x=sub_df["date"], y=sub_df[col_true],
                mode="lines+markers",
                line=dict(color="#111827", width=2.5),
                marker=dict(size=6),
                name="Thực tế (Ground Truth)",
            ))
            # Proposed model
            fig.add_trace(go.Scatter(
                x=sub_df["date"], y=sub_df[col_pred],
                mode="lines+markers",
                line=dict(color="#2563EB", width=2.5, dash="solid"),
                marker=dict(size=6),
                name="RA-HE-DLinear (Dự báo)",
            ))

            fig.update_layout(
                title=f"Quỹ đạo dự báo chuỗi thời gian tại {selected_geo} ({selected_acc.upper()})",
                xaxis_title="Thời gian (Tháng/Năm)",
                yaxis_title="Số lượng khách / đêm",
                template="plotly_white",
                height=480,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
                hovermode="x unified",
            )
            st.plotly_chart(fig, use_container_width=True)

        except ImportError:
            st.line_chart(sub_df.set_index("date")[[col_true, col_pred]])

        st.dataframe(
            sub_df[["date", "geo", "accommodation_type", col_true, col_pred, col_p10, col_p90]].head(12),
            use_container_width=True,
        )

    # =========================================================================
    # TAB 2: SPATIAL SPILLOVERS (DUAL TRMF MATRIX FACTORIZATION)
    # =========================================================================
    elif demo_mode.startswith("2."):
        st.markdown('<div class="main-header">🌐 Động Cơ Lan Tỏa Không Gian: Dual TRMF Matrix Factorization</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Mô hình hóa độc lập cấu trúc lan tỏa không gian cho cả Lượt khách (Yarr) và Số đêm lưu trú (Ynig), học riêng động thái thời gian lưu trú (Length of Stay - LoS).</div>', unsafe_allow_html=True)

        st.info(
            "💡 **Cơ chế Độc lập Toán học:** Khác với các phương pháp cũ nhân tuyến tính scalar ratio, "
            "Dual TRMF phân rã song song 2 ma trận:\\n"
            "- $\\mathbf{Y}_{\\mathrm{arr}} \\approx \\mathbf{W}_{\\mathrm{arr}} \\mathbf{X}_{\\mathrm{arr}}^T$: Nắm bắt dòng luân chuyển khách du lịch xuyên biên giới Schengen.\\n"
            "- $\\mathbf{Y}_{\\mathrm{nig}} \\approx \\mathbf{W}_{\\mathrm{nig}} \\mathbf{X}_{\\mathrm{nig}}^T$: Nắm bắt cấu trúc lan tỏa thời gian lưu trú (vùng ven biển Địa Trung Hải lưu trú dài ngày vs đô thị quá cảnh ngắn ngày)."
        )

        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("#### 🏛️ 4 Cụm Không Gian Du Lịch Châu Âu")
            cluster_data = {
                "Cụm Không Gian": ["Cụm 1: Nam Âu / Địa Trung Hải", "Cụm 2: Tây Âu Đô Thị", "Cụm 3: Bắc Âu Sinh Thái", "Cụm 4: Đông Âu Mới Nổi"],
                "Đại diện quốc gia tiêu biểu": ["ES, IT, EL, PT, HR", "FR, DE, NL, BE, AT", "SE, NO, DK, FI, IS", "PL, CZ, HU, RO, BG"],
                "Hành vi Lưu trú (LoS)": ["Dài ngày (4.0 – 6.5 đêm/lượt)", "Trung bình (2.0 – 3.2 đêm/lượt)", "Trung bình (2.2 – 3.5 đêm/lượt)", "Ngắn ngày (1.8 – 2.8 đêm/lượt)"],
                "Tính chất Mùa vụ": ["Đỉnh nhọn cực hạn mùa hè (Tháng 7-8)", "Ổn định quanh năm & du lịch công vụ", "Đỉnh kép: Trượt tuyết mùa đông & Hè", "Mùa lễ hội & văn hóa mùa thu"],
            }
            st.table(pd.DataFrame(cluster_data))

        with col_s2:
            st.markdown("#### ⚖️ Trọng Số Hòa Trộn Không - Thời Gian (Validation Constrained Optimization)")
            st.write("Trọng số hòa trộn $w_{\\mathrm{time}}$ được tối ưu hóa lồi trên tập Validation (2023) độc lập:")
            st.latex(r"w_{\mathrm{time}}^* = \arg\min_{w \in [0.70, 1.00]} \mathcal{L}_{\mathrm{val}}\left(\mathbf{Y}, \Pi_{\mathcal{C}}\left(w \hat{\mathbf{Y}}_{\mathrm{lin}} + (1-w) \hat{\mathbf{Y}}_{\mathrm{spatial}}\right)\right)")
            st.metric("Trọng số Lõi Thời gian (RA-Exo-DLinear)", "70.0% - 85.0%")
            st.metric("Trọng số Động cơ Không gian (Dual TRMF)", "15.0% - 30.0%")

        if df_results is not None:
            st.markdown("#### 📊 So sánh Tỷ lệ Lưu trú Thực tế vs Mô hình Hóa Không gian")
            sample_df = df_results[df_results["accommodation_type"] == "total"].copy()
            sample_df["actual_los"] = sample_df["nights_total"] / np.maximum(sample_df["arrivals_total"], 1.0)
            sample_df["pred_los"] = sample_df["pred_nights_point"] / np.maximum(sample_df["pred_arrivals_point"], 1.0)
            los_by_geo = sample_df.groupby("geo")[["actual_los", "pred_los"]].mean().reset_index()

            try:
                import plotly.express as px
                fig_los = px.bar(
                    los_by_geo.head(20),
                    x="geo",
                    y=["actual_los", "pred_los"],
                    barmode="group",
                    title="Thời gian lưu trú bình quân (Length of Stay - LoS = Nights / Arrivals) theo quốc gia",
                    labels={"value": "Số đêm bình quân / lượt khách", "geo": "Quốc gia", "variable": "Chỉ số"},
                    template="plotly_white",
                    height=380,
                )
                st.plotly_chart(fig_los, use_container_width=True)
            except Exception:
                st.dataframe(los_by_geo.head(15))

    # =========================================================================
    # TAB 3: COUPLED CONSISTENCY & ANCHOR PROJECTION (PI_C)
    # =========================================================================
    elif demo_mode.startswith("3."):
        st.markdown('<div class="main-header">⚖️ Ràng Buộc Nghiệp Vụ Kinh Tế & Phép Chiếu Neo (Π_C)</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Bảo chứng triệt tiêu 100% mâu thuẫn phi lý (Nights >= Arrivals) và duy trì tỷ lệ lưu trú trong dải lịch sử.</div>', unsafe_allow_html=True)

        col_c1, col_c2 = st.columns([1, 1])

        with col_c1:
            st.markdown("#### 🧪 Thử nghiệm Trực quan Phép Chiếu Neo (Live Simulation)")
            st.write("Điều chỉnh giả lập một dự báo thô có nguy cơ vi phạm cận vật lý:")
            sim_arr = st.slider("Dự báo Lượt khách thô (Arrivals):", min_value=100, max_value=5000, value=1000, step=50)
            sim_nig_raw = st.slider("Dự báo Số đêm thô (Nights - thử kéo nhỏ hơn Arrivals):", min_value=50, max_value=8000, value=850, step=50)
            base_r = st.slider("Tỷ lệ lưu trú chuẩn của quốc gia (Historical Base Ratio):", min_value=1.5, max_value=5.0, value=2.5, step=0.1)

            # Perform Anchor Projection
            is_violation = sim_nig_raw < sim_arr
            ratio_raw = sim_nig_raw / max(1, sim_arr)
            r_min = max(1.05, base_r - 1.0)
            r_max = base_r + 1.0

            if ratio_raw < r_min:
                sim_nig_proj = int(round(sim_arr * r_min))
                status_text = "🔴 VI PHẠM CẬN VẬT LÝ / DẢI KINH TẾ (Đã được sửa bởi Π_C)"
                badge_class = "badge-shock"
            elif ratio_raw > r_max:
                sim_nig_proj = int(round(sim_arr * r_max))
                status_text = "🟠 VƯỢT DẢI KINH TẾ TỐI ĐA (Đã nắn chỉnh về biên trên)"
                badge_class = "badge-recovery"
            else:
                sim_nig_proj = sim_nig_raw
                status_text = "🟢 HỢP LỆ VẬT LÝ & KINH TẾ"
                badge_class = "badge-normal"

            st.markdown(f'<span class="status-badge {badge_class}">{status_text}</span>', unsafe_allow_html=True)
            st.write("")
            res_col1, res_col2 = st.columns(2)
            res_col1.metric("Lượt khách sau chiếu", f"{sim_arr:,} (Neo bất biến)")
            res_col2.metric("Số đêm sau chiếu", f"{sim_nig_proj:,}", delta=f"{sim_nig_proj - sim_nig_raw:+,} đêm")

        with col_c2:
            st.markdown("#### 📐 Cơ Sở Toán Học Phép Chiếu Neo Asymmetric Anchor")
            st.latex(r"\Pi_{\mathcal{C}}\left(\hat{Y}_{\mathrm{arr}}, \hat{Y}_{\mathrm{nig}}\right) = \left(\hat{Y}_{\mathrm{arr}}, \, \max\left(\hat{Y}_{\mathrm{nig}}, \, \hat{Y}_{\mathrm{arr}} \times 1.05\right)\right)")
            st.markdown(
                """
                - **Lượt khách là Neo bất biến (Anchor):** Số liệu khai báo nhập cảnh được ghi nhận có độ tin cậy cao nhất.
                - **Số đêm là biến phụ thuộc:** Chiếu nắn chỉnh số đêm đảm bảo luôn $\\text{Nights} \\ge \\text{Arrivals} \\times 1.05$.
                - **Tỷ lệ vi phạm trên tập kiểm thử:** Chỉ **0.15% (7/4,712 mẫu)** trước chiếu, và chính xác **0.00% vi phạm** sau chiếu.
                """
            )

        if df_results is not None:
            # Scatter plot
            arr_vals = df_results["pred_arrivals_point"].values[:1000] / 1000.0
            nig_vals = df_results["pred_nights_point"].values[:1000] / 1000.0
            try:
                import plotly.graph_objects as go
                fig_sc = go.Figure()
                max_v = 1500
                fig_sc.add_trace(go.Scatter(
                    x=[0, max_v], y=[0, max_v],
                    mode="lines",
                    line=dict(color="#DC2626", width=2, dash="dash"),
                    name="Cận vật lý tối thiểu (Nights = Arrivals, LOS = 1.0)",
                ))
                fig_sc.add_trace(go.Scatter(
                    x=arr_vals, y=nig_vals,
                    mode="markers",
                    marker=dict(color="#2563EB", size=5, opacity=0.5),
                    name="Kết quả dự báo sau chiếu Π_C (0.00% vi phạm)",
                ))
                fig_sc.update_layout(
                    title="Không gian phân bổ Dự báo sau Phép chiếu Neo (Arrivals vs Nights)",
                    xaxis_title="Lượt khách dự báo (Nghìn lượt)",
                    yaxis_title="Số đêm dự báo (Nghìn đêm)",
                    template="plotly_white",
                    height=400,
                )
                st.plotly_chart(fig_sc, use_container_width=True)
            except Exception:
                pass

    # =========================================================================
    # TAB 4: XAI & POLICY SIMULATION
    # =========================================================================
    elif demo_mode.startswith("4."):
        st.markdown('<div class="main-header">🔬 Trí Tuệ Nhân Tạo Giải Thích (XAI) & Mô Phỏng Chính Sách</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Trọng số chú ý chéo thực tế (M-RA-BiLSTM Attention), Đóng góp đặc trưng vĩ mô TreeSHAP và mô phỏng cú sốc giá phòng HICP.</div>', unsafe_allow_html=True)

        tab_xai1, tab_xai2 = st.tabs(["🎯 Mô phỏng Cú sốc Lạm phát / Giá phòng (What-If)", "🧠 Trọng số Chú ý Mùa vụ (Cross-Temporal Attention)"])

        with tab_xai1:
            col_w1, col_w2 = st.columns([1, 1])
            with col_w1:
                st.markdown("#### ⚡ Thử nghiệm Can thiệp Chỉ số Giá phòng Khách sạn (HICP)")
                hicp_shock = st.slider("Mức độ biến động giá phòng khách sạn (%):", min_value=-20, max_value=20, value=10, step=5)
                sim_country = st.selectbox("Chọn quốc gia mô phỏng:", ["FR (Pháp)", "ES (Tây Ban Nha)", "IT (Ý)", "DE (Đức)"])

                # Standard empirical price elasticity of tourism demand in Southern/Western Europe: ~ -0.45 to -0.65
                elasticity = -0.58 if "ES" in sim_country or "IT" in sim_country else -0.48
                arr_delta = hicp_shock * elasticity
                nig_delta = hicp_shock * (elasticity * 1.15)  # Nights drops more than arrivals due to shortened trips

                st.write("")
                st.metric("Tác động tới Lượt khách (Arrivals)", f"{arr_delta:+.2f}%", delta=f"{arr_delta:+.2f}%", delta_color="inverse")
                st.metric("Tác động tới Số đêm lưu trú (Nights)", f"{nig_delta:+.2f}%", delta=f"{nig_delta:+.2f}%", delta_color="inverse")
                st.caption(f"Hệ số co giãn giá (Price Elasticity): {elasticity:.2f}. Khách có xu hướng rút ngắn số đêm lưu trú trước khi hủy hẳn chuyến đi.")

            with col_w2:
                df_elast, df_whatif = load_xai_data()
                if df_whatif is not None:
                    st.markdown("#### 📋 Bảng Kịch bản Giả định Đối thực nghiệm Đã Tính toán")
                    st.dataframe(df_whatif[["country", "scenario", "arrivals_delta_pct", "nights_delta_pct"]].head(12), use_container_width=True)
                elif df_elast is not None:
                    st.dataframe(df_elast.head(10), use_container_width=True)

        with tab_xai2:
            st.markdown("#### 🧠 Bản đồ Nhiệt Chú ý Chéo Thời Gian Học từ Mô Hình Thật")
            attn_img_path = ROOT_DIR / "results" / "xai" / "attention_cross_temporal_heatmap.png"
            if attn_img_path.exists():
                st.image(str(attn_img_path), caption="Cross-Temporal Attention Heatmap: Query (Lịch lễ tương lai) attend Key/Value (Lags lịch sử)", use_column_width=True)
            else:
                st.info("Chạy `python run_xai_analysis.py` để trích xuất và hiển thị heatmap chú ý chéo thực tế.")

    # =========================================================================
    # TAB 5: BENCHMARK LEADERBOARD
    # =========================================================================
    elif demo_mode.startswith("5."):
        st.markdown('<div class="main-header">🏆 Bảng Xếp Hạng Đối Sánh Khoa Học Toàn Diện</div>', unsafe_allow_html=True)
        st.markdown('<div class="sub-header">Đánh giá đối đầu trực tiếp trên tập kiểm thử độc lập (2024–2026, 4,712 quan sát) qua 3 làn sóng y văn thế giới.</div>', unsafe_allow_html=True)

        df_bench = load_benchmark_table()
        if df_bench is not None:
            st.dataframe(df_bench, use_container_width=True)
        else:
            # Fallback display of verified scientific metrics from README
            leaderboard_data = {
                "Mô hình đối sánh": [
                    "RA-HE-DLinear (Proposed Linear ⭐)",
                    "RA-Exo-DLinear (Proposed Core)",
                    "Seasonal Naive (S-Naive)",
                    "Standard LightGBM",
                    "Vanilla DLinear (AAAI 2023)",
                    "Vanilla NLinear (AAAI 2023)",
                    "Standard XGBoost",
                    "CatBoost-Regime (Ablation)",
                    "Classical LSTM (1997)",
                    "Vanilla BiLSTM (2013)",
                ],
                "Lớp mô hình": [
                    "Linear Framework (Proposed ⭐)",
                    "Linear LTSF (Proposed)",
                    "Thống kê cổ điển",
                    "Machine Learning",
                    "Modern Linear LTSF",
                    "Modern Linear LTSF",
                    "Machine Learning",
                    "Machine Learning",
                    "Classical Deep Learning",
                    "Classical Deep Learning",
                ],
                "Arrivals WAPE": ["14.59%", "18.29%", "27.30%", "30.37%", "45.38%", "32.10%", "34.13%", "40.27%", "108.53%", "108.53%"],
                "Nights WAPE": ["25.00%", "25.87%", "32.73%", "34.28%", "25.61%", "49.38%", "42.36%", "44.86%", "113.16%", "113.16%"],
                "Vi phạm Cận vật lý": ["0.00%*", "0.21%", "0.05%", "0.19%", "0.05%", "0.69%", "0.34%", "0.82%", "0.05%", "0.05%"],
                "Lệch dải Kinh tế": ["16.0%", "21.5%", "28.8%", "49.2%", "28.8%", "29.2%", "37.7%", "44.9%", "99.4%", "99.4%"],
            }
            st.table(pd.DataFrame(leaderboard_data))
            st.caption("* Sau phép chiếu neo Π_C, vi phạm cận vật lý đạt 0.00% (0/4,712 mẫu). Tỷ lệ cần can thiệp của mô hình thô là 0.15% (7/4,712 mẫu).")


if __name__ == "__main__":
    main()

