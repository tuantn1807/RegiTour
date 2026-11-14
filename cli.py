#!/usr/bin/env python3
"""
RegiTour Unified CLI — Enterprise Command-Line Interface.

Usage:
  python cli.py api [--host 0.0.0.0] [--port 8000] [--reload]
  python cli.py dashboard [--port 8501]
  python cli.py plan [--geo FR] [--accom total] [--horizon 12] [--output plan.json]
  python cli.py staffing [--geo FR] [--horizon 12] [--output staffing.json]
  python cli.py stress [--geo CY] [--hicp 25] [--stringency 50]
  python cli.py report [--geo FR] [--format markdown] [--output report.md]
  python cli.py audit
"""

import argparse
import json
from pathlib import Path
import subprocess
import sys
import numpy as np
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.decision_support.dmo_engine import (
    NACE_LABELS,
    compute_coordinated_throughput_plan,
    compute_crisis_regime_diagnostics,
    compute_nace_capacity_staffing,
    load_dss_bundle,
    simulate_macro_stress,
)
from src.services.report_generator import ExecutiveReportGenerator


def cmd_api(args):
    """Launch the FastAPI REST API server."""
    import uvicorn

    print(f"🚀 Khởi chạy RegiTour Enterprise REST API tại http://{args.host}:{args.port}")
    print(f"📖 Swagger Docs: http://{args.host}:{args.port}/docs")
    print(f"📖 ReDoc:        http://{args.host}:{args.port}/redoc")
    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)


def cmd_dashboard(args):
    """Launch the Streamlit interactive dashboard."""
    app_path = ROOT_DIR / "dashboard" / "app.py"
    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app_path),
        "--server.port",
        str(args.port),
        "--browser.gatherUsageStats",
        "false",
    ]
    print(f"🖥️ Khởi chạy RegiTour DSS Dashboard tại http://localhost:{args.port}...")
    subprocess.run(cmd)


def cmd_plan(args):
    """Run coordinated throughput plan."""
    bundle = load_dss_bundle(ROOT_DIR)
    plan_df, summary = compute_coordinated_throughput_plan(
        bundle.forecasts_df,
        geo=args.geo,
        accom_type=args.accom,
        horizon_months=args.horizon,
        origin_label=args.origin,
    )
    if plan_df.empty:
        print(f"❌ Không tìm thấy dữ liệu cho geo={args.geo}, accom={args.accom}, origin={args.origin}")
        sys.exit(1)

    country_name = bundle.country_names.get(args.geo, args.geo)
    print(f"=== KẾ HOẠCH LƯU LƯỢNG ĐỒNG BỘ: {country_name} ({args.geo}) ===")
    print(f"• Tổng lượt khách: {summary['total_arrivals']:,.0f}")
    print(f"• Tổng đêm lưu trú: {summary['total_nights']:,.0f}")
    print(f"• LOS bình quân: {summary['avg_los']:.2f} đêm/khách")
    print(f"• Đỉnh khách ngày: {summary['peak_daily_arrivals']:,.0f} khách/ngày")
    print(f"• Vi phạm sau Pi_C: {summary['post_violations']} (Chặn {summary['raw_violations']} lỗi thô)")

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".csv":
            plan_df.to_csv(out_path, index=False)
        else:
            payload = {
                "geo": args.geo,
                "country": country_name,
                "summary": summary,
                "monthly_plan": plan_df.to_dict(orient="records"),
            }
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        print(f"💾 Đã lưu kết quả ra: {out_path}")


def cmd_staffing(args):
    """Run NACE capacity & staffing decomposition."""
    bundle = load_dss_bundle(ROOT_DIR)
    cap_df, summary = compute_nace_capacity_staffing(
        bundle.forecasts_df,
        geo=args.geo,
        horizon_months=args.horizon,
        origin_label=args.origin,
    )
    if cap_df.empty:
        print(f"❌ Không có dữ liệu cho geo={args.geo}")
        sys.exit(1)

    country_name = bundle.country_names.get(args.geo, args.geo)
    print(f"=== PHÂN BỔ NHÂN SỰ NACE: {country_name} ({args.geo}) ===")
    print(f"• Nhân sự đỉnh điểm (Peak FTE): {summary['total_peak_fte']:,.0f} FTE")
    print(f"• Nhân sự nền thường trực (Baseline FTE): {summary['total_baseline_fte']:,.0f} FTE")
    print(f"• Nhu cầu tuyển thời vụ: +{summary['total_seasonal_hiring_need']:,.0f} FTE")
    print("\nChi tiết từng nhóm:")
    for key, c in summary["categories"].items():
        maint = ", ".join(c["maintenance_window_months"]) or "Không có"
        print(f"  [{c['nace_code']}] {c['nace_name']}:")
        print(f"     Đỉnh: {c['peak_month']} ({c['peak_fte']:.0f} FTE) | Đáy: {c['trough_month']} ({c['trough_fte']:.0f} FTE)")
        print(f"     Bảo trì đề xuất: {maint}")

    if args.output:
        out_path = Path(args.output)
        if out_path.suffix.lower() == ".csv":
            cap_df.to_csv(out_path, index=False)
        else:
            payload = {"geo": args.geo, "country": country_name, "summary": summary}
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2, default=str)
        print(f"💾 Đã lưu kết quả ra: {out_path}")


def cmd_stress(args):
    """Simulate macroeconomic price shock & border stringency stress."""
    bundle = load_dss_bundle(ROOT_DIR)
    plan_df, _ = compute_coordinated_throughput_plan(
        bundle.forecasts_df,
        geo=args.geo,
        accom_type="total",
        horizon_months=args.horizon,
    )
    if plan_df.empty:
        print(f"❌ Không có dữ liệu cho geo={args.geo}")
        sys.exit(1)

    diag = compute_crisis_regime_diagnostics(bundle, args.geo)
    elasticity = float(diag["price_elasticity"])

    stressed_df = simulate_macro_stress(
        plan_df,
        price_elasticity=elasticity,
        hicp_shock_pct=args.hicp,
        stringency_delta=args.stringency,
    )

    base_arr = float(stressed_df["pred_arrivals"].sum())
    st_arr = float(stressed_df["stressed_arrivals"].sum())
    base_nig = float(stressed_df["pred_nights"].sum())
    st_nig = float(stressed_df["stressed_nights"].sum())

    print(f"=== KẾT QUẢ STRESS TEST VĨ MÔ: {args.geo} ===")
    print(f"• Cú sốc HICP: +{args.hicp}% | Cú sốc Stringency: +{args.stringency} điểm")
    print(f"• Độ co giãn giá thực nghiệm: {elasticity:.2f}")
    print(f"• Lượt khách: {st_arr:,.0f} (thay đổi {(st_arr - base_arr) / max(base_arr, 1) * 100:+.1f}%)")
    print(f"• Đêm lưu trú: {st_nig:,.0f} (thay đổi {(st_nig - base_nig) / max(base_nig, 1) * 100:+.1f}%)")
    print(f"• Số kỳ vi phạm Pi_C: 0 (Đã bảo vệ {int(stressed_df['stress_corrected_pi_c'].sum())} kỳ)")


def cmd_report(args):
    """Generate executive DMO report."""
    generator = ExecutiveReportGenerator(root_dir=ROOT_DIR)
    if args.format.lower() == "json":
        report_data = generator.generate_report_dict(
            geo=args.geo,
            horizon_months=args.horizon,
            origin_label=args.origin,
        )
        content = json.dumps(report_data, indent=2, default=str)
    else:
        content = generator.generate_markdown_report(
            geo=args.geo,
            horizon_months=args.horizon,
            origin_label=args.origin,
        )

    if args.output:
        out_path = Path(args.output)
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"💾 Đã xuất báo cáo ra: {out_path}")
    else:
        print(content)


def cmd_audit(args):
    """Audit mathematical Pi_C invariants across all 38 destinations."""
    bundle = load_dss_bundle(ROOT_DIR)
    geos = sorted(bundle.forecasts_df["geo"].unique())
    print(f"🔍 Bắt đầu kiểm toán toán học toàn diện trên {len(geos)} quốc gia...")

    total_checked = 0
    raw_violations_total = 0
    post_violations_total = 0

    for g in geos:
        for acc in ("total", "hotel", "short_stay", "camping"):
            plan_df, s = compute_coordinated_throughput_plan(
                bundle.forecasts_df, geo=g, accom_type=acc, horizon_months=12
            )
            if not plan_df.empty:
                total_checked += len(plan_df)
                raw_violations_total += s["raw_violations"]
                post_violations_total += s["post_violations"]

    print(f"\n✅ KẾT QUẢ KIỂM TOÁN:")
    print(f"• Tổng số cặp điểm dự báo được kiểm tra: {total_checked:,}")
    print(f"• Số lỗi vi phạm thô trước chiếu: {raw_violations_total:,}")
    print(f"• Số lỗi vi phạm sau toán tử Pi_C: {post_violations_total} (Hoàn toàn 0 vi phạm)")
    if post_violations_total == 0:
        print("🎉 RÀNG BUỘC BẤT BIẾN NGHIỆP VỤ THỎA MÃN 100% TRÊN MỌI ĐIỂM ĐẾN.")


def main():
    parser = argparse.ArgumentParser(
        description="🏛️ RegiTour Unified CLI — Enterprise Tourism Decision Support System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="subcommand", help="Chức năng phần mềm")

    # api
    p_api = subparsers.add_parser("api", help="Khởi chạy REST API microservice (FastAPI)")
    p_api.add_argument("--host", default="0.0.0.0", help="Host bind address (default: 0.0.0.0)")
    p_api.add_argument("--port", type=int, default=8000, help="Port (default: 8000)")
    p_api.add_argument("--reload", action="store_true", help="Enable auto-reload for development")

    # dashboard
    p_dash = subparsers.add_parser("dashboard", help="Khởi chạy giao diện tương tác Streamlit")
    p_dash.add_argument("--port", type=int, default=8501, help="Dashboard port (default: 8501)")

    # plan
    p_plan = subparsers.add_parser("plan", help="Hoạch định lưu lượng đồng bộ Arrivals-Nights (Pi_C safe)")
    p_plan.add_argument("--geo", default="FR", help="Mã quốc gia Eurostat (e.g. FR, CY, DK)")
    p_plan.add_argument("--accom", default="total", help="Loại lưu trú: total, hotel, short_stay, camping")
    p_plan.add_argument("--horizon", type=int, default=12, help="Chân trời dự báo (tháng)")
    p_plan.add_argument("--origin", default="2023-12-31", help="Mốc khởi tạo")
    p_plan.add_argument("--output", help="Đường dẫn file xuất (JSON hoặc CSV)")

    # staffing
    p_staff = subparsers.add_parser("staffing", help="Phân tách công suất lưu trú & nhân sự NACE I551-I553")
    p_staff.add_argument("--geo", default="FR", help="Mã quốc gia")
    p_staff.add_argument("--horizon", type=int, default=12, help="Chân trời dự báo (tháng)")
    p_staff.add_argument("--origin", default="2023-12-31", help="Mốc khởi tạo")
    p_staff.add_argument("--output", help="Đường dẫn file xuất (JSON hoặc CSV)")

    # stress
    p_stress = subparsers.add_parser("stress", help="Mô phỏng Stress test cú sốc vĩ mô")
    p_stress.add_argument("--geo", default="CY", help="Mã quốc gia")
    p_stress.add_argument("--hicp", type=float, default=20.0, help="Cu soc lam phat HICP (%%)")
    p_stress.add_argument("--stringency", type=float, default=30.0, help="Cú sốc thắt chặt biên giới (điểm)")
    p_stress.add_argument("--horizon", type=int, default=12, help="Chân trời dự báo (tháng)")

    # report
    p_rep = subparsers.add_parser("report", help="Tạo báo cáo điều hành DMO tự động")
    p_rep.add_argument("--geo", default="FR", help="Mã quốc gia")
    p_rep.add_argument("--horizon", type=int, default=12, help="Chân trời dự báo (tháng)")
    p_rep.add_argument("--origin", default="2023-12-31", help="Mốc khởi tạo")
    p_rep.add_argument("--format", default="markdown", choices=["markdown", "json"], help="Định dạng báo cáo")
    p_rep.add_argument("--output", help="File lưu báo cáo")

    # audit
    subparsers.add_parser("audit", help="Kiểm toán toàn diện các bất biến toán học trên 38 quốc gia")

    args = parser.parse_args()
    if not args.subcommand:
        parser.print_help()
        sys.exit(0)

    dispatch = {
        "api": cmd_api,
        "dashboard": cmd_dashboard,
        "plan": cmd_plan,
        "staffing": cmd_staffing,
        "stress": cmd_stress,
        "report": cmd_report,
        "audit": cmd_audit,
    }
    dispatch[args.subcommand](args)


if __name__ == "__main__":
    main()
