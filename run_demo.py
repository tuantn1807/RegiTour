#!/usr/bin/env python3
"""
Convenient launcher for RegiTour Interactive Dashboard & Demos.
Automatically detects environment capabilities:
  - If streamlit is installed: launches Streamlit Web Dashboard on port 8501
  - If streamlit is not installed: launches built-in zero-dependency Web Dashboard on port 8501
  - If --cli is passed: runs fast terminal CLI demonstration (demo.py)

Usage:
    python3 run_demo.py          # Launches interactive Web Dashboard
    python3 run_demo.py --cli    # Runs fast terminal CLI demo
"""

import os
import sys
import subprocess
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent

def check_package(pkg_name: str) -> bool:
    try:
        __import__(pkg_name)
        return True
    except ImportError:
        return False

def find_streamlit_python() -> str | None:
    if check_package("streamlit"):
        return sys.executable
    venv_py = ROOT_DIR / ".venv" / "bin" / "python"
    if venv_py.exists():
        res = subprocess.run(
            [str(venv_py), "-c", "import streamlit"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        if res.returncode == 0:
            return str(venv_py)
    return None

def main():
    if "--cli" in sys.argv:
        cli_script = ROOT_DIR / "demo.py"
        subprocess.run([sys.executable, str(cli_script)], check=True)
        return

    streamlit_python = find_streamlit_python()

    if streamlit_python is not None:
        dashboard_script = ROOT_DIR / "dashboard" / "app.py"
        cmd = [
            streamlit_python,
            "-m",
            "streamlit",
            "run",
            str(dashboard_script),
            "--server.headless=true",
            "--server.address=0.0.0.0",
            "--server.port=8501",
        ]
        extra_args = [arg for arg in sys.argv[1:] if arg != "--web"]
        cmd.extend(extra_args)

        print("\n" + "=" * 70)
        print("🚀 Khởi chạy RegiTour Streamlit Dashboard trên cổng 8501...")
        print("👉 Mở trình duyệt truy cập: http://localhost:8501 (hoặc http://127.0.0.1:8501)")
        print("=" * 70 + "\n")
        try:
            subprocess.run(cmd)
        except KeyboardInterrupt:
            print("\n👋 Dashboard đã dừng.")
    else:
        # Fallback to Python built-in HTTP server for dashboard/index.html (zero extra dependencies)
        dashboard_dir = ROOT_DIR / "dashboard"
        port = 8501
        print("\n" + "=" * 75)
        print("⚠️ Gói 'streamlit' chưa được cài đặt trong môi trường ảo (.venv).")
        print("💡 Để dùng Streamlit đầy đủ:  pip install streamlit plotly")
        print("-" * 75)
        print(f"🌐 ĐANG KHỞI CHẠY BẢNG ĐIỀU KHIỂN WEB TƯƠNG TÁC (TÍCH HỢP SẴN) TRÊN CỔNG {port}...")
        print(f"👉 Mở trình duyệt truy cập: http://localhost:{port} (hoặc http://127.0.0.1:{port})")
        print("   (Nhấn Ctrl + C để dừng máy chủ)")
        print("=" * 75 + "\n")

        import http.server
        import socketserver

        os.chdir(str(dashboard_dir))
        Handler = http.server.SimpleHTTPRequestHandler
        
        # Allow immediate port reuse
        socketserver.TCPServer.allow_reuse_address = True
        try:
            with socketserver.TCPServer(("0.0.0.0", port), Handler) as httpd:
                httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n👋 Máy chủ Web đã dừng.")

if __name__ == "__main__":
    main()
