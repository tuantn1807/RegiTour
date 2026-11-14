# BÁO CÁO ĐỒ ÁN / LUẬN VĂN TỐT NGHIỆP BẰNG LATEX (RA-HE-DLINEAR)

Bộ mã nguồn LaTeX báo cáo hoàn chỉnh được xây dựng chính xác theo đề cương yêu cầu và chuẩn trình bày của **Trường Đại học Công nghệ Thông tin và Truyền thông Việt -- Hàn (VKU)**, lấy mẫu trang bìa giống tệp `Thực tập tốt nghiệp.pdf`.

---

## 1. Cấu Trúc Thư Mục và Các Tệp Thành Phần

```
report/
├── main.tex                    # Tệp gốc điều khiển biên dịch toàn bộ báo cáo
├── cover.tex                   # Trang bìa ngoài (Hoa văn khung viền & họa tiết gốc từ Thực tập tốt nghiệp.pdf)
├── cover_sub.tex               # Trang bìa phụ (Bìa trong)
├── review.tex                  # Nhận xét của Giảng viên hướng dẫn (Hàng chấm chấm chuẩn)
├── acknowledgments.tex         # Lời cảm ơn (Tri ân GVHD TS. Hoàng Nguyên Vũ, Khoa KHMT, Nhà trường)
├── abstract.tex                # Tóm tắt đề tài song ngữ (Tiếng Việt & Tiếng Anh)
├── references.bib              # Danh mục tài liệu tham khảo BibTeX theo chuẩn APA
├── appendices.tex              # Phụ lục A (Bổ đề toán học), Phụ lục B (Danh mục 38 nước), Phụ lục C (Đa chân trời)
├── summary_ieee.tex            # Tóm tắt báo cáo dạng bài báo khoa học chuẩn IEEE (tích hợp trong báo cáo)
├── summary_paper_ieee.tex      # Tệp độc lập biên dịch bài báo khoa học IEEE (<= 10 trang)
├── IEEEtran.cls                # Class chính thức của IEEE Transactions
├── assets/                     # Tài nguyên đồ họa trang bìa
│   ├── border_frame.png        # Khung viền hoa văn đôi trích xuất từ Thực tập tốt nghiệp.pdf
│   └── decorative_divider.png  # Thanh họa tiết phân cách trang trọng
├── figures/                    # Toàn bộ biểu đồ và hình ảnh thực nghiệm
│   ├── ra_he_dlinear_architecture.png
│   ├── forecast_trajectory_eurostat.png
│   ├── consistency_projection_analysis.png
│   ├── consistency_projection_scatter.png
│   ├── los_distribution_economic_band.png
│   ├── hyperparameter_sensitivity.png
│   ├── shap_feature_importance.png
│   └── attention_cross_temporal_heatmap.png
└── chapters/                   # 5 chương nội dung chính
    ├── ch1_introduction.tex          # Chương 1: Giới thiệu (Motivation, Problem, Objectives, Scopes, Structure)
    ├── ch2_literature_review.tex     # Chương 2: Tổng quan tài liệu & Cơ sở lý thuyết (DLinear, HMM, FiLM, TRMF, ...)
    ├── ch3_proposed_method.tex       # Chương 3: Phương pháp nghiên cứu đề xuất (Kiến trúc, RA-Exo, Projection, DCQF)
    ├── ch4_experimental_results.tex  # Chương 4: Kết quả thực nghiệm (Eurostat, World Bank, Baselines, Ablation, XAI)
    └── ch5_conclusion.tex            # Chương 5: Kết luận & Hướng phát triển
```

---

## 2. Đối Chiếu Đáp Ứng Đề Cương Yêu Cầu

| Mục đề cương | Yêu cầu trang | Hiện thực hóa trong LaTeX |
| :--- | :--- | :--- |
| **Trang tiêu đề (Title page)** | $=1$ trang | `cover.tex` & `cover_sub.tex` với đầy đủ tên đề tài, họ tên sinh viên (Trần Nguyễn Tuấn - 22IT329, Phan Hoài Bão), GVHD (TS. Hoàng Nguyên Vũ), kèm khung viền và họa tiết chuẩn từ `Thực tập tốt nghiệp.pdf`. |
| **Trang nội dung (Content page)** | $\le 5$ trang | Tự động tạo bằng `\tableofcontents`, `\listoffigures`, `\listoftables`, `DANH MỤC TỪ VIẾT TẮT`. Đánh số mục theo phong cách chuẩn **1.1.1, 1.1.2...** |
| **Chương 1. Giới thiệu** | 5--7 trang | `chapters/ch1_introduction.tex`: Đầy đủ 5 mục bắt buộc: Lý do chọn đề tài (Motivation), Phát biểu bài toán (Problem Statement), Mục tiêu & Đóng góp, Đối tượng & Phạm vi (Scopes), Cấu trúc đồ án (Thesis Structure). |
| **Chương 2. Tổng quan & Cơ sở lý thuyết** | $\le 25$ trang | `chapters/ch2_literature_review.tex`: Cơ sở lý thuyết chi tiết (LTSF-Linear, HMM, FiLM, TRMF, Conformal, SHAP); Các nghiên cứu liên quan; Bảng phân tích khoảng trống nghiên cứu (Research Gaps). |
| **Chương 3. Phương pháp đề xuất** | $\le 20$ trang | `chapters/ch3_proposed_method.tex`: Sơ đồ khối kiến trúc hệ thống, động cơ thời gian RA-Exo-DLinear, bóc tách TRMF, toán tử chiếu mốc neo $\Pi_{\mathcal{C}}$, lượng hóa độ bất định DCQF, thuật toán huấn luyện liên hợp `Algorithm`. |
| **Chương 4. Kết quả thực nghiệm** | $\le 25$ trang | `chapters/ch4_experimental_results.tex`: Môi trường máy chủ, mô tả dữ liệu Eurostat (38 nước) & World Bank (203 nền kinh tế), kiểm thử Expanding Window, các chỉ số (WAPE, RMSE, VR, HBC, ECR), bảng so sánh 8 baselines, phân tích quỹ đạo, phân tích chiếu ràng buộc, ablation study, độ nhạy, XAI SHAP & Attention. |
| **Chương 5. Kết luận & Hướng phát triển** | $\le 5$ trang | `chapters/ch5_conclusion.tex`: Đóng góp chính, hạn chế còn tồn tại, kiến nghị chính sách và hướng phát triển trong tương lai. |
| **Tài liệu tham khảo (References)** | $\le 3$ trang | Định dạng trích dẫn chuẩn **APA (American Psychological Association)** tích hợp trong `main.tex` và `references.bib`. |
| **Phụ lục (Appendices)** | $\le 10$ trang | `appendices.tex`: Phụ lục A (Chứng minh bổ đề tối ưu phép chiếu), Phụ lục B (Danh mục 38 quốc gia và biến ngoại sinh), Phụ lục C (Bảng phân rã kết quả theo đa chân trời $H=3, 6, 12$). |
| **Summary / Tóm tắt bài báo IEEE** | $\le 10$ trang | `summary_ieee.tex` (tóm tắt tích hợp) và `summary_paper_ieee.tex` (bài báo khoa học hoàn chỉnh theo chuẩn IEEE 2 cột). |

---

## 3. Hướng Dẫn Biên Dịch Sang PDF

### Cách 1: Sử dụng Overleaf (Khuyên Dùng)
1. Nén toàn bộ thư mục `report/` thành tệp `report.zip`.
2. Truy cập [Overleaf](https://www.overleaf.com), chọn **New Project** $\to$ **Upload Project** và tải lên `report.zip`.
3. Chọn trình biên dịch **pdfLaTeX** hoặc **XeLaTeX**, chọn `main.tex` làm Main document và nhấn **Recompile**.

### Cách 2: Biên dịch qua dòng lệnh với TeX Live
```bash
cd /home/tuantn1807/project/nckh_2027/report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

### Cách 3: Biên dịch bài báo tóm tắt IEEE độc lập
```bash
cd /home/tuantn1807/project/nckh_2027/report
pdflatex summary_paper_ieee.tex
```
