# 📈 BÁO CÁO ĐÁNH GIÁ KHOA HỌC KẾT QUẢ DỰ BÁO (SCIENTIFIC EVALUATION REPORT)
## Kiến trúc đề xuất: RegiTour Framework (RA-HE-DLinear)
> **Tập dữ liệu kiểm thử (Test Split):** Giai đoạn 2024 – 2026 (4,712 bản ghi, 38 Quốc gia Châu Âu, 4 Loại hình lưu trú NACE).  
> **Thời điểm xuất báo cáo:** Tự động đồng bộ từ pipeline đánh giá chuẩn hóa.

---

### 1. BẢNG TỔNG HỢP HIỆU NĂNG TOÀN DIỆN (OVERALL PERFORMANCE)

| Chỉ tiêu / Mục tiêu | RMSE | MAE | WAPE (%) | Conformal Coverage [P10, P90] | Khung mô hình đề xuất |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Lượt khách (Arrivals)** | **`589,486`** | **`213,562`** | **`14.67%`** | **`91.4%`** | **RA-HE-DLinear ⭐** *(Mô hình phân rã tuyến tính đề xuất)* |
| **Số đêm lưu trú (Nights)** | `3,441,309` | **`983,447`** | **`25.00%`** | **`91.4%`** | **RA-HE-DLinear ⭐** *(Mô hình phân rã tuyến tính đề xuất)* |
| **Lượt khách (Arrivals)** | `733,884` | `289,365` | `19.88%` | `94.2%` | RA-HE-BiLSTM *(Mô hình học sâu kết hợp)* |
| **Số đêm lưu trú (Nights)** | **`2,827,337`** | `1,028,961` | `26.16%` | `94.1%` | RA-HE-BiLSTM *(Mô hình học sâu kết hợp)* |

---

### 2. PHÂN RÃ HIỆU NĂNG THEO 4 LOẠI HÌNH LƯU TRÚ NACE (ACCOMMODATION BREAKDOWN)

| Loại hình lưu trú (NACE) | Số bản ghi | Arrivals WAPE (%) | Arrivals MAE | Nights WAPE (%) | Nights MAE |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **`camping`** | 1,178 | 26.10% | 72,544 | 35.00% | 388,644 |
| **`hotel`** | 1,178 | 14.70% | 304,992 | 22.66% | 1,104,384 |
| **`short_stay`** | 1,178 | 34.27% | 165,748 | 38.43% | 676,153 |
| **`total`** | 1,178 | 18.25% | 507,111 | 22.78% | 1,718,211 |

---

### 3. KIỂM ĐỊNH TÍNH NHẤT QUÁN NGHIỆP VỤ & BẢO CHỨNG RỦI RO

1. **Ràng buộc nhất quán Kinh tế du lịch (Economic Consistency):**
   - **Cận vật lý cơ bản ($\text{Nights} \ge \text{Arrivals}$):** Toán tử chiếu mốc neo bất đối xứng $\Pi_{\mathcal{C}}$ bảo đảm đầu ra tuân thủ cận vật lý tối thiểu ($<0.01\%$), trong khi hàm phạt $\mathcal{L}_{\text{cons}}$ giúp mô hình nơ-ron tự kiểm soát vi phạm tự nhiên trước chiếu ở mức chỉ $0.15\%$ (7/4,712 mẫu).
   - **Dải tỷ lệ kinh tế lịch sử theo quốc gia ($[r_{\min}, r_{\max}]$):** RA-HE-DLinear kiểm soát độ lệch dải ở mức tối ưu **$16.02\%$**, vượt trội so với LightGBM ($49.19\%$), XGBoost ($37.71\%$) và Seasonal Naive ($28.82\%$). Mức lệch $16.02\%$ này phản ánh trung thực biến động hành vi du lịch hậu COVID (bản thân Ground Truth cũng lệch $22.18\%$ so với lịch sử tiền COVID).

2. **Bảo chứng toán học về độ bao phủ không phụ thuộc phân phối (Conformal Prediction):**
   - Động cơ DCQF đạt độ bao phủ thực tế **94.25%** (Arrivals) và **94.14%** (Nights) tại mức danh định $90\%$ ($\alpha = 0.10$), hoàn toàn thỏa mãn định lý biên toán học $\mathbb{P}(y \in [P_{10}, P_{90}]) \ge 1 - \alpha = 90\%$.
   - Giúp các nhà hoạch định chính sách du lịch Châu Âu có cơ sở quản trị rủi ro vững chắc cho kịch bản thắt chặt ($P_{10}$) và bùng nổ mùa hè ($P_{90}$).

3. **Cơ chế trọng tài động MoE:**
   - Trọng số trung bình trên tập Test: **30% Trend (Vĩ mô) + 26% Seasonal (Lịch lễ) + 44% Spatial (Lan tỏa Schengen)**, khẳng định vai trò thiết yếu của việc mô hình hóa không gian liên quốc gia mà không cần dùng GNN cồng kềnh.

---
*Báo cáo được lưu trữ tự động tại `results/evaluation_report.md` và các bảng dữ liệu số tại `results/tables/`.*

---

### 4. BẢNG SO SÁNH ĐỐI KHÁNG TRỰC TIẾP VỚI CÁC MÔ HÌNH BASELINE (BENCHMARK COMPARISON)

Bảng dưới đây tổng hợp kết quả đối sánh thực nghiệm giữa kiến trúc đề xuất **RA-HE-DLinear** và các mô hình đại diện cho các thế hệ phương pháp trong kinh tế học du lịch (trích xuất từ `results/tables/benchmark_model_comparison.csv`):

| Mô hình đối sánh (Model) | Lớp mô hình (Family) | Arrivals WAPE (%) | Arrivals MAE | Nights WAPE (%) | Nights MAE | Vi phạm cận vật lý (< 1.0) | Lệch dải kinh tế ([r_min, r_max]) | Cải thiện sai số vs S-Naive |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| Seasonal Naive (S-Naive) | Statistical | 27.30% | 397,503 | 32.73% | 1,287,411 | 0.05% | 28.8% | +0.0% |
| Standard LightGBM | Machine Learning | 30.37% | 442,199 | 34.28% | 1,348,103 | 0.19% | 49.2% | -3.07% |
| Standard XGBoost | Machine Learning | 34.13% | 496,861 | 42.36% | 1,666,128 | 0.34% | 37.7% | -6.83% |
| Vanilla DLinear (AAAI 2023) | Linear LTSF (Modern Baseline) | 45.38% | 660,721 | 25.61% | 1,007,190 | 0.05% | 28.8% | -18.08% |
| Vanilla NLinear (AAAI 2023) | Linear LTSF (Modern Baseline) | 32.10% | 467,388 | 49.38% | 1,942,248 | 0.69% | 29.2% | -4.8% |
| Standard LSTM (Hochreiter, 1997) | Deep Learning (Classical) | 108.53% | 1,580,071 | 113.16% | 4,450,857 | 0.05% | 99.4% | -81.23% |
| Vanilla BiLSTM (Graves, 2013) | Deep Learning (Classical) | 108.53% | 1,580,071 | 113.16% | 4,450,857 | 0.05% | 99.4% | -81.23% |
| CatBoost-Regime (Ablation) | Machine Learning (Ablation) | 40.27% | 586,231 | 44.86% | 1,764,508 | 0.82% | 44.9% | -12.97% |
| **RA-Exo-DLinear (Proposed Linear ⭐)** | Linear LTSF (Proposed ⭐) | **18.29%** | 266,307 | **25.87%** | 1,017,349 | 0.21% | 21.5% | +9.01% |
| **RA-HE-DLinear (Proposed Linear Framework ⭐)** | Linear Framework (Proposed ⭐) | **14.59%** | 212,341 | **25.00%** | 983,447 | **0.00% (sau chiếu; 0.15% cần sửa)** | **16.0%** | **+12.71%** |

#### 💡 Phân Tích Khoa Học & Bàn Luận (Scientific Discussion):
1. **Sự vượt trội của Framework Tuyến tính Đề xuất (RA-HE-DLinear Breakthrough):**
   - **`RA-HE-DLinear`** thiết lập kỷ lục sai số thấp nhất toàn bảng với **WAPE `14.59%` (Arrivals)** và **`25.00%` (Nights)**, cải thiện `+12.71%` so với Seasonal Naive (`27.30%`) và đánh bại các baseline AAAI 2023 (*Vanilla DLinear* `45.38%`, *Vanilla NLinear* `32.10%`).
   - Việc kết hợp lõi thời gian `RA-Exo-DLinear` với thành phần ma trận lan tỏa không gian `TRMF-Spatial` giúp bổ khuyết hoàn hảo 2 chiều Không - Thời gian mà không bị cồng kềnh hay chồng chéo chức năng.
2. **Làm rõ tính chân thực của chỉ số nhất quán (Dismantling the "0% Fallacy"):**
   - Việc tuyên bố một mô hình dự báo đạt "0.0% vi phạm logic" nếu chỉ dựa vào phép gán cứng hậu xử lý là một ngụy tạo thống kê. Trong dữ liệu thực nghiệm (Ground Truth), do sai số ghi nhận thực tế cũng có $0.05\%$ quan sát có Nights < Arrivals và $22.18\%$ quan sát lệch khỏi dải lưu trú lịch sử do biến động hành vi du lịch hậu đại dịch.
   - Điểm khác biệt khoa học thực chất nằm ở **Độ lệch dải kinh tế lịch sử**: trong khi LightGBM lệch tới $49.2\%$ và XGBoost lệch $37.7\%$, RA-HE-DLinear duy trì độ lệch ở mức tối ưu **$16.0\%$**, bám sát đặc thù lưu trú của từng quốc gia.
   - Về cận vật lý, hàm phạt $\mathcal{L}_{\text{cons}}$ giúp RA-HE-DLinear kiểm soát vi phạm tự nhiên trước chiếu ở mức $0.15\%$ (chỉ 7/4,712 mẫu cần can thiệp chiếu mốc neo), chứng minh mô hình nơ-ron thực sự học được mối liên kết giữa hai chuỗi chứ không chỉ phụ thuộc vào bộ gán cứng hậu xử lý.
