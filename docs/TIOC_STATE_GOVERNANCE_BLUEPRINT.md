# 🏛️ Đề Án Xây Dựng Hệ Thống Quản Lý Nhà Nước Về Du Lịch Thông Minh (TIOC — RegiTour)
### Áp dụng cho: Cục Du Lịch Quốc Gia & Sở Du Lịch các Tỉnh/Thành Phố

---

## 1. Bối Cảnh & Sự Cần Thiết Đầu Tư
Trong công tác quản lý nhà nước về du lịch hiện nay, các cơ quan quản lý (Cục Du lịch Quốc gia, Sở Du lịch các tỉnh/thành) đang đối mặt với 3 thách thức lớn:
1. **Thiếu công cụ dự báo định lượng tin cậy:** Việc giao chỉ tiêu lượt khách, ngày khách lưu trú và lập dự toán ngân sách xúc tiến hàng năm phần lớn dựa trên số liệu lịch sử quá khứ theo phương pháp ngoại suy cảm tính.
2. **Số liệu thống kê bị phân mảnh và mâu thuẫn:** Tình trạng số liệu dự báo lượt khách ($Y^{arr}$) và số đêm lưu trú ($Y^{nig}$) không đồng bộ, thậm chí vi phạm quy luật vật lý ($LOS < 1.0$), gây khó khăn khi giải trình trước Hội đồng Nhân dân và UBND tỉnh.
3. **Bị động trước các cú sốc vĩ mô & quá tải điểm đến:** Không có kịch bản diễn tập định lượng khi xảy ra biến động giá vé máy bay, lạm phát giá phòng, dịch bệnh hoặc quá tải hạ tầng đô thị trong mùa cao điểm hè.

**Hệ thống RegiTour TIOC (Tourism Intelligent Operations Center)** được thiết kế để giải quyết trọn vẹn các bài toán trên bằng cách kết hợp giữa **mô hình AI chuỗi thời gian kinh tế lượng tiên tiến (RA-HE-DLinear)** và **ma trận điều hành quản lý nhà nước liên ngành**.

---

## 2. Kiến Trúc 6 Phân Hệ Quản Lý Nhà Nước Chuyên Sâu

```
┌─────────────────────────────────────────────────────────────────────────────┐
│            TRUNG TÂM GIÁM SÁT & ĐIỀU HÀNH DU LỊCH THÔNG MINH (TIOC)         │
│               (Dành cho Lãnh đạo Cục / Giám đốc Sở Du lịch)                 │
└──────────────────────────────────────┬──────────────────────────────────────┘
                                       │
     ┌───────────────────┬─────────────┴───────┬───────────────────┐
     │                   │                     │                   │
┌────▼─────────────┐┌────▼──────────────┐┌─────▼────────────┐┌─────▼────────────┐
│ Phân hệ 1        ││ Phân hệ 2         ││ Phân hệ 3        ││ Phân hệ 4        │
│ QUY HOẠCH & GIAO ││ KIỂM SOÁT TẢI TRỌNG││ ĐỊNH HƯỚNG NHÂN  ││ CẤP PHÉP BẢO TRÌ│
│ CHỈ TIÊU KPI NĂM ││ & CHỐNG QUÁ TẢI   ││ LỰC THỜI VỤ      ││ CƠ SỞ VẬT CHẤT   │
│ • Lượt khách arr ││ • Đỉnh khách ngày ││ • Nhu cầu +FTE   ││ • Công suất <35% │
│ • Số đêm nig     ││ • Sức chứa buồng  ││ • Đào tạo nghề   ││ • Tránh mùa đỉnh │
│ • Dải Conformal  ││ • Phân luồng xe   ││   trước 60 ngày  ││   cao điểm hè    │
└──────────────────┘└───────────────────┘└───────────────────┘└───────────────────┘
     │                   │                     │                   │
     └───────────────────┼─────────────────────┴───────────────────┘
                         │
     ┌───────────────────┴─────────────────────┐
     │                                         │
┌────▼───────────────────────────────────┐┌────▼────────────────────────────────┐
│ Phân hệ 5                              ││ Phân hệ 6                           │
│ DIỄN TẬP CÚ SỐC VĨ MÔ (STRESS TEST)    ││ VĂN PHÒNG ĐIỆN TỬ & TỜ TRÌNH UBND   │
│ • Mô phỏng Lạm phát phòng (HICP shock) ││ • Xuất dự thảo Tờ trình UBND tỉnh   │
│ • Mô phỏng siết chặt biên giới / visa  ││ • Báo cáo kỳ họp HĐND tỉnh          │
│ • Kịch bản Sàn bảo hộ dòng tiền        ││ • Kiểm toán 100% chuẩn UNWTO (Π_C)  │
└────────────────────────────────────────┘└─────────────────────────────────────┘
```

---

## 3. Chi Tiết Các Quy Trình Nghiệp Vụ Của Cơ Quan Nhà Nước

### 3.1. Quy trình 1: Lập Quy Hoạch & Giao Chỉ Tiêu Hàng Năm
1. **Đầu vào:** Cơ quan quản lý chọn địa bàn (quốc gia / tỉnh / thành), phân khúc lưu trú (Tổng hợp, Khách sạn, Căn hộ, Cắm trại) và chân trời $H=12$ tháng.
2. **Xử lý AI:**
   - Phân tích chuỗi mùa vụ và xu hướng tăng trưởng bằng **RA-Exo-DLinear**.
   - Bất biến $\Pi_{\mathcal{C}}$ bảo đảm $Y^{nig} \ge Y^{arr}$ và $LOS \ge 1.0$.
   - Tính toán dải tin cậy $\hat{\mathcal{I}}_{0.90}$ bao gồm:
     - **Kịch bản Cơ sở (Baseline Point Forecast):** Dùng để giao chỉ tiêu phấn đấu cho toàn ngành.
     - **Kịch bản Sàn (Lower 90% Revenue-at-Risk):** Dùng để thẩm định dự toán thu ngân sách an toàn.
     - **Kịch bản Trần (Upper 90% Capacity-Strain):** Dùng để dự trù công suất hạ tầng giao thông và điện nước.

### 3.2. Quy trình 2: Điều Phối Hành Động Liên Ngành Chống Quá Tải (Anti-Overtourism)
Khi hệ thống phát hiện các tháng có chỉ số lấp đầy $\ge 85\%$ hoặc lượng khách ngày vượt ngưỡng phục vụ tối đa:
* **Lệnh điều phối sang Công an & Sở Giao thông Vận tải:**
  - Thiết lập phương án cấm xe du lịch trên 45 chỗ vào trung tâm trong các khung giờ cao điểm.
  - Phối hợp Cục Hàng không và Hải quan mở 100% cửa kiểm soát tự động tại sân bay.
* **Lệnh điều phối sang Ban Quản lý Di tích & Danh lam Thắng cảnh:**
  - Triển khai bán vé điện tử theo khung giờ (Time-slotted ticketing) để giãn dòng khách.

### 3.3. Quy trình 3: Điều Tiết Đào Tạo Nghề Du Lịch Trước Mùa Vụ
* Hệ thống tính toán tổng số lao động thiếu hụt: $\Delta \text{FTE} = \text{Peak FTE} - \text{Baseline FTE}$.
* Phân tách chính xác theo từng nhóm NACE:
  - Khách sạn & tương đương (I551): Cần bao nhiêu buồng phòng, lễ tân.
  - Căn hộ ngắn ngày (I552): Nhu cầu dọn dẹp, quản gia.
  - Khu cắm trại / du lịch sinh thái (I553): Nhân viên an toàn, hướng dẫn viên.
* **Hành động nhà nước:** Sở Du lịch ký văn bản phối hợp với Sở Lao động - Thương binh & Xã hội đặt hàng các trường cao đẳng/trung cấp nghề mở các khóa đào tạo cấp tốc trước ngày 01 tháng 05 hàng năm.

### 3.4. Quy trình 4: Cấp Phép Xây Dựng & Bảo Trì Điểm Đến
* Hệ thống tự động xác định các tháng có tỷ lệ lấp đầy $< 35\%$ (**Cửa sổ bảo trì**).
* **Hành động nhà nước:** Sở Du lịch ban hành văn bản quy định khung thời gian khuyến nghị nâng cấp cơ sở lưu trú và chỉnh trang đô thị vào đúng các tháng này. Kiên quyết không cấp phép sửa chữa lớn vào giai đoạn cao điểm hè.

### 3.5. Quy trình 5: Tự Động Khởi Tạo Dự Thảo Tờ Trình & Báo Cáo Hành Chính
* Hệ thống tích hợp sẵn mẫu văn bản hành chính nhà nước chuẩn:
  - **Tờ trình UBND tỉnh:** Đề xuất phê duyệt phương án chỉ tiêu và dự toán xúc tiến du lịch.
  - **Báo cáo Hội đồng Nhân dân:** Phục vụ các kỳ họp thường lệ cuối năm.
* Toàn bộ bảng biểu, biểu đồ và phụ lục được đính kèm tự động chỉ với 1 cú click chuột (`.md`, `.json`, `.pdf`).

---

## 4. Hướng Dẫn Vận Hành Dành Cho Cán Bộ Quản Lý Nhà Nước

### Bước 1: Khởi động Hệ thống
```bash
# Khởi chạy giao diện điều hành dành cho Lãnh đạo
python cli.py dashboard --port 8501
```
* Trên thanh điều hướng bên trái (Sidebar), tại mục **👤 Vai trò khai thác**, chọn:
  👉 **`🏛️ Cơ quan Quản lý Nhà nước (Cục / Sở Du lịch)`**.

### Bước 2: Xem Trung Tâm Điều Hành (TIOC)
* Màn hình chính sẽ hiển thị ngay **Ma trận Điều phối Hành động Liên Ngành** gồm 4 trụ cột: Công an/Giao thông, Lao động/Đào tạo, Xây dựng/Bảo trì và Tài chính/Ngân sách.

### Bước 3: Xuất Tờ Trình UBND Tỉnh
* Vào phân hệ **`🔌 Tích hợp API & Báo cáo Tự động`** -> Chọn tab **`📄 Khởi Tạo Báo Cáo Điều Hành`**.
* Chọn định dạng: **`🏛️ Tờ trình UBND / Báo cáo Cấp trên (.md)`** -> Nhấn nút **Tải Dự Thảo Tờ Trình UBND** để trích xuất văn bản phục vụ trình ký.

