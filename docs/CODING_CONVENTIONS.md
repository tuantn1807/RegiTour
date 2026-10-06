# 📐 Quy Chuẩn Lập Trình & Thiết Kế Phần Mềm (Coding Conventions)

Tài liệu này quy định các tiêu chuẩn kỹ thuật lập trình thống nhất cho toàn bộ đề tài **RegiTour / NCKH 2027**. Mọi thành viên và các module mới đều phải tuân thủ nghiêm ngặt các quy chuẩn dưới đây để đảm bảo tính đồng bộ, sạch sẽ và chuẩn mực khoa học.

---

## 1. QUY CHUẨN ĐẶT TÊN (NAMING CONVENTIONS)

| Đối tượng | Quy tắc đặt tên | Ví dụ thực tế trong dự án |
| :--- | :--- | :--- |
| **Tệp mã nguồn (Files)** | `snake_case` | `ra_exo_dlinear.py`, `dataset_loader.py`, `benchmark_models.py` |
| **Tên Lớp (Classes)** | `PascalCase` | `RAExoDLinearNet`, `TourismDataLoader`, `CoupledConsistencyOptimizer` |
| **Hàm & Phương thức** | `snake_case` (có động từ) | `calculate_metrics()`, `load_data()`, `predict_conformal_intervals()` |
| **Biến số (Variables)** | `snake_case` (rõ nghĩa) | `y_true_arr`, `trend_pred`, `loss_total`, `base_ratios` |
| **Hằng số (Constants)** | `UPPER_CASE` | `PROJECT_ROOT`, `FUTURE_CALENDAR_COLS`, `TARGET_ARRIVALS` |

---

## 2. KIỂU DỮ LIỆU (TYPE HINTS) & CHÚ THÍCH (DOCSTRINGS)

### a. Bắt buộc Type Annotations
Tất cả các hàm và phương thức trong thư mục `src/` phải khai báo Type Hints rõ ràng:
```python
from typing import Dict, List, Optional, Tuple, Union
import numpy as np
import torch

def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    ...
```

### b. Chú thích Tensor Shapes trong Deep Learning
Mọi khối mạng nơ-ron PyTorch phải ghi chú rõ kích thước tensor (`Shape Comments`) ở từng dòng tính toán để dễ dàng gỡ lỗi và kiểm tra toán học:
```python
def forward(
    self,
    x_past: torch.Tensor,    # (B, L, C_past)
    x_future: torch.Tensor,  # (B, H, D_future)
    regime_id: torch.Tensor, # (B,)
) -> Tuple[torch.Tensor, torch.Tensor]:
    # (B, L, C) -> (B, L, C)
    seasonal, trend = self.decomp(x_past)
    # (B, L, C) -> (B, H, C)
    trend_out = self.linear_trend(trend.permute(0, 2, 1)).permute(0, 2, 1)
```

---

## 3. NGUYÊN TẮC THIẾT KẾ MÔ HÌNH HỌC SÂU (PYTORCH BEST PRACTICES)

1. **Khả năng Tái lập Kết quả (Reproducibility):**  
   Mọi class huấn luyện bắt buộc hỗ trợ tham số `random_state: int = 42` để cố định seed cho cả CPU và GPU:
   ```python
   torch.manual_seed(self.random_state)
   np.random.seed(self.random_state)
   ```
2. **Quản lý Thiết bị (Device-Agnostic):**  
   Không hardcode `.cuda()` hoặc `.cpu()`. Luôn tạo tensor trung gian trên cùng thiết bị với tensor đầu vào: `device=x_past.device`.
3. **An toàn Môi trường (Graceful Fallback):**  
   Bọc các module PyTorch trong cờ `HAS_TORCH` để mã nguồn vẫn có thể import và chạy phần tiền xử lý bảng ngay cả khi môi trường chưa cài PyTorch.
4. **Hỗ trợ Giải thích XAI (Full Interpretability):**  
   Các mô hình thuộc họ Linear Decomposition bắt buộc hỗ trợ cờ `return_components: bool = False` để có thể trích xuất đồ thị phân rã thành phần phục vụ bài báo.

---

## 4. QUY CHUẨN XỬ LÝ DỮ LIỆU & RÀNG BUỘC KINH TẾ (DOMAIN LOGIC)

1. **Không Hardcode Đường dẫn Tuyệt đối:**  
   Luôn sử dụng `Path(__file__).resolve().parent` hoặc nạp từ file cấu hình trung tâm [`config.py`](file:///home/tuantn1807/project/nckh_2027/config.py).
2. **Bảo toàn Dữ liệu:**  
   Luôn dùng `.copy()` khi cắt lát (`slice`) hoặc phân tách tập dữ liệu để tránh lỗi `SettingWithCopyWarning` trong Pandas.
3. **Tuân thủ Tuyệt đối Ràng buộc Nghiệp vụ Du lịch:**
   * Dự báo lượt khách và số đêm phải luôn không âm: `.clip(lower=0)`.
   * Tỷ lệ số đêm lưu trú trên lượt khách phải luôn lớn hơn hoặc bằng $1.0$ (Khách nghỉ tối thiểu 1 đêm):
     $$\text{nights\_total} \ge \text{arrivals\_total} \quad \Longleftrightarrow \quad LOS \ge 1.0$$

---

## 5. CẤU TRÚC PHÂN ĐOẠN FILE MÃ NGUỒN CHUẨN

Mọi file Python mới phải tuân theo trật tự 6 phân đoạn chuẩn sau:

```python
"""
Header: Tên module, chức năng nghiệp vụ và đóng góp phương pháp luận.
"""

# 1. Standard Libraries
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 2. Third-Party Data Libraries
import numpy as np
import pandas as pd

# 3. Deep Learning Libraries (with safe fallback)
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 4. Local Project Imports
from src.data.dataset_loader import TourismDataLoader

# 5. Core Neural Modules / Classes

# 6. High-Level Scikit-Learn Forecaster Wrapper (.fit / .predict)
```
