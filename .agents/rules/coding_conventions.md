# 📐 Quy Chuẩn Lập Trình (Project Coding Conventions)

Tài liệu này định nghĩa các quy tắc lập trình bắt buộc áp dụng cho toàn bộ mã nguồn của đề tài **RegiTour / NCKH 2027**. Mọi đoạn mã mới hoặc chỉnh sửa đều phải tuân thủ nghiêm ngặt các nguyên tắc sau:

---

## 1. QUY CHUẨN ĐẶT TÊN (NAMING CONVENTIONS)
* **Tên tệp (File names):** Sử dụng `snake_case` (ví dụ: `ra_exo_dlinear.py`, `dataset_loader.py`, `catboost_regime.py`).
* **Tên lớp (Classes):** Sử dụng `PascalCase` (ví dụ: `RAExoDLinearNet`, `TourismDataLoader`, `CoupledConsistencyOptimizer`).
* **Tên hàm & Phương thức (Functions & Methods):** Sử dụng `snake_case` có động từ chỉ hành vi (ví dụ: `calculate_metrics()`, `load_data()`, `predict_conformal_intervals()`).
* **Tên biến (Variables):** Sử dụng `snake_case` ngắn gọn, tường minh ngữ cảnh (ví dụ: `y_true_arr`, `loss_trend`, `attn_weights`).
* **Hằng số (Constants):** Sử dụng `UPPER_CASE` (ví dụ: `FUTURE_CALENDAR_COLS`, `GEO_LIST`, `PROJECT_ROOT`).

---

## 2. TYPE HINTS & DOCSTRINGS
* **Type Annotations:** Bắt buộc áp dụng Type Hints đầy đủ cho tham số đầu vào và kiểu dữ liệu trả về của tất cả các hàm/phương thức public:
  ```python
  from typing import Dict, List, Optional, Tuple, Union
  import numpy as np
  import torch

  def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
      ...
  ```
* **Docstrings:** Áp dụng chuẩn Google Docstring. Với các khối Deep Learning / PyTorch, **bắt buộc chú thích Tensor Shapes** trong docstring hoặc inline comments:
  ```python
  def forward(
      self,
      x_past: torch.Tensor,    # Shape: (B, L, C_past)
      x_future: torch.Tensor,  # Shape: (B, H, D_future)
  ) -> Tuple[torch.Tensor, torch.Tensor]:
      """
      Forward pass cho mô hình dự báo đa bước.

      Args:
          x_past: Chuỗi quan sát quá khứ độ dài L.
          x_future: Các biến ngoại sinh lịch lễ biết trước độ dài H.

      Returns:
          y_arr: Dự báo lượt khách đến (B, H).
          y_nig: Dự báo số đêm lưu trú (B, H).
      """
  ```

---

## 3. QUY CHUẨN PYTORCH & DEEP LEARNING
1. **Khả năng tái lập (Reproducibility):** Mọi lớp Forecaster đều phải hỗ trợ tham số `random_state: int = 42` để cố định seed cho `torch.manual_seed` và `np.random.seed`.
2. **Quản lý thiết bị (Device Agnostic):** Tạo tensor trên cùng thiết bị với input (`device=x.device`).
3. **Graceful Fallbacks:** Đóng gói module PyTorch bằng cờ `HAS_TORCH` để script vẫn import an toàn khi môi trường thiếu CUDA/Torch:
   ```python
   try:
       import torch
       import torch.nn as nn
       HAS_TORCH = True
   except ImportError:
       HAS_TORCH = False
   ```
4. **Trực quan hóa Phân rã (XAI):** Các mô hình tuyến tính phân rã (như `RA-Exo-DLinear`) phải hỗ trợ cờ `return_components: bool = False` để bóc tách các thành phần độc lập (`trend`, `seasonal`, `future_exo`, `past_exo`).

---

## 4. QUY CHUẨN XỬ LÝ DỮ LIỆU & ĐƯỜNG DẪN
1. **Tuyệt đối không hardcode đường dẫn tuyệt đối:** Luôn dùng `Path(__file__).resolve().parent` hoặc tham chiếu qua [`config.py`](file:///home/tuantn1807/project/nckh_2027/config.py).
2. **Không làm thay đổi DataFrame gốc bất ngờ:** Luôn dùng `.copy()` khi tạo lát cắt dữ liệu hoặc filter split.
3. **Tuân thủ logic nghiệp vụ du lịch:**
   * Mọi dự báo số lượng khách (`arrivals`) và số đêm (`nights`) phải không âm: `.clip(lower=0)`.
   * Số đêm lưu trú không được phép nhỏ hơn số lượt khách: $\text{nights} \ge \text{arrivals}$ ($LOS \ge 1.0$).

---

## 5. CẤU TRÚC TỆP MÃ NGUỒN CHUẨN (STANDARD FILE STRUCTURE)
Mọi tệp Python trong dự án phải tuân theo thứ tự phân đoạn sau:
```python
"""
Module Header: Tên module, chức năng chính và các đóng góp khoa học.
"""

# 1. Standard Library Imports
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

# 2. Third-Party Imports
import numpy as np
import pandas as pd

# 3. Optional Deep Learning Imports with Fallback
try:
    import torch
    import torch.nn as nn
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

# 4. Local Project Imports
from src.data.dataset_loader import TourismDataLoader

# 5. Constants & Configuration

# 6. Core Classes & Neural Network Modules

# 7. High-Level Forecaster Wrappers (Scikit-Learn API style)
```
