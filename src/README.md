
---

## Hướng dẫn tổ chức thư mục

1. **`app/`**  
   - Chứa các ứng dụng triển khai mô hình:
     - `camera_demo.py`: demo sử dụng camera offline.
     - `streamlit_app.py`: ứng dụng web realtime bằng Streamlit.
   
2. **`data/`**  
   - Chứa dataset gốc, các folder theo class (`glass/`, `metal/`, ...).  
   - Không thay đổi dữ liệu gốc, chỉ đọc để chia.

3. **`dataset/`**  
   - Chứa dữ liệu đã chia:
     ```
     dataset/
         train/
             glass/
             metal/
             ...
         val/
             glass/
             ...
         test/
             glass/
             ...
     ```

4. **`scripts/`**  
   - Chứa các script xử lý, train, đánh giá:
     - `preprocess.py`: tạo generator dữ liệu cho train/val/test.
     - `train.py`: huấn luyện model (MobileNetV2/EfficientNetB0).
     - `evaluate.py`: đánh giá model trên tập test.
     - `model/`: lưu model đã train (`model.h5`).

5. **`split_data.py`**  
   - Script độc lập, đọc `data/` và tạo `dataset/` theo tỷ lệ train/val/test.

---

Nếu muốn, tui có thể viết luôn **phiên bản hình minh họa cây thư mục dạng diagram trực quan** để chèn vào báo cáo PDF/PowerPoint, nhìn là hiểu ngay workflow.  

Muốn tui vẽ luôn khum?

