# Cấu trúc thư mục

```text
src/
├── app/
│   ├── camera_demo.py        # Ứng dụng demo camera offline
│   └── streamlit_app.py      # Ứng dụng web realtime với Streamlit
├── data/                     # Thư mục chứa dataset gốc
├── dataset/                  # Thư mục chứa dữ liệu đã chia train/val/test
├── scripts/
│   ├── model/
│   │   └── model.h5          # Model đã train lưu lại
│   ├── evaluate.py           # Script đánh giá model trên test set
│   ├── preprocess.py         # Script tạo ImageDataGenerator cho train/val/test
│   └── train.py              # Script huấn luyện model
└── split_data.py             # Script chia dữ liệu từ data → dataset
```

---

## Hướng dẫn tổ chức thư mục

Các file được tải về cần sửa các đường dẫn (file path) nếu có trong code và được tổ chức như sau

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

