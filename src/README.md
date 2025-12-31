# Cấu trúc thư mục

```text
src/
├── app/
│   ├── camera_demo.py        # Ứng dụng demo camera offline sử dụng model MobileNetv2
│   └── streamlit_app.py      # Ứng dụng web realtime với Streamlit sử dụng model MobileNetv2
│   └──  camera_demo_eff.py   # Ứng dụng demo camera offline sử dụng model EfficientNetB0
│   └── streamlit_app_eff.py  # Ứng dụng web realtime với Streamlit sử dụng model EfficientNetB0
├── data/                     # Thư mục chứa data.zip
├── scripts/
│   ├── model/
│   │   └── model.h5                # Model MobileNetv2 đã train lưu lại
│   │   └── efficientnetb0.keras    # Model EfficientNetB0 đã train lưu lại
│   ├── YOLO
│   │   └── YOLO.ipynb                #chứa src YOLO
│   ├── efficientnetb0
│   │   └── efficientnetb0.ipynb      #chứa src efficientnetb0
│   ├── mobilenet
│   │   └── mobilenetv2.ipynb         #chứa src mobilenetv2
```

---

## Hướng dẫn tổ chức thư mục

Các file được tải về cần sửa các đường dẫn (file path) nếu có trong code và được tổ chức như sau

1. **`app/`**  
   - Chứa các ứng dụng triển khai mô hình:
     - `camera_demo.py`: demo sử dụng camera offline sử dụng model MobileNetv2.
     - `streamlit_app.py`: ứng dụng web realtime bằng Streamlit sử dụng model MobileNetv2.
     - `camera_demo.py`: demo sử dụng camera offline sử dụng model EfficientNetB0.
     - - `streamlit_app.py`: ứng dụng web realtime bằng Streamlit sử dụng model EfficientNetB0.
2. **`data/`**  
   - Chứa dataset gốc, các folder theo class (`glass/`, `metal/`, ...).  
   - Không thay đổi dữ liệu gốc, chỉ đọc để chia.


3. **`scripts/`**  
   - Chứa các script xử lý, train, đánh giá của từng model.
   - Chứa các model sau train, phần này nặng, không có trên git, các model sẽ xuất hiện khi train xong trong các file notebook.
     
---

