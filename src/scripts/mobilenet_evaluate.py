# evaluate.py
import tensorflow as tf
from tensorflow.keras.models import load_model
from preprocess import get_data_generators
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    f1_score,
    recall_score,
    classification_report,
    confusion_matrix
)
import numpy as np

if __name__ == "__main__":
    # -----------------------------
    # 1) Load test generator
    # -----------------------------
    # Chỉ cần generator test, shuffle=False để giữ thứ tự ảnh
    _, _, test_gen = get_data_generators()

    # -----------------------------
    # 2) Load model đã train
    # -----------------------------
    model = load_model("model/model.h5")

    # -----------------------------
    # 3) Dự đoán trên toàn bộ test set
    # -----------------------------
    test_gen.reset()  # reset iterator để dự đoán từ đầu
    preds = model.predict(test_gen, verbose=1)

    # Chuyển kết quả softmax sang nhãn class (index lớn nhất)
    y_pred = np.argmax(preds, axis=1)

    # Lấy nhãn thật từ generator
    y_true = test_gen.classes

    # Lấy tên class
    class_names = list(test_gen.class_indices.keys())

    # -----------------------------
    # 4) Tính metrics chính
    # -----------------------------
    # Accuracy: tỉ lệ dự đoán đúng trên tổng số mẫu
    print("Accuracy:", accuracy_score(y_true, y_pred) * 100)

    # Precision: độ chính xác của dự đoán (weighted average: tính theo tỉ lệ số mẫu mỗi class)
    print("Precision:", precision_score(y_true, y_pred, average='weighted') * 100)

    # Recall: độ nhạy (weighted average)
    print("Recall:", recall_score(y_true, y_pred, average='weighted') * 100)

    # F1-score: harmonic mean giữa precision và recall (weighted average)
    print("F1:", f1_score(y_true, y_pred, average='weighted') * 100)

    # -----------------------------
    # 5) Confusion Matrix
    # -----------------------------
    # Ma trận nhầm lẫn thể hiện số lượng dự đoán đúng/sai cho từng class
    # Hàng: nhãn thật, cột: nhãn dự đoán
    print("Confusion Matrix:")
    print(confusion_matrix(y_true, y_pred))

    # -----------------------------
    # 6) Classification Report
    # -----------------------------
    # In báo cáo chi tiết theo từng class: precision, recall, f1-score, support
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))
