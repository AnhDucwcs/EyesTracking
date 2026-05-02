# Eye Tracking

Hệ thống theo dõi ánh mắt (Eye Tracking) sử dụng webcam thông thường, dựa trên MediaPipe Face Mesh và Machine Learning để ước tính vị trí nhìn trên màn hình.

## Tính năng

- **Theo dõi ánh mắt realtime** sử dụng webcam thông thường
- **Calibration tự động** với 9 điểm trên màn hình
- **Head pose estimation** để bù trừ chuyển động đầu
- **Filtering** (EMA, One Euro Filter) để làm mượt dữ liệu
- **Thu thập và huấn luyện dữ liệu** để cải thiện độ chính xác
- **Hỗ trợ nhiều điều kiện ánh sáng** (ban ngày, ban đêm)


## Sử dụng

### 1. Chạy ứng dụng chính

```bash
python src/app/main_app.py
```

### 2. Quy trình Calibration

1. Nhấn **SPACE** để bắt đầu calibration
2. Nhìn vào từng điểm đỏ xuất hiện trên màn hình (9 điểm)
3. Giữ đầu thẳng và nhìn chằm chằm vào mỗi điểm trong khoảng 2 giây
4. Sau khi hoàn tất, hệ thống sẽ tự động huấn luyện mô hình

### 3. Sử dụng Eye Tracking

- Sau calibration, hệ thống sẽ hiển thị điểm gaze (màu xanh lá) trên màn hình
- Nhấn **ESC** hoặc **Q** để thoát

### 4. Thu thập dữ liệu (nâng cao)

```bash
python src/core/data_collector.py
```

### 5. Huấn luyện mô hình (nâng cao)

```bash
# Gộp dữ liệu từ nhiều nguồn
python combine_data.py

# Huấn luyện mô hình
python train_model.py
```

### 6. Kiểm tra dữ liệu

```bash
python inspect_data.py
```


```

## Troubleshooting

### Camera không hoạt động

```python
# Thử thay đổi camera index trong capture.py
self.cap = cv2.VideoCapture(0)  # Thử 0, 1, 2, ...
```

### Calibration không chính xác

- Đảm bảo ánh sáng đủ và đều
- Giữ đầu thẳng và không di chuyển trong khi calibration
- Tăng số mẫu thu thập trong `calib_callback`:

```python
# Tăng từ 50 lên 90
samples_per_point=50  
```

### Gaze point bị rung

```python
# Tăng min_cutoff lên 0.8 để giảm rung và giảm beta xuống 0.05 hoặc 0.01 để giảm nhạy khi di chuyển quá nhanh
gaze_filter = OneEuroWrapper(min_cutoff = 0.5, beta = 0.1)  
```

## Đánh giá hiệu suất

Độ chính xác phụ thuộc vào:
- Chất lượng calibration
- Điều kiện ánh sáng
- Độ phân giải camera
- Khoảng cách từ camera đến người dùng

### Kết quả huấn luyện mô hình

Sau khi huấn luyện với 48,077 mẫu dữ liệu (38,461 mẫu training):

| Mô hình | RMSE | R² Score | Ghi chú |
|---------|------|----------|---------|
| **RandomForest** | 0.110965 | 0.8621 | RMSE thấp nhất |
| **KNeighbors** | 0.113184 | 0.8565 | Đã lưu |
| **MLP** | 0.116763 | 0.8472 | **Ổn định nhất trong thực tế** |

**Nhận xét thực tế:**
- Cả 3 mô hình đều hoạt động tốt trên trục hoành (X)
- Cả 3 mô hình đều yếu hơn trên trục tung (Y)
- Mặc dù RMSE cao hơn, **mô hình MLP cho kết quả ổn định nhất trong thực tế** và được khuyến nghị sử dụng


## Tài liệu tham khảo

- [MediaPipe Face Mesh](https://google.github.io/mediapipe/solutions/face_mesh.html)
- [OpenCV Documentation](https://docs.opencv.org/)
- [Head Pose Estimation using OpenCV and Dlib](https://learnopencv.com/head-pose-estimation-using-opencv-and-dlib/)
- [One Euro Filter](http://cristal.univ-lille.fr/~casiez/1euro/)

---
