import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import time
import numpy as np
from collections import deque
from core.capture import Camera
from core.detection import FaceMeshDetector
from core.gaze_estimator import GazeEstimator
import app.ui_overlay as ui

HISTORY_LEN = 200
COLORS_FEAT = [
    (255, 100, 100), (100, 255, 100), (100, 100, 255), (255, 255, 100), # Feat 0-3: Mắt (Xanh/Cam/Đỏ...)
    (255, 0, 255), (0, 255, 255), (255, 255, 255),                      # Feat 4-6: Góc Quay (Tím/Vàng/Trắng)
    (100, 100, 100), (150, 150, 150), (0, 165, 255)                     # Feat 7-9: Tịnh tiến (Xám/Cam đậm)
]

def main():
    cam = Camera()
    detector = FaceMeshDetector()
    estimator = GazeEstimator()
    
    FRAME_WIDTH, FRAME_HEIGHT = cam.width, cam.height
    print("Đang khởi động camera...")
    cam.start()
    time.sleep(1.0) 

    WINDOW_NAME = "Threshold Finder (Dashboard)"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    # Khởi tạo bộ nhớ lưu lịch sử 10 đặc trưng
    feat_history = [deque(maxlen=HISTORY_LEN) for _ in range(10)]
    
    # Khởi tạo Min/Max toàn cục
    global_min = [float('inf')] * 10
    global_max = [float('-inf')] * 10
    
    paused = False

    print("--- HƯỚNG DẪN ---")
    print("1. Quan sát các biểu đồ.")
    print("2. Thực hiện các hành động (nháy mắt, xoay đầu).")
    print("3. Ghi lại các giá trị Min/Max 'bình thường' để làm ngưỡng.")
    print("Phím tắt: 'r': Reset Min/Max | 'SPACE': Pause | 'q': Quit")

    while True:
        if not paused:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            
            frame = cv2.flip(frame, 1)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = detector.process(frame_rgb)

            # Tạo Dashboard đen (Rộng 1200, Cao 800)
            dashboard = np.zeros((800, 1200, 3), dtype=np.uint8)
            
            # 1. XỬ LÝ DỮ LIỆU
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                landmarks_array = detector.landmarks_to_array(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
                
                # Vẽ landmarks lên frame webcam gốc
                ui.draw_key_landmarks(frame, landmarks_array)
                
                # --- TRÍCH XUẤT ĐẶC TRƯNG (Dùng hàm của bạn) ---
                # Hàm này trả về mảng 10 phần tử: [Eye(4), Rot(3), Trans(3)]
                features = estimator.extract_features(landmarks_array, FRAME_WIDTH, FRAME_HEIGHT)
                
                # Cập nhật lịch sử Min/Max
                for i, val in enumerate(features):
                    feat_history[i].append(val)
                    if val < global_min[i]: global_min[i] = val
                    if val > global_max[i]: global_max[i] = val

            # 2. VẼ DASHBOARD
            
            # A. Đặt webcam thu nhỏ vào góc trái trên
            thumb_h, thumb_w = 240, 426
            frame_small = cv2.resize(frame, (thumb_w, thumb_h))
            dashboard[500:500+thumb_h, 0:thumb_w] = frame_small
            ui.draw_text(dashboard, "Camera Feed", (10, 520), scale=0.6)

            # B. Vẽ các biểu đồ (Sử dụng hàm draw_chart trong ui_overlay)
            chart_h = 100
            chart_w = 400
            col1_x = 120  # Cột Mắt
            col2_x = 540  # Cột Đầu
            
            # --- CỘT 1: 4 ĐẶC TRƯNG MẮT (Feat 0-3) ---
            for i in range(4):
                y = 10 + i * (chart_h + 20)
                ui.draw_chart(dashboard, feat_history[i], col1_x, y, chart_w, chart_h, 
                              global_min[i], global_max[i], f"{i}", COLORS_FEAT[i])

            # --- CỘT 2: GÓC QUAY (Feat 4-6) ---
            ui.draw_text(dashboard, "HEAD ROTATION (Do)", (col2_x, 20), scale=0.6)
            rot_labels = ["Yaw (Trai/Phai)", "Pitch (Len/Xuong)", "Roll (Nghieng)"]
            for i in range(3):
                idx = 4 + i
                y = 30 + i * (chart_h + 10)
                ui.draw_chart(dashboard, feat_history[idx], col2_x, y, chart_w, chart_h, 
                              global_min[idx], global_max[idx], f"{idx}: {rot_labels[i]}", COLORS_FEAT[idx])

            # --- CỘT 2: TỊNH TIẾN (Feat 7-9) ---
            trans_start_y = 30 + 3 * (chart_h + 10) + 20
            ui.draw_text(dashboard, "HEAD TRANSLATION", (col2_x, trans_start_y - 5), scale=0.6)
            trans_labels = ["TX", "TY", "TZ"]
            for i in range(3):
                idx = 7 + i
                y = trans_start_y + i * (chart_h + 10)
                ui.draw_chart(dashboard, feat_history[idx], col2_x, y, chart_w, chart_h, 
                              global_min[idx], global_max[idx], f"{idx}: {trans_labels[i]}", COLORS_FEAT[idx])

            # C. Hướng dẫn
            ui.draw_text(dashboard, "'r': Reset Min/Max | 'SPACE': Pause | 'q': Quit", (20, 780), color=(0,255,0))

            cv2.imshow(WINDOW_NAME, dashboard)

        # Xử lý phím
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        elif key == ord(' '):
            paused = not paused
        elif key == ord('r'):
            print("Resetting Min/Max...")
            global_min = [float('inf')] * 10
            global_max = [float('-inf')] * 10
            for q in feat_history: q.clear()

    cam.stop()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()