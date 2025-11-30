import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import time
import numpy as np
import joblib
import tkinter as tk
from core.capture import Camera
from core.detection import FaceMeshDetector
from core.gaze_estimator import GazeEstimator
from core.calibration import Calibrator
from core.filters import EMAFilter, OneEuroWrapper
from core.utils import get_screen_dimensions
import app.ui_overlay as ui



SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
# MODEL_TO_TEST = 'model_RandomForest.pkl'
# MODEL_TO_TEST = 'model_KNeighbors.pkl'
MODEL_TO_TEST = 'model_MLP.pkl'
MODEL_PATH = os.path.join(ROOT_DIR, 'models', MODEL_TO_TEST)
SCALER_PATH = os.path.join(ROOT_DIR, 'models', 'scaler.pkl')
print(f"--- ĐANG THỬ NGHIỆM VỚI MÔ HÌNH: {MODEL_TO_TEST} ---")


def main():
    SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_dimensions()
    cam = Camera()
    detector = FaceMeshDetector()
    estimator = GazeEstimator()
    gaze_filter = OneEuroWrapper(min_cutoff = 0.5, beta = 0.1)
    
    FRAME_WIDTH, FRAME_HEIGHT = cam.width, cam.height
    print(f"Kích thước Frame: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("Kích thước màn hình: ", SCREEN_WIDTH, "x", SCREEN_HEIGHT)
    
    
    print("Đang tải mô hình và scaler...")
    try:
        model = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        print("Tải mô hình thành công.")
    except FileNotFoundError:
        print(f"LỖI: Không tìm thấy file mô hình/scaler tại '{MODEL_PATH}'")
        print("Vui lòng chạy train_model.py trước.")
        return

    calibrator = Calibrator(model, scaler, samples_per_point=50, grid_size=3)

    # Khởi động camera
    print("Đang khởi động camera...")
    cam.start()
    time.sleep(1.0) 

    WINDOW_NAME = "Eyes Tracker"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.setWindowProperty(WINDOW_NAME, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    print("Đang vào chế độ toàn màn hình...")
    dummy_canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
    
    # (Tùy chọn) Sử dụng hàm draw_text của bạn để vẽ chữ "Đang tải"
    try:
        ui.draw_text(dummy_canvas, "LOADING...", 
                  (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2), 
                  scale=1.5)
    except:
        pass # Bỏ qua nếu chưa import được hàm

    cv2.imshow(WINDOW_NAME, dummy_canvas)
    cv2.waitKey(1)

    print("--- Khởi tạo hoàn tất ---")
    print("Nhấn 'c' để bắt đầu hiệu chỉnh.")
    print("Nhấn 'Esc' để thoát.")
    

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1, dst=frame)
        ui_frame = frame.copy()
        canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
        
        # --- Detection ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        # --- Feature Extraction ---
        features = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            landmarks_array = detector.landmarks_to_array(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
            features = estimator.extract_features(landmarks_array, FRAME_WIDTH, FRAME_HEIGHT)
            ui.draw_key_landmarks(ui_frame, landmarks_array)
            canvas[SCREEN_HEIGHT - FRAME_HEIGHT : SCREEN_HEIGHT, 
               SCREEN_WIDTH - FRAME_WIDTH : SCREEN_WIDTH] = ui_frame
        
        key = cv2.waitKey(1) & 0xFF   
        # --- Calibration ---
        if calibrator.is_calibrating:       
            target_coords_norm = calibrator.get_current_target_norm_coords()  
            if target_coords_norm:
                norm_x, norm_y = target_coords_norm
                pixel_x = int(norm_x * SCREEN_WIDTH)
                pixel_y = int(norm_y * SCREEN_HEIGHT)
                progress_text = None
                if not calibrator.is_waiting_for_trigger:
                    progress_text = calibrator.collection_progress_text
                ui.draw_calibration_dot(canvas, (pixel_x, pixel_y), progress_text=progress_text)
            if key == ord(' '):
                calibrator.trigger_collection()
            if features is not None:
                calibrator.collect_sample(features)
        elif calibrator.is_finished:
            if features is not None:
                # Dự đoán tọa độ chuẩn hóa (0.0 -> 1.0)
                raw_gaze_norm = calibrator.get_estimated_gaze(features)
                
                if raw_gaze_norm is not None:
                    gaze_norm_filtered = gaze_filter.update(raw_gaze_norm)
                    # Chuyển đổi tọa độ đã lọc sang pixel
                    gaze_px_x = int(gaze_norm_filtered[0] * SCREEN_WIDTH)
                    gaze_px_y = int(gaze_norm_filtered[1] * SCREEN_HEIGHT)
                    ui.draw_gaze_dot(canvas, (gaze_px_x, gaze_px_y))
        else:
            if key == ord('c'):
                calibrator.start()
        
        status_text = calibrator.get_progress_text()
        ui.draw_text(canvas, status_text, (50, 70), color=ui.COLOR_WHITE)
        
        # --- Hiện thị ---
        cv2.imshow(WINDOW_NAME, canvas) 

        
        if key == ord('q') or key == 27:
            break

    cam.stop()
    cv2.destroyAllWindows()
        



if __name__ == '__main__':
    main()