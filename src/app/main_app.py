import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
import cv2
import time
import numpy as np
import tkinter as tk
from core.capture import Camera
from core.detection import FaceMeshDetector
from core.gaze_estimator import GazeEstimator
from core.filters import EMAFilter, OneEuroWrapper
from core.calibration import Calibrator
import app.ui_overlay as ui


def main():
    
    SCREEN_WIDTH, SCREEN_HEIGHT = 1280, 720
    
    cam = Camera(width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    detector = FaceMeshDetector()
    estimator = GazeEstimator()
    gaze_filter = OneEuroWrapper(min_cutoff = 0.1, beta = 1.0)
    calibrator = Calibrator(estimator, samples_per_point = 70, grid_size = 4)
    # Khởi động camera
    print("Đang khởi động camera...")
    cam.start()
    time.sleep(1.0) 

    WINDOW_NAME = "Eyes Tracker"
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("--- Khởi tạo hoàn tất ---")
    print("Nhấn 'c' để bắt đầu hiệu chỉnh.")
    print("Nhấn 'Esc' để thoát.")
    

    while True:
        frame = cam.read()
        if frame is None:
            time.sleep(0.01)
            continue
        frame = cv2.flip(frame, 1, dst=frame)
        frame_height, frame_width = frame.shape[:2]
        
        # --- Detection ---
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = detector.process(frame_rgb)

        # --- Feature Extraction ---
        features = None
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0].landmark
            landmarks_array = detector.landmarks_to_array(face_landmarks, frame_width, frame_height)
            features = estimator.extract_features(landmarks_array)
            
        # --- Calibration ---
        if calibrator.is_calibrating:
            if features is not None:
                # Cập nhật calibrator với các đặc trưng mắt hiện tại
                calibrator.update(features)
        
            target_coords_norm = calibrator.get_current_target_norm_coords()  
            if target_coords_norm:
                norm_x, norm_y = target_coords_norm
                pixel_x = int(norm_x * frame_width)
                pixel_y = int(norm_y * frame_height) 
                ui.draw_calibration_dot(frame, (pixel_x, pixel_y))
        elif calibrator.is_finished:
            if features is not None:
                # Dự đoán tọa độ chuẩn hóa (0.0 -> 1.0)
                raw_gaze_norm = estimator.predict(features) # (x_norm, y_norm)
                
                if raw_gaze_norm:
                    gaze_norm_filtered = gaze_filter.update(raw_gaze_norm)
                    # Chuyển đổi tọa độ đã lọc sang pixel
                    gaze_px_x = int(gaze_norm_filtered[0] * frame_width)
                    gaze_px_y = int(gaze_norm_filtered[1] * frame_height)
                    ui.draw_gaze_dot(frame, (gaze_px_x, gaze_px_y))
        
        status_text = calibrator.get_progress_text()
        ui.draw_text(frame, status_text, (50, 70), color=ui.COLOR_WHITE)
        
        # --- Hiện thị ---
        cv2.imshow(WINDOW_NAME, frame) 

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
        if key == ord('c'):
            calibrator.start()
        if key == ord(' '):
            calibrator.trigger_collection()

    cam.stop()
    cv2.destroyAllWindows()
        



if __name__ == '__main__':
    main()