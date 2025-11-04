import sys, os
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from capture import Camera
from detection import FaceMeshDetector
from gaze_estimator import GazeEstimator
from utils import generate_calibration_points, get_screen_dimensions
import app.ui_overlay as ui
import numpy as np
import cv2
import csv
import time
import random

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.dirname(SCRIPT_DIR)
ROOT_DIR = os.path.dirname(SRC_DIR)
DATA_DIR = os.path.join(ROOT_DIR, 'data')
OUTPUT_CSV = os.path.join(DATA_DIR, 'collected_data.csv')

os.makedirs(DATA_DIR, exist_ok=True)

CALIB_GRID_SIZE = 5
CALIB_POINT_MARGIN = 0.1



def update_natural_target(target_pos, target_vel, frame_width, frame_height):
    """Di chuyển mục tiêu và làm nó "nảy" (bounce) trên tường."""
    x, y = target_pos
    vx, vy = target_vel
    x += vx
    y += vy
    if x <= 0 or x >= frame_width: vx = -vx
    if y <= 0 or y >= frame_height: vy = -vy
    return (x, y), (vx, vy)


def main():
    cam = Camera()
    SCREEN_WIDTH, SCREEN_HEIGHT = get_screen_dimensions()
    FRAME_WIDTH, FRAME_HEIGHT = cam.width, cam.height
    print(f"Kích thước Frame: {FRAME_WIDTH}x{FRAME_HEIGHT}")
    print("Kích thước màn hình: ", SCREEN_WIDTH, "x", SCREEN_HEIGHT)
    cam.start()
    time.sleep(1.0)
    detector = FaceMeshDetector()
    estimator = GazeEstimator()
    
    WINDOW_NAME = 'Data Collector'
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
    
    
    normalized_points = generate_calibration_points(CALIB_GRID_SIZE, CALIB_POINT_MARGIN)
    calibration_points_pixel = []
    for nx, ny in normalized_points:
        px = int(nx * SCREEN_WIDTH)  # <-- Nhân với SCREEN_WIDTH
        py = int(ny * SCREEN_HEIGHT) # <-- Nhân với SCREEN_HEIGHT
        calibration_points_pixel.append((px, py))
    num_points = len(calibration_points_pixel)
    compensation_target = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    natural_target = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    natural_velocity = (random.choice([-3, 3]), random.choice([-3, 3]))
    
    current_scenario = 0  # 0=Idle, 1=Calib, 2=Comp, 3=Natural
    calib_point_index = 0
    sc2_is_recording = False
    sc3_is_recording = False
    
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = []
        headers.extend("feat_"+str(i) for i in range(10))
        headers.extend(['target_x', 'target_y'])
        csv_writer.writerow(headers)
        
        print(f"Bắt đầu... Ghi dữ liệu vào {OUTPUT_CSV}")
        print("Nhấn '1', '2', '3' để bắt đầu. Nhấn '0' để tạm dừng. Nhấn 'Q' để thoát.")
        
        while True:
            frame = cam.read()
            if frame is None:
                time.sleep(0.01)
                continue
            frame = cv2.flip(frame, 1, dst=frame)
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ui_frame = frame.copy()
            canvas = np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH, 3), dtype=np.uint8)
            
            
            target_pos = None
            data_to_write = None
            key = cv2.waitKey(1) & 0xFF
            
            results = detector.process(frame_rgb)
            features = None
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                landmarks_array = detector.landmarks_to_array(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
                features = estimator.extract_features(landmarks_array, FRAME_WIDTH, FRAME_HEIGHT)
                ui.draw_key_landmarks(ui_frame, landmarks_array)
            canvas[SCREEN_HEIGHT - FRAME_HEIGHT : SCREEN_HEIGHT, 
               SCREEN_WIDTH - FRAME_WIDTH : SCREEN_WIDTH] = ui_frame
            
            
            if current_scenario == 1:
                target_pos = calibration_points_pixel[calib_point_index]
                if key == ord(' '):  # Phím cách
                    label_x_norm = target_pos[0] / SCREEN_WIDTH
                    label_y_norm = target_pos[1] / SCREEN_HEIGHT
                    data_to_write = np.concatenate((features, [label_x_norm, label_y_norm]))
                    print(f"Ghi điểm {calib_point_index + 1}/{num_points} tại {target_pos}")
                elif key == ord('n'):
                    calib_point_index = (calib_point_index + 1) % num_points
                    print(f"--- Chuyển sang điểm {calib_point_index + 1} ---")
            elif current_scenario == 2:
                target_pos = compensation_target
                if key == ord(' '):
                    sc2_is_recording = not sc2_is_recording
                    if sc2_is_recording:
                        print("--- Kịch bản 2: Bắt đầu ghi ---")
                    else:
                        print("--- Kịch bản 2: TẠM DỪNG ghi ---")
                if sc2_is_recording:
                    label_x_norm = target_pos[0] / SCREEN_WIDTH
                    label_y_norm = target_pos[1] / SCREEN_HEIGHT
                    data_to_write = np.concatenate((features, [label_x_norm, label_y_norm]))
            elif current_scenario == 3:
                if key == ord(' '):
                    sc3_is_recording = not sc3_is_recording # Đảo ngược trạng thái
                    
                    if sc3_is_recording:
                        print("--- Kịch bản 3: BẮT ĐẦU ghi ---")
                    else:
                        print("--- Kịch bản 3: TẠM DỪNG ghi ---")
                if sc3_is_recording:
                    natural_target, natural_velocity = update_natural_target(
                        natural_target, natural_velocity, SCREEN_WIDTH, SCREEN_HEIGHT
                    )
                    target_pos = (int(natural_target[0]), int(natural_target[1]))
                    label_x_norm = target_pos[0] / SCREEN_WIDTH
                    label_y_norm = target_pos[1] / SCREEN_HEIGHT
                    data_to_write = np.concatenate((features, [label_x_norm, label_y_norm]))
                else:
                    # Nếu đang tạm dừng, mục tiêu sẽ đứng yên
                    target_pos = (int(natural_target[0]), int(natural_target[1]))
                
                
            if data_to_write is not None:
                csv_writer.writerow(data_to_write)
            
            
            if current_scenario == 0:
                ui.draw_text(canvas, "IDLE. Press 1, 2, or 3. Press Q to quit.", (10, FRAME_HEIGHT - 20))
            elif current_scenario == 1:
                ui.draw_text(canvas, f"SCENARIO 1: Look at target {calib_point_index+1}/{num_points}. Press SPACE to record.", (10, SCREEN_HEIGHT - 60))
                ui.draw_text(canvas, "Press 'n' to move to the next point.", (10, SCREEN_HEIGHT - 20), color=ui.COLOR_GREEN)
            elif current_scenario == 2:
                if not sc2_is_recording:
                    ui.draw_text(canvas, "SCENARIO 2: Stare at target. Press SPACE to start/stop recording.", (10, SCREEN_HEIGHT - 60))
                else:
                    ui.draw_text(canvas, "SCENARIO 2: RECORDING... Move head (L/R, U/D, Fwd/Back).", (10, SCREEN_HEIGHT - 60), color=ui.COLOR_GREEN)
            elif current_scenario == 3:
                if not sc3_is_recording:
                    ui.draw_text(canvas, "SCENARIO 3: Follow the target. Press SPACE to start/stop recording.", (10, SCREEN_HEIGHT - 60))
                else:
                    ui.draw_text(canvas, "SCENARIO 3: RECORDING... Follow the target.", (10, SCREEN_HEIGHT - 60), color=ui.COLOR_GREEN)
                
            if target_pos:
                ui.draw_calibration_dot(canvas, target_pos)

            ui.draw_text(canvas, "Press '0' to pause", (10, 30))
            cv2.imshow("Data Collector", canvas)
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                if current_scenario != 1:
                    current_scenario = 1
                    calib_point_index = 0
                    sc2_is_recording = False 
                    sc3_is_recording = False
            elif key == ord('2'):
                if current_scenario != 2:
                    current_scenario = 2
                    sc2_is_recording = False 
                    sc3_is_recording = False
            elif key == ord('3'):
                if current_scenario != 3:
                    current_scenario = 3
                    sc2_is_recording = False 
                    sc3_is_recording = False
            elif key == ord('0'):
                if current_scenario != 0:
                    current_scenario = 0
                    sc2_is_recording = False 
                    sc3_is_recording = False
        
        cam.stop()
        cv2.destroyAllWindows()
        print(f"Đã thu thập xong. Dữ liệu được lưu tại '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()