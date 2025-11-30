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
OUTPUT_CSV = os.path.join(DATA_DIR, 'data_nightlight.csv')

os.makedirs(DATA_DIR, exist_ok=True)

# Cấu hình ngưỡng để lọc dữ liệu bất thường
EYE_FEATURE_MIN = -2.0
EYE_FEATURE_MAX = 2.0
HEAD_Z_MIN = 350.0
HEAD_Z_MAX = 1500.0
HEAD_PITCH_MIN = -0.2
HEAD_PITCH_MAX = 0.1
HEAD_YAW_MIN = -0.85 
HEAD_YAW_MAX = 0.85 


SAMPLES_PER_POINT_SC1 = 50
CAPTURE_INTERVAL = 0.1  # giây giữa các lần chụp trong Scenario 2 và 3
JUMP_INTERVAL_SC3 = 4.0  # giây giữa các lần nhảy của mục tiêu trong Scenario 3


def is_data_sane(features):
    # Tách các đặc trưng từ mảng 10 phần tử
    eye_feats = features[0:4]   # 4 đặc trưng mắt
    yaw = features[4]           # Góc quay trái/phải
    pitch = features[5]         # Góc cúi/ngẩng
    # roll = features[6]        # Góc nghiêng (không lọc)
    tz = features[9]            # Khoảng cách Z

    # --- 1. KIỂM TRA MẮT (Lọc nhiễu "gai nhọn") ---
    indices_to_check = [1, 3] 
    for i in indices_to_check:
        val = eye_feats[i]
        if not (EYE_FEATURE_MIN < val < EYE_FEATURE_MAX):
            return False, f"Nhieu Mat Feat_{i} ({val:.2f})"

    # --- 2. KIỂM TRA KHOẢNG CÁCH (Trục Z) ---
    if not (HEAD_Z_MIN < tz < HEAD_Z_MAX):
        return False, f"Sai Khoang cach (tz={tz:.0f})"

    # # --- 3. KIỂM TRA PITCH (Cúi/Ngẩng) ---
    # if not (HEAD_PITCH_MIN < pitch < HEAD_PITCH_MAX):
    #     return False, f"Cui/Ngua qua muc ({pitch:.2f})"

    # # --- 4. KIỂM TRA YAW (Trái/Phải) ---
    # if not (HEAD_YAW_MIN < yaw < HEAD_YAW_MAX):
    #     return False, f"Xoay Trai/Phai qua muc ({yaw:.2f})"

    # Dữ liệu hợp lệ
    return True, ""

def update_natural_target(target_pos, target_vel, frame_width, frame_height, last_jump_time):
    current_time = time.time()
    did_jump = False
    if (current_time - last_jump_time) > JUMP_INTERVAL_SC3:
        x = random.randint(int(frame_width * 0.1), int(frame_width * 0.9))
        y = random.randint(int(frame_height * 0.1), int(frame_height * 0.9))
        vx = random.choice([-3, 3])
        vy = random.choice([-3, 3])
        last_jump_time = current_time
        did_jump = True
        return (x, y), (vx, vy), last_jump_time, did_jump
    
    x, y = target_pos
    vx, vy = target_vel
    x += vx
    y += vy
    if x <= 0 or x >= frame_width:
        vx = -vx
        x = max(0, min(x, frame_width)) 
    if y <= 0 or y >= frame_height:
        vy = -vy
        y = max(0, min(y, frame_height)) 
    return (x, y), (vx, vy), last_jump_time, did_jump


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
    
    try:
        ui.draw_text(dummy_canvas, "LOADING...", 
                  (SCREEN_WIDTH // 2 - 200, SCREEN_HEIGHT // 2), 
                  scale=1.5)
    except:
        pass # Bỏ qua nếu chưa import được hàm

    cv2.imshow(WINDOW_NAME, dummy_canvas)
    cv2.waitKey(1)
    
    
    normalized_points = generate_calibration_points(5, 0.1)
    calibration_points_pixel = []
    for nx, ny in normalized_points:
        px = int(nx * SCREEN_WIDTH)  # <-- Nhân với SCREEN_WIDTH
        py = int(ny * SCREEN_HEIGHT) # <-- Nhân với SCREEN_HEIGHT
        calibration_points_pixel.append((px, py))
    calib_num_points = len(calibration_points_pixel)
    
    compensation_norm_points = generate_calibration_points(3, 0.2)
    compensation_points_pixel = []
    for nx, ny in compensation_norm_points:
        px = int(nx * SCREEN_WIDTH)
        py = int(ny * SCREEN_HEIGHT)
        compensation_points_pixel.append((px, py))
    compensation_num_points = len(compensation_points_pixel)
    natural_target = (SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2)
    natural_velocity = (random.choice([-3, 3]), random.choice([-3, 3]))
    
    current_scenario = 0  # 0=Idle, 1=Calib, 2=Comp, 3=Natural
    calib_point_index = 0
    compensation_point_index = 0
    sc1_is_recording = False
    sc2_is_recording = False
    sc3_is_recording = False
    sc1_samples_collected = 0
    last_capture_time = 0.0
    last_jump_time = 0.0
    
    with open(OUTPUT_CSV, mode='w', newline='') as csvfile:
        csv_writer = csv.writer(csvfile)
        headers = []
        headers.extend("feat_"+str(i) for i in range(10))
        headers.extend(['target_x', 'target_y'])
        headers.append('scenario_id')
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
            
            if current_scenario == 1:
                target_pos = calibration_points_pixel[calib_point_index]
                if key == ord('n'):
                    calib_point_index = (calib_point_index + 1) % calib_num_points
                    sc1_is_recording = False
                    sc1_samples_collected = 0
                    print(f"--- Chuyển sang điểm {calib_point_index + 1} ---")
                elif key == ord(' '):
                    if not sc1_is_recording:
                        sc1_is_recording = True
                        sc1_samples_collected = 0
            elif current_scenario == 2:
                target_pos = compensation_points_pixel[compensation_point_index]
                if key == ord(' '):
                    sc2_is_recording = not sc2_is_recording
                    last_capture_time = time.time()
                if key == ord('n'):
                    compensation_point_index = (compensation_point_index + 1) % compensation_num_points
                    sc2_is_recording = False
            elif current_scenario == 3:
                if key == ord(' '):
                    sc3_is_recording = not sc3_is_recording
                    last_capture_time = time.time()
                    last_jump_time = time.time()
                if sc3_is_recording:
                    (natural_target, natural_velocity, last_jump_time, did_jump) = update_natural_target(
                                                                                            natural_target, natural_velocity, 
                                                                                            SCREEN_WIDTH, SCREEN_HEIGHT, 
                                                                                            last_jump_time
                                                                                        )
                    target_pos = (int(natural_target[0]), int(natural_target[1]))
                    if did_jump:
                        sc3_is_recording = False  
                target_pos = (int(natural_target[0]), int(natural_target[1]))
            
            results = detector.process(frame_rgb)
            features = None
            
            is_tracking_lost = True 
            tracking_error_msg = "No Face"
            
            if results.multi_face_landmarks:
                face_landmarks = results.multi_face_landmarks[0].landmark
                landmarks_array = detector.landmarks_to_array(face_landmarks, FRAME_WIDTH, FRAME_HEIGHT)
                features = estimator.extract_features(landmarks_array, FRAME_WIDTH, FRAME_HEIGHT)
                ui.draw_key_landmarks(ui_frame, landmarks_array)
            
                is_sane, error_msg = is_data_sane(features)
                if not is_sane:
                    is_tracking_lost = True
                    tracking_error_msg = error_msg
                else:
                    is_tracking_lost = False
                    tracking_error_msg = ""
                
                if not is_tracking_lost:    
                    if current_scenario == 1:
                        if sc1_samples_collected < SAMPLES_PER_POINT_SC1 and sc1_is_recording:
                                label_x_norm = target_pos[0] / SCREEN_WIDTH
                                label_y_norm = target_pos[1] / SCREEN_HEIGHT
                                data_to_write = np.concatenate((features, [label_x_norm, label_y_norm, 1]))
                                
                                sc1_samples_collected += 1
                                if sc1_samples_collected >= SAMPLES_PER_POINT_SC1:
                                    sc1_is_recording = False
                        else:
                            sc1_is_recording = False
                            
                    elif current_scenario == 2:
                        if sc2_is_recording:
                            current_time = time.time()
                            if (current_time - last_capture_time) > CAPTURE_INTERVAL:
                                last_capture_time = current_time
                                label_x_norm = target_pos[0] / SCREEN_WIDTH
                                label_y_norm = target_pos[1] / SCREEN_HEIGHT
                                data_to_write = np.concatenate((features, [label_x_norm, label_y_norm, 2]))
                            
                    elif current_scenario == 3:    
                        if sc3_is_recording and not did_jump:
                            current_time = time.time()
                            if (current_time - last_capture_time) > CAPTURE_INTERVAL:
                                last_capture_time = current_time 
                                label_x_norm = target_pos[0] / SCREEN_WIDTH
                                label_y_norm = target_pos[1] / SCREEN_HEIGHT
                                data_to_write = np.concatenate((features, [label_x_norm, label_y_norm, 3]))
                             
            if data_to_write is not None:
                csv_writer.writerow(data_to_write)
            
            
            canvas[SCREEN_HEIGHT - FRAME_HEIGHT : SCREEN_HEIGHT, 
               SCREEN_WIDTH - FRAME_WIDTH : SCREEN_WIDTH] = ui_frame
            
            progress_text_to_draw = None
            if current_scenario == 0:
                ui.draw_text(canvas, "IDLE. Press 1, 2, or 3. Press Q to quit.", (10, FRAME_HEIGHT - 20))
            elif current_scenario == 1:
                progress_text = f"[{sc1_samples_collected}/{SAMPLES_PER_POINT_SC1}]"
                progress_text_to_draw = progress_text
                ui.draw_text(canvas, f"SCENARIO 1: Look at target {calib_point_index+1}/{calib_num_points}. Press SPACE to record.", (10, SCREEN_HEIGHT - 60))
                ui.draw_text(canvas, "Press 'n' to move to the next point.", (10, SCREEN_HEIGHT - 20), color=ui.COLOR_GREEN)
            elif current_scenario == 2:
                if not sc2_is_recording:
                    ui.draw_text(canvas, "SCENARIO 2: Stare at target. Press SPACE to start/resume recording. Press 'n' to move to the next point.", (10, SCREEN_HEIGHT - 60))
                else:
                    ui.draw_text(canvas, "SCENARIO 2: RECORDING... Move head (L/R, U/D, Fwd/Back).", (10, SCREEN_HEIGHT - 60), color=ui.COLOR_GREEN)
            elif current_scenario == 3:
                if not sc3_is_recording:
                    ui.draw_text(canvas, "SCENARIO 3: Follow the target. Press SPACE to start/resume recording.", (10, SCREEN_HEIGHT - 60))
                else:
                    ui.draw_text(canvas, "SCENARIO 3: RECORDING... Follow the target.", (10, SCREEN_HEIGHT - 60), color=ui.COLOR_GREEN)
            
            if is_tracking_lost and features is not None: 
                # (features is not None để chỉ hiện khi có mặt nhưng bị lỗi pose/eye)
                ui.draw_text(canvas, "TRACKING LOST!", (SCREEN_WIDTH//2 - 150, 100), 
                             color=ui.COLOR_RED, scale=1.5)
                ui.draw_text(canvas, f"Ly do: {tracking_error_msg}", (SCREEN_WIDTH//2 - 200, 150), 
                             color=ui.COLOR_RED, scale=1.0)
               
            if target_pos:
                ui.draw_calibration_dot(canvas, target_pos, progress_text=progress_text_to_draw)

            ui.draw_text(canvas, "Press '0' to pause", (10, 30))
            cv2.imshow("Data Collector", canvas)
            
            if key == ord('q') or key == 27:
                break
            elif key == ord('1'):
                if current_scenario != 1:
                    current_scenario = 1
                    calib_point_index = 0
                    sc1_is_recording = False
                    sc2_is_recording = False 
                    sc3_is_recording = False
                    sc1_samples_collected = 0
            elif key == ord('2'):
                if current_scenario != 2:
                    current_scenario = 2
                    sc1_is_recording = False
                    sc2_is_recording = False 
                    sc3_is_recording = False
                    sc1_samples_collected = 0
            elif key == ord('3'):
                if current_scenario != 3:
                    current_scenario = 3
                    sc1_is_recording = False
                    sc2_is_recording = False 
                    sc3_is_recording = False
                    sc1_samples_collected = 0
            elif key == ord('0'):
                if current_scenario != 0:
                    current_scenario = 0
                    sc1_is_recording = False
                    sc2_is_recording = False 
                    sc3_is_recording = False  
                    sc1_samples_collected = 0

        cam.stop()
        cv2.destroyAllWindows()
        print(f"Đã thu thập xong. Dữ liệu được lưu tại '{OUTPUT_CSV}'")


if __name__ == "__main__":
    main()