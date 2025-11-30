import cv2
import time
import numpy as np


# --- Các hằng số về màu sắc (định dạng BGR) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

COLOR_GRID = (50, 50, 50)
COLOR_CHART_LINE = (0, 255, 0)

# --- Cài đặt Font ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.7
FONT_SCALE_LARGE = 1.0
TEXT_THICKNESS = 1

# --- Cài đặt Gaze Dot ---
GAZE_DOT_RADIUS = 7
GAZE_DOT_THICKNESS = -1 # -1 để tô đầy (filled)

# --- Cài đặt Calibration Dot ---
CALIB_OUTER_RADIUS = 15
CALIB_INNER_RADIUS = 3
CALIB_CIRCLE_THICKNESS = 2

# --- Điểm mốc quan trọng trên khuôn mặt ---
KEY_LANDMARK_INDICES = [
    1,
    33,
    263,
    133,
    159,
    145,
    362,
    386,
    374,
    473,
    468,
    152,
    61,
    291
]


def draw_text(frame, text, position, color = COLOR_WHITE, scale = FONT_SCALE_SMALL):
    # Vẽ viền đen
    cv2.putText(frame, text, position, FONT, scale, COLOR_BLACK, TEXT_THICKNESS + 2, cv2.LINE_AA)
    # Vẽ văn bản chính
    cv2.putText(frame, text, position, FONT, scale, color, TEXT_THICKNESS, cv2.LINE_AA)



def draw_calibration_dot(frame, center_coords, progress_text=None):
    if center_coords is None:
        return
    # Vòng tròn trắng bên ngoài
    cv2.circle(frame, center_coords, CALIB_OUTER_RADIUS, COLOR_WHITE, CALIB_CIRCLE_THICKNESS, cv2.LINE_AA)
    # Chấm đỏ bên trong (tô đầy)
    cv2.circle(frame, center_coords, CALIB_INNER_RADIUS, COLOR_RED, -1, cv2.LINE_AA)
    
    if progress_text:
        font_scale = FONT_SCALE_SMALL
        (text_width, text_height), baseline = cv2.getTextSize(progress_text, FONT, font_scale, TEXT_THICKNESS)
        text_x = center_coords[0] - (text_width // 2)
        text_y = center_coords[1] + CALIB_OUTER_RADIUS + text_height + 10
        
        frame_height = frame.shape[0]
        if text_y > frame_height - 10:
            text_y = center_coords[1] - CALIB_OUTER_RADIUS - 10
        draw_text(frame, progress_text, (text_x, text_y), color=COLOR_WHITE, scale=font_scale)
    

def draw_gaze_dot(frame, center_coords):
    if center_coords is None:
        return
    # Vẽ vòng tròn xanh lá cây (tô đầy)
    cv2.circle(frame, center_coords, GAZE_DOT_RADIUS, COLOR_GREEN, GAZE_DOT_THICKNESS, cv2.LINE_AA)


def draw_key_landmarks(frame, landmarks_array):
    for idx in KEY_LANDMARK_INDICES:
        x = int(landmarks_array[idx][0])
        y = int(landmarks_array[idx][1])
        cv2.circle(frame, (x, y), 3, COLOR_GREEN, -1, cv2.LINE_AA)
    

def draw_chart(frame, history, x, y, w, h, min_val, max_val, label, color=COLOR_GREEN):
    """
    Vẽ biểu đồ dòng lên frame.
    - history: danh sách các giá trị (deque)
    - x, y, w, h: vị trí và kích thước biểu đồ
    - min_val, max_val: giá trị nhỏ nhất/lớn nhất toàn cục
    """
    # 1. Vẽ khung nền
    cv2.rectangle(frame, (x, y), (x + w, y + h), COLOR_GRID, 1)
    
    # 2. Vẽ thông tin (Label, Min, Max) bằng hàm draw_text có sẵn
    # Lấy giá trị hiện tại (nếu có)
    cur_val = history[-1] if len(history) > 0 else 0.0
    
    info_text = f"{label}: Cur={cur_val:.2f} | Min={min_val:.2f} | Max={max_val:.2f}"
    # Dùng hàm draw_text của bạn để chữ rõ hơn
    draw_text(frame, info_text, (x + 5, y + 90), color, scale=0.45)
    if len(history) < 2:
        return

    # 3. Tính toán tỉ lệ để vẽ đường dây
    # Tránh chia cho 0 nếu max == min
    if max_val == float('-inf') or min_val == float('inf') or max_val == min_val:
        range_val = 1.0
        display_min = min_val if min_val != float('inf') else 0
    else:
        range_val = max_val - min_val
        display_min = min_val
        
    # Padding trên dưới để đường không chạm sát mép
    padding = 5
    plot_h = h - (padding * 2)
    
    points = []
    for i, val in enumerate(history):
        # Chuẩn hóa giá trị về 0.0 -> 1.0
        normalized_y = (val - display_min) / range_val
        
        # Chuyển sang tọa độ pixel
        # 1.0 - normalized_y để đảo ngược (giá trị cao ở trên)
        plot_y = int(y + padding + ((1.0 - normalized_y) * plot_h))
        
        # Tọa độ X chia đều theo chiều rộng
        plot_x = int(x + (i / len(history)) * w)
        
        points.append((plot_x, plot_y))

    # 4. Vẽ đường nối các điểm
    if points:
        pts = np.array(points, np.int32)
        pts = pts.reshape((-1, 1, 2))
        cv2.polylines(frame, [pts], isClosed=False, color=color, thickness=1)