import cv2
import time
import numpy as np


# --- Các hằng số về màu sắc (định dạng BGR) ---
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_BLUE = (255, 0, 0)

# --- Cài đặt Font ---
FONT = cv2.FONT_HERSHEY_SIMPLEX
FONT_SCALE_SMALL = 0.7
FONT_SCALE_LARGE = 1.0
TEXT_THICKNESS = 2

# --- Cài đặt Gaze Dot ---
GAZE_DOT_RADIUS = 7
GAZE_DOT_THICKNESS = -1 # -1 để tô đầy (filled)

# --- Cài đặt Calibration Dot ---
CALIB_OUTER_RADIUS = 15
CALIB_INNER_RADIUS = 3
CALIB_CIRCLE_THICKNESS = 2

# --- Điểm mốc quan trọng trên khuôn mặt ---
KEY_LANDMARK_INDICES = [
    1,   # 1. Chóp mũi
    152, # 2. Cằm
    33,  # 3. Khóe mắt trái
    263, # 4. Khóe mắt phải
    61,  # 5. Khóe miệng trái
    291,  # 6. Khóe miệng phải
    133,
    159,
    145,
    362,
    386,
    374,
    473,
    468
]


def draw_text(frame, text, position, color = COLOR_WHITE, scale = FONT_SCALE_SMALL):
    # Vẽ viền đen
    cv2.putText(frame, text, position, FONT, scale, COLOR_BLACK, TEXT_THICKNESS + 2, cv2.LINE_AA)
    # Vẽ văn bản chính
    cv2.putText(frame, text, position, FONT, scale, color, TEXT_THICKNESS, cv2.LINE_AA)



def draw_calibration_dot(frame, center_coords):
    if center_coords is None:
        return
    # Vòng tròn trắng bên ngoài
    cv2.circle(frame, center_coords, CALIB_OUTER_RADIUS, COLOR_WHITE, CALIB_CIRCLE_THICKNESS, cv2.LINE_AA)
    # Chấm đỏ bên trong (tô đầy)
    cv2.circle(frame, center_coords, CALIB_INNER_RADIUS, COLOR_RED, -1, cv2.LINE_AA)
    

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
    