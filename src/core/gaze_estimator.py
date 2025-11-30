import cv2
import numpy as np
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from core.utils import calculate_horizontal_ratio, calculate_vertical_ratio


class GazeEstimator:
    def __init__(self):
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('mlp', MLPRegressor(
                hidden_layer_sizes=(32, 16),
                max_iter=500,
                activation='relu',
                solver='adam', 
                random_state=42,
                early_stopping=True
            ))
        ])
        self.last_rvec = None
        self.last_tvec = None


    def extract_features(self, lm_array, frame_width, frame_height):
        # Mắt trái
        left_outer = lm_array[33][:2]
        left_inner = lm_array[133][:2]
        left_top   = lm_array[159][:2]
        left_bottom= lm_array[145][:2]
        iris_left_center = lm_array[473][:2]

        # Mắt phải
        right_outer = lm_array[263][:2]
        right_inner = lm_array[362][:2]
        right_top   = lm_array[386][:2]
        right_bottom= lm_array[374][:2]
        iris_right_center = lm_array[468][:2]

        h_ratio_left = calculate_horizontal_ratio(iris_left_center, left_outer, left_inner)
        h_ratio_right = calculate_horizontal_ratio(iris_right_center, right_inner, right_outer)
        v_ratio_left = calculate_vertical_ratio(iris_left_center, left_top, left_bottom)
        v_ratio_right = calculate_vertical_ratio(iris_right_center, right_top, right_bottom)
        
        eye_features = np.array([
                h_ratio_left, v_ratio_left,
                h_ratio_right, v_ratio_right
        ])
        
        
        
        # Các điểm 3D của một khuôn mặt "mẫu"
        model_points = np.array([
                            (0.0, 0.0, 0.0),           # 1. Nose tip
                            (-45.0, -30.0, 50.0),      # 2. Left eye left corner
                            (45.0, -30.0, 50.0),       # 3. Right eye right corne
                            (-30.0, 50.0, 40.0),       # 4. Left Mouth corner
                            (30.0, 50.0, 40.0),        # 5. Right mouth corner
                            (0.0, 120.0, 40.0),        # 6. Chin
                        ])

        # Các điểm 2D tương ứng từ MediaPipe (chỉ lấy x, y)
        image_points = np.array([
            lm_array[1][:2],    # 1. Chóp mũi (Index 1)
            lm_array[33][:2],   # 2. Mắt trái ngoài (Index 33)
            lm_array[263][:2],  # 3. Mắt phải ngoài (Index 263)
            lm_array[61][:2],   # 4. Mép miệng trái
            lm_array[291][:2],  # 5. Mép miệng phải
            lm_array[152][:2]   # 6. Cằm (Index 152)
        ], dtype=np.float64)

        # Ước tính Camera Matrix (giả định đơn giản)
        focal_length = frame_width
        center = (frame_width / 2, frame_height / 2)
        camera_matrix = np.array(
            [[focal_length, 0, center[0]],
             [0, focal_length, center[1]],
             [0, 0, 1]], dtype=np.float64
        )

        # Giả sử không có méo ống kính
        dist_coeffs = np.zeros((4, 1), dtype=np.float64)

        
        if self.last_rvec is not None:
            rvec = self.last_rvec
            tvec = self.last_tvec
            use_guess = True
        else:
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
            use_guess = False
            
        # Tính toán tư thế đầu
        try:
            success, rvec, tvec = cv2.solvePnP(
                model_points, 
                image_points, 
                camera_matrix, 
                dist_coeffs, 
                rvec=rvec,
                tvec=tvec,
                useExtrinsicGuess=use_guess,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if success:
                self.last_rvec = rvec
                self.last_tvec = tvec
            else:
                self.last_rvec = None
                self.last_tvec = None
                rvec = np.zeros((3, 1))
                tvec = np.zeros((3, 1))
                
        except Exception:
            self.last_rvec = None
            self.last_tvec = None
            rvec = np.zeros((3, 1))
            tvec = np.zeros((3, 1))
        

        pose_features = np.concatenate((rvec.flatten(), tvec.flatten()))


        features = np.concatenate((eye_features, pose_features)).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0) 
        
        return features