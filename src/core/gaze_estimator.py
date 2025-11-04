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
        self.fitted = False


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
            (0.0, 0.0, 0.0),       # 1. Chóp mũi (Index 1)
            (0.0, -330.0, -65.0),  # 2. Cằm (Index 152)
            (-225.0, 170.0, -135.0),# 3. Khóe mắt trái (Index 33)
            (225.0, 170.0, -135.0), # 4. Khóe mắt phải (Index 263)
            (-150.0, -150.0, -125.0),# 5. Khóe miệng trái (Index 61)
            (150.0, -150.0, -125.0) # 6. Khóe miệng phải (Index 291)
        ], dtype=np.float64)

        # Các điểm 2D tương ứng từ MediaPipe (chỉ lấy x, y)
        image_points = np.array([
            lm_array[1][:2],    # 1. Chóp mũi
            lm_array[152][:2],  # 2. Cằm
            lm_array[33][:2],   # 3. Khóe mắt trái
            lm_array[263][:2],  # 4. Khóe mắt phải
            lm_array[61][:2],   # 5. Khóe miệng trái
            lm_array[291][:2]   # 6. Khóe miệng phải
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

        # Tính toán tư thế đầu
        (_, rvec, tvec) = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs)
        
        # Dàn phẳng rvec (3 giá trị) và tvec (3 giá trị)
        pose_features = np.concatenate((rvec.flatten(), tvec.flatten()))


        features = np.concatenate((eye_features, pose_features)).astype(np.float32)
        features = np.nan_to_num(features, nan=0.0) 
        
        return features


    def fit(self, X_feats, Yx, Yy):
        """
        Huấn luyện mô hình.
        X_feats: danh sách các mảng đặc trưng (đầu vào X)
        Yx: danh sách tọa độ x (đầu ra Y)
        Yy: danh sách tọa độ y (đầu ra Y)
        """
        X = np.array(X_feats)
        Y = np.column_stack((Yx, Yy))
        
        print(f"Bắt đầu huấn luyện MLPRegressor trên {len(X)} mẫu, {X.shape[1]} đặc trưng...")
        self.model.fit(X, Y)
        self.fitted = True
        print("Huấn luyện MLPRegressor hoàn tất.")


    def predict(self, feat):
        if not self.fitted:
            raise RuntimeError('GazeEstimator chưa được huấn luyện (fitted)')
            
        F = feat.reshape(1, -1)
        predicted_coords = self.model.predict(F)
        coords = predicted_coords[0]
        return (float(coords[0]), float(coords[1]))