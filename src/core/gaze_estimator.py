import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from .utils import calculate_horizontal_ratio, calculate_vertical_ratio


class GazeEstimator:
    def __init__(self):
        self.model = KNeighborsRegressor(n_neighbors=3, weights='distance')
        self.fitted = False


    def extract_features(self, lm_array):
        # Trả về bộ đặc trưng gaze từ mảng điểm đặc trưng mặt
        if lm_array is None or len(lm_array) < 474:
            return np.zeros((4,), dtype=np.float32)

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


        feat = np.array([
            h_ratio_left,
            h_ratio_right,
            v_ratio_left,
            v_ratio_right,
        ], dtype=np.float32)
        return feat


    def fit(self, X_feats, Yx, Yy):
        """
        Huấn luyện mô hình.
        X_feats: danh sách các mảng đặc trưng (đầu vào X)
        Yx: danh sách tọa độ x (đầu ra Y)
        Yy: danh sách tọa độ y (đầu ra Y)
        """
        X = np.vstack(X_feats)
        Y = np.column_stack((Yx, Yy))
        
            
        # Huấn luyện mô hình duy nhất trên toàn bộ dữ liệu
        self.model.fit(X, Y)
        self.fitted = True
        print("Mô hình GazeEstimator đã được huấn luyện.")


    def predict(self, feat):
        """Dự đoán tọa độ (x, y) từ một vector đặc trưng."""
        if not self.fitted:
            raise RuntimeError('GazeEstimator chưa được huấn luyện (fitted)')
            
        F = feat.reshape(1, -1)

        # self.model.predict(F) sẽ trả về một mảng 2D: [[x, y]]
        predicted_coords = self.model.predict(F)
        
        # Lấy kết quả [x, y] từ phần tử đầu tiên
        coords = predicted_coords[0]
        x = float(coords[0])
        y = float(coords[1])
        
        return (x, y)