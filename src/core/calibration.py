import numpy as np
from sklearn.neighbors import KNeighborsRegressor
from .utils import generate_calibration_points


class Calibrator:
    def __init__(self, base_model, base_scaler, samples_per_point=50, grid_size=3):
        self.samples_per_point = samples_per_point
        self.base_model = base_model
        self.base_scaler = base_scaler
        self.error_model = KNeighborsRegressor(n_neighbors=7, weights='distance')

        self.points_to_calibrate = generate_calibration_points(grid_size, 0.1)
        self.total_points = len(self.points_to_calibrate)
        print(f"Calibrator duoc khoi tao voi luoi {grid_size}x{grid_size} ({self.total_points} diem).")
        
        self.current_point_index = 0
        self._is_calibrating = False
        self._is_finished = False
        self._is_waiting_for_trigger = True
        
        self.collected_base_preds = []
        self.collected_errors = []
        self.samples_collected = 0
        
    
    
    def start(self):
        print("Bắt đầu hiệu chỉnh...")
        self._is_calibrating = True
        self._is_finished = False
        self.current_point_index = 0
        self.samples_collected = 0

        self._is_waiting_for_trigger = True
        # Xóa dữ liệu cũ
        self.collected_base_preds.clear()
        self.collected_errors.clear()
    
    def trigger_collection(self):
        if self._is_calibrating and self._is_waiting_for_trigger:
            self._is_waiting_for_trigger = False
            print("Bắt đầu thu thập dữ liệu cho điểm hiệu chỉnh hiện tại.")
            
    def _move_to_next_point(self):
        self.current_point_index += 1
        
        if self.current_point_index >= self.total_points:
            self._finish_calibration()
        else:
            print(f"Chuyển sang điểm {self.current_point_index + 1}...")
            self._is_waiting_for_trigger = True
            self.samples_collected = 0
    
    def stop(self):
        print("Hủy bỏ hiệu chỉnh.")
        self._is_calibrating = False
        
    def collect_sample(self, features):
        if not self._is_calibrating or self._is_waiting_for_trigger:
            return
        
        target_norm = self.points_to_calibrate[self.current_point_index]
        features_scaled = self.base_scaler.transform(features.reshape(1, -1))
        predicted_norm = self.base_model.predict(features_scaled)[0]
        
        error = np.array(target_norm) - np.array(predicted_norm)
        
        self.collected_base_preds.append(predicted_norm)
        self.collected_errors.append(error)
        self.samples_collected += 1
        
        if self.samples_collected >= self.samples_per_point:
            self._move_to_next_point()
        
    def _finish_calibration(self):
        print("Thu thập dữ liệu hoàn tất. Đang huấn luyện mô hình...")
        self._is_calibrating = False
        self._is_finished = True

        if len(self.collected_base_preds) > 0:
            self.error_model.fit(self.collected_base_preds, self.collected_errors)
            self._is_finished = True
            print("[Calibrator] Error model trained. Calibration complete.")
        else:
            print("[Calibrator] Lỗi: Không có dữ liệu để huấn luyện.")
            
    def get_estimated_gaze(self, features):
        if not self._is_finished:
            raise RuntimeError('Calibrator chưa hoàn tất hiệu chỉnh')
        
        features_scaled = self.base_scaler.transform(features.reshape(1, -1))
        base_prediction = self.base_model.predict(features_scaled)
        error_correction = self.error_model.predict(base_prediction)
        
        corrected_gaze = base_prediction + error_correction
        
        return corrected_gaze[0]

    def get_current_target_norm_coords(self):
        if not self._is_calibrating:
            return None
        
        return self.points_to_calibrate[self.current_point_index]

    def get_progress_text(self):
        """Lấy văn bản trạng thái để hiển thị cho người dùng."""
        if self._is_finished:
            return "Hieu chinh Hoan tat! 'C' de hieu chinh. 'Esc' de thoat."
        if self._is_calibrating:
            if self._is_waiting_for_trigger:
                return (f"Nhin vao diem ({self.current_point_index + 1}/{self.total_points}) "
                        f"& Press SPACE to collect [{self.samples_collected}/{self.samples_per_point}]")
        return "Nhan 'C' de bat dau hieu chinh."



    # --- Thuộc tính (Properties) để truy cập an toàn ---
    @property
    def is_calibrating(self):
        return self._is_calibrating

    @property
    def is_finished(self):
        return self._is_finished
    
    @property
    def is_waiting_for_trigger(self):
        return self._is_waiting_for_trigger
    
    @property
    def collection_progress_text(self):
        return f"[{self.samples_collected}/{self.samples_per_point}]"