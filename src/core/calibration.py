import numpy as np
from .gaze_estimator import GazeEstimator


class Calibrator:
    def __init__(self, estimator: GazeEstimator, samples_per_point=120, grid_size=3, point_margin=0.1):
        self.estimator = estimator
        self.samples_per_point = samples_per_point

        self.points_to_calibrate = self._generate_calibration_points(grid_size, point_margin)
        self.total_points = len(self.points_to_calibrate)
        print(f"Calibrator duoc khoi tao voi luoi {grid_size}x{grid_size} ({self.total_points} diem).")
        
        self.current_point_index = 0.
        self.current_sample_count = 0.
        self._is_calibrating = False
        self._is_finished = False
        self._is_waiting_for_trigger = True
        
        self.collected_features = []
        self.collected_targets_x = []
        self.collected_targets_y = []
        
    def _generate_calibration_points(self, grid_size, margin):
        points = []
        if grid_size == 1: # Trường hợp đặc biệt: chỉ một điểm ở giữa
            return [(0.5, 0.5)]
            
        # Tạo ra các giá trị tọa độ cách đều nhau, có tính đến lề
        # np.linspace tạo ra các điểm từ 'margin' đến '1.0 - margin'
        coords = np.linspace(margin, 1.0 - margin, grid_size)
        for y in coords:
            for x in coords:
                points.append((x, y))
        return points
    
    def start(self):
        print("Bắt đầu hiệu chỉnh...")
        self._is_calibrating = True
        self._is_finished = False
        self.current_point_index = 0
        self.current_sample_count = 0
        self._is_waiting_for_trigger = True
        
        # Xóa dữ liệu cũ
        self.collected_features.clear()
        self.collected_targets_x.clear()
        self.collected_targets_y.clear()
    
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
            self.current_sample_count = 0
    
    def stop(self):
        print("Hủy bỏ hiệu chỉnh.")
        self._is_calibrating = False
        
    def update(self, feature_vector):
        if not self._is_calibrating or self._is_finished:
            return
        if self._is_waiting_for_trigger:
            return
        target_coords = self.points_to_calibrate[self.current_point_index]
        self.collected_features.append(feature_vector)
        self.collected_targets_x.append(target_coords[0])
        self.collected_targets_y.append(target_coords[1])
        self.current_sample_count += 1
        
        if self.current_sample_count >= self.samples_per_point:
            self._move_to_next_point()
        
    def _finish_calibration(self):
        print("Thu thập dữ liệu hoàn tất. Đang huấn luyện mô hình...")
        self._is_calibrating = False
        self._is_finished = True

        try:
            self.estimator.fit(
                self.collected_features,
                self.collected_targets_x,
                self.collected_targets_y
            )
        except Exception as e:
            print(f"LỖI trong quá trình huấn luyện GazeEstimator: {e}")
            self._is_finished = False # Đánh dấu là chưa hoàn thành nếu huấn luyện lỗi

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
                        f"& Nhan [SPACE]...")
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