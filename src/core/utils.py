import numpy as np


def calculate_horizontal_ratio(iris_coords, left_coords, right_coords):
    """
    Tính H_ratio từ các mảng tọa độ [x, y].
    Giá trị trả về:
    ~0.0: Liếc hết sang trái
    ~0.5: Nhìn thẳng
    ~1.0: Liếc hết sang phải
    """
    try:
        # Lấy tọa độ x từ chỉ số [0]
        iris_x = iris_coords[0]
        left_x = left_coords[0]
        right_x = right_coords[0]

        eye_width = right_x - left_x
        if eye_width == 0:
            return 0.5

        relative_iris_pos = iris_x - left_x
        ratio = relative_iris_pos / eye_width #Chuẩn hóa về [0,1]
        return ratio

    except Exception as e:
        return 0.5

def calculate_vertical_ratio(iris_coords, top_coords, bottom_coords):
    """
    Tính V_ratio từ các mảng tọa độ [x, y].
    Giá trị trả về:
    ~0.0: Liếc hết lên trên
    ~0.5: Nhìn thẳng
    ~1.0: Liếc hết xuống dưới
    """
    try:
        # Lấy tọa độ y từ chỉ số [1]
        iris_y = iris_coords[1]
        top_y = top_coords[1]
        bottom_y = bottom_coords[1]

        eye_height = bottom_y - top_y
        if eye_height == 0:
            return 0.5

        relative_iris_pos = iris_y - top_y
        ratio = relative_iris_pos / eye_height
        return ratio

    except Exception as e:
        return 0.5

def generate_calibration_points(grid_size, margin):
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
    
import tkinter

def get_screen_dimensions():
    """Lấy kích thước màn hình (width, height) bằng tkinter."""
    try:
        root = tkinter.Tk()
        root.withdraw() # Ẩn cửa sổ tkinter
        width = root.winfo_screenwidth()
        height = root.winfo_screenheight()
        root.destroy()
        return width, height
    except Exception as e:
        print(f"Lỗi khi lấy kích thước màn hình: {e}. Dùng tạm 1920x1080.")
        return 1920, 1080