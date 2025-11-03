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

