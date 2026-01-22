import pandas as pd
import os

# 1. Đường dẫn đến các file
file1_path = 'data/data_nightlight_right.csv'
file2_path = 'data/training_master.csv'
output_path = 'data/training_master.csv'

# 2. Đọc dữ liệu
print("Đang đọc file...")
try:
    df1 = pd.read_csv(file1_path)
    df2 = pd.read_csv(file2_path)
    
    # 3. Gộp dữ liệu (Vertical Concatenation)
    master_df = pd.concat([df1, df2], ignore_index=True)
    
    print(f"File 1 có: {len(df1)} dòng")
    print(f"File 2 có: {len(df2)} dòng")
    print(f"Tổng cộng: {len(master_df)} dòng")

    # 4. Lưu ra file mới
    master_df.to_csv(output_path, index=False)
    print(f"Đã lưu file gộp tại: {output_path}")

except FileNotFoundError as e:
    print(f"Lỗi: Không tìm thấy file. {e}")