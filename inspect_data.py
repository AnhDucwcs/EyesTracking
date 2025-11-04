import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/collected_data.csv")
eye_cols = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
head_cols = ['feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9']


print("--- Dữ liệu thiếu (NaN) ---")
print(df.isnull().sum())
print("-" * 30)
rows_before = len(df)
df = df.dropna()
rows_after = len(df)
print(f"Đã xóa {rows_before - rows_after} hàng bị thiếu dữ liệu.")


print("\n--- Thống kê tóm tắt (Kiểm tra min/max) ---")
pd.set_option('display.float_format', '{:.2f}'.format)
print(df.describe())
print("-" * 30)
invalid_targets = df[
    (df['target_x'] < 0.0) | (df['target_x'] > 1.0) |
    (df['target_y'] < 0.0) | (df['target_y'] > 1.0)
]
print(f"Tìm thấy {len(invalid_targets)} hàng có target nằm ngoài phạm vi [0, 1].")
if len(invalid_targets) > 0:
    df = df[
        (df['target_x'] >= 0.0) & (df['target_x'] <= 1.0) &
        (df['target_y'] >= 0.0) & (df['target_y'] <= 1.0)
    ]
    print("Đã xóa các hàng target không hợp lệ.")


# print("\nĐang tạo biểu đồ Box Plot (hãy kiểm tra cửa sổ mới)...")
# plt.figure(figsize=(15, 5))
# sns.boxplot(data=df[head_cols])
# plt.title("Box Plot")
# plt.show()


# df[eye_cols].plot(figsize=(15, 7), 
#                          title="Biểu đồ 4 Đặc trưng Mắt (Tìm 'vực sâu' = nháy mắt)",
#                          grid=True, subplots=False) # 'subplots=False' để vẽ chung 1 biểu đồ

# plt.xlabel("Chỉ số hàng (Thời gian)")
# plt.ylabel("Giá trị đặc trưng")
# plt.show()

"""
Sau khi xem các biểu đồ, có thể thấy có nhiều điểm ở feat_9 âm, đây là đặc trưng biểu diễn giá trị tz aka khoảng cách, 
do đó giá trị này âm mang ý nghĩa giá trị này là một giá trị lỗi, ngoài ra còn thấy nhiễu ở feat_7 và feat_8.
    df = df[df['feat_9'] > 0]
    df = df[(df['feat_7'] > -1000) & (df['feat_7'] < 1000)]
    df = df[(df['feat_8'] > -1000) & (df['feat_8'] < 1000)]
Kiểm tra Biểu đồ 4 đặc trưng mắt (Tìm 'vực sâu' = nháy mắt): Ta thấy được hai giá trị cột 2654 và 2655 có giá trị thấp đột ngột,
có thể đây là lúc người dùng nháy mắt, ta có thể loại bỏ hai hàng này nếu muốn.
    df = df.drop(index=[2654, 2655, 2656])
"""
df = df.drop(index=[2654, 2655, 2656])
df = df[df['feat_9'] > 0]


df[eye_cols].plot(figsize=(15, 7), 
                         title="Biểu đồ 4 Đặc trưng Mắt (Tìm 'vực sâu' = nháy mắt)",
                         grid=True, subplots=False) # 'subplots=False' để vẽ chung 1 biểu đồ

plt.xlabel("Chỉ số hàng (Thời gian)")
plt.ylabel("Giá trị đặc trưng")
plt.show()

print(f"Dữ liệu sạch cuối cùng có {len(df)} hàng.")
output_filename = 'data/collected_data_cleaned.csv'
df.to_csv(output_filename, index=False)