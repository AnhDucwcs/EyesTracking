import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("data/data_nightlight.csv")
eye_cols = ['feat_0', 'feat_1', 'feat_2', 'feat_3']
head_cols = ['feat_4', 'feat_5', 'feat_6', 'feat_7', 'feat_8', 'feat_9']

print(f"Dữ liệu ban đầu có {len(df)} hàng\n")


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


# df = df.drop(index=[])
# df = df[(df['feat_9'] > 0) & (df['feat_9'] < 6000)]
# df = df[(df['feat_7'] > -10000) & (df['feat_7'] < 10000)]
# df = df[(df['feat_8'] > -10000) & (df['feat_8'] < 10000)]

# df = df[(df['feat_0'] < 3.2) & (df['feat_0'] > 2.7)]
# df = df[(df['feat_1'] < 1.0) & (df['feat_1'] > -0)]
# df = df[(df['feat_2'] < -1.8) & (df['feat_2'] > -2.2)]
# df = df[(df['feat_3'] < 1.0) & (df['feat_3'] > -0.1)]

# print(f"Dữ liệu sau làm sạch có {len(df)} hàng.")



print("\nĐang tạo biểu đồ Box Plot (hãy kiểm tra cửa sổ mới)...")
plt.figure(figsize=(15, 5))
sns.boxplot(data=df[head_cols])
plt.title("Box Plot")
plt.show()


df[eye_cols].plot(figsize=(15, 7), 
                         title="Biểu đồ 4 Đặc trưng Mắt (Tìm 'vực sâu' = nháy mắt)",
                         grid=True, subplots=True) # 'subplots=False' để vẽ chung 1 biểu đồ

plt.xlabel("Chỉ số hàng (Thời gian)")
plt.ylabel("Giá trị đặc trưng")
plt.show()

plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='target_x', y='target_y') 
plt.title("Mối quan hệ Giữa target_x và target_y")
plt.show()

# print(f"Dữ liệu sạch cuối cùng có {len(df)} hàng.")
# output_filename = 'data/data_cafe_0_cleaned1.csv'
# df.to_csv(output_filename, index=False)