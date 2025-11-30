import time
import pandas as pd
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score

DATA_FILE = 'data/training_master.csv'
MODEL_DIR = 'models'
SCALER_PATH = os.path.join(MODEL_DIR, 'scaler.pkl')

print(f"Đang tải dữ liệu sạch từ {DATA_FILE}...")
try:
    df = pd.read_csv(DATA_FILE)
except FileNotFoundError:
    print(f"Lỗi: Không tìm thấy file {DATA_FILE}.")
    print("Vui lòng chạy file inspect_data.py trước để tạo file này.")
    exit()
    
print(f"Đã tải {len(df)} hàng dữ liệu.")

X = df.iloc[:, :-3]
y = df.iloc[:, -3:-1]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"Kích thước bộ huấn luyện: {len(X_train)} mẫu")


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.values)
X_test_scaled = scaler.transform(X_test.values)
print("\nĐã chuẩn hóa đặc trưng.")

models_to_train = {
    "RandomForest": RandomForestRegressor(
        n_estimators=100, 
        max_depth=20, 
        random_state=42, 
        n_jobs=-1,
        min_samples_leaf=5
    ),
    "KNeighbors": KNeighborsRegressor(
        n_neighbors=9, 
        n_jobs=-1
    ),
    "MLP": MLPRegressor(
        hidden_layer_sizes=(256, 128, 64), 
        activation='relu',
        solver='adam',
        learning_rate_init=0.01,
        alpha=0.001,
        max_iter=1000, 
        random_state=42,
        early_stopping=True
    )
}
print("\n--- BẮT ĐẦU HUẤN LUYỆN & ĐÁNH GIÁ ---")

results = {}
best_model = None
best_model_name = ""
best_rmse = float('inf') # Đặt RMSE tốt nhất là vô cực

for name, model in models_to_train.items():
    print(f"\nĐang huấn luyện: {name}...")
    start_time = time.time()
    
    # Huấn luyện mô hình
    model.fit(X_train_scaled, y_train)
    
    train_time = time.time() - start_time
    print(f"Đã huấn luyện xong {name} (Thời gian: {train_time:.2f}s)")
    
    # Đánh giá trên bộ Test
    y_pred = model.predict(X_test_scaled)
    
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**0.5
    r2 = r2_score(y_test, y_pred)
    
    print(f"--- Kết quả của {name} ---")
    print(f"RMSE: {rmse:.6f} (Càng thấp càng tốt)")
    print(f"R2 Score: {r2:.4f} (Càng gần 1.0 càng tốt)")
    
    current_model_path = os.path.join(MODEL_DIR, f'model_{name}.pkl')
    joblib.dump(model, current_model_path)
    print(f"ĐÃ LƯU: {current_model_path}")
    
    # Lưu kết quả
    results[name] = {'rmse': rmse, 'r2': r2, 'model': model}
    
    # Kiểm tra xem đây có phải mô hình tốt nhất không
    if rmse < best_rmse:
        best_rmse = rmse
        best_model = model
        best_model_name = name
        
print("\n\n--- KẾT QUẢ CUỐI CÙNG ---")
print(f"Mô hình tốt nhất (RMSE thấp nhất) là: {best_model_name}")
print(f"Với RMSE = {best_rmse:.6f} và R2 Score = {results[best_model_name]['r2']:.4f}")


print("\n-------------------------------------------")
joblib.dump(scaler, SCALER_PATH)
print(f"Đã lưu scaler vào: {SCALER_PATH}")
print("Quá trình hoàn tất.")

