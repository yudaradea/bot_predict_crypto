import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math
from predictor import apply_all_strategies

def load_and_preprocess_data(csv_file, window_size=60):
    df = pd.read_csv(csv_file)
    
    if df.empty:
        raise ValueError("DataFrame kosong.")
    
    if 'timestamp' not in df.columns or 'price' not in df.columns:
        raise ValueError("Kolom 'timestamp' atau 'price' tidak ditemukan dalam file.")
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df.set_index('timestamp', inplace=True)
    
    # Terapkan semua strategi
    df = apply_all_strategies(df)
    
    # Pastikan data mencukupi untuk membuat window
    if len(df) <= window_size:
        raise ValueError("Jumlah data tidak mencukupi untuk membuat window.")
    
    # Normalisasi harga
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['scaled_price'] = scaler.fit_transform(df[['price']])
    
    # Buat data untuk pelatihan
    X, y = [], []
    for i in range(len(df) - window_size):
        X.append(df['scaled_price'].iloc[i:i + window_size].values)
        y.append(df['scaled_price'].iloc[i + window_size])
    
    # Konversi ke array NumPy
    X, y = np.array(X), np.array(y)
    
    if X.ndim == 2:
        X = X[:, :, np.newaxis]
    
    print(f"Dimensi data untuk {csv_file}: X={X.shape}, y={y.shape}")
    
    return X, y, scaler, df

def build_model(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def plot_predictions(y_true, y_pred, scaler, df, coin_id):
    plt.figure(figsize=(14, 7))
    plt.plot(df.index[len(df) - len(y_true):], scaler.inverse_transform(y_true.reshape(-1, 1)), label='Data Asli', color='blue')
    plt.plot(df.index[len(df) - len(y_pred):], scaler.inverse_transform(y_pred.reshape(-1, 1)), label='Prediksi', color='red')
    
    rmse = math.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    
    plt.xlabel('Tanggal')
    plt.ylabel('Harga')
    plt.title(f'Prediksi Harga untuk {coin_id}')
    plt.legend()
    
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}', 
             verticalalignment='top', horizontalalignment='left', 
             transform=plt.gca().transAxes, fontsize=12, bbox=dict(facecolor='white', alpha=0.5))
    
    plt.show()

def calculate_percentage_change(true_price, predicted_price):
    return (predicted_price - true_price) / true_price * 100

def main():
    coin_id = input("Masukkan ID Koin (e.g., 'bitcoin'): ").strip()
    
    if not os.path.exists('models'):
        os.makedirs('models')
    
    csv_file = f'data/{coin_id}_data.csv'
    if not os.path.isfile(csv_file):
        print(f"File {csv_file} tidak ditemukan. Lewati koin ini.")
        return
    
    try:
        X, y, scaler, df = load_and_preprocess_data(csv_file)
        
        if X.shape[0] == 0 or X.shape[1] == 0 or X.shape[2] == 0:
            print(f"Data kosong untuk {coin_id}. Lewati koin ini.")
            return
        
        model = build_model((X.shape[1], X.shape[2]))
        
        early_stopping = EarlyStopping(monitor='val_loss', patience=10)
        
        history = model.fit(X, y, epochs=50, batch_size=32, validation_split=0.1, callbacks=[early_stopping])
        
        model_file_path = f'models/{coin_id}_lstm_model.h5'
        scaler_file_path = f'models/{coin_id}_scaler.npy'
        
        model.save(model_file_path)
        np.save(scaler_file_path, scaler.scale_)
        
        print(f"Model dan scaler untuk {coin_id} disimpan sebagai {model_file_path} dan {scaler_file_path}")
        
        # Prediksi menggunakan model
        y_pred = model.predict(X)
        
        # Plot hasil prediksi
        plot_predictions(y, y_pred, scaler, df, coin_id)
        
        # Tampilkan persentase perubahan untuk data terakhir
        last_true_price = y[-1]
        last_predicted_price = y_pred[-1]
        percentage_change = calculate_percentage_change(last_true_price, last_predicted_price)
        
        print(f"Harga Sebenarnya Terakhir: ${scaler.inverse_transform([[last_true_price]])[0][0]:.2f}")
        print(f"Prediksi Harga Terakhir: ${scaler.inverse_transform([[last_predicted_price]])[0][0]:.2f}")
        print(f"Perubahan Persentase Terakhir: {percentage_change[0]:.2f}%")
    
    except ValueError as e:
        print(f"Error saat memproses {coin_id}: {e}")

if __name__ == "__main__":
    main()