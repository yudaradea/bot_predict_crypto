import os
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
from keras.models import load_model
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')  # Atau 'Qt5Agg', 'Agg', 'MacOSX', dll., sesuai dengan sistem Anda

def fetch_coin_data(coin_id, days):
    file_path = f'data/{coin_id}_data.csv'
    if not os.path.exists(file_path):
        print(f"File data '{file_path}' tidak ditemukan.")
        return None
    
    df = pd.read_csv(file_path)
    
    try:
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    except:
        df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    df.set_index('timestamp', inplace=True)
    df = df.tail(days + 60)  # Ambil data lebih banyak untuk window pelatihan
    return df

def predict_price(model, df, n_steps=60, forecast_days=1):
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df[['price']])
    
    X = []
    for i in range(len(scaled_data) - n_steps):
        X.append(scaled_data[i:i+n_steps])
    
    X = np.array(X)
    predictions = []
    current_step = X[-1]  # Mulai dari data terakhir
    
    for _ in range(forecast_days):
        pred = model.predict(current_step[np.newaxis, :, :])
        predictions.append(pred[0, 0])
        current_step = np.append(current_step[1:], pred, axis=0)
    
    predictions = np.array(predictions)
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1))
    
    return predictions

def format_currency(value):
    # Pembulatan harga ke nilai bulat
    rounded_value = round(value)
    return "Rp {:,.0f}".format(rounded_value).replace(',', '.')

def main():
    coin_id = input("Masukkan ID Koin (e.g., 'bitcoin'): ").strip()
    days = int(input("Masukkan Jumlah Hari untuk Prediksi (e.g., 1 atau 30): ").strip())
    
    model_path = f'models/{coin_id}_lstm_model.h5'
    if not os.path.exists(model_path):
        print(f"Model file '{model_path}' tidak ditemukan.")
        return
    
    model = load_model(model_path)
    
    df = fetch_coin_data(coin_id, days)
    if df is None:
        return
    
    predictions = predict_price(model, df, forecast_days=days)
    
    last_price = df['price'].values[-1]
    predicted_prices = predictions.flatten()
    
    print(f"Harga Sekarang: {format_currency(last_price)}")
    
    for i, pred in enumerate(predicted_prices):
        date = df.index[-1] + pd.DateOffset(days=i + 1)
        change = pred - last_price
        percentage_change = (change / last_price) * 100
        last_price = pred  # Update last_price untuk hari berikutnya
        print(f"Tanggal: {date.date()} - Prediksi Harga: {format_currency(pred)} - Perubahan: {format_currency(change)} ({percentage_change:.2f}%)")

if __name__ == "__main__":
    main()
