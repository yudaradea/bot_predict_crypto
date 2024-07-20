import pandas as pd
import numpy as np

def macd(price, short_window=12, long_window=26, signal_window=9):
    short_ema = price.ewm(span=short_window, adjust=False).mean()
    long_ema = price.ewm(span=long_window, adjust=False).mean()
    macd_line = short_ema - long_ema
    macd_signal = macd_line.ewm(span=signal_window, adjust=False).mean()
    return macd_line, macd_signal

def add_rsi(df, period=14):
    delta = df['price'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

def add_bollinger_bands(df, window=20):
    rolling_mean = df['price'].rolling(window=window).mean()
    rolling_std = df['price'].rolling(window=window).std()
    df['bollinger_upper'] = rolling_mean + (rolling_std * 2)
    df['bollinger_lower'] = rolling_mean - (rolling_std * 2)
    return df

def add_volume(df):
    df['volume'] = np.random.randint(100, 1000, size=len(df))  # Gantilah dengan data volume asli jika tersedia
    return df

def add_lagged_features(df, lags=[1, 2, 3]):
    for lag in lags:
        df[f'price_lag_{lag}'] = df['price'].shift(lag)
    return df

def apply_macd_strategy(df):
    df['MACD'], df['MACD_signal'] = macd(df['price'])
    df['Signal'] = np.where(df['MACD'] > df['MACD_signal'], 1, 0)
    df['MACD_divergence'] = df['price'] - df['MACD']
    return df

def apply_rsi_strategy(df):
    df = add_rsi(df)
    df['RSI_signal'] = np.where(df['RSI'] < 30, 1, np.where(df['RSI'] > 70, -1, 0))
    return df

def apply_bollinger_bands_strategy(df):
    df = add_bollinger_bands(df)
    df['Bollinger_signal'] = np.where(df['price'] > df['bollinger_upper'], -1,
                                      np.where(df['price'] < df['bollinger_lower'], 1, 0))
    return df

def apply_volume_strategy(df):
    df = add_volume(df)
    df['Volume_signal'] = np.where(df['volume'] > df['volume'].rolling(window=20).mean(), 1, 0)
    return df

def apply_all_strategies(df):
    df = apply_macd_strategy(df)
    df = apply_rsi_strategy(df)
    df = apply_bollinger_bands_strategy(df)
    df = apply_volume_strategy(df)
    df = add_lagged_features(df)
    return df
