import requests
import pandas as pd
import os

def fetch_data_from_coingecko(coin_id, days):
    url = f"https://api.coingecko.com/api/v3/coins/{coin_id}/market_chart"
    params = {
        'vs_currency': 'idr',
        'days': days,
      
    }
    
    try:
        print(f"Fetching data for {coin_id} for the last {days} days...")
        response = requests.get(url, params=params)
        response.raise_for_status()  # Menangkap kesalahan HTTP
        
        data = response.json()
        
        if 'prices' not in data:
            print(f"Error: Data tidak ditemukan untuk {coin_id}.")
            return None
        
        prices = data['prices']
        if not prices:
            print(f"Error: Tidak ada data harga untuk {coin_id}.")
            return None
        
        df = pd.DataFrame(prices, columns=['timestamp', 'price'])
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df.set_index('timestamp', inplace=True)
        
        # Cek dan buat folder 'data' jika tidak ada
        if not os.path.exists('data'):
            os.makedirs('data')
        
        # Simpan data ke CSV
        filename = f'data/{coin_id}_data.csv'
        df.to_csv(filename)
        print(f"Data berhasil disimpan dalam file {filename}")
        
        return df
    
    except requests.HTTPError as e:
        print(f"HTTP Error: {e}")
        return None
    except requests.RequestException as e:
        print(f"Request Exception: {e}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

def main():
    coin_id = input("Masukkan ID Koin (e.g., 'bitcoin'): ").strip()
    days = int(input("Masukkan Jumlah Hari untuk Data (e.g., 1 atau 30): ").strip())
    
    df = fetch_data_from_coingecko(coin_id, days)
    
    if df is not None:
        print(f"Data berhasil diambil dan disimpan untuk {coin_id}.")
        print(df.head())
    else:
        print(f"Data tidak berhasil diambil untuk {coin_id}.")

if __name__ == "__main__":
    main()
