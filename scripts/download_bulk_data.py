import requests
import os
import pandas as pd

BASE_URL = "https://premier.72-60-245-2.sslip.io"
RAW_DIR = "data/raw"

datasets = {
    "players": f"{BASE_URL}/export/players",
    "matches": f"{BASE_URL}/export/matches",
    "events": f"{BASE_URL}/export/events",
    "player_history": f"{BASE_URL}/export/player_history"
}

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    results = {}
    
    for name, url in datasets.items():
        print(f"Descargando {name} desde {url}...")
        try:
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            file_path = os.path.join(RAW_DIR, f"{name}.csv")
            with open(file_path, "wb") as f:
                for chunk in response.iter_content(chunk_size=8192):
                    f.write(chunk)
            
            # Count records (excluding header)
            # Using pandas for accurate count and verification
            df = pd.read_csv(file_path)
            results[name] = len(df)
            print(f"  ✓ {name}.csv descargado. Registros: {results[name]}")
            
        except Exception as e:
            print(f"  ✗ Error descargando {name}: {e}")
            results[name] = f"Error: {e}"
            
    return results

if __name__ == "__main__":
    download_data()
