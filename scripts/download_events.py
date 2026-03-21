import requests
import os

url = "https://premier.72-60-245-2.sslip.io/export/events?format=csv"
output_path = "data/raw/events.csv"

def download_events():
    print(f"Descargando eventos desde {url}...")
    try:
        with requests.get(url, stream=True) as r:
            r.raise_for_status()
            with open(output_path, 'wb') as f:
                for chunk in r.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
        print("Descarga completa.")
    except Exception as e:
        print(f"Error durante la descarga: {e}")

if __name__ == "__main__":
    download_events()
