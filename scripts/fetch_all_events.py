import json
import requests
import csv
import os

MATCHES_FILE = "data/raw/matches.json"
OUTPUT_FILE = "data/raw/events.csv"
BASE_URL = "https://premier.72-60-245-2.sslip.io"

def fetch_events():
    if not os.path.exists(MATCHES_FILE):
        print(f"Error: {MATCHES_FILE} no encontrado.")
        return

    with open(MATCHES_FILE, 'r') as f:
        matches_data = json.load(f)
    
    match_ids = [m['id'] for m in matches_data.get('matches', [])]
    total_matches = len(match_ids)
    print(f"Identificados {total_matches} partidos. Iniciando descarga de eventos...")

    header_written = False
    total_records = 0

    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = None
        
        for i, match_id in enumerate(match_ids):
            url = f"{BASE_URL}/matches/{match_id}/events"
            try:
                response = requests.get(url)
                response.raise_for_status()
                events = response.json()
                
                # Manejar respuesta si es un objeto con 'events' o directamente una lista
                if isinstance(events, dict) and 'events' in events:
                    events_list = events['events']
                elif isinstance(events, list):
                    events_list = events
                else:
                    print(f"[{i+1}/{total_matches}] Match {match_id}: Formato inesperado.")
                    continue

                if not events_list:
                    continue

                # Preparar el writer con las llaves del primer evento
                if not header_written:
                    keys = events_list[0].keys()
                    writer = csv.DictWriter(csvfile, fieldnames=keys)
                    writer.writeheader()
                    header_written = True

                for event in events_list:
                    # Opcional: limpiar qualifiers si es dict/list para que entre en CSV como string
                    if 'qualifiers' in event and isinstance(event['qualifiers'], (list, dict)):
                        event['qualifiers'] = json.dumps(event['qualifiers'])
                    writer.writerow(event)
                
                total_records += len(events_list)
                if (i + 1) % 10 == 0 or (i + 1) == total_matches:
                    print(f"[{i+1}/{total_matches}] Descargados {total_records} registros...")

            except Exception as e:
                print(f"Error procesando partido {match_id}: {e}")

    print(f"Proceso finalizado. Total de registros: {total_records}")

if __name__ == "__main__":
    fetch_events()
