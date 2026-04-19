import json
import requests
import csv
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

MATCHES_FILE = "data/raw/matches.json"
OUTPUT_FILE = "data/raw/events.csv"
BASE_URL = "https://premier.72-60-245-2.sslip.io"
MAX_WORKERS = 10

def fetch_events_for_match(match_id):
    url = f"{BASE_URL}/matches/{match_id}/events"
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        events = response.json()
        if isinstance(events, dict) and 'events' in events:
            return events['events']
        elif isinstance(events, list):
            return events
    except Exception as e:
        print(f"Error en partido {match_id}: {e}")
    return []

def main():
    if not os.path.exists(MATCHES_FILE):
        print(f"Error: {MATCHES_FILE} no encontrado.")
        return

    with open(MATCHES_FILE, 'r') as f:
        matches_data = json.load(f)
    
    match_ids = [m['id'] for m in matches_data.get('matches', [])]
    total_matches = len(match_ids)
    print(f"Identificados {total_matches} partidos. Iniciando descarga paralela...")

    all_events = []
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_match = {executor.submit(fetch_events_for_match, mid): mid for mid in match_ids}
        for i, future in enumerate(as_completed(future_to_match)):
            match_id = future_to_match[future]
            res = future.result()
            all_events.extend(res)
            if (i + 1) % 50 == 0 or (i + 1) == total_matches:
                print(f"[{i+1}/{total_matches}] Partidos procesados. Total eventos acumulados: {len(all_events)}")

    if not all_events:
        print("No se obtuvieron eventos.")
        return

    print(f"Escribiendo {len(all_events)} eventos a {OUTPUT_FILE}...")
    keys = all_events[0].keys()
    with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=keys)
        writer.writeheader()
        for event in all_events:
            if 'qualifiers' in event and isinstance(event['qualifiers'], (list, dict)):
                event['qualifiers'] = json.dumps(event['qualifiers'])
            writer.writerow(event)

    print(f"Proceso finalizado. Total de registros: {len(all_events)}")

if __name__ == "__main__":
    main()
