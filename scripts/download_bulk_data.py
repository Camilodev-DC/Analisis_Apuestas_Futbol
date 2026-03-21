import requests
import os
import pandas as pd
import time

BASE_URL = "https://premier.72-60-245-2.sslip.io"
RAW_DIR = "data/raw"

# ─── Datasets simples (bulk export) ─────────────────────────────────────────
BULK_DATASETS = {
    "players":        f"{BASE_URL}/export/players",
    "matches":        f"{BASE_URL}/export/matches",
    "player_history": f"{BASE_URL}/export/player_history",
}

def download_bulk(name, url):
    """Descarga un CSV desde el endpoint de bulk export."""
    print(f"Descargando {name}...")
    response = requests.get(url, stream=True, timeout=60)
    response.raise_for_status()
    path = os.path.join(RAW_DIR, f"{name}.csv")
    with open(path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    df = pd.read_csv(path)
    print(f"  ✓ {name}.csv — {len(df):,} registros")
    return len(df)

def download_events_with_qualifiers():
    """
    Descarga eventos partido por partido para obtener el campo 'qualifiers' completo.
    Usa el endpoint /matches/{id}/events que devuelve el JSON con qualifiers.
    Resultado: data/raw/events.csv con columna 'qualifiers' incluida.
    """
    print("\nDescargando events con qualifiers (partido por partido)...")
    
    # 1. Obtener todos los match_ids
    resp = requests.get(f"{BASE_URL}/matches?limit=400", timeout=30)
    resp.raise_for_status()
    matches_data = resp.json()
    match_ids = [m["id"] for m in matches_data.get("matches", [])]
    total = len(match_ids)
    print(f"  Partidos encontrados: {total}")

    all_events = []
    errors = []

    for i, match_id in enumerate(match_ids, 1):
        try:
            url = f"{BASE_URL}/matches/{match_id}/events?limit=5000"
            r = requests.get(url, timeout=30)
            r.raise_for_status()
            data = r.json()
            events = data.get("events", [])
            # Serializar qualifiers como string JSON para guardar en CSV
            for ev in events:
                if isinstance(ev.get("qualifiers"), list):
                    import json
                    ev["qualifiers"] = json.dumps(ev["qualifiers"])
            all_events.extend(events)
            if i % 25 == 0 or i == total:
                print(f"  [{i}/{total}] Partido {match_id} — {len(events)} eventos (total: {len(all_events):,})")
            time.sleep(0.05)  # Rate limiting suave
        except Exception as e:
            errors.append((match_id, str(e)))
            print(f"  ✗ Error partido {match_id}: {e}")

    if all_events:
        df = pd.DataFrame(all_events)
        path = os.path.join(RAW_DIR, "events.csv")
        df.to_csv(path, index=False)
        print(f"\n  ✓ events.csv — {len(df):,} eventos con qualifiers guardados en {path}")
        if errors:
            print(f"  ⚠️  {len(errors)} partidos con error: {[m for m,e in errors]}")
        return len(df)
    else:
        print("  ✗ No se pudieron descargar eventos.")
        return 0

def download_data():
    os.makedirs(RAW_DIR, exist_ok=True)
    results = {}

    # Bulk datasets (players, matches, player_history)
    for name, url in BULK_DATASETS.items():
        try:
            results[name] = download_bulk(name, url)
        except Exception as e:
            print(f"  ✗ Error descargando {name}: {e}")
            results[name] = f"Error: {e}"

    # Events con qualifiers (partido por partido)
    try:
        results["events"] = download_events_with_qualifiers()
    except Exception as e:
        print(f"  ✗ Error descargando events: {e}")
        results["events"] = f"Error: {e}"

    print("\n" + "="*50)
    print("RESUMEN DE DESCARGA:")
    for name, count in results.items():
        print(f"  {name:<20}: {count}")
    print("="*50)
    return results

if __name__ == "__main__":
    download_data()
