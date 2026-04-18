import requests
import pandas as pd
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
REPORTS_DIR = "analysis/reports"
os.makedirs(REPORTS_DIR, exist_ok=True)

def generate_leaderboards():
    print("Obteniendo datos de jugadores (FPL API)...")
    players = []
    for offset in [0, 500]:
        resp = requests.get(f"{BASE_URL}/players?limit=500&offset={offset}")
        if resp.status_code == 200 and "players" in resp.json():
            players.extend(resp.json()["players"])
    df_players = pd.DataFrame(players)
    
    # Check if red_cards exists
    if 'red_cards' in df_players.columns:
        top_reds = df_players.nlargest(10, 'red_cards')[['web_name', 'team', 'red_cards', 'minutes']]
    else:
        print("Buscando columnas relacionadas con tarjetas rojas...")
        cols = [c for c in df_players.columns if 'card' in c.lower() or 'red' in c.lower()]
        print("Columnas encontradas:", cols)
        if len(cols) > 0:
            col_name = cols[0]
            top_reds = df_players.nlargest(10, col_name)[['web_name', 'team', col_name, 'minutes']]
        else:
            top_reds = pd.DataFrame()
    
    print("Obteniendo todos los tiros para calcular fallas (WhoScored API)...")
    shots_resp = requests.get(f"{BASE_URL}/events?is_shot=true&limit=5000")
    df_shots = pd.DataFrame(shots_resp.json()["events"])
    
    # Fallas: tiros que NO fueron gol
    misses = df_shots[df_shots['is_goal'] == False]
    
    # Agrupar por jugador
    top_misses = misses.groupby(['player_name', 'team_name']).size().reset_index(name='fallas')
    top_misses = top_misses.sort_values(by='fallas', ascending=False).head(10)
    
    # Generar Markdown
    report_path = os.path.join(REPORTS_DIR, "leaderboards.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Leaderboards: Indisciplina y Fallas de Cara a Portería\n\n")
        f.write("A continuación se presentan los jugadores con más tarjetas rojas registradas en su historial y aquellos que lideran la liga en disparos fallados (tiros que no resultaron en gol).\n\n")
        
        f.write("## 🟥 Top 10: Jugadores con más Tarjetas Rojas\n\n")
        if not top_reds.empty:
            f.write(top_reds.to_markdown(index=False))
        else:
            f.write("> No se encontraron datos de tarjetas rojas en la API FPL para esta temporada.")
        
        f.write("\n\n")
        
        f.write("## ❌ Top 10: Jugadores con más Tiros Fallados\n\n")
        f.write(top_misses.to_markdown(index=False))
        f.write("\n\n")
        
        f.write("*Datos obtenidos directamente del endpoint de eventos (últimos 5000 tiros) y el registro de la FPL.*\n")
    
    print(f"Reporte generado exitosamente en: {report_path}")

if __name__ == "__main__":
    generate_leaderboards()
