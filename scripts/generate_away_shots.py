import requests
import pandas as pd
import matplotlib.pyplot as plt
from mplsoccer import Pitch
import os

# Configuración
BASE_URL = "https://premier.72-60-245-2.sslip.io"
PLOTS_DIR = "analysis/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def main():
    print("Obteniendo información de los partidos para identificar visitantes...")
    matches_resp = requests.get(f"{BASE_URL}/matches?limit=500")
    df_matches = pd.DataFrame(matches_resp.json()["matches"])
    
    # Crear diccionario match_id -> away_team_name
    # Según la exploración anterior, las columnas son 'id' y 'away_team'
    away_teams_map = dict(zip(df_matches['id'], df_matches['away_team']))
    print(f"Partidos mapeados: {len(away_teams_map)}")
    
    print("Obteniendo todos los tiros de la temporada...")
    resp = requests.get(f"{BASE_URL}/events?is_shot=true&limit=5000")
    shots = resp.json()["events"]
    df_shots = pd.DataFrame(shots)
    print(f"Total de tiros obtenidos: {len(df_shots)}")
    
    # Filtrar solo tiros donde el team_name coincida con el away_team_name de ese match_id
    # Creamos una columna 'is_away' cruzando el match_id
    df_shots['away_team'] = df_shots['match_id'].map(away_teams_map)
    df_away_shots = df_shots[df_shots['team_name'] == df_shots['away_team']].copy()
    
    print(f"Tiros de equipos visitantes: {len(df_away_shots)}")
    
    print("Generando mapa de tiros profesionales con mplsoccer...")
    
    # mplsoccer pitch (opta uses 100x100 para coordenadas de X e Y)
    # Pitch color oscuro para un diseño más moderno y profesional
    pitch = Pitch(pitch_type='opta', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(12, 8))
    fig.patch.set_facecolor('#22312b')
    
    # Diferenciar goles de fallos
    goals = df_away_shots[df_away_shots['is_goal'] == True]
    non_goals = df_away_shots[df_away_shots['is_goal'] == False]
    
    # Dibujar no goles (transparencia)
    pitch.scatter(non_goals.x, non_goals.y, alpha=0.4, s=30, c="#ea6969", edgecolors="black", ax=ax, label="Fallo / Atajada")
    
    # Dibujar goles (más grandes y visibles, color cian/verde para resaltar en fondo oscuro)
    pitch.scatter(goals.x, goals.y, alpha=0.9, s=120, c="#00ff85", marker="*", edgecolors="white", linewidth=1, zorder=5, ax=ax, label="Gol")
    
    # Añadir título y subtítulo
    ax.text(50, 105, "Mapa de Tiros - Equipos Visitantes", color="white", fontsize=20, ha="center", fontfamily="sans-serif", fontweight="bold")
    ax.text(50, 101, f"Premier League 2025/26 - Total tiros: {len(df_away_shots)} ({len(goals)} goles)", color="#c7d5cc", fontsize=12, ha="center", fontfamily="sans-serif")
    
    # Leyenda profesional
    legend = ax.legend(loc='lower left', facecolor='#22312b', edgecolor='none', labelcolor='white')
    
    output_path = os.path.join(PLOTS_DIR, "away_shots.png")
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches='tight', dpi=300)
    print(f"Gráfico guardado en: {output_path}")

if __name__ == "__main__":
    main()
