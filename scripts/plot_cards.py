import requests
import pandas as pd
import matplotlib.pyplot as plt
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
PLOTS_DIR = "analysis/plots"
os.makedirs(PLOTS_DIR, exist_ok=True)

def generate_card_charts():
    print("Obteniendo datos de jugadores (FPL API)...")
    players = []
    for offset in [0, 500]:
        resp = requests.get(f"{BASE_URL}/players?limit=500&offset={offset}")
        if resp.status_code == 200 and "players" in resp.json():
            players.extend(resp.json()["players"])
    
    df = pd.DataFrame(players)
    
    # Asegurarnos de que las columnas existan
    if 'yellow_cards' not in df.columns or 'red_cards' not in df.columns:
        print("Error: No se encontraron las columnas de tarjetas en FPL.")
        return
    
    df['total_cards'] = df['yellow_cards'] + df['red_cards']
    
    # --- 1. Top Teams ---
    df_teams = df.groupby('team')[['yellow_cards', 'red_cards', 'total_cards']].sum().reset_index()
    df_teams = df_teams.sort_values('total_cards', ascending=True) # Ascending para barra horizontal
    
    # --- 2. Top Players ---
    df_players = df.sort_values('total_cards', ascending=True).tail(15) # Top 15 jugadores
    
    # --- Graficar ---
    print("Generando gráficos de barras...")
    plt.style.use("dark_background")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
    fig.patch.set_facecolor('#1e1e1e')
    
    # Equipos
    ax1.set_facecolor('#1e1e1e')
    ax1.barh(df_teams['team'], df_teams['yellow_cards'], color='#f1c40f', label='Amarillas', edgecolor='#1e1e1e')
    ax1.barh(df_teams['team'], df_teams['red_cards'], left=df_teams['yellow_cards'], color='#e74c3c', label='Rojas', edgecolor='#1e1e1e')
    ax1.set_title("Tarjetas Acumuladas por Equipo", fontsize=16, pad=15, color='white')
    ax1.set_xlabel("Cantidad de Tarjetas", color='#aaaaaa')
    ax1.tick_params(colors='#aaaaaa')
    ax1.legend(facecolor='#1e1e1e', edgecolor='none', labelcolor='white')
    ax1.grid(axis='x', color='#333333', linestyle='--', alpha=0.7)
    
    # Jugadores
    ax2.set_facecolor('#1e1e1e')
    labels = df_players['web_name'] + " (" + df_players['team'].astype(str) + ")"
    ax2.barh(labels, df_players['yellow_cards'], color='#f1c40f', label='Amarillas', edgecolor='#1e1e1e')
    ax2.barh(labels, df_players['red_cards'], left=df_players['yellow_cards'], color='#e74c3c', label='Rojas', edgecolor='#1e1e1e')
    ax2.set_title("Top 15 Jugadores con Más Tarjetas", fontsize=16, pad=15, color='white')
    ax2.set_xlabel("Cantidad de Tarjetas", color='#aaaaaa')
    ax2.tick_params(colors='#aaaaaa')
    ax2.grid(axis='x', color='#333333', linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    output_path = os.path.join(PLOTS_DIR, "cards_bar_charts.png")
    fig.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Gráfico guardado en: {output_path}")

if __name__ == "__main__":
    generate_card_charts()
