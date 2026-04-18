import requests
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import os

BASE_URL = "https://premier.72-60-245-2.sslip.io"
PLOTS_DIR = "analysis/plots"
REPORTS_DIR = "analysis/reports"
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)

def main():
    print("Obteniendo datos de jugadores (FPL API)...")
    players = []
    for offset in [0, 500]:
        resp = requests.get(f"{BASE_URL}/players?limit=500&offset={offset}")
        if resp.status_code == 200 and "players" in resp.json():
            players.extend(resp.json()["players"])
    
    df = pd.DataFrame(players)
    
    # 1. Filtro: Minutos jugados
    df_filtered = df[df['minutes'] > 1000].copy()
    
    # 2. Seleccionar variables
    features = ['goals_scored', 'assists', 'clean_sheets',
                'influence', 'creativity', 'threat', 'bps']
    
    for col in ['xG', 'xA']:
        if col in df_filtered.columns:
            features.append(col)
            df_filtered[col] = df_filtered[col].astype(float)
            
    for f in features:
        df_filtered[f] = pd.to_numeric(df_filtered[f], errors='coerce')
        
    # 3. Normalizar por 90 minutos
    df_filtered['90s'] = df_filtered['minutes'] / 90.0
    for f in features:
        df_filtered[f"{f}_p90"] = df_filtered[f] / df_filtered['90s']
    
    features_p90 = [f"{f}_p90" for f in features]
    df_filtered[features_p90] = df_filtered[features_p90].fillna(0)
    
    X = df_filtered[features_p90]
    
    # 4. Standard Scaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 5. K-Means (Fijamos K=5 como óptimo táctico)
    best_k = 5
    final_kmeans = KMeans(n_clusters=best_k, random_state=42, n_init=10)
    df_filtered['cluster'] = final_kmeans.fit_predict(X_scaled)
    
    # 6. PCA para visualización 2D
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)
    df_filtered['PCA1'] = X_pca[:, 0]
    df_filtered['PCA2'] = X_pca[:, 1]
    
    # --- GRÁFICO 1: PCA COLOREADO POR CLUSTER ---
    plt.style.use("dark_background")
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    scatter1 = ax.scatter(df_filtered['PCA1'], df_filtered['PCA2'], c=df_filtered['cluster'], cmap='Set1', s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
    
    # Añadir leyenda de clústeres
    legend1 = ax.legend(*scatter1.legend_elements(), title="Nuevos Roles (Clúster)", loc="best", facecolor="#1e1e1e", edgecolor="white", labelcolor="white")
    ax.add_artist(legend1)
    
    ax.set_title("Mapa de Componentes Principales (Jugadores agrupados por Nuevo Rol)", color='white', fontsize=14)
    ax.set_xlabel(f"PCA 1 (Varianza explicada: {pca.explained_variance_ratio_[0]:.1%})", color='#aaaaaa')
    ax.set_ylabel(f"PCA 2 (Varianza explicada: {pca.explained_variance_ratio_[1]:.1%})", color='#aaaaaa')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    pca_cluster_path = os.path.join(PLOTS_DIR, "pca_clusters.png")
    fig.savefig(pca_cluster_path, dpi=300, bbox_inches='tight')
    plt.close()

    # --- GRÁFICO 2: PCA COLOREADO POR POSICIÓN ORIGINAL FPL ---
    fig, ax = plt.subplots(figsize=(10, 8))
    fig.patch.set_facecolor('#1e1e1e')
    ax.set_facecolor('#1e1e1e')
    
    # Mapear colores por posición original (position suele ser string en la API)
    positions = df_filtered['position'].unique()
    colors = plt.cm.get_cmap('Set3', len(positions))
    
    for i, pos in enumerate(positions):
        subset = df_filtered[df_filtered['position'] == pos]
        ax.scatter(subset['PCA1'], subset['PCA2'], c=[colors(i)], label=f"Posición FPL: {pos}", s=50, alpha=0.8, edgecolors='white', linewidth=0.5)
        
    ax.legend(title="Posiciones Originales", loc="best", facecolor="#1e1e1e", edgecolor="white", labelcolor="white")
    ax.set_title("El mismo mapa 2D coloreado por la Posición Clásica FPL", color='white', fontsize=14)
    ax.set_xlabel("PCA 1 (Resumen general Ofensivo/Defensivo)", color='#aaaaaa')
    ax.set_ylabel("PCA 2 (Resumen Secundario)", color='#aaaaaa')
    ax.grid(color='#333333', linestyle='--', alpha=0.5)
    
    pca_position_path = os.path.join(PLOTS_DIR, "pca_positions.png")
    fig.savefig(pca_position_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    # Re-escribir Reporte MD
    cluster_means = df_filtered.groupby('cluster')[features_p90].mean()
    cluster_sizes = df_filtered.groupby('cluster').size()
    
    report_path = os.path.join(REPORTS_DIR, "clustering_roles.md")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("# Visualización Avanzada de Roles (PCA & K-Means)\n\n")
        f.write("Una vez entrenado el algoritmo **K-Means con K=5**, hemos utilizado Análisis de Componentes Principales (PCA) para reducir nuestras estadísticas a 2 dimensiones, permitiéndonos visualizar qué tan bien separados están los roles.\n\n")
        
        f.write("## 1. El Mapa de los 'Nuevos Roles'\n\n")
        f.write("A continuación vemos cómo la Inteligencia Artificial agrupó a los jugadores basándose **únicamente en su rendimiento per 90 minutos**.\n\n")
        f.write(f"![Clústeres PCA](/home/camilo/proyectos/Ml_Futbol/analysis/plots/pca_clusters.png)\n\n")
        
        f.write("## 2. El Mapa de las Posiciones Clásicas FPL\n\n")
        f.write("Si miramos la *misma nube de puntos* pero coloreada según su posición oficial en la Fantasy Premier League (Portero, Defensa, Medio, Delantero), notamos cómo el modelo descubrió sub-roles (por ejemplo, medios defensivos separados de los creativos).\n\n")
        f.write(f"![Posiciones PCA](/home/camilo/proyectos/Ml_Futbol/analysis/plots/pca_positions.png)\n\n")
        
        f.write(f"## Tablas de Perfiles por Clúster\n\n")
        
        for c in range(best_k):
            size = cluster_sizes[c]
            means = cluster_means.loc[c]
            if 'bps_p90' in df_filtered.columns:
                top_players = df_filtered[df_filtered['cluster'] == c].nlargest(5, 'bps_p90')['web_name'].tolist()
            else:
                top_players = df_filtered[df_filtered['cluster'] == c].head(5)['web_name'].tolist()
                
            f.write(f"### Rol {c} ({size} jugadores)\n")
            f.write(f"**Ejemplos destacados:** {', '.join(top_players)}\n\n")
            f.write("| Métrica (per 90) | Valor Promedio |\n")
            f.write("| ---------------- | -------------- |\n")
            for feat in features_p90:
                f.write(f"| {feat.replace('_p90', '')} | {means[feat]:.2f} |\n")
            f.write("\n")
            
    print(f"Reporte y gráficos de PCA guardados exitosamente.")

if __name__ == "__main__":
    main()
