import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from api_client import PremierLeagueAPI

def run_exploration():
    client = PremierLeagueAPI()
    
    # 1. Verificar salud
    print("Verificando estado de la API...")
    health = client.check_health()
    print(f"API {health['status']} - Partidos disponibles: {health['data_counts']['matches']}")
    
    # 2. Cargas partidos
    print("\nCargando datos de todos los partidos de la temporada...")
    df_matches = client.get_matches()
    print(f"Partidos cargados: {len(df_matches)}")
    
    # 3. Exploración rápida de resultados
    print("\nDistribución de resultados finales (ftr):")
    dist_ftr = df_matches['ftr'].value_counts(normalize=True).round(3)
    print(dist_ftr)
    
    # 4. Exploración rápida de goles
    promedio_goles = (df_matches['fthg'] + df_matches['ftag']).mean()
    print(f"\nPromedio de goles totales por partido: {promedio_goles:.2f}")

if __name__ == "__main__":
    run_exploration()
