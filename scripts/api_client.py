import requests
import pandas as pd
from config import BASE_URL

class PremierLeagueAPI:
    def __init__(self, base_url=BASE_URL):
        self.base_url = base_url

    def check_health(self):
        """Verifica el estado de la API."""
        response = requests.get(f"{self.base_url}/health")
        response.raise_for_status()
        return response.json()

    def get_matches(self, limit=500):
        """Obtiene el histórico de partidos de la temporada."""
        url = f"{self.base_url}/matches"
        params = {"limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["matches"])

    def get_players(self, limit=900):
        """Obtiene la lista de jugadores con estadísticas FPL."""
        url = f"{self.base_url}/players"
        params = {"limit": limit}
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        return pd.DataFrame(data["players"])

    def get_match_events(self, match_id):
        """Obtiene eventos detallados de un partido específico."""
        url = f"{self.base_url}/matches/{match_id}/events"
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
