import pandas as pd
import requests
import os
from dotenv import load_dotenv

load_dotenv()

class DataLoader:
    def __init__(self, base_url=None):
        self.base_url = base_url or os.getenv("API_BASE_URL", "https://premier.72-60-245-2.sslip.io")

    def fetch_matches(self, limit=500):
        """Fetch match data from the API."""
        endpoint = f"{self.base_url}/matches?limit={limit}"
        response = requests.get(endpoint)
        response.raise_for_status()
        return pd.DataFrame(response.json())

    def save_raw_data(self, df, filename="matches_raw.csv"):
        """Save raw data to the data/raw directory."""
        os.makedirs("data/raw", exist_ok=True)
        path = os.path.join("data/raw", filename)
        df.to_csv(path, index=False)
        return path
