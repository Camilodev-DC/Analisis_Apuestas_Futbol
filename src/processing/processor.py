import pandas as pd

class DataProcessor:
    def __init__(self):
        pass

    def clean_data(self, df):
        """Basic data cleaning."""
        # Remove duplicates, handle missing values
        df = df.drop_duplicates().copy()
        return df

    def feature_engineering(self, df):
        """Create features for the models."""
        # Example: SOTDiff (Home Shots on Target - Away Shots on Target)
        if 'hst' in df.columns and 'ast' in df.columns:
            df['SOTDiff'] = df['hst'] - df['ast']
        
        # Example: Probabilidades implícitas si las cuotas están presentes
        # (b365h, b365d, b365a)
        return df
