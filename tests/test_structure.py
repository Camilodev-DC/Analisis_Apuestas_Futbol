from src.data_loader.loader import DataLoader
from src.processing.processor import DataProcessor
from src.models.trainer import ModelTrainer
import pandas as pd

def test_modularity():
    """Test that all modules can be imported and initialized."""
    loader = DataLoader()
    processor = DataProcessor()
    trainer = ModelTrainer()
    assert loader is not None
    assert processor is not None
    assert trainer is not None

def test_feature_engineering():
    """Test feature engineering logic."""
    processor = DataProcessor()
    df = pd.DataFrame({'hst': [5], 'ast': [2]})
    df_feat = processor.feature_engineering(df)
    assert 'SOTDiff' in df_feat.columns
    assert df_feat['SOTDiff'].iloc[0] == 3
