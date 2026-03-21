from src.data_loader.loader import DataLoader
from src.processing.processor import DataProcessor
from src.models.trainer import ModelTrainer

def main():
    print("Iniciando Pipeline de ML Premier League...")
    
    # 1. Carga de datos
    loader = DataLoader()
    # df = loader.fetch_matches() # Descomentar cuando la API esté disponible
    # path = loader.save_raw_data(df)
    
    # 2. Procesamiento
    processor = DataProcessor()
    # df_clean = processor.clean_data(df)
    
    # 3. Modelado
    trainer = ModelTrainer()
    
    print("Pipeline completado estructuralmente.")

if __name__ == "__main__":
    main()
