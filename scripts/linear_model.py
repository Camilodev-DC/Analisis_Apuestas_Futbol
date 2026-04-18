import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
from api_client import PremierLeagueAPI
from config import FEATURES_LINEAL, TARGET_LINEAL

def train_linear_model():
    client = PremierLeagueAPI()
    
    # 1. Obtener datos
    print("Obteniendo datos para regresión lineal...")
    df = client.get_matches()
    
    # 2. Selección de features y target (goles locales)
    X = df[FEATURES_LINEAL].astype(float)
    y = df[TARGET_LINEAL].astype(float)
    
    # 3. Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # 4. Entrenamiento
    model = LinearRegression()
    model.fit(X_train, y_train)
    
    # 5. Evaluación
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    
    print(f"\nResultados Regresión Lineal (Target: {TARGET_LINEAL}):")
    print(f"R-squared: {r2:.3f}")
    print(f"RMSE: {rmse:.3f}")
    
    # 6. Interpretación de coeficientes
    coefs = pd.DataFrame({'Feature': FEATURES_LINEAL, 'Coeficiente': model.coef_})
    print("\nCoeficientes del modelo:")
    print(coefs.sort_values('Coeficiente', ascending=False).to_string(index=False))

    return model

if __name__ == "__main__":
    train_linear_model()
