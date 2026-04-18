import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from api_client import PremierLeagueAPI
from config import FEATURES_LOGISTICA_ODDS, TARGET_LOGISTICA

def prepare_odds_features(df):
    """Convierte las cuotas Bet365 en probabilidades normalizadas."""
    X_odds = df[FEATURES_LOGISTICA_ODDS].astype(float)
    X_probs = 1 / X_odds
    
    # Normalización para eliminar el margen de la casa (implied probability)
    row_sums = X_probs.sum(axis=1)
    X_probs = X_probs.div(row_sums, axis=0)
    X_probs.columns = ["prob_H", "prob_D", "prob_A"]
    
    return X_probs

def train_logistic_model():
    client = PremierLeagueAPI()
    
    # 1. Obtener datos
    print("Obteniendo datos para regresión logística...")
    df = client.get_matches()
    
    # Limpiamos NaNs si hay problemas con cuotas (common issue mentioned in PDF)
    df_clean = df.dropna(subset=FEATURES_LOGISTICA_ODDS + [TARGET_LOGISTICA])
    
    # 2. Preparar features (cuotas normalizadas) e incluir stats clave
    X_probs = prepare_odds_features(df_clean)
    X_stats = df_clean[["hs", "as_", "hst", "ast"]].astype(float) # Stats básicas sugeridas por workshop
    
    X = pd.concat([X_probs, X_stats], axis=1)
    y = df_clean[TARGET_LOGISTICA]
    
    # 3. Split y entrenamiento (Estratificado por el desbalance de clases)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    model = LogisticRegression(multi_class='multinomial', max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    
    # 4. Evaluación
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    
    print(f"\nResultados Regresión Logística (Target: {TARGET_LOGISTICA}):")
    print(f"Accuracy: {acc:.1%}")
    print("\nReporte de Clasificación:")
    print(classification_report(y_test, y_pred))

    # 5. Baseline (Bet365 - resultado con menor odd / mayor probabilidad)
    baseline_pred = X_probs.loc[X_test.index].idxmax(axis=1).map({"prob_H": "H", "prob_D": "D", "prob_A": "A"})
    baseline_acc = accuracy_score(y_test, baseline_pred)
    print(f"\nAccuracy Baseline (Bet365): {baseline_acc:.1%}")
    print(f"Mejora respecto al mercado: {(acc - baseline_acc)*100:+.1f} puntos porcentuales")

    return model

if __name__ == "__main__":
    train_logistic_model()
