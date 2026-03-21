"""
run_all_eda.py — Orquestador del EDA Completo
Ejecuta los 4 scripts de EDA en secuencia.
Uso: python3 Research/EDA/scripts_EDA/run_all_eda.py
"""
import subprocess
import sys
import time
from pathlib import Path

SCRIPTS_DIR = Path(__file__).resolve().parent

scripts = [
    ("players.csv",        "eda_players.py"),
    ("matches.csv",        "eda_matches.py"),
    ("events.csv",         "eda_events.py"),
    ("player_history.csv", "eda_player_history.py"),
]

print("=" * 60)
print("  EDA COMPLETO — Análisis de Apuestas Premier League")
print("=" * 60)

total_start = time.time()
for dataset, script_name in scripts:
    script_path = SCRIPTS_DIR / script_name
    print(f"\n{'─'*60}")
    print(f"  ▶ Procesando: {dataset}")
    print(f"{'─'*60}")
    start = time.time()
    result = subprocess.run([sys.executable, str(script_path)])
    elapsed = time.time() - start
    status = "✅" if result.returncode == 0 else "❌"
    print(f"  {status} {dataset} — {elapsed:.1f}s")

elapsed_total = time.time() - total_start
print(f"\n{'='*60}")
print(f"  ✅ EDA FINALIZADO en {elapsed_total:.1f}s")
print(f"  Gráficas en Research/EDA/<base>/graficas/")
print(f"{'='*60}")
