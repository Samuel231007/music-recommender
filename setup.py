"""
setup.py
Script de inicialización: genera datos procesados y entrena modelos.
Ejecutar antes de lanzar la app si data/processed/ está vacío.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from data_loader import load_clean
from synthetic_users import generate_interactions, INTERACTIONS_PATH
from collab_model import train_svd, MODEL_PATH
import pandas as pd

print("=== Setup del Recomendador Híbrido ===")

print("\n[1/3] Cargando y limpiando dataset...")
df = load_clean(force_reprocess=True)
print(f"    ✓ {len(df)} canciones listas")

print("\n[2/3] Generando usuarios sintéticos...")
if INTERACTIONS_PATH.exists():
    interactions = pd.read_csv(INTERACTIONS_PATH)
    print(f"    ✓ Cargado desde caché ({len(interactions)} interacciones)")
else:
    interactions = generate_interactions(df)
    print(f"    ✓ {len(interactions)} interacciones generadas")

print("\n[3/3] Entrenando modelo SVD...")
if MODEL_PATH.exists():
    print("    ✓ Modelo cargado desde caché")
else:
    model = train_svd(interactions)
    print("    ✓ Modelo entrenado y guardado")

print("\n✅ Setup completo. Lanza la app con: streamlit run app/streamlit_app.py")
