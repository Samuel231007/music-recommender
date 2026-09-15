"""
user_manager.py
Gestión de usuarios reales, persistencia de calificaciones (likes/dislikes)
y re-entrenamiento del modelo SVD con retroalimentación en vivo.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys

PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed"
REAL_INTERACTIONS_PATH = PROCESSED_DIR / "real_interactions.csv"

def get_real_interactions() -> pd.DataFrame:
    """Carga las interacciones registradas de usuarios reales."""
    if REAL_INTERACTIONS_PATH.exists():
        try:
            df = pd.read_csv(REAL_INTERACTIONS_PATH)
            if not df.empty and "user_id" in df.columns:
                return df
        except Exception:
            pass
    return pd.DataFrame(columns=["user_id", "archetype", "track_id", "rating", "timestamp"])

def save_user_rating(user_name: str, track_id: str, rating: float) -> pd.DataFrame:
    """
    Registra o actualiza el voto de un usuario real.
    user_name: Nombre o apodo del usuario (ej. 'Samuel')
    rating: 5.0 (Me gusta) o 1.0 (No me gusta)
    """
    clean_user = user_name.strip()
    if not clean_user:
        clean_user = "Invitado"
    
    user_id = f"real_{clean_user.lower().replace(' ', '_')}"
    
    df_real = get_real_interactions()
    
    # Si ya votó por esta canción, actualizar; si no, agregar fila
    mask = (df_real["user_id"] == user_id) & (df_real["track_id"] == track_id)
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    if mask.any():
        df_real.loc[mask, "rating"] = rating
        df_real.loc[mask, "timestamp"] = now_str
    else:
        new_row = pd.DataFrame([{
            "user_id": user_id,
            "archetype": "real_user",
            "track_id": track_id,
            "rating": rating,
            "timestamp": now_str
        }])
        df_real = pd.concat([df_real, new_row], ignore_index=True)
        
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    df_real.to_csv(REAL_INTERACTIONS_PATH, index=False)
    return df_real

def get_combined_interactions(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Combina usuarios sintéticos con las interacciones reales recopiladas."""
    real_df = get_real_interactions()
    if real_df.empty:
        return synthetic_df
    
    # Asegurar columnas comunes
    cols = ["user_id", "archetype", "track_id", "rating"]
    combined = pd.concat([
        synthetic_df[cols],
        real_df[cols]
    ], ignore_index=True)
    return combined

def get_real_user_votes(user_name: str) -> dict:
    """Devuelve un diccionario {track_id: rating} para el usuario actual."""
    clean_user = user_name.strip()
    user_id = f"real_{clean_user.lower().replace(' ', '_')}"
    df_real = get_real_interactions()
    if df_real.empty:
        return {}
    user_votes = df_real[df_real["user_id"] == user_id]
    return dict(zip(user_votes["track_id"], user_votes["rating"]))
