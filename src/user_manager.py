"""
user_manager.py
Gestión de usuarios reales, persistencia de calificaciones (likes/dislikes)
y re-entrenamiento del modelo SVD con retroalimentación en vivo.
Usa Supabase (PostgreSQL) en lugar de un CSV local, para que los votos
persistan entre reinicios y redeploys de la app.
"""
import pandas as pd
from db import load_interactions as load_all_interactions, add_interaction


def get_real_interactions() -> pd.DataFrame:
    """Carga las interacciones registradas de usuarios reales desde Supabase."""
    df = load_all_interactions()
    if df.empty:
        return pd.DataFrame(columns=["user_id", "archetype", "track_id", "rating", "timestamp"])
    real_df = df[df["archetype"] == "real_user"].copy()
    real_df = real_df.rename(columns={"created_at": "timestamp"})
    return real_df[["user_id", "archetype", "track_id", "rating", "timestamp"]]


def save_user_rating(user_name: str, track_id: str, rating: float) -> pd.DataFrame:
    """
    Registra o actualiza el voto de un usuario real en Supabase.
    user_name: Nombre o apodo del usuario (ej. 'Samuel')
    rating: 5.0 (Me gusta) o 1.0 (No me gusta)
    """
    clean_user = user_name.strip()
    if not clean_user:
        clean_user = "Invitado"

    user_id = f"real_{clean_user.lower().replace(' ', '_')}"
    add_interaction(user_id=user_id, track_id=track_id, rating=rating, archetype="real_user")
    return get_real_interactions()


def get_combined_interactions(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Combina usuarios sintéticos con las interacciones reales recopiladas."""
    real_df = get_real_interactions()
    if real_df.empty:
        return synthetic_df

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