"""
user_manager.py
Gestión de usuarios reales, persistencia en Supabase (PostgreSQL),
normalización de nombres únicos y re-entrenamiento del modelo SVD.
"""
import re
import unicodedata
import pandas as pd
import streamlit as st

try:
    from db import (
        load_interactions as db_load_interactions,
        load_real_interactions as db_load_real_interactions,
        add_interaction as db_add_interaction,
        get_or_create_user as db_get_or_create_user,
        test_connection as db_test_connection
    )
    HAS_DB = True
except Exception:
    HAS_DB = False

def normalize_username(user_name: str) -> str:
    """
    Normaliza un nombre removiendo tildes, caracteres especiales y espacios extra.
    Ejemplo: ' Sámuel   Pérez ' -> 'samuel_perez'
    """
    if not user_name:
        return "invitado"
    text = user_name.strip().lower()
    text = unicodedata.normalize('NFD', text)
    text = re.sub(r'[\u0300-\u036f]', '', text)
    text = re.sub(r'\s+', '_', text)
    text = re.sub(r'[^a-z0-9_]', '', text)
    return text if text else "invitado"

def _init_local_session():
    """Inicializa la estructura de datos local en session_state."""
    if "local_user_votes" not in st.session_state:
        st.session_state["local_user_votes"] = {}

def get_real_interactions() -> pd.DataFrame:
    """
    Carga las interacciones registradas de usuarios reales desde Supabase.
    Combina con la sesión en memoria para reflejar votos instantáneos.
    """
    _init_local_session()

    db_df = pd.DataFrame()
    if HAS_DB:
        try:
            db_df = db_load_real_interactions()
            if not db_df.empty and "created_at" in db_df.columns:
                db_df = db_df.rename(columns={"created_at": "timestamp"})
        except Exception as e:
            print(f"Nota: Leyendo local por error en DB: {e}")

    # Votos de la sesión actual
    local_rows = []
    for (uid, tid), rat in st.session_state["local_user_votes"].items():
        local_rows.append({
            "user_id": uid,
            "archetype": "real_user",
            "track_id": tid,
            "rating": rat,
            "timestamp": "Sesión Actual"
        })
    local_df = pd.DataFrame(local_rows)

    if db_df.empty and local_df.empty:
        return pd.DataFrame(columns=["user_id", "archetype", "track_id", "rating", "timestamp"])

    combined = pd.concat([db_df, local_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=["user_id", "track_id"], keep="last")
    return combined

def is_username_taken(user_name: str) -> bool:
    """Verifica si el nombre normalizado ya tiene registros."""
    user_id = f"real_{normalize_username(user_name)}"
    df_real = get_real_interactions()
    if df_real.empty:
        return False
    return user_id in df_real["user_id"].values

def save_user_rating(user_name: str, track_id: str, rating: float) -> pd.DataFrame:
    """
    Registra o actualiza el voto en Supabase y en la sesión local.
    Garantiza que el perfil de usuario existe en la tabla 'users'.
    """
    _init_local_session()
    clean_id = normalize_username(user_name)
    user_id = f"real_{clean_id}"

    # 1. Actualizar estado en sesión local inmediata
    st.session_state["local_user_votes"][(user_id, track_id)] = rating

    # 2. Persistir en Supabase
    if HAS_DB:
        try:
            # Asegurar usuario en tabla users
            db_get_or_create_user(username=clean_id, display_name=user_name.strip().title())
            # Insertar interacción
            db_add_interaction(
                user_id=user_id,
                track_id=track_id,
                rating=float(rating),
                archetype="real_user",
                interaction_type="like" if rating >= 4.0 else "dislike"
            )
        except Exception as e:
            print(f"Advertencia al persistir en Supabase: {e}")

    # Limpiar caché de datos para refrescar estadísticas
    if hasattr(st, "cache_data"):
        st.cache_data.clear()
    return get_real_interactions()

def get_combined_interactions(synthetic_df: pd.DataFrame) -> pd.DataFrame:
    """Combina la base sintética con todas las interacciones reales recopiladas."""
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
    """Devuelve un diccionario {track_id: rating} para el usuario normalizado actual."""
    _init_local_session()
    clean_id = normalize_username(user_name)
    user_id = f"real_{clean_id}"

    df_real = get_real_interactions()
    if df_real.empty:
        return {}

    user_votes = df_real[df_real["user_id"] == user_id]
    if user_votes.empty:
        return {}

    return dict(zip(user_votes["track_id"], user_votes["rating"]))
