"""
user_manager.py
Gestión de usuarios reales, persistencia de calificaciones (likes/dislikes),
normalización de nombres únicos y re-entrenamiento del modelo SVD.
"""
import re
import unicodedata
import pandas as pd
import streamlit as st

try:
    from db import load_interactions as load_all_interactions, add_interaction
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
    """Carga las interacciones registradas de usuarios reales desde Supabase o la sesión local."""
    _init_local_session()
    
    db_df = pd.DataFrame()
    if HAS_DB:
        try:
            df = load_all_interactions()
            if not df.empty and "archetype" in df.columns:
                real_df = df[df["archetype"] == "real_user"].copy()
                if "created_at" in real_df.columns:
                    real_df = real_df.rename(columns={"created_at": "timestamp"})
                db_df = real_df[["user_id", "archetype", "track_id", "rating", "timestamp"]]
        except Exception:
            db_df = pd.DataFrame()

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
    """Verifica si el nombre normalizado ya tiene registros en la base de datos o sesión."""
    user_id = f"real_{normalize_username(user_name)}"
    df_real = get_real_interactions()
    if df_real.empty:
        return False
    return user_id in df_real["user_id"].values


def save_user_rating(user_name: str, track_id: str, rating: float) -> pd.DataFrame:
    """Registra o actualiza el voto asegurando un user_id único y normalizado."""
    _init_local_session()
    clean_id = normalize_username(user_name)
    user_id = f"real_{clean_id}"

    # 1. Actualizar estado en sesión local
    st.session_state["local_user_votes"][(user_id, track_id)] = rating

    # 2. Persistir en base de datos
    if HAS_DB:
        try:
            add_interaction(user_id=user_id, track_id=track_id, rating=rating, archetype="real_user")
        except Exception as e:
            print(f"Advertencia al guardar en la BD: {e}")

    # 3. Limpiar caché global de Streamlit
    st.cache_data.clear()
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
