"""
db.py
Capa de persistencia y conexión con Supabase (PostgreSQL).
Soporta comunicación directa vía REST API sobre HTTPS (resiliente a IPv6/firewalls)
y sincronización en vivo con perfiles de usuarios reales.
"""
import requests
import pandas as pd
import streamlit as st
from typing import Optional, Dict, Any, List
from datetime import datetime, timezone

# Credenciales por defecto (proporcionadas por el usuario para este proyecto)
DEFAULT_SUPABASE_URL = "https://acmzwxjlejmikoodtzcv.supabase.co"
DEFAULT_SUPABASE_KEY = "sb_publishable_ywZIWzyyWppD87HejlQaGQ_21leYrEu"

def get_supabase_config() -> tuple[str, str]:
    """Obtiene la URL y clave de Supabase desde st.secrets o valores por defecto."""
    url = DEFAULT_SUPABASE_URL
    key = DEFAULT_SUPABASE_KEY

    try:
        if hasattr(st, "secrets"):
            if "supabase" in st.secrets:
                url = st.secrets["supabase"].get("url", url)
                key = st.secrets["supabase"].get("key", key)
            elif "connections" in st.secrets and "supabase" in st.secrets["connections"]:
                sb = st.secrets["connections"]["supabase"]
                url = sb.get("url", url)
                key = sb.get("key", key)
    except Exception:
        pass

    # Normalizar URL base (sin trailing slash ni /rest/v1)
    url = url.rstrip("/")
    if url.endswith("/rest/v1"):
        url = url[:-8]
    return url, key

def _get_headers() -> Dict[str, str]:
    url, key = get_supabase_config()
    return {
        "apikey": key,
        "Authorization": f"Bearer {key}",
        "Content-Type": "application/json",
        "Prefer": "return=minimal"
    }

def test_connection() -> Dict[str, Any]:
    """Verifica el estado de conexión con la base de datos Supabase."""
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/user_interactions?select=count"
    headers = _get_headers()
    headers["Range-Unit"] = "items"
    headers["Prefer"] = "count=exact"

    try:
        r = requests.get(endpoint, headers=headers, timeout=5)
        if r.status_code in [200, 206]:
            cr = r.headers.get("Content-Range", "")
            total = int(cr.split("/")[-1]) if "/" in cr else 0
            return {"connected": True, "total_interactions": total, "error": None}
        return {"connected": False, "total_interactions": 0, "error": f"HTTP {r.status_code}: {r.text}"}
    except Exception as e:
        return {"connected": False, "total_interactions": 0, "error": str(e)}

def get_or_create_user(username: str, display_name: Optional[str] = None) -> Dict[str, Any]:
    """
    Crea o actualiza un usuario real en la tabla 'users'.
    Garantiza que cada usuario tenga registro permanente.
    """
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/users"
    headers = _get_headers()
    headers["Prefer"] = "resolution=merge-duplicates,return=representation"

    clean_user = username.strip().lower()
    user_id = f"real_{clean_user}"
    disp = display_name if display_name else username.strip().title()
    now_iso = datetime.now(timezone.utc).isoformat()

    payload = {
        "id": user_id,
        "username": clean_user,
        "display_name": disp,
        "is_synthetic": False,
        "archetype": "real_user",
        "last_active_at": now_iso
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=5)
        if r.status_code in [200, 201]:
            data = r.json()
            return data[0] if data else payload
    except Exception as e:
        print(f"Error al sincronizar usuario en Supabase: {e}")

    return payload

def load_interactions() -> pd.DataFrame:
    """
    Carga todas las interacciones desde Supabase (sintéticas y reales).
    Si falla la conexión, devuelve un DataFrame vacío con esquema intacto.
    """
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/user_interactions?select=user_id,track_id,rating,archetype,created_at&limit=30000"
    headers = _get_headers()

    try:
        r = requests.get(endpoint, headers=headers, timeout=8)
        if r.status_code == 200:
            data = r.json()
            if data:
                return pd.DataFrame(data)
    except Exception as e:
        print(f"Error al cargar interacciones de Supabase: {e}")

    return pd.DataFrame(columns=["user_id", "track_id", "rating", "archetype", "created_at"])

def load_real_interactions() -> pd.DataFrame:
    """Carga exclusivamente las interacciones de personas reales."""
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/user_interactions?archetype=eq.real_user&select=user_id,track_id,rating,archetype,created_at&order=created_at.desc"
    headers = _get_headers()

    try:
        r = requests.get(endpoint, headers=headers, timeout=5)
        if r.status_code == 200:
            data = r.json()
            if data:
                return pd.DataFrame(data)
    except Exception as e:
        print(f"Error al cargar interacciones reales: {e}")

    return pd.DataFrame(columns=["user_id", "track_id", "rating", "archetype", "created_at"])

def add_interaction(user_id: str, track_id: str, rating: float, archetype: str = "real_user", interaction_type: str = "like") -> bool:
    """
    Inserta o actualiza un voto (upsert) en la tabla 'user_interactions'.
    Retorna True si fue exitoso, False si ocurrió algún fallo de red.
    """
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/user_interactions"
    headers = _get_headers()
    headers["Prefer"] = "resolution=merge-duplicates"

    now_iso = datetime.now(timezone.utc).isoformat()
    payload = {
        "user_id": user_id,
        "track_id": track_id,
        "rating": float(rating),
        "archetype": archetype,
        "created_at": now_iso
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=5)
        return r.status_code in [200, 201, 204]
    except Exception as e:
        print(f"Error al guardar voto en Supabase: {e}")
        return False

def log_recommendation(user_id: str, seed_track_id: Optional[str], rec_row: Dict[str, Any], alpha: float, rank: int) -> bool:
    """Registra la auditoría de recomendaciones servidas al usuario."""
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/recommendation_history"
    headers = _get_headers()

    payload = {
        "user_id": user_id,
        "seed_track_id": seed_track_id,
        "recommended_track_id": rec_row.get("track_id"),
        "rank_position": rank,
        "alpha_weight": float(alpha),
        "content_score": float(rec_row.get("content_score_norm", 0.0)),
        "collab_score": float(rec_row.get("collab_score_norm", 0.0)),
        "hybrid_score": float(rec_row.get("hybrid_score", 0.0)),
        "source": rec_row.get("source", "hybrid"),
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=3)
        return r.status_code in [200, 201, 204]
    except Exception:
        return False

def log_chat_query(user_id: Optional[str], prompt: str, mood: Optional[str], response: str) -> bool:
    """Registra las consultas conversacionales con el DJ Virtual."""
    url, _ = get_supabase_config()
    endpoint = f"{url}/rest/v1/dj_chat_logs"
    headers = _get_headers()

    payload = {
        "user_id": user_id,
        "prompt": prompt,
        "mood_detected": mood,
        "bot_response": response[:1000] if response else "",
        "created_at": datetime.now(timezone.utc).isoformat()
    }

    try:
        r = requests.post(endpoint, headers=headers, json=payload, timeout=3)
        return r.status_code in [200, 201, 204]
    except Exception:
        return False
