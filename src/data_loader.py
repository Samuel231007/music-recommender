"""
data_loader.py
Carga y limpieza del dataset de Spotify.
"""
import pandas as pd
import numpy as np
from pathlib import Path

RAW_PATH = Path(__file__).parent.parent / "data" / "raw" / "train.csv"
PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed" / "tracks_clean.csv"

AUDIO_FEATURES = [
    "danceability", "energy", "loudness", "speechiness",
    "acousticness", "instrumentalness", "liveness", "valence", "tempo"
]

def load_raw(path: Path = RAW_PATH) -> pd.DataFrame:
    """Carga el CSV raw."""
    df = pd.read_csv(path)
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])
    return df

def clean(df: pd.DataFrame) -> pd.DataFrame:
    """Limpieza completa del dataset."""
    # Normalizar texto
    df["track_name"] = df["track_name"].str.strip().str.lower()
    df["artists"] = df["artists"].str.strip().str.lower()
    df["album_name"] = df["album_name"].str.strip().str.lower()
    df["track_genre"] = df["track_genre"].str.strip().str.lower()

    # Eliminar duplicados por track_id
    before = len(df)
    df = df.drop_duplicates(subset=["track_id"])
    print(f"Duplicados por track_id eliminados: {before - len(df)}")

    # Eliminar duplicados por nombre+artista (quedarse con mayor popularidad)
    df = df.sort_values("popularity", ascending=False)
    df = df.drop_duplicates(subset=["track_name", "artists"], keep="first")
    print(f"Duplicados por nombre+artista eliminados. Quedan: {len(df)} canciones")

    # Tratar nulos en audio features con mediana
    for col in AUDIO_FEATURES:
        if df[col].isna().any():
            median_val = df[col].median()
            df[col] = df[col].fillna(median_val)
            print(f"  Nulos en {col} imputados con mediana={median_val:.3f}")

    df = df.reset_index(drop=True)
    return df

def load_clean(force_reprocess: bool = False) -> pd.DataFrame:
    """Carga el dataset limpio (desde cache si existe)."""
    if PROCESSED_PATH.exists() and not force_reprocess:
        return pd.read_csv(PROCESSED_PATH)
    df = load_raw()
    df = clean(df)
    PROCESSED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(PROCESSED_PATH, index=False)
    print(f"Dataset limpio guardado en {PROCESSED_PATH}")
    return df

if __name__ == "__main__":
    df = load_clean(force_reprocess=True)
    print(f"\nResumen final:")
    print(f"  Canciones: {len(df)}")
    print(f"  Géneros: {df['track_genre'].nunique()}")
    print(f"  Artistas: {df['artists'].nunique()}")
    print(f"  Nulos restantes:\n{df[AUDIO_FEATURES].isna().sum()}")
