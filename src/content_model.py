"""
content_model.py
Modelo de recomendación basado en contenido (similitud coseno sobre audio features).
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))
from data_loader import load_clean, AUDIO_FEATURES

def build_feature_matrix(df: pd.DataFrame) -> np.ndarray:
    """Normaliza las audio features y devuelve la matriz."""
    scaler = MinMaxScaler()
    matrix = scaler.fit_transform(df[AUDIO_FEATURES].values)
    return matrix

def get_content_recommendations(
    track_id: str,
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    n: int = 10
) -> pd.DataFrame:
    """
    Dado un track_id, devuelve las N canciones más similares
    usando similitud coseno sobre audio features.
    """
    if track_id not in df["track_id"].values:
        raise ValueError(f"track_id '{track_id}' no encontrado en el dataset.")

    idx = df.index[df["track_id"] == track_id][0]
    song_vector = feature_matrix[idx].reshape(1, -1)

    # Similitud solo contra la canción de consulta (lazy — no pre-calcula toda la matriz)
    similarities = cosine_similarity(song_vector, feature_matrix)[0]

    # Excluir la canción de consulta
    similarities[idx] = -1

    top_indices = np.argsort(similarities)[::-1][:n]
    results = df.iloc[top_indices].copy()
    results["content_score"] = similarities[top_indices]
    results["source"] = "content"

    return results[["track_id", "track_name", "artists", "track_genre",
                     "popularity", "content_score", "source"]].reset_index(drop=True)

if __name__ == "__main__":
    df = load_clean()
    matrix = build_feature_matrix(df)

    # Prueba manual
    sample_track = df.iloc[0]
    print(f"Canción de consulta: {sample_track['track_name']} — {sample_track['artists']}")
    print(f"Género: {sample_track['track_genre']}\n")

    recs = get_content_recommendations(sample_track["track_id"], df, matrix, n=5)
    print("Top 5 recomendaciones basadas en contenido:")
    print(recs[["track_name", "artists", "track_genre", "content_score"]].to_string(index=False))
