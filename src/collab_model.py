"""
collab_model.py
Modelo de filtrado colaborativo por factorización de matrices (SVD).
Usa scikit-surprise para entrenamiento y predicción.
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split

MODEL_PATH = Path(__file__).parent.parent / "data" / "processed" / "svd_model.pkl"
INTERACTIONS_PATH = Path(__file__).parent.parent / "data" / "processed" / "interactions.csv"

def train_svd(interactions: pd.DataFrame, n_factors: int = 50, n_epochs: int = 20) -> SVD:
    """Entrena el modelo SVD sobre la tabla de interacciones."""
    reader = Reader(rating_scale=(0.5, 5.0))
    data = Dataset.load_from_df(
        interactions[["user_id", "track_id", "rating"]], reader
    )
    trainset, testset = train_test_split(data, test_size=0.15, random_state=42)

    model = SVD(n_factors=n_factors, n_epochs=n_epochs, random_state=42, verbose=False)
    model.fit(trainset)

    predictions = model.test(testset)
    rmse = accuracy.rmse(predictions, verbose=False)
    print(f"SVD entrenado — RMSE en test: {rmse:.4f}")

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Modelo guardado en {MODEL_PATH}")
    return model

def load_model() -> SVD:
    """Carga el modelo SVD desde disco."""
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Modelo no encontrado en {MODEL_PATH}. Ejecuta train_svd primero.")
    return joblib.load(MODEL_PATH)

def get_collab_recommendations(
    user_id: str,
    df_tracks: pd.DataFrame,
    interactions: pd.DataFrame,
    model: SVD,
    n: int = 10
) -> pd.DataFrame:
    """
    Para un user_id dado, predice el rating para todas las canciones
    que aún no ha escuchado y devuelve las top N.
    """
    listened = set(interactions[interactions["user_id"] == user_id]["track_id"].values)
    all_tracks = df_tracks["track_id"].values
    unlistened = [t for t in all_tracks if t not in listened]

    predictions = [(t, model.predict(user_id, t).est) for t in unlistened]
    predictions.sort(key=lambda x: x[1], reverse=True)
    top = predictions[:n]

    top_ids = [p[0] for p in top]
    top_scores = {p[0]: p[1] for p in top}

    results = df_tracks[df_tracks["track_id"].isin(top_ids)].copy()
    results["collab_score"] = results["track_id"].map(top_scores)
    results["source"] = "collab"

    return results[["track_id", "track_name", "artists", "track_genre",
                     "popularity", "collab_score", "source"]].sort_values(
        "collab_score", ascending=False
    ).reset_index(drop=True)

if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from data_loader import load_clean
    from synthetic_users import generate_interactions

    df = load_clean()
    interactions = generate_interactions(df)
    model = train_svd(interactions)

    # Prueba con primer usuario
    test_user = interactions["user_id"].iloc[0]
    archetype = interactions[interactions["user_id"] == test_user]["archetype"].iloc[0]
    print(f"\nRecomendaciones para {test_user} (arquetipo: {archetype}):")
    recs = get_collab_recommendations(test_user, df, interactions, model, n=5)
    print(recs[["track_name", "artists", "track_genre", "collab_score"]].to_string(index=False))
