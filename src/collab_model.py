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
    Para un user_id dado, predice el rating de manera vectorizada
    (instantánea) y devuelve las top N canciones recomendadas.
    """
    listened = set(interactions[interactions["user_id"] == user_id]["track_id"].values)

    try:
        inner_uid = model.trainset.to_inner_uid(user_id)
        # Vectorized SVD: mu + bu + bi + (qi . pu)
        inner_scores = (
            model.trainset.global_mean
            + model.bu[inner_uid]
            + model.bi
            + np.dot(model.qi, model.pu[inner_uid])
        )
        
        # Mapear a raw track_ids
        raw_items = [model.trainset.to_raw_iid(i) for i in range(len(inner_scores))]
        
        # Filtrar ya escuchadas
        candidates = [(tid, score) for tid, score in zip(raw_items, inner_scores) if tid not in listened]
        candidates.sort(key=lambda x: x[1], reverse=True)
        top = candidates[:n]
        
    except (ValueError, KeyError):
        # Cold-start / usuario nuevo: usar popularidad
        unheard = df_tracks[~df_tracks["track_id"].isin(listened)]
        top_popular = unheard.sort_values("popularity", ascending=False).head(n)
        top = [(r["track_id"], r["popularity"] / 20.0) for _, r in top_popular.iterrows()]

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
