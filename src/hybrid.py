"""
hybrid.py
Combina recomendaciones basadas en contenido y colaborativas con peso ajustable.
alpha = 1.0 → puro contenido
alpha = 0.0 → puro colaborativo
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from data_loader import load_clean, AUDIO_FEATURES
from content_model import build_feature_matrix, get_content_recommendations
from collab_model import load_model, get_collab_recommendations

def _normalize_scores(series: pd.Series) -> pd.Series:
    """Normaliza scores a [0, 1]."""
    min_v, max_v = series.min(), series.max()
    if max_v == min_v:
        return pd.Series(np.ones(len(series)), index=series.index)
    return (series - min_v) / (max_v - min_v)

def get_hybrid_recommendations(
    track_id: str,
    user_id: str | None,
    df: pd.DataFrame,
    feature_matrix: np.ndarray,
    interactions: pd.DataFrame,
    model,
    alpha: float = 0.5,
    n: int = 10,
    n_candidates: int = 50
) -> pd.DataFrame:
    """
    Combina recomendaciones de contenido y colaborativas.

    Args:
        track_id: Canción de consulta para el modelo de contenido.
        user_id: Usuario para el modelo colaborativo. Si es None, solo usa contenido.
        alpha: Peso del modelo de contenido [0, 1].
        n: Número de recomendaciones finales.
        n_candidates: Pool de candidatos de cada modelo antes de combinar.
    """
    # --- Contenido ---
    content_recs = get_content_recommendations(track_id, df, feature_matrix, n=n_candidates)
    content_recs["content_score_norm"] = _normalize_scores(content_recs["content_score"])

    # --- Colaborativo (si hay usuario) ---
    if user_id is not None and alpha < 1.0:
        collab_recs = get_collab_recommendations(user_id, df, interactions, model, n=n_candidates)
        collab_recs["collab_score_norm"] = _normalize_scores(collab_recs["collab_score"])
    else:
        collab_recs = pd.DataFrame(columns=["track_id", "collab_score_norm"])
        collab_recs["collab_score_norm"] = pd.Series(dtype=float)

    # --- Merge y combinación ---
    merged = content_recs[["track_id", "track_name", "artists", "track_genre",
                             "popularity", "content_score_norm"]].copy()
    if not collab_recs.empty:
        collab_subset = collab_recs[["track_id", "collab_score_norm"]]
        merged = merged.merge(collab_subset, on="track_id", how="outer")
    else:
        merged["collab_score_norm"] = 0.0

    merged["content_score_norm"] = merged["content_score_norm"].fillna(0.0)
    merged["collab_score_norm"] = merged["collab_score_norm"].fillna(0.0)

    # Llenar metadata faltante del merge outer
    missing_mask = merged["track_name"].isna()
    if missing_mask.any():
        missing_ids = merged[missing_mask]["track_id"].values
        fill_data = df[df["track_id"].isin(missing_ids)][
            ["track_id", "track_name", "artists", "track_genre", "popularity"]
        ]
        merged = merged.merge(fill_data, on="track_id", how="left", suffixes=("", "_fill"))
        for col in ["track_name", "artists", "track_genre", "popularity"]:
            fill_col = f"{col}_fill"
            if fill_col in merged.columns:
                merged[col] = merged[col].fillna(merged[fill_col])
                merged = merged.drop(columns=[fill_col])

    # Score híbrido
    merged["hybrid_score"] = alpha * merged["content_score_norm"] + (1 - alpha) * merged["collab_score_norm"]

    # Excluir la canción de consulta
    merged = merged[merged["track_id"] != track_id]

    # Determinar fuente dominante
    def _source(row):
        if row["content_score_norm"] > row["collab_score_norm"]:
            return "content"
        elif row["collab_score_norm"] > row["content_score_norm"]:
            return "collab"
        return "hybrid"

    merged["source"] = merged.apply(_source, axis=1)

    result = merged.sort_values("hybrid_score", ascending=False).head(n)
    return result[["track_id", "track_name", "artists", "track_genre",
                    "popularity", "hybrid_score", "content_score_norm",
                    "collab_score_norm", "source"]].reset_index(drop=True)

if __name__ == "__main__":
    from synthetic_users import generate_interactions

    df = load_clean()
    matrix = build_feature_matrix(df)
    interactions = generate_interactions(df)
    model = load_model()

    sample = df.iloc[100]
    test_user = interactions["user_id"].iloc[0]
    print(f"Consulta: {sample['track_name']} — {sample['artists']}")
    print(f"Usuario: {test_user} | alpha=0.5\n")

    recs = get_hybrid_recommendations(
        track_id=sample["track_id"],
        user_id=test_user,
        df=df,
        feature_matrix=matrix,
        interactions=interactions,
        model=model,
        alpha=0.5,
        n=5
    )
    print(recs[["track_name", "artists", "hybrid_score", "source"]].to_string(index=False))
