"""
synthetic_users.py
Genera usuarios sintéticos con arquetipos de oyente y tabla de interacciones.
"""
import pandas as pd
import numpy as np
from pathlib import Path
from data_loader import load_clean, AUDIO_FEATURES

PROCESSED_PATH = Path(__file__).parent.parent / "data" / "processed"
INTERACTIONS_PATH = PROCESSED_PATH / "interactions.csv"

N_USERS = 500
ARCHETYPES = {
    "genre_fan":      0.30,  # Fan de 1-2 géneros
    "artist_follower": 0.20,  # Seguidor de artistas específicos
    "eclectic":        0.25,  # Oyente ecléctico
    "nostalgic":       0.15,  # Prefiere épocas antiguas
    "energetic":       0.10,  # Prefiere canciones con alta energía
}

SONGS_PER_USER_MIN = 20
SONGS_PER_USER_MAX = 80

def _rating(row, base_score: float) -> float:
    """Genera un rating implícito (0-5) con algo de ruido."""
    noise = np.random.normal(0, 0.3)
    return float(np.clip(base_score * 5 + noise, 0.5, 5.0))

def generate_interactions(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    np.random.seed(seed)
    genres = df["track_genre"].unique()
    artists = df["artists"].unique()
    rows = []

    n_per_archetype = {k: int(v * N_USERS) for k, v in ARCHETYPES.items()}
    # Ajustar para que sumen exactamente N_USERS
    diff = N_USERS - sum(n_per_archetype.values())
    n_per_archetype["eclectic"] += diff

    user_id = 0
    for archetype, n_users in n_per_archetype.items():
        for _ in range(n_users):
            n_songs = np.random.randint(SONGS_PER_USER_MIN, SONGS_PER_USER_MAX + 1)

            if archetype == "genre_fan":
                fav_genres = np.random.choice(genres, size=np.random.randint(1, 3), replace=False)
                pool = df[df["track_genre"].isin(fav_genres)]
                if len(pool) < n_songs:
                    pool = df
                scores = pool["popularity"] / 100.0

            elif archetype == "artist_follower":
                fav_artists = np.random.choice(artists, size=np.random.randint(3, 8), replace=False)
                pool = df[df["artists"].isin(fav_artists)]
                if len(pool) < n_songs:
                    pool = df
                scores = pool["popularity"] / 100.0

            elif archetype == "eclectic":
                pool = df
                scores = pd.Series(np.ones(len(df)), index=df.index)

            elif archetype == "nostalgic":
                # Simulamos "antiguas" usando baja popularidad como proxy
                # (el dataset no tiene año, usamos acousticness alta como proxy)
                pool = df[df["acousticness"] > 0.4]
                if len(pool) < n_songs:
                    pool = df
                scores = pool["acousticness"]

            elif archetype == "energetic":
                pool = df[(df["energy"] > 0.6) & (df["danceability"] > 0.5)]
                if len(pool) < n_songs:
                    pool = df
                scores = (pool["energy"] + pool["danceability"]) / 2.0

            # Muestrear canciones ponderadas por score usando numpy (evita bug de pandas)
            pool = pool.copy()
            scores = scores.reindex(pool.index).fillna(0.5)
            raw_weights = np.clip(scores.values.astype(float), 1e-9, None)
            probs = raw_weights / raw_weights.sum()
            n_songs = min(n_songs, len(pool))
            chosen_indices = np.random.choice(len(pool), size=n_songs, replace=False, p=probs)
            selected = pool.iloc[chosen_indices]

            for _, song in selected.iterrows():
                base_score = scores.loc[song.name] if song.name in scores.index else 0.5
                rows.append({
                    "user_id": f"user_{user_id:04d}",
                    "archetype": archetype,
                    "track_id": song["track_id"],
                    "rating": _rating(song, base_score)
                })

            user_id += 1

    interactions = pd.DataFrame(rows)
    PROCESSED_PATH.mkdir(parents=True, exist_ok=True)
    interactions.to_csv(INTERACTIONS_PATH, index=False)
    print(f"Interacciones generadas: {len(interactions)} filas, {interactions['user_id'].nunique()} usuarios")
    return interactions

if __name__ == "__main__":
    df = load_clean()
    interactions = generate_interactions(df)
    print(interactions.groupby("archetype")["user_id"].nunique())
