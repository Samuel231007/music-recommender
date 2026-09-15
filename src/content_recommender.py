import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

class ContentBasedRecommender:
    def __init__(self, tracks_df: pd.DataFrame):
        self.df = tracks_df.copy()
        self.feature_cols = [
            "scaled_danceability", "scaled_energy", "scaled_valence", "scaled_tempo",
            "scaled_acousticness", "scaled_instrumentalness", "scaled_speechiness", "scaled_loudness"
        ]
        # Generar matriz de características
        self.features_matrix = self.df[self.feature_cols].values
        
        # OHE para géneros y agregarlo a la matriz de características con un peso
        genres_encoded = pd.get_dummies(self.df["genre"]).values
        # Concatenar características de audio con géneros (dando un peso extra al género de 1.5)
        self.full_features = np.hstack([self.features_matrix, genres_encoded * 1.5])
        
        # Calcular similitud coseno entre todas las canciones
        self.similarity_matrix = cosine_similarity(self.full_features)
        
        # Mapeo rápido de track_id a índice
        self.track_to_idx = {row["track_id"]: idx for idx, row in self.df.iterrows()}
        self.idx_to_track = {idx: row["track_id"] for idx, row in self.df.iterrows()}

    def recommend_by_track(self, track_id: str, top_k: int = 10):
        """
        Recomienda canciones similares a una canción semilla basándose en similitud de contenido.
        """
        if track_id not in self.track_to_idx:
            return pd.DataFrame()
            
        idx = self.track_to_idx[track_id]
        
        # Obtener los scores de similitud para esta canción
        sim_scores = list(enumerate(self.similarity_matrix[idx]))
        
        # Ordenar de mayor a menor similitud (excluyendo la propia canción que tiene score 1.0 en idx)
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [item for item in sim_scores if item[0] != idx][:top_k]
        
        recommendations = []
        for rank, (sim_idx, score) in enumerate(sim_scores):
            track_info = self.df.iloc[sim_idx].to_dict()
            track_info["similarity_score"] = float(score)
            track_info["rank"] = rank + 1
            recommendations.append(track_info)
            
        return pd.DataFrame(recommendations)

    def recommend_for_user_profile(self, user_interactions: pd.DataFrame, track_ids_list: list, top_k: int = 10):
        """
        Genera recomendaciones de contenido para un usuario construyendo un perfil vectorial
        basado en el promedio ponderado de las canciones que le han gustado (rating >= 3).
        """
        liked_tracks = user_interactions[user_interactions["rating"] >= 3.0]
        if liked_tracks.empty:
            # Fallback: canciones populares
            return self.df.sort_values(by="popularity", ascending=False).head(top_k)
            
        # Obtener los índices de las canciones que le gustaron al usuario
        liked_indices = [self.track_to_idx[tid] for tid in liked_tracks["track_id"] if tid in self.track_to_idx]
        
        if not liked_indices:
            return self.df.sort_values(by="popularity", ascending=False).head(top_k)
            
        # Pesos basados en los ratings
        ratings = liked_tracks[liked_tracks["track_id"].isin(self.track_to_idx.keys())]["rating"].values
        ratings_normalized = ratings / np.sum(ratings)
        
        # Perfil promedio del usuario
        user_profile = np.dot(ratings_normalized, self.full_features[liked_indices])
        user_profile = user_profile.reshape(1, -1)
        
        # Calcular similitud del perfil del usuario con todas las canciones
        sim_scores = cosine_similarity(user_profile, self.full_features).flatten()
        
        # Filtrar canciones que el usuario ya escuchó
        exclude_ids = set(liked_tracks["track_id"])
        
        sim_scores_ranked = sorted(
            [(idx, score) for idx, score in enumerate(sim_scores) if self.idx_to_track[idx] not in exclude_ids],
            key=lambda x: x[1], reverse=True
        )[:top_k]
        
        recommendations = []
        for rank, (sim_idx, score) in enumerate(sim_scores_ranked):
            track_info = self.df.iloc[sim_idx].to_dict()
            track_info["similarity_score"] = float(score)
            track_info["rank"] = rank + 1
            recommendations.append(track_info)
            
        return pd.DataFrame(recommendations)
