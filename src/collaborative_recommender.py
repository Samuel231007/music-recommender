import pandas as pd
import numpy as np
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
import os

class CollaborativeRecommender:
    def __init__(self, interactions_df: pd.DataFrame, tracks_df: pd.DataFrame):
        self.interactions_df = interactions_df.copy()
        self.tracks_df = tracks_df.copy()
        
        # Configurar Surprise Reader con escala de ratings de 1 a 5
        reader = Reader(rating_scale=(1.0, 5.0))
        self.data = Dataset.load_from_df(
            self.interactions_df[["user_id", "track_id", "rating"]], 
            reader
        )
        
        # Inicializar y entrenar el modelo SVD (Matrix Factorization)
        self.model = SVD(n_factors=50, random_state=42)
        self.trainset = self.data.build_full_trainset()
        self.model.fit(self.trainset)
        
        # Guardar lista de tracks e items para búsquedas rápidas
        self.all_track_ids = self.tracks_df["track_id"].unique().tolist()
        self.track_info_map = self.tracks_df.set_index("track_id").to_dict(orient="index")

    def predict_rating(self, user_id: str, track_id: str) -> float:
        """
        Predice el rating esperado para un usuario y una canción.
        """
        pred = self.model.predict(user_id, track_id)
        return float(pred.est)

    def recommend_for_user(self, user_id: str, top_k: int = 10, exclude_interacted: bool = True):
        """
        Predice ratings para todas las canciones que el usuario no ha escuchado,
        y devuelve las top K con el score estimado más alto.
        """
        # Canciones ya interactuadas por el usuario
        interacted_tracks = set()
        if exclude_interacted:
            interacted_tracks = set(
                self.interactions_df[self.interactions_df["user_id"] == user_id]["track_id"]
            )
            
        predictions = []
        for track_id in self.all_track_ids:
            if track_id in interacted_tracks:
                continue
                
            pred_rating = self.predict_rating(user_id, track_id)
            predictions.append((track_id, pred_rating))
            
        # Ordenar por el rating predicho descendente
        predictions = sorted(predictions, key=lambda x: x[1], reverse=True)[:top_k]
        
        recommendations = []
        for rank, (track_id, est_rating) in enumerate(predictions):
            info = self.track_info_map.get(track_id, {}).copy()
            info["track_id"] = track_id
            info["collaborative_score"] = est_rating
            info["rank"] = rank + 1
            recommendations.append(info)
            
        return pd.DataFrame(recommendations)
