import pandas as pd
import numpy as np
try:
    from src.content_recommender import ContentBasedRecommender
    from src.collaborative_recommender import CollaborativeRecommender
except ModuleNotFoundError:
    from content_recommender import ContentBasedRecommender
    from collaborative_recommender import CollaborativeRecommender

class HybridRecommender:
    def __init__(self, content_rec: ContentBasedRecommender, collab_rec: CollaborativeRecommender):
        self.content_rec = content_rec
        self.collab_rec = collab_rec
        self.tracks_df = content_rec.df.copy()
        
    def recommend(self, user_id: str, alpha: float = 0.5, top_k: int = 10):
        """
        Combina el filtrado basado en contenido y el colaborativo.
        alpha: peso del modelo basado en contenido [0.0, 1.0].
        (1 - alpha) será el peso del modelo colaborativo.
        """
        # 1. Obtener interacciones de este usuario
        user_interactions = self.collab_rec.interactions_df[
            self.collab_rec.interactions_df["user_id"] == user_id
        ]
        
        # Si el usuario no existe en interacciones (Cold Start), usamos 100% Contenido
        if user_interactions.empty:
            print(f"Usuario {user_id} es nuevo. Aplicando estrategia Cold Start basada únicamente en popularidad y contenido.")
            # Buscar canciones populares como seed
            top_pop = self.tracks_df.sort_values(by="popularity", ascending=False).head(5)
            seed_tracks = top_pop["track_id"].tolist()
            # Generar por contenido basado en estas populares
            recs = self.content_rec.recommend_for_user_profile(
                pd.DataFrame(columns=["user_id", "track_id", "rating"]), 
                seed_tracks, 
                top_k=top_k
            )
            recs["source"] = "Cold Start (Content-Based)"
            return recs

        # 2. Obtener recomendaciones de Contenido y Colaborativo
        # Para contenido, creamos el perfil del usuario a partir de sus likes pasados
        df_content_recs = self.content_rec.recommend_for_user_profile(user_interactions, [], top_k=50)
        # Para colaborativo, obtenemos predicciones para el catálogo completo
        df_collab_recs = self.collab_rec.recommend_for_user(user_id, top_k=50)

        # 3. Normalizar scores de ambos recomendadores al rango [0, 1] para combinarlos limpiamente
        # Contenido: similarity_score (ya está aprox entre 0 y 1 o podemos normalizarlo)
        if not df_content_recs.empty:
            min_c, max_c = df_content_recs["similarity_score"].min(), df_content_recs["similarity_score"].max()
            denom_c = (max_c - min_c) if (max_c - min_c) > 0 else 1
            df_content_recs["norm_content_score"] = (df_content_recs["similarity_score"] - min_c) / denom_c
        else:
            df_content_recs["norm_content_score"] = 0.0

        # Colaborativo: collaborative_score (ratings estimados, ej 1.0 a 5.0)
        if not df_collab_recs.empty:
            min_col, max_col = df_collab_recs["collaborative_score"].min(), df_collab_recs["collaborative_score"].max()
            denom_col = (max_col - min_col) if (max_col - min_col) > 0 else 1
            df_collab_recs["norm_collab_score"] = (df_collab_recs["collaborative_score"] - min_col) / denom_col
        else:
            df_collab_recs["norm_collab_score"] = 0.0

        # 4. Mezclar scores
        # Mapeamos track_id -> score
        content_scores = dict(zip(df_content_recs["track_id"], df_content_recs["norm_content_score"]))
        collab_scores = dict(zip(df_collab_recs["track_id"], df_collab_recs["norm_collab_score"]))
        
        # Todos los tracks candidatos
        candidate_ids = set(content_scores.keys()).union(set(collab_scores.keys()))
        
        hybrid_results = []
        for track_id in candidate_ids:
            c_score = content_scores.get(track_id, 0.0)
            col_score = collab_scores.get(track_id, 0.0)
            
            # Combinación lineal con el peso alpha
            final_score = alpha * c_score + (1 - alpha) * col_score
            
            hybrid_results.append({
                "track_id": track_id,
                "hybrid_score": float(final_score),
                "content_score": float(c_score),
                "collaborative_score": float(col_score)
            })
            
        df_hybrid = pd.DataFrame(hybrid_results)
        df_hybrid = df_hybrid.sort_values(by="hybrid_score", ascending=False).head(top_k)
        
        # Enriquecer con metadatos de las canciones
        track_info_map = self.tracks_df.set_index("track_id").to_dict(orient="index")
        
        detailed_recs = []
        for rank, row in enumerate(df_hybrid.to_dict(orient="records")):
            tid = row["track_id"]
            info = track_info_map.get(tid, {}).copy()
            info.update(row)
            info["rank"] = rank + 1
            info["source"] = "Hybrid"
            detailed_recs.append(info)
            
        return pd.DataFrame(detailed_recs)
