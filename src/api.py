from fastapi import FastAPI, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
import pandas as pd
import uvicorn
import os

try:
    from src.preprocessor import DataPreprocessor
    from src.content_recommender import ContentBasedRecommender
    from src.collaborative_recommender import CollaborativeRecommender
    from src.hybrid_recommender import HybridRecommender
except ModuleNotFoundError:
    from preprocessor import DataPreprocessor
    from content_recommender import ContentBasedRecommender
    from collaborative_recommender import CollaborativeRecommender
    from hybrid_recommender import HybridRecommender

app = FastAPI(
    title="Spotify-style Recommendation API",
    description="API híbrida de recomendación musical que combina filtrado colaborativo e inferencias basadas en contenido de audio.",
    version="1.0.0"
)

# Permitir CORS para desarrollo local cómodo
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicializar datasets y recomendadores
BASE_DIR = "C:/Users/s6535/.gemini/antigravity/scratch/music-recommender"
PROCESSED_TRACKS = os.path.join(BASE_DIR, "data/processed/tracks_processed.csv")
PROCESSED_INTERACTIONS = os.path.join(BASE_DIR, "data/processed/interactions_processed.csv")

# Instanciar en memoria al arrancar la app
tracks_df = pd.read_csv(PROCESSED_TRACKS)
interactions_df = pd.read_csv(PROCESSED_INTERACTIONS)

content_recommender = ContentBasedRecommender(tracks_df)
collaborative_recommender = CollaborativeRecommender(interactions_df, tracks_df)
hybrid_recommender = HybridRecommender(content_recommender, collaborative_recommender)

@app.get("/api/tracks")
def get_tracks(limit: int = 50):
    """Obtiene una muestra de canciones populares para mostrar en la interfaz."""
    sample = tracks_df.sort_values(by="popularity", ascending=False).head(limit)
    return sample.to_dict(orient="records")

@app.get("/api/users")
def get_users():
    """Obtiene el listado de IDs de usuarios del dataset."""
    users = sorted(interactions_df["user_id"].unique().tolist())
    return {"users": users}

@app.get("/api/recommend/content")
def recommend_content(track_id: str, limit: int = 10):
    """Recomendaciones basadas puramente en similitud de audio features (Contenido)."""
    recs = content_recommender.recommend_by_track(track_id, top_k=limit)
    return recs.to_dict(orient="records")

@app.get("/api/recommend/collaborative")
def recommend_collab(user_id: str, limit: int = 10):
    """Recomendaciones basadas en interacciones de usuarios similares (Colaborativo)."""
    recs = collaborative_recommender.recommend_for_user(user_id, top_k=limit)
    return recs.to_dict(orient="records")

@app.get("/api/recommend/hybrid")
def recommend_hybrid(user_id: str, alpha: float = Query(0.5, ge=0.0, le=1.0), limit: int = 10):
    """Recomendaciones híbridas personalizadas combinando ambos enfoques."""
    recs = hybrid_recommender.recommend(user_id, alpha=alpha, top_k=limit)
    return recs.to_dict(orient="records")

# Montar frontend para servir los archivos de la interfaz gráfica
frontend_dir = os.path.join(BASE_DIR, "frontend")
if os.path.exists(frontend_dir):
    app.mount("/static", StaticFiles(directory=frontend_dir), name="static")
    
    @app.get("/")
    def read_index():
        return FileResponse(os.path.join(frontend_dir, "index.html"))

if __name__ == "__main__":
    uvicorn.run("api:app", host="127.0.0.1", port=8000, reload=True)
