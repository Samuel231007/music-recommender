import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import os

class DataPreprocessor:
    def __init__(self):
        self.scaler = MinMaxScaler()
        self.audio_cols = [
            "danceability", "energy", "valence", "tempo", 
            "acousticness", "instrumentalness", "speechiness", "loudness"
        ]

    def fit_transform_tracks(self, tracks_path: str, output_path: str) -> pd.DataFrame:
        """
        Carga, limpia y escala las características de audio del dataset de canciones.
        """
        if not os.path.exists(tracks_path):
            raise FileNotFoundError(f"Archivo de canciones no encontrado en {tracks_path}")
            
        df = pd.read_csv(tracks_path)
        
        # Eliminar duplicados de canciones por ID
        df = df.drop_duplicates(subset=["track_id"])
        
        # Escalar columnas de audio para que estén en el rango [0, 1]
        df_scaled_cols = pd.DataFrame(
            self.scaler.fit_transform(df[self.audio_cols]),
            columns=[f"scaled_{col}" for col in self.audio_cols],
            index=df.index
        )
        
        # Combinar columnas escaladas al dataframe original
        df_processed = pd.concat([df, df_scaled_cols], axis=1)
        
        # Crear carpeta de salida si no existe
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df_processed.to_csv(output_path, index=False)
        print(f"Canciones procesadas guardadas en '{output_path}'")
        return df_processed

    def prepare_interactions(self, interactions_path: str, output_path: str) -> pd.DataFrame:
        """
        Carga y valida las interacciones de usuario.
        """
        if not os.path.exists(interactions_path):
            raise FileNotFoundError(f"Archivo de interacciones no encontrado en {interactions_path}")
            
        df = pd.read_csv(interactions_path)
        
        # Limpieza básica
        df = df.dropna(subset=["user_id", "track_id", "rating"])
        df["rating"] = df["rating"].astype(float)
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Interacciones procesadas guardadas en '{output_path}'")
        return df

if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    base_dir = "C:/Users/s6535/.gemini/antigravity/scratch/music-recommender"
    preprocessor.fit_transform_tracks(
        os.path.join(base_dir, "data/raw/tracks.csv"),
        os.path.join(base_dir, "data/processed/tracks_processed.csv")
    )
    preprocessor.prepare_interactions(
        os.path.join(base_dir, "data/raw/interactions.csv"),
        os.path.join(base_dir, "data/processed/interactions_processed.csv")
    )
