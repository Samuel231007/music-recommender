import pandas as pd
import numpy as np
import os
import json

def generate_synthetic_data(output_dir: str, num_tracks: int = 1000, num_users: int = 200, num_interactions: int = 15000):
    """
    Genera un conjunto de datos sintético y coherente de canciones e interacciones de usuario.
    Las canciones contienen características de audio similares a la API de Spotify.
    Las interacciones se basan en perfiles de afinidad de usuario para simular patrones reales.
    """
    os.makedirs(output_dir, exist_ok=True)
    np.random.seed(42)

    # 1. Definir géneros y sus perfiles de audio típicos
    genres_profiles = {
        "Rock": {"danceability": (0.3, 0.6), "energy": (0.6, 0.9), "valence": (0.3, 0.7), "tempo": (110, 160), "acousticness": (0.0, 0.3)},
        "Pop": {"danceability": (0.6, 0.85), "energy": (0.5, 0.85), "valence": (0.5, 0.9), "tempo": (95, 130), "acousticness": (0.05, 0.4)},
        "Jazz": {"danceability": (0.4, 0.7), "energy": (0.2, 0.5), "valence": (0.3, 0.6), "tempo": (70, 110), "acousticness": (0.5, 0.9)},
        "Classical": {"danceability": (0.1, 0.4), "energy": (0.05, 0.35), "valence": (0.05, 0.4), "tempo": (60, 120), "acousticness": (0.8, 1.0)},
        "Electronic": {"danceability": (0.7, 0.9), "energy": (0.7, 0.95), "valence": (0.4, 0.8), "tempo": (120, 140), "acousticness": (0.0, 0.2)},
        "Acoustic/Folk": {"danceability": (0.4, 0.65), "energy": (0.2, 0.55), "valence": (0.3, 0.7), "tempo": (80, 120), "acousticness": (0.6, 0.9)}
    }

    genres = list(genres_profiles.keys())
    artists_by_genre = {
        "Rock": ["The Rolling Stones", "AC/DC", "Led Zeppelin", "Queen", "Muse", "Foo Fighters"],
        "Pop": ["Dua Lipa", "Taylor Swift", "The Weeknd", "Billie Eilish", "Ed Sheeran", "Ariana Grande"],
        "Jazz": ["Miles Davis", "John Coltrane", "Ella Fitzgerald", "Bill Evans", "Norah Jones"],
        "Classical": ["Ludwig van Beethoven", "Wolfgang Amadeus Mozart", "Johann Sebastian Bach", "Frédéric Chopin"],
        "Electronic": ["Daft Punk", "Deadmau5", "Disclosure", "Avicii", "Calvin Harris", "Flume"],
        "Acoustic/Folk": ["Mumford & Sons", "Iron & Wine", "Bon Iver", "The Lumineers", "Fleet Foxes"]
    }

    # 2. Generar Canciones
    tracks_data = []
    for i in range(num_tracks):
        track_id = f"track_{i:04d}"
        genre = np.random.choice(genres)
        artist = np.random.choice(artists_by_genre[genre])
        title = f"{genre} Song #{np.random.randint(1, 100)}"
        
        # Audio features segun perfil
        prof = genres_profiles[genre]
        danceability = np.random.uniform(*prof["danceability"])
        energy = np.random.uniform(*prof["energy"])
        valence = np.random.uniform(*prof["valence"])
        tempo = np.random.uniform(*prof["tempo"])
        acousticness = np.random.uniform(*prof["acousticness"])
        instrumentalness = np.random.uniform(0.0, 0.2) if genre not in ["Classical", "Electronic", "Jazz"] else np.random.uniform(0.5, 0.95)
        speechiness = np.random.uniform(0.02, 0.12)
        loudness = np.random.uniform(-15.0, -3.0)
        
        tracks_data.append({
            "track_id": track_id,
            "track_name": title,
            "artist_name": artist,
            "genre": genre,
            "danceability": round(danceability, 4),
            "energy": round(energy, 4),
            "valence": round(valence, 4),
            "tempo": round(tempo, 1),
            "acousticness": round(acousticness, 4),
            "instrumentalness": round(instrumentalness, 4),
            "speechiness": round(speechiness, 4),
            "loudness": round(loudness, 2),
            "popularity": np.random.randint(10, 100)
        })
        
    df_tracks = pd.DataFrame(tracks_data)
    df_tracks.to_csv(os.path.join(output_dir, "tracks.csv"), index=False)
    print(f"Generadas {num_tracks} canciones en 'tracks.csv'")

    # 3. Generar Usuarios y sus Perfiles de Interés
    users_profiles = []
    for u in range(num_users):
        user_id = f"user_{u:04d}"
        # Cada usuario tiene un género favorito principal y uno secundario
        fav_genres = np.random.choice(genres, size=2, replace=False)
        users_profiles.append({
            "user_id": user_id,
            "primary_genre": fav_genres[0],
            "secondary_genre": fav_genres[1]
        })
    
    # 4. Generar Interacciones (Likes, Escuchas) con sesgo según afinidad
    interactions = []
    # Indexar canciones por género para acceso rápido
    tracks_by_genre = {g: df_tracks[df_tracks["genre"] == g]["track_id"].tolist() for g in genres}
    
    for u_prof in users_profiles:
        u_id = u_prof["user_id"]
        # Número de interacciones de este usuario
        n_inter = np.random.randint(20, 120)
        
        # Probabilidades de interactuar con géneros
        p_genres = []
        for g in genres:
            if g == u_prof["primary_genre"]:
                p_genres.append(0.50)
            elif g == u_prof["secondary_genre"]:
                p_genres.append(0.30)
            else:
                p_genres.append(0.20 / (len(genres) - 2))
                
        # Normalizar probabilidades
        p_genres = np.array(p_genres) / sum(p_genres)
        
        interacted_tracks = set()
        for _ in range(n_inter):
            # Elegir género basado en preferencias
            chosen_genre = np.random.choice(genres, p=p_genres)
            # Elegir canción aleatoria de ese género
            possible_tracks = tracks_by_genre[chosen_genre]
            if not possible_tracks:
                continue
            track_id = np.random.choice(possible_tracks)
            
            if track_id in interacted_tracks:
                continue
            interacted_tracks.add(track_id)
            
            # Simular rating (ej. 1 a 5 estrellas)
            # Damos más probabilidad de ratings altos si es de su género primario
            if chosen_genre == u_prof["primary_genre"]:
                rating = np.random.choice([3, 4, 5], p=[0.1, 0.3, 0.6])
            elif chosen_genre == u_prof["secondary_genre"]:
                rating = np.random.choice([2, 3, 4, 5], p=[0.1, 0.2, 0.4, 0.3])
            else:
                rating = np.random.choice([1, 2, 3, 4, 5], p=[0.2, 0.3, 0.3, 0.15, 0.05])
                
            interactions.append({
                "user_id": u_id,
                "track_id": track_id,
                "rating": int(rating),
                "play_count": int(np.random.geometric(p=0.3) if rating >= 3 else np.random.randint(1, 3))
            })
            
    df_interactions = pd.DataFrame(interactions)
    df_interactions.to_csv(os.path.join(output_dir, "interactions.csv"), index=False)
    print(f"Generadas {len(df_interactions)} interacciones de usuario en 'interactions.csv'")

if __name__ == "__main__":
    generate_synthetic_data("C:/Users/s6535/.gemini/antigravity/scratch/music-recommender/data/raw")
