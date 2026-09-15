"""
chatbot.py
Motor de Inteligencia Artificial para el Asistente Musical / DJ Virtual (BeatBot).
Interpreta intenciones en lenguaje natural, analiza estados de ánimo y actividades,
y ejecuta búsquedas en el espacio vectorial acústico y colaborativo.
"""
import re
import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Any

MOOD_VECTORS = {
    "entrenar": {"energy": (0.75, 1.0), "danceability": (0.6, 1.0), "tempo": (120, 190), "valence": (0.5, 1.0)},
    "ejercicio": {"energy": (0.75, 1.0), "danceability": (0.6, 1.0), "tempo": (120, 190), "valence": (0.5, 1.0)},
    "gym": {"energy": (0.8, 1.0), "danceability": (0.6, 1.0), "tempo": (125, 200)},
    "estudiar": {"acousticness": (0.4, 1.0), "energy": (0.0, 0.45), "instrumentalness": (0.2, 1.0), "speechiness": (0.0, 0.15)},
    "concentracion": {"acousticness": (0.4, 1.0), "energy": (0.0, 0.45), "instrumentalness": (0.3, 1.0)},
    "relax": {"valence": (0.4, 0.8), "energy": (0.0, 0.4), "acousticness": (0.5, 1.0)},
    "dormir": {"energy": (0.0, 0.25), "acousticness": (0.7, 1.0), "loudness": (-40, -15)},
    "fiesta": {"danceability": (0.75, 1.0), "energy": (0.7, 1.0), "valence": (0.6, 1.0)},
    "bailar": {"danceability": (0.8, 1.0), "energy": (0.65, 1.0)},
    "triste": {"valence": (0.0, 0.35), "energy": (0.0, 0.45), "acousticness": (0.3, 1.0)},
    "depresion": {"valence": (0.0, 0.3), "energy": (0.0, 0.4)},
    "feliz": {"valence": (0.7, 1.0), "energy": (0.6, 1.0), "danceability": (0.6, 1.0)},
    "alegre": {"valence": (0.7, 1.0), "energy": (0.6, 1.0)},
    "motivacion": {"energy": (0.7, 1.0), "valence": (0.6, 1.0)},
    "viajar": {"valence": (0.5, 0.9), "energy": (0.5, 0.85), "danceability": (0.5, 0.85)},
    "romantico": {"valence": (0.4, 0.7), "acousticness": (0.3, 0.9), "energy": (0.2, 0.6)}
}

SYSTEM_EXPLANATIONS = {
    "svd": (
        "🔍 **Factorización de Matrices SVD en BeatMatch:**\n\n"
        "El modelo colaborativo descompone la matriz usuario-canción $R$ en tres matrices $U \\Sigma V^T$ "
        "con **50 factores latentes**. Esto permite predecir la calificación implícita que un usuario le daría "
        "a una pista que nunca ha escuchado: $\\hat{r}_{ui} = \\mu + b_u + b_i + \\mathbf{q}_i^T \\mathbf{p}_u$."
    ),
    "contenido": (
        "🎵 **Filtrado Basado en Contenido:**\n\n"
        "Compara el vector acústico de 9 dimensiones (*danceability, energy, acousticness, valence, etc.*) "
        "de la pista de origen contra el catálogo entero usando **Similitud Coseno**. Si dos canciones tienen "
        "un perfil sonoro similar, tendrán una afinidad cercana a 1.0 independientemente de si los usuarios las conocen."
    ),
    "hibrido": (
        "⚡ **Fusión Híbrida Ponderada:**\n\n"
        "Combina lo mejor de ambos mundos mediante el parámetro $\\alpha$:\n"
        "$$\\text{Score Final} = \\alpha \\cdot \\text{Score Contenido} + (1 - \\alpha) \\cdot \\text{Score Colaborativo}$$\n"
        "- $\\alpha = 1.0$: Emparejamiento por ADN acústico.\n"
        "- $\\alpha = 0.0$: Tendencias de la comunidad con gustos similares.\n"
        "- $\\alpha = 0.5$: El balance óptimo demostrado en benchmarks."
    )
}

def analyze_chat_query(query: str, df: pd.DataFrame, n_results: int = 5) -> Dict[str, Any]:
    """
    Analiza el texto del usuario y devuelve una respuesta estructurada con
    comentario conversacional y canciones recomendadas relevantes.
    """
    q = query.lower().strip()

    # 1. Detección de preguntas conceptuales
    for key, text in SYSTEM_EXPLANATIONS.items():
        if key in q or (key == "svd" and ("colaborativo" in q or "factorizacion" in q)):
            return {
                "type": "explanation",
                "message": text,
                "tracks": None
            }

    # 2. Detección de estados de ánimo o actividades
    matched_moods = [mood for mood in MOOD_VECTORS if re.search(r'\b' + re.escape(mood) + r'\b', q)]
    
    if matched_moods:
        mood = matched_moods[0]
        bounds = MOOD_VECTORS[mood]
        
        filtered = df.copy()
        for feat, (low, high) in bounds.items():
            if feat in filtered.columns:
                filtered = filtered[(filtered[feat] >= low) & (filtered[feat] <= high)]
        
        if len(filtered) < n_results:
            filtered = df[(df["valence"] > 0.5) & (df["energy"] > 0.5)]

        # Ordenar por popularidad y una muestra representativa
        selected = filtered.sort_values("popularity", ascending=False).head(40).sample(
            min(n_results, len(filtered)), random_state=np.random.randint(1, 1000)
        )
        
        mood_titles = {
            "entrenar": "🔥 Selección de Alta Energía para Entrenar",
            "gym": "💪 Potencia y Ritmo Intenso para el Gimnasio",
            "estudiar": "📚 Sonoridad Acústica e Instrumental para Máxima Concentración",
            "relax": "🌿 Ondas Calmas y Sonidos Tranquilos para Relajarte",
            "dormir": "🌙 Frecuencias Suaves y Envolventes para Conciliar el Sueño",
            "fiesta": "🎉 Éxitos Bailables y Euforia para Prender la Fiesta",
            "bailar": "💃 Pistas con Máxima Bailabilidad y Groove",
            "triste": "🌧️ Melancolía y Baladas Emotivas para Días Grises",
            "feliz": "☀️ Pistas con Alta Positividad y Energía Positiva",
            "viajar": "🚗 El Soundtrack Ideal para la Carretera y Aventuras"
        }
        title = mood_titles.get(mood, f"🎶 Selección Especial para '{mood.capitalize()}'")

        return {
            "type": "tracks",
            "mood": mood,
            "message": f"¡Entendido! Diseñé esta playlist con base en tu vibra: **{title}**.\nAnalicé las variables acústicas para encontrar la resonancia perfecta:",
            "tracks": selected
        }

    # 3. Detección de canciones o artistas en la consulta
    words = [w for w in re.split(r'\s+', q) if len(w) > 3 and w not in ["para", "como", "quiero", "cancion", "canciones", "busca", "musica", "recomienda", "algo"]]
    if words:
        search_pattern = "|".join(re.escape(w) for w in words)
        candidate_matches = df[
            df["track_name"].str.contains(search_pattern, na=False) |
            df["artists"].str.contains(search_pattern, na=False) |
            df["track_genre"].str.contains(search_pattern, na=False)
        ]
        
        if not candidate_matches.empty:
            seed_song = candidate_matches.sort_values("popularity", ascending=False).iloc[0]
            # Muestrear del mismo género o artista
            genre_pool = df[df["track_genre"] == seed_song["track_genre"]]
            if len(genre_pool) < n_results:
                genre_pool = df
            recs = genre_pool[genre_pool["track_id"] != seed_song["track_id"]].sort_values(
                "popularity", ascending=False
            ).head(30).sample(min(n_results, len(genre_pool)), random_state=42)
            
            return {
                "type": "tracks",
                "message": f"Identifiqué que te interesa el estilo de **{seed_song['artists'].title()}** o el género `{seed_song['track_genre'].title()}`. Aquí tienes una selección con perfil sonoro compatible:",
                "tracks": recs
            }

    # 4. Respuesta general
    sample_tracks = df.sort_values("popularity", ascending=False).head(50).sample(n_results)
    return {
        "type": "tracks",
        "message": (
            "¡Hola! Soy **BeatBot**, tu DJ Virtual con Inteligencia Artificial. 🎧\n\n"
            "Puedes pedirme:\n"
            "- 🏃 *'Recomiéndame música para entrenar con fuerza'*\n"
            "- ☕ *'Quiero canciones acústicas para estudiar'*\n"
            "- 🌧️ *'Busco baladas tristes o melancólicas'*\n"
            "- 🔬 *'¿Cómo funciona el SVD?'* o *'Explícame la similitud coseno'*\n\n"
            "Mientras tanto, aquí tienes algunos de los temas mejor valorados de nuestro catálogo:"
        ),
        "tracks": sample_tracks
    }
