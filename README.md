# 🎵 Music Recommender — Sistema Híbrido

Recomendador de música que combina **filtrado basado en contenido** (audio features de Spotify) con **filtrado colaborativo** (SVD sobre interacciones sintéticas de usuarios), con **peso ajustable desde la interfaz**.

---

## Demo

> *[Link al deploy — se agrega tras el despliegue en Streamlit Cloud]*

---

## Arquitectura

`
Canción seleccionada ──► Modelo Contenido (coseno) ──┐
                                                       ├──► Score Híbrido (alpha) ──► Top-N
Usuario seleccionado ──► Modelo Colaborativo (SVD) ──┘
`

**Content-based filtering**: Calcula similitud coseno entre la canción de consulta y todas las demás, usando 9 audio features de Spotify (danceability, energy, loudness, etc.).

**Collaborative filtering**: Factorización de matrices SVD entrenada sobre 500 usuarios sintéticos con 5 arquetipos de oyente. Predice ratings para canciones no escuchadas.

**Híbrido**: score_final = α × score_contenido + (1-α) × score_colaborativo donde α se ajusta con un slider en la interfaz.

---

## Dataset

- **Fuente**: [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset) — Kaggle
- **Licencia**: [CC0 1.0 Universal](https://creativecommons.org/publicdomain/zero/1.0/) (Dominio Público)
- **Tamaño**: ~114,000 canciones, 114 géneros, 31,437 artistas
- **Features de audio**: danceability, energy, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo

> **Disclaimer**: Este proyecto usa el dataset bajo licencia CC0. Los usuarios sintéticos son datos generados artificialmente, no datos reales de ningún servicio de streaming.

---

## Instalación local

`ash
git clone https://github.com/TU_USUARIO/music-recommender
cd music-recommender

pip install -r requirements.txt

# Descargar dataset (requiere cuenta Kaggle)
kaggle datasets download -d thedevastator/spotify-tracks-genre-dataset --path data/raw --unzip

# Preprocesar datos y entrenar modelos
python src/data_loader.py
python src/synthetic_users.py
python src/collab_model.py

# Lanzar la app
streamlit run app/streamlit_app.py
`

---

## Estructura del proyecto

`
music-recommender/
├── data/
│   ├── raw/                # Dataset original (no versionado en git)
│   └── processed/          # Datos limpios y modelos pre-entrenados
├── notebooks/              # EDA y evaluación
├── src/
│   ├── data_loader.py      # Carga y limpieza
│   ├── synthetic_users.py  # Generación de usuarios con arquetipos
│   ├── content_model.py    # Modelo basado en contenido
│   ├── collab_model.py     # Filtrado colaborativo (SVD)
│   └── hybrid.py           # Lógica de combinación híbrida
├── app/
│   └── streamlit_app.py    # Interfaz Streamlit
├── tests/
│   └── test_models.py      # Tests automatizados
├── requirements.txt
└── README.md
`

---

## Decisiones de diseño

| Decisión | Elección | Motivo |
|----------|----------|--------|
| Dataset | Completo (114k canciones) | Mayor cobertura del catálogo |
| Usuarios | 500 sintéticos con 5 arquetipos | Interacciones creíbles sin datos reales |
| Modelo colaborativo | SVD (Surprise) | Balance entre calidad y velocidad de entrenamiento |
| Peso híbrido | Slider ajustable (α) | Control explícito del usuario sobre el sistema |
| Interfaz | Streamlit | Prototipado rápido, deploy sencillo |

---

## Limitaciones

- Los usuarios sintéticos simplifican el comportamiento real de oyentes
- No hay retroalimentación real del usuario (ratings implícitos simulados)
- La similitud coseno en contenido no captura relaciones semánticas (letra, estado de ánimo)
- Cold-start: usuarios nuevos sin historial solo reciben recomendaciones por contenido

---

## Tests

`ash
pytest tests/test_models.py -v
`

---

## Licencia del código

MIT
