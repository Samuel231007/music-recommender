"""
streamlit_app.py
Interfaz principal del recomendador híbrido de música.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import plotly.express as px

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_clean, AUDIO_FEATURES
from content_model import build_feature_matrix, get_content_recommendations
from collab_model import load_model, get_collab_recommendations, train_svd
from hybrid import get_hybrid_recommendations
from synthetic_users import generate_interactions, INTERACTIONS_PATH

# ─────────────────────────────────────────────
# Configuración de la página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="🎵 Music Recommender",
    page_icon="🎵",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# Carga de datos con cache
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando dataset...")
def load_data():
    return load_clean()

@st.cache_resource(show_spinner="Construyendo modelo de contenido...")
def load_content_model(df):
    return build_feature_matrix(df)

@st.cache_data(show_spinner="Cargando interacciones de usuarios...")
def load_interactions(df):
    if INTERACTIONS_PATH.exists():
        return pd.read_csv(INTERACTIONS_PATH)
    return generate_interactions(df)

@st.cache_resource(show_spinner="Cargando modelo colaborativo...")
def load_collab_model(df, interactions):
    from collab_model import MODEL_PATH
    if MODEL_PATH.exists():
        return load_model()
    return train_svd(interactions)

# ─────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────
with st.sidebar:
    st.title("🎵 Music Recommender")
    st.markdown("Sistema de recomendación **híbrido** que combina similitud de audio con preferencias de usuario.")
    st.divider()

    st.subheader("⚙️ Configuración")
    n_recs = st.slider("Número de recomendaciones", min_value=5, max_value=20, value=10, step=5)

    alpha = st.slider(
        "⚖️ Peso del modelo",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="0 = Solo colaborativo | 1 = Solo contenido de audio"
    )
    col1, col2 = st.columns(2)
    col1.caption("🤝 Colaborativo")
    col2.caption("🎵 Contenido")

    st.divider()
    st.caption("Dataset: Spotify Tracks (CC0) | Usuarios: sintéticos")

# ─────────────────────────────────────────────
# Carga de modelos
# ─────────────────────────────────────────────
df = load_data()
feature_matrix = load_content_model(df)
interactions = load_interactions(df)
collab_model = load_collab_model(df, interactions)

users = interactions["user_id"].unique().tolist()

# ─────────────────────────────────────────────
# Pantalla principal
# ─────────────────────────────────────────────
st.title("🎵 Recomendador de Música Híbrido")
st.markdown("Busca una canción y descubre nuevas recomendaciones personalizadas.")

col_search, col_user = st.columns([2, 1])

with col_search:
    # Búsqueda por nombre de canción
    search_query = st.text_input("🔍 Busca una canción o artista", placeholder="Ej: bohemian rhapsody")
    if search_query:
        mask = (
            df["track_name"].str.contains(search_query.lower(), na=False) |
            df["artists"].str.contains(search_query.lower(), na=False)
        )
        filtered = df[mask].head(50)
    else:
        filtered = df.head(50)

    track_options = [
        f"{row['track_name'].title()} — {row['artists'].title()} [{row['track_genre']}]"
        for _, row in filtered.iterrows()
    ]
    track_ids = filtered["track_id"].values

    selected_label = st.selectbox("Selecciona una canción", track_options)

with col_user:
    use_user = st.checkbox("Usar perfil de usuario", value=True)
    if use_user:
        selected_user = st.selectbox("Usuario sintético", users)
        archetype = interactions[interactions["user_id"] == selected_user]["archetype"].iloc[0]
        st.info(f"Arquetipo: **{archetype}**")
    else:
        selected_user = None

# ─────────────────────────────────────────────
# Generar recomendaciones
# ─────────────────────────────────────────────
if selected_label and len(track_ids) > 0:
    selected_idx = track_options.index(selected_label)
    selected_track_id = track_ids[selected_idx]
    selected_track = df[df["track_id"] == selected_track_id].iloc[0]

    # Info de la canción seleccionada
    with st.expander("🎧 Canción seleccionada", expanded=True):
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Artista", selected_track["artists"].title())
        c2.metric("Género", selected_track["track_genre"].title())
        c3.metric("Popularidad", f"{selected_track['popularity']}/100")
        c4.metric("Energy", f"{selected_track['energy']:.2f}")

        # Radar de features de audio
        features_vals = selected_track[["danceability", "energy", "speechiness",
                                         "acousticness", "instrumentalness",
                                         "liveness", "valence"]].values
        features_names = ["Danceability", "Energy", "Speechiness",
                           "Acousticness", "Instrumentalness", "Liveness", "Valence"]
        fig = px.bar(
            x=features_names, y=features_vals,
            labels={"x": "Feature", "y": "Valor"},
            title="Audio Features",
            color=features_vals,
            color_continuous_scale="Viridis"
        )
        fig.update_layout(height=250, showlegend=False, coloraxis_showscale=False)
        st.plotly_chart(fig, use_container_width=True)

    st.divider()

    # Botón de recomendación
    if st.button("🎯 Obtener recomendaciones", type="primary", use_container_width=True):
        with st.spinner("Calculando recomendaciones..."):
            try:
                recs = get_hybrid_recommendations(
                    track_id=selected_track_id,
                    user_id=selected_user if use_user else None,
                    df=df,
                    feature_matrix=feature_matrix,
                    interactions=interactions,
                    model=collab_model,
                    alpha=alpha,
                    n=n_recs
                )

                st.subheader(f"🎵 Top {n_recs} Recomendaciones")
                st.caption(f"Alpha={alpha:.2f} | {'Con perfil de usuario' if use_user else 'Solo por contenido'}")

                source_colors = {"content": "🎵", "collab": "🤝", "hybrid": "⚡"}

                for i, row in recs.iterrows():
                    with st.container():
                        col_num, col_info, col_score = st.columns([0.5, 4, 1.5])
                        col_num.markdown(f"### {i+1}")
                        with col_info:
                            icon = source_colors.get(row["source"], "🎵")
                            st.markdown(f"**{row['track_name'].title()}**  {icon}")
                            st.caption(f"{row['artists'].title()} · {row['track_genre'].title()} · ⭐ {row['popularity']}/100")
                        with col_score:
                            st.metric("Score", f"{row['hybrid_score']:.3f}")
                            st.caption(f"C:{row['content_score_norm']:.2f} | CF:{row['collab_score_norm']:.2f}")
                    st.divider()

                # Leyenda
                st.markdown("**Leyenda fuente:** 🎵 Contenido de audio | 🤝 Filtrado colaborativo | ⚡ Híbrido")

                # Distribución de géneros en las recomendaciones
                genre_dist = recs["track_genre"].value_counts().reset_index()
                genre_dist.columns = ["género", "count"]
                fig2 = px.pie(genre_dist, values="count", names="género",
                               title="Géneros en las recomendaciones")
                fig2.update_layout(height=300)
                st.plotly_chart(fig2, use_container_width=True)

            except Exception as e:
                st.error(f"Error al generar recomendaciones: {str(e)}")
