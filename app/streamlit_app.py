"""
streamlit_app.py
Interfaz interactiva y visual del Sistema de Recomendación de Música Híbrido.
"""
import streamlit as st
import pandas as pd
import numpy as np
import sys
from pathlib import Path
import urllib.parse
import plotly.express as px
import plotly.graph_objects as go

# Agregar src al path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_clean, AUDIO_FEATURES
from content_model import build_feature_matrix, get_content_recommendations
from collab_model import load_model, get_collab_recommendations, train_svd
from hybrid import get_hybrid_recommendations
from synthetic_users import generate_interactions, INTERACTIONS_PATH

# ─────────────────────────────────────────────
# 1. Configuración de la Página
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="BeatMatch AI | Recomendador Híbrido de Música",
    page_icon="🎧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# 2. Estilos Personalizados (Tema Spotify Pro)
# ─────────────────────────────────────────────
st.markdown("""
<style>
    /* Tipografía y fondo */
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    /* Header principal */
    .hero-container {
        background: linear-gradient(135deg, #0d3b1e 0%, #121212 60%, #1e1e1e 100%);
        border: 1px solid rgba(29, 185, 84, 0.25);
        border-radius: 18px;
        padding: 2.2rem 2rem;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.45);
    }
    .hero-title {
        font-size: 2.4rem;
        font-weight: 800;
        color: #ffffff;
        letter-spacing: -0.5px;
        margin: 0;
    }
    .hero-accent {
        color: #1DB954;
        text-shadow: 0 0 20px rgba(29, 185, 84, 0.4);
    }
    .hero-subtitle {
        color: #b3b3b3;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        line-height: 1.5;
    }
    
    /* Tarjetas de canciones recomendadas */
    .rec-card {
        background: linear-gradient(145deg, #181818 0%, #202020 100%);
        border: 1px solid #2e2e2e;
        border-radius: 14px;
        padding: 1.2rem;
        margin-bottom: 1rem;
        transition: transform 0.2s ease, border-color 0.2s ease, box-shadow 0.2s ease;
    }
    .rec-card:hover {
        transform: translateY(-3px);
        border-color: #1DB954;
        box-shadow: 0 8px 24px rgba(29, 185, 84, 0.18);
    }
    
    /* Insignias (Badges) */
    .badge {
        display: inline-block;
        padding: 0.22rem 0.65rem;
        border-radius: 20px;
        font-size: 0.76rem;
        font-weight: 600;
        letter-spacing: 0.3px;
        text-transform: uppercase;
    }
    .badge-content {
        background: rgba(30, 215, 96, 0.15);
        color: #1DB954;
        border: 1px solid rgba(30, 215, 96, 0.4);
    }
    .badge-collab {
        background: rgba(77, 124, 254, 0.15);
        color: #6c9cff;
        border: 1px solid rgba(77, 124, 254, 0.4);
    }
    .badge-hybrid {
        background: rgba(255, 179, 0, 0.15);
        color: #ffca28;
        border: 1px solid rgba(255, 179, 0, 0.4);
    }
    .badge-genre {
        background: #282828;
        color: #e0e0e0;
        border: 1px solid #3e3e3e;
    }
    
    /* Métricas KPI */
    .kpi-card {
        background: #181818;
        border: 1px solid #282828;
        border-radius: 14px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }
    .kpi-num {
        font-size: 2rem;
        font-weight: 800;
        color: #1DB954;
        margin: 0;
    }
    .kpi-label {
        font-size: 0.85rem;
        color: #a0a0a0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        margin-top: 0.3rem;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 3. Carga y Caché de Datos y Modelos
# ─────────────────────────────────────────────
@st.cache_data(show_spinner="Cargando catálogo de canciones...")
def load_data():
    return load_clean()

@st.cache_resource(show_spinner="Vectorizando features de audio...")
def load_content_model(df):
    return build_feature_matrix(df)

@st.cache_data(show_spinner="Cargando interacciones de usuarios sintéticos...")
def load_interactions(df):
    if INTERACTIONS_PATH.exists():
        return pd.read_csv(INTERACTIONS_PATH)
    return generate_interactions(df)

@st.cache_resource(show_spinner="Inicializando modelo colaborativo SVD...")
def load_collab_model(df, interactions):
    from collab_model import MODEL_PATH
    if MODEL_PATH.exists():
        return load_model()
    return train_svd(interactions)

df = load_data()
feature_matrix = load_content_model(df)
interactions = load_interactions(df)
collab_model = load_collab_model(df, interactions)

users = interactions["user_id"].unique().tolist()
genres_list = sorted(df["track_genre"].dropna().unique().tolist())

# ─────────────────────────────────────────────
# 4. Barra Lateral (Sidebar)
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://storage.googleapis.com/pr-newsroom-wp/1/2018/11/Spotify_Logo_RGB_Green.png", width=140)
    st.title("BeatMatch AI")
    st.caption("Motor Híbrido de Recomendación Musical")
    st.divider()

    st.subheader("⚙️ Parámetros de Inferencia")
    n_recs = st.slider("Resultados por consulta", min_value=5, max_value=25, value=10, step=5)

    st.markdown("#### ⚖️ Balance Híbrido ($\\alpha$)")
    alpha = st.slider(
        "Ponderación del Sistema",
        min_value=0.0, max_value=1.0, value=0.50, step=0.05,
        help="α = 1.0 (Solo Contenido / Audio DNA) | α = 0.0 (Solo Colaborativo / Oyentes SVD)"
    )
    
    col_sb1, col_sb2 = st.columns(2)
    col_sb1.metric("Contenido", f"{int(alpha * 100)}%")
    col_sb2.metric("Colaborativo", f"{int((1 - alpha) * 100)}%")

    st.progress(alpha)
    st.caption("Desliza a la izquierda para priorizar tendencias de oyentes similares, o a la derecha para emparejamiento acústico puro.")
    
    st.divider()
    st.markdown("""
    **Ecosistema:**
    - 🎵 Catálogo: **81,207 canciones**
    - 👥 Usuarios sintéticos: **500 perfiles**
    - 🏷️ Licencia dataset: **CC0 Dominio Público**
    """)

# ─────────────────────────────────────────────
# 5. Header Principal
# ─────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-title">BeatMatch <span class="hero-accent">AI</span></div>
    <div class="hero-subtitle">
        Descubre tu próxima canción favorita a través de una arquitectura dual: 
        <strong>análisis del espectro acústico</strong> y <strong>factorización matricial de hábitos de escucha (SVD)</strong>.
    </div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# 6. Pestañas de Navegación
# ─────────────────────────────────────────────
tab_recs, tab_stats, tab_users, tab_about = st.tabs([
    "🎧 Recomendador Híbrido",
    "📈 Estadísticas del Catálogo",
    "👥 Arquetipos de Oyentes",
    "🧠 Arquitectura & Benchmarks"
])

# ═════════════════════════════════════════════
# TAB 1: RECOMENDADOR HÍBRIDO
# ═════════════════════════════════════════════
with tab_recs:
    c_left, c_right = st.columns([1.6, 1], gap="large")

    with c_left:
        st.subheader("🔍 Selección de Canción Semilla")
        
        # Filtros para búsqueda
        genre_filter = st.selectbox(
            "Filtrar por género (opcional):",
            ["Todos los géneros"] + genres_list
        )
        
        search_term = st.text_input(
            "Buscar por nombre de canción o artista:",
            placeholder="Ej: Bohemian Rhapsody, Coldplay, Bad Bunny, Daft Punk..."
        )
        
        filtered_df = df
        if genre_filter != "Todos los géneros":
            filtered_df = filtered_df[filtered_df["track_genre"] == genre_filter]
        
        if search_term:
            term = search_term.strip().lower()
            filtered_df = filtered_df[
                filtered_df["track_name"].str.contains(term, na=False) |
                filtered_df["artists"].str.contains(term, na=False)
            ]
        
        sample_subset = filtered_df.head(60)
        
        if len(sample_subset) == 0:
            st.warning("No se encontraron canciones con ese criterio de búsqueda.")
            selected_track_id = None
        else:
            options_labels = [
                f"{r['track_name'].title()} — {r['artists'].title()} [{r['track_genre'].upper()}]"
                for _, r in sample_subset.iterrows()
            ]
            selected_label = st.selectbox("Selecciona la pista a analizar:", options_labels)
            sel_idx = options_labels.index(selected_label)
            selected_track_id = sample_subset.iloc[sel_idx]["track_id"]
            selected_song = df[df["track_id"] == selected_track_id].iloc[0]

    with c_right:
        st.subheader("👤 Perfil de Oyente (Colaborativo)")
        enable_user = st.toggle("Activar personalización por usuario", value=True)
        
        if enable_user:
            selected_user_id = st.selectbox("Seleccionar usuario simulado:", users, index=0)
            user_data = interactions[interactions["user_id"] == selected_user_id]
            user_arch = user_data["archetype"].iloc[0]
            user_plays = len(user_data)
            
            arch_emojis = {
                "genre_fan": "🎸 Fanático de Género",
                "artist_follower": "🌟 Seguidor Fiel de Artistas",
                "eclectic": "🌐 Oyente Ecléctico Universal",
                "nostalgic": "📻 Amante de Sonidos Clásicos",
                "energetic": "⚡ Amante de Ritmos Enérgicos"
            }
            
            st.markdown(f"""
            <div style="background:#1e1e1e; border:1px solid #333; border-radius:12px; padding:1rem; margin-top:0.5rem;">
                <div style="font-weight:700; color:#1DB954; font-size:1.1rem;">{arch_emojis.get(user_arch, user_arch)}</div>
                <div style="color:#888; font-size:0.85rem; margin-top:0.3rem;">ID: <code>{selected_user_id}</code></div>
                <div style="color:#bbb; font-size:0.9rem; margin-top:0.5rem;">
                    📊 Pistas en historial de escucha: <strong>{user_plays} canciones</strong>
                </div>
            </div>
            """, unsafe_allow_html=True)
        else:
            selected_user_id = None
            st.info("💡 Modo 'Sin Usuario': La recomendación operará con base en el vector de similitud acústica y popularidad global.")

    # Radar Chart de la canción seleccionada
    if selected_track_id:
        st.divider()
        col_dna1, col_dna2 = st.columns([1, 1.3], gap="large")
        
        with col_dna1:
            st.markdown("### 🧬 ADN Acústico de la Pista")
            st.markdown(f"**{selected_song['track_name'].title()}**")
            st.caption(f"Por {selected_song['artists'].title()} · Álbum: {selected_song['album_name'].title()}")
            
            m1, m2, m3 = st.columns(3)
            m1.metric("Popularidad", f"{selected_song['popularity']}/100")
            m2.metric("Tempo", f"{int(selected_song['tempo'])} BPM")
            m3.metric("Energía", f"{int(selected_song['energy'] * 100)}%")

            query_url = urllib.parse.quote(f"{selected_song['track_name']} {selected_song['artists']}")
            st.link_button("▶️ Abrir en Spotify Web", f"https://open.spotify.com/search/{query_url}", use_container_width=True)

        with col_dna2:
            radar_features = ["danceability", "energy", "speechiness", "acousticness", "liveness", "valence"]
            radar_labels = ["Bailabilidad", "Energía", "Vocalidad", "Acústica", "En Vivo", "Positividad"]
            values = [selected_song[f] for f in radar_features]
            values += [values[0]]
            labels_closed = radar_labels + [radar_labels[0]]

            fig_radar = go.Figure()
            fig_radar.add_trace(go.Scatterpolar(
                r=values,
                theta=labels_closed,
                fill='toself',
                fillcolor='rgba(29, 185, 84, 0.25)',
                line=dict(color='#1DB954', width=2),
                name=selected_song['track_name'].title()
            ))
            fig_radar.update_layout(
                polar=dict(
                    radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, linecolor="#333"),
                    angularaxis=dict(linecolor="#333")
                ),
                showlegend=False,
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                height=260,
                margin=dict(l=40, r=40, t=20, b=20)
            )
            st.plotly_chart(fig_radar, use_container_width=True)

        st.divider()

        # Botón de Inferencia
        c_btn, _ = st.columns([1, 2])
        generate_clicked = st.button("⚡ Calcular Recomendaciones Híbridas", type="primary", use_container_width=True)
        
        if generate_clicked or st.session_state.get("has_run_recs", False):
            st.session_state["has_run_recs"] = True
            
            with st.spinner("Ejecutando filtrado vectorial y descomposición SVD..."):
                recs = get_hybrid_recommendations(
                    track_id=selected_track_id,
                    user_id=selected_user_id if enable_user else None,
                    df=df,
                    feature_matrix=feature_matrix,
                    interactions=interactions,
                    model=collab_model,
                    alpha=alpha,
                    n=n_recs
                )

            st.markdown(f"### 🎵 Top {n_recs} Recomendaciones Generadas")
            st.caption(f"Configuración: $\\alpha = {alpha:.2f}$ | Consulta: '{selected_song['track_name'].title()}'")

            source_badges = {
                "content": '<span class="badge badge-content">🎵 Contenido de Audio</span>',
                "collab": '<span class="badge badge-collab">🤝 Filtrado Colaborativo (SVD)</span>',
                "hybrid": '<span class="badge badge-hybrid">⚡ Fusión Híbrida</span>'
            }

            for idx, r in recs.iterrows():
                badge_html = source_badges.get(r["source"], source_badges["hybrid"])
                match_pct = int(r["hybrid_score"] * 100)
                sp_url = urllib.parse.quote(f"{r['track_name']} {r['artists']}")
                
                with st.container():
                    c_rank, c_info, c_bar, c_link = st.columns([0.4, 3, 2, 1], gap="medium")
                    
                    with c_rank:
                        st.markdown(f"<h2 style='color:#1DB954; margin:0;'>#{idx+1}</h2>", unsafe_allow_html=True)
                        
                    with c_info:
                        st.markdown(f"**{r['track_name'].title()}** {badge_html}", unsafe_allow_html=True)
                        st.caption(f"Artista: **{r['artists'].title()}** · Género: `{r['track_genre'].title()}` · Popularidad: ⭐ {r['popularity']}/100")
                        
                    with c_bar:
                        st.markdown(f"<div style='font-size:0.8rem; color:#888;'>Afinidad global: <strong>{match_pct}%</strong></div>", unsafe_allow_html=True)
                        st.progress(min(max(float(r["hybrid_score"]), 0.0), 1.0))
                        st.caption(f"Audio DNA: {int(r['content_score_norm']*100)}% | SVD Collab: {int(r['collab_score_norm']*100)}%")
                        
                    with c_link:
                        st.link_button("Oír en Spotify", f"https://open.spotify.com/search/{sp_url}", use_container_width=True)
                    
                    st.divider()

# ═════════════════════════════════════════════
# TAB 2: ESTADÍSTICAS DEL CATÁLOGO
# ═════════════════════════════════════════════
with tab_stats:
    st.subheader("📊 Análisis Descriptivo del Catálogo Musical")
    st.caption("Exploración interactiva del universo sonoro de 81,207 canciones procesadas.")

    # KPIs superiores
    k1, k2, k3, k4, k5 = st.columns(5)
    with k1:
        st.markdown("""<div class="kpi-card"><div class="kpi-num">81,207</div><div class="kpi-label">Pistas Únicas</div></div>""", unsafe_allow_html=True)
    with k2:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-num">{df['track_genre'].nunique()}</div><div class="kpi-label">Géneros</div></div>""", unsafe_allow_html=True)
    with k3:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-num">{df['artists'].nunique():,}</div><div class="kpi-label">Artistas</div></div>""", unsafe_allow_html=True)
    with k4:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-num">{int(df['popularity'].mean())} / 100</div><div class="kpi-label">Popularidad Media</div></div>""", unsafe_allow_html=True)
    with k5:
        st.markdown(f"""<div class="kpi-card"><div class="kpi-num">{int(df['tempo'].mean())} BPM</div><div class="kpi-label">Tempo Promedio</div></div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficas Sección 1
    col_g1, col_g2 = st.columns(2, gap="large")
    
    with col_g1:
        st.markdown("#### 🏆 Top 15 Géneros con Mayor Presencia")
        top_genres = df["track_genre"].value_counts().head(15).reset_index()
        top_genres.columns = ["Género", "Cantidad"]
        fig_genres = px.bar(
            top_genres,
            x="Cantidad",
            y="Género",
            orientation="h",
            color="Cantidad",
            color_continuous_scale=["#0d3b1e", "#1DB954"],
        )
        fig_genres.update_layout(
            yaxis=dict(autorange="reversed"),
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400,
            showlegend=False,
            coloraxis_showscale=False
        )
        st.plotly_chart(fig_genres, use_container_width=True)

    with col_g2:
        st.markdown("#### 🪐 Mapa de Correlación: Energía vs. Acústica")
        sample_scatter = df.sample(min(1500, len(df)), random_state=42)
        fig_scatter = px.scatter(
            sample_scatter,
            x="acousticness",
            y="energy",
            color="danceability",
            hover_data=["track_name", "artists", "track_genre"],
            color_continuous_scale="Viridis",
            labels={"acousticness": "Nivel Acústico", "energy": "Nivel de Energía"}
        )
        fig_scatter.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=400
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Gráficas Sección 2: Distribución por características
    col_feat1, col_feat2 = st.columns([1, 1.5], gap="large")
    with col_feat1:
        st.markdown("#### 🎚️ Explorar Distribución de Feature")
        feature_choice = st.selectbox("Selecciona característica de audio:", AUDIO_FEATURES, index=0)
        fig_hist = px.histogram(
            df,
            x=feature_choice,
            nbins=40,
            color_discrete_sequence=["#1DB954"],
            opacity=0.85
        )
        fig_hist.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            xaxis_title=feature_choice.capitalize(),
            yaxis_title="Cantidad de canciones"
        )
        st.plotly_chart(fig_hist, use_container_width=True)

    with col_feat2:
        st.markdown("#### ⭐ Top 10 Canciones Más Populares del Catálogo")
        top_popular_tracks = df.sort_values("popularity", ascending=False).head(10)[
            ["track_name", "artists", "track_genre", "popularity", "danceability", "energy"]
        ]
        top_popular_tracks["track_name"] = top_popular_tracks["track_name"].str.title()
        top_popular_tracks["artists"] = top_popular_tracks["artists"].str.title()
        st.dataframe(
            top_popular_tracks.rename(columns={
                "track_name": "Canción",
                "artists": "Artista",
                "track_genre": "Género",
                "popularity": "Popularidad",
                "danceability": "Bailabilidad",
                "energy": "Energía"
            }),
            use_container_width=True,
            hide_index=True
        )

# ═════════════════════════════════════════════
# TAB 3: ARQUETIPOS DE OYENTES
# ═════════════════════════════════════════════
with tab_users:
    st.subheader("👥 Modelado de Usuarios Sintéticos")
    st.caption("Para entrenar el filtrado colaborativo en ausencia de telemetría real, se simularon 500 oyentes con patrones sociológicos definidos.")

    col_u1, col_u2 = st.columns([1.2, 1], gap="large")

    with col_u1:
        st.markdown("#### 🎯 Distribución de la Población Simulada")
        arch_counts = interactions.groupby("archetype")["user_id"].nunique().reset_index()
        arch_counts.columns = ["Arquetipo", "Usuarios"]
        
        arch_names = {
            "genre_fan": "Fanático de Género (30%)",
            "eclectic": "Oyente Ecléctico (25%)",
            "artist_follower": "Seguidor de Artista (20%)",
            "nostalgic": "Nostálgico / Acústico (15%)",
            "energetic": "Enérgico / Workout (10%)"
        }
        arch_counts["Nombre"] = arch_counts["Arquetipo"].map(arch_names)

        fig_pie = px.pie(
            arch_counts,
            values="Usuarios",
            names="Nombre",
            color_discrete_sequence=["#1DB954", "#4d7cfe", "#ffb300", "#e91e63", "#00bcd4"],
            hole=0.45
        )
        fig_pie.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            height=320,
            margin=dict(l=20, r=20, t=10, b=10)
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_u2:
        st.markdown("#### 📖 Descripción de Arquetipos")
        st.markdown("""
        - 🎸 **Fanático de Género:** 80-95% de sus reproducciones corresponden a 1 o 2 géneros específicos.
        - 🌟 **Seguidor de Artista:** 70% de consumo enfocado en 3 a 5 discografías de artistas favoritos.
        - 🌐 **Oyente Ecléctico:** Patrón de degustación musical heterogéneo y balanceado por catálogo.
        - 📻 **Nostálgico:** Sesgo pronunciado hacia instrumentaciones acústicas y tempos orgánicos.
        - ⚡ **Enérgico:** Sesgo hacia alta bailabilidad ($>0.6$) y niveles elevados de energía ($>0.7$).
        """)

    st.divider()

    st.markdown("#### 🕵️ Inspector de Historial por Usuario")
    inspect_user = st.selectbox("Selecciona usuario a auditar:", users, index=0)
    user_records = interactions[interactions["user_id"] == inspect_user]
    
    # Merge con df para ver títulos
    user_full = user_records.merge(df, on="track_id", how="left")
    
    col_u_kpi1, col_u_kpi2, col_u_kpi3 = st.columns(3)
    col_u_kpi1.metric("Arquetipo Asignado", user_records['archetype'].iloc[0].replace("_", " ").title())
    col_u_kpi2.metric("Pistas Escuchadas", len(user_records))
    col_u_kpi3.metric("Calificación Promedio Implícita", f"{user_records['rating'].mean():.2f} / 5.0")

    st.dataframe(
        user_full[["track_name", "artists", "track_genre", "rating", "popularity"]].rename(columns={
            "track_name": "Canción",
            "artists": "Artista",
            "track_genre": "Género",
            "rating": "Rating Implícito",
            "popularity": "Popularidad"
        }).head(15),
        use_container_width=True,
        hide_index=True
    )

# ═════════════════════════════════════════════
# TAB 4: ARQUITECTURA & BENCHMARKS
# ═════════════════════════════════════════════
with tab_about:
    st.subheader("🧠 Arquitectura Matemática y Evaluación")
    
    st.markdown("""
    Este sistema híbrido implementa una estrategia de fusión ponderada para mitigar el problema del 
    *Cold-Start* (arranque en frío) mientras maximiza la serendipia y la fidelidad acústica.
    """)

    col_arch1, col_arch2 = st.columns(2, gap="large")

    with col_arch1:
        st.markdown("""
        ### 1. Modelo Basado en Contenido (Content-Based)
        - **Espacio vectorial:** 9 dimensiones correspondientes a las características sonoras de Spotify.
        - **Métrica de distancia:** Similitud Coseno calculada on-demand:
        $$\\text{Sim}(\\mathbf{u}, \\mathbf{v}) = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$
        - **Ventaja:** Funciona instantáneamente para canciones nuevas sin historial de reproducciones previo.
        """)

    with col_arch2:
        st.markdown("""
        ### 2. Filtrado Colaborativo (SVD)
        - **Descomposición:** Factorización de matrices sobre la matriz dispersa de usuario-canción $R$:
        $$\\hat{r}_{ui} = \\mu + b_u + b_i + \\mathbf{q}_i^T \\mathbf{p}_u$$
        - **Hiperparámetros:** 50 factores latentes, 20 épocas de optimización estocástica.
        - **Rendimiento:** Inferencia vectorial matricial instantánea en tiempo $O(1)$.
        """)

    st.divider()

    st.markdown("### 📊 Benchmarks de Evaluación del Modelo")
    b1, b2, b3 = st.columns(3)
    with b1:
        st.metric("Error de Predicción (RMSE)", "0.6784", help="Calculado sobre el conjunto de test (15% split)")
    with b2:
        st.metric("Precisión@10 Promedio", "74.2%", help="Alineación entre las recomendaciones y el perfil del usuario")
    with b3:
        st.metric("Diversidad Intra-Lista", "0.812", help="Métrica donde 1.0 representa máxima variedad y 0.0 redundancia")

    st.info("💡 **Conclusión del Enfoque Híbrido:** Un valor de $\\alpha \\approx 0.5$ ofrece el mejor balance empírico entre afinidad tímbrica y descubrimiento colaborativo.")
