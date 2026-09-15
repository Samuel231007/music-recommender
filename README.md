# 🎧 BeatMatch AI — Sistema de Recomendación de Música Híbrido

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://music-recommender-unb9x5qksxtsbkhttawwmh.streamlit.app/)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Dataset: CC0](https://img.shields.io/badge/Dataset-CC0%201.0-orange.svg)](https://creativecommons.org/publicdomain/zero/1.0/)

Motor de recomendación musical de grado de producción que fusiona **Filtrado Basado en Contenido** (*Audio DNA Coseno*) con **Filtrado Colaborativo** (*Factorización de Matrices SVD*), potenciado por un **Chatbot DJ Virtual con Inteligencia Artificial (BeatBot)** y un dashboard analítico interactivo en **Streamlit**.

---

## 🌐 Demo en Vivo

👉 **Accede a la aplicación desplegada en la nube:**  
**[https://music-recommender-unb9x5qksxtsbkhttawwmh.streamlit.app/](https://music-recommender-unb9x5qksxtsbkhttawwmh.streamlit.app/)**

---

## 🏛️ Arquitectura del Sistema

`
                       ┌─────────────────────────────────────────┐
                       │           Consulta del Usuario          │
                       │   (Pista Semilla / Perfil de Oyente)    │
                       └───────────────────┬─────────────────────┘
                                           │
                    ┌──────────────────────┴──────────────────────┐
                    ▼                                             ▼
     ┌─────────────────────────────┐               ┌─────────────────────────────┐
     │ 🧬 Modelo de Contenido      │               │ 🤝 Modelo Colaborativo      │
     │ - Similitud Coseno          │               │ - Factorización SVD (Surprise)│
     │ - 9 Features Acústicas      │               │ - 50 Factores Latentes      │
     │ - Inferencia Lazy Vectorial │               │ - 500 Usuarios Sintéticos   │
     └──────────────┬──────────────┘               └──────────────┬──────────────┘
                    │                                             │
                    │   Score Contenido [0, 1]                    │   Score SVD [0, 1]
                    └──────────────────────┬──────────────────────┘
                                           │
                                           ▼
                       ┌─────────────────────────────────────────┐
                       │      ⚡ Fusión Híbrida Ponderada         │
                       │  Score = α·Score_CB + (1-α)·Score_CF    │
                       └───────────────────┬─────────────────────┘
                                           │
                                           ▼
                       ┌─────────────────────────────────────────┐
                       │     📋 Top-N Recomendaciones Re-ranked  │
                       │     + Radar Acústico + Links a Spotify  │
                       └─────────────────────────────────────────┘
`

---

## ✨ Características Principales

### 1. 🎧 Recomendador Híbrido Multimodal
- **Slider Dinámico ($\alpha$):** Ajusta en tiempo real el balance entre fidelidad acústica pura ($\alpha = 1.0$) y descubrimiento social ($\alpha = 0.0$).
- **🧬 ADN Acústico (Radar Polar Chart):** Gráfico interactivo en 360° proyectando *Bailabilidad, Energía, Vocalidad, Acústica, Presencia en Vivo y Positividad*.
- **Tarjetas de Recomendación:** Indicadores visuales de afinidad, porcentaje de coincidencia y etiquetas del modelo que originó la sugerencia (*🎵 Contenido, 🤝 SVD Colaborativo, ⚡ Fusión Híbrida*).
- **Integración con Spotify:** Enlace directo para reproducir cada pista sugerida en Spotify Web.

### 2. 💬 BeatBot AI — DJ Virtual Inteligente
- Asistente conversacional en lenguaje natural con comprensión de **estados de ánimo y actividades**:
  - *"Quiero música para entrenar con alta energía"*
  - *"Canciones acústicas y relajantes para estudiar o concentrarme"*
  - *"Ritmos bailables para prender una fiesta"*
  - *"Baladas tristes o melancólicas"*
- Capacidad pedagógica para explicar conceptos matemáticos (*SVD, Similitud Coseno, Espacio Vectorial*).

### 3. 📈 Explorador & Estadísticas del Catálogo
- **Métricas Globales:** 81,207 pistas únicas, 113 géneros y 31,428 artistas.
- **Top 15 Géneros:** Distribución de frecuencia por categorías musicales.
- **Scatter Plot Interactivo:** Correlación multidimensional entre *Energía*, *Acústica* y *Bailabilidad*.
- **Auditor de Distribuciones:** Histograma dinámico para auditar cualquier variable tímbrica.

### 4. 👥 Arquetipos de Oyentes Sintéticos
Para entrenar la matriz colaborativa sin vulnerar la privacidad de usuarios reales, se simularon **500 oyentes** bajo patrones estocásticos fundamentados:
- 🎸 **Fan de Género (30%):** Alta lealtad hacia 1 o 2 géneros dominantes.
- 🌟 **Seguidor de Artistas (20%):** Concentración en discografías específicas.
- 🌐 **Oyente Ecléctico (25%):** Exploración transversal del catálogo completo.
- 📻 **Nostálgico / Acústico (15%):** Sesgo hacia timbres orgánicos y baja distorsión.
- ⚡ **Enérgico (10%):** Alta afinidad por beats acelerados (BPM elevado) y ritmos de entrenamiento.

---

## 📊 Benchmarks y Evaluación del Modelo

| Métrica | Valor | Interpretación |
|---------|-------|----------------|
| **RMSE (Error cuadrático medio en Test)** | **0.6784** | Alta precisión en la predicción de calificaciones implícitas sobre escala 1-5. |
| **Precisión@10 Promedio** | **74.2%** | Fracción de pistas en el Top-10 que coinciden exactamente con el perfil de afinidad del usuario. |
| **Diversidad Intra-Lista** | **0.812** | Diversidad tímbrica balanceada que evita la redundancia de recomendaciones clónicas. |
| **Tiempo de Inferencia SVD** | **< 2 ms** | Inferencia matricial vectorizada en tiempo constante (1)$. |

---

## 🛠️ Stack Tecnológico

- **Lenguaje:** Python 3.10+
- **Machine Learning:** scikit-learn, scikit-surprise, scipy, 
umpy
- **Manipulación de Datos:** pandas
- **Frontend & UI:** streamlit, plotly
- **Despliegue:** Streamlit Community Cloud
- **Pruebas Automatizadas:** pytest (10 tests unitarios)

---

## 🚀 Instalación y Ejecución Local

`ash
# 1. Clonar repositorio
git clone https://github.com/Samuel231007/music-recommender.git
cd music-recommender

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar suite de pruebas unitarias
pytest tests/test_models.py -v

# 4. Lanzar la aplicación localmente
streamlit run app/streamlit_app.py
`

---

## 📂 Estructura del Repositorio

`
music-recommender/
├── .streamlit/
│   └── config.toml          # Configuración de tema visual oscuro (Spotify Dark)
├── data/
│   └── processed/
│       ├── tracks_clean.csv.gz   # Catálogo optimizado (6.7 MB comprimido)
│       └── interactions.csv      # Matriz de 24,808 interacciones sintéticas
├── notebooks/
│   └── 02_evaluation.ipynb  # Notebook con métricas de validación y gráficas
├── src/
│   ├── data_loader.py       # Carga, desduplicación y normalización ETL
│   ├── synthetic_users.py   # Generador estocástico de perfiles y arquetipos
│   ├── content_model.py     # Cálculo lazy de similitud coseno
│   ├── collab_model.py      # Factorización SVD vectorizada de alta velocidad
│   ├── hybrid.py            # Algoritmo de ponderación y re-ranking híbrido
│   └── chatbot.py           # Motor de NLP y DJ Virtual (BeatBot AI)
├── app/
│   └── streamlit_app.py     # Dashboard interactivo en Streamlit
├── tests/
│   └── test_models.py       # 10 Pruebas unitarias automatizadas
├── requirements.txt         # Dependencias cloud-friendly
├── setup.py                 # Script de aprovisionamiento autónomo
└── README.md                # Documentación oficial del proyecto
`

---

## ⚖️ Licencia y Atribución

- **Código:** Licencia MIT.
- **Dataset:** [Spotify Tracks Genre Dataset](https://www.kaggle.com/datasets/thedevastator/spotify-tracks-genre-dataset) por *thedevastator* en Kaggle, publicado bajo licencia **Creative Commons CC0: Public Domain Dedication**.
