"""
test_models.py
Tests básicos para los módulos del recomendador híbrido.
"""
import sys
from pathlib import Path
import pandas as pd
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_loader import load_clean, AUDIO_FEATURES
from content_model import build_feature_matrix, get_content_recommendations
from synthetic_users import generate_interactions

@pytest.fixture(scope="module")
def df():
    return load_clean()

@pytest.fixture(scope="module")
def feature_matrix(df):
    return build_feature_matrix(df)

@pytest.fixture(scope="module")
def interactions(df):
    return generate_interactions(df, seed=42)

def test_data_loader_no_nulls(df):
    """El dataset limpio no debe tener nulos en audio features."""
    assert df[AUDIO_FEATURES].isna().sum().sum() == 0, "Hay nulos en audio features"

def test_data_loader_track_ids_unique(df):
    """Todos los track_id deben ser únicos."""
    assert df["track_id"].nunique() == len(df), "Hay track_ids duplicados"

def test_content_model_returns_n_results(df, feature_matrix):
    """El modelo de contenido debe devolver exactamente N resultados."""
    sample_id = df["track_id"].iloc[0]
    recs = get_content_recommendations(sample_id, df, feature_matrix, n=10)
    assert len(recs) == 10

def test_content_model_excludes_query(df, feature_matrix):
    """La canción de consulta no debe aparecer en los resultados."""
    sample_id = df["track_id"].iloc[0]
    recs = get_content_recommendations(sample_id, df, feature_matrix, n=10)
    assert sample_id not in recs["track_id"].values

def test_content_model_scores_range(df, feature_matrix):
    """Los scores de similitud deben estar en [-1, 1]."""
    sample_id = df["track_id"].iloc[5]
    recs = get_content_recommendations(sample_id, df, feature_matrix, n=10)
    assert recs["content_score"].between(-1, 1).all()

def test_synthetic_users_count(interactions):
    """Deben generarse exactamente 500 usuarios."""
    assert interactions["user_id"].nunique() == 500

def test_synthetic_users_archetypes(interactions):
    """Todos los arquetipos esperados deben estar presentes."""
    expected = {"genre_fan", "artist_follower", "eclectic", "nostalgic", "energetic"}
    actual = set(interactions["archetype"].unique())
    assert actual == expected

def test_synthetic_users_ratings_range(interactions):
    """Los ratings deben estar en el rango [0.5, 5.0]."""
    assert interactions["rating"].between(0.5, 5.0).all()

def test_chatbot_mood_analysis(df):
    """El chatbot debe responder con pistas acordes al estado de animo solicitado."""
    from chatbot import analyze_chat_query
    res = analyze_chat_query("quiero musica para entrenar en el gym", df, n_results=5)
    assert res["type"] == "tracks"
    assert len(res["tracks"]) == 5
    # Verificar que el promedio de energía es alto para entrenar
    assert res["tracks"]["energy"].mean() >= 0.65

def test_chatbot_explanation(df):
    """El chatbot debe reconocer preguntas teoricas de SVD y responder explicaciones."""
    from chatbot import analyze_chat_query
    res = analyze_chat_query("como funciona el algoritmo SVD?", df)
    assert res["type"] == "explanation"
    assert "SVD" in res["message"]

def test_real_user_voting(df, interactions):
    """Prueba que el guardado de votos reales y combinación de interacciones funciona."""
    from user_manager import save_user_rating, get_real_user_votes, get_combined_interactions
    sample_tid = df["track_id"].iloc[0]
    save_user_rating("TestSamuel", sample_tid, 5.0)
    votes = get_real_user_votes("TestSamuel")
    assert sample_tid in votes
    assert votes[sample_tid] == 5.0
    
    combined = get_combined_interactions(interactions)
    assert len(combined) > len(interactions)
    assert "real_testsamuel" in combined["user_id"].values
