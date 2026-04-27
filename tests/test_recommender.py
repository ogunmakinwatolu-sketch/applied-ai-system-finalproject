"""
Test suite for VibeFinder AI 1.0.

Run from the project root:
    pytest
"""

from unittest.mock import MagicMock

from src.recommender import Song, UserProfile, Recommender, score_song, recommend_songs
from src.ai_recommender import (
    AIRecommender,
    build_context,
    confidence_label,
    generate_recommendation,
    load_genre_guide,
    retrieve_genre_context,
)


# ── Shared fixtures ───────────────────────────────────────────────────────────

def _make_recommender() -> Recommender:
    return Recommender([
        Song(
            id=1, title="Test Pop Track", artist="Test Artist",
            genre="pop", mood="happy", energy=0.8,
            tempo_bpm=120, valence=0.9, danceability=0.8, acousticness=0.2,
        ),
        Song(
            id=2, title="Chill Lofi Loop", artist="Test Artist",
            genre="lofi", mood="chill", energy=0.4,
            tempo_bpm=80, valence=0.6, danceability=0.5, acousticness=0.9,
        ),
    ])


_POP_USER = UserProfile(
    favorite_genre="pop", favorite_mood="happy",
    target_energy=0.8, likes_acoustic=False,
)

_LOFI_USER = UserProfile(
    favorite_genre="lofi", favorite_mood="chill",
    target_energy=0.4, likes_acoustic=True,
)

_SCORED_SONGS = [
    (
        {"title": "Test Pop Track", "artist": "Test Artist",
         "genre": "pop", "mood": "happy", "energy": 0.8, "acousticness": 0.2},
        0.90,
        "Matches your pop genre preference. Matches your happy mood.",
    ),
    (
        {"title": "Chill Lofi Loop", "artist": "Test Artist",
         "genre": "lofi", "mood": "chill", "energy": 0.4, "acousticness": 0.9},
        0.20,
        "Broad match based on your profile.",
    ),
]


# ── Recommender core tests ────────────────────────────────────────────────────

def test_recommend_returns_songs_sorted_by_score():
    results = _make_recommender().recommend(_POP_USER, k=2)
    assert len(results) == 2
    assert results[0].genre == "pop"
    assert results[0].mood == "happy"


def test_recommend_respects_k():
    results = _make_recommender().recommend(_POP_USER, k=1)
    assert len(results) == 1


def test_explain_recommendation_returns_non_empty_string():
    rec = _make_recommender()
    explanation = rec.explain_recommendation(_POP_USER, rec.songs[0])
    assert isinstance(explanation, str)
    assert explanation.strip() != ""


def test_score_song_genre_match_adds_weight():
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    song = {"title": "T", "genre": "pop", "mood": "sad", "energy": 0.8, "acousticness": 0.2}
    score, reasons = score_song(user_prefs, song)
    assert score >= 0.40
    assert any("pop" in r for r in reasons)


def test_score_song_no_match_returns_only_energy_component():
    user_prefs = {"genre": "jazz", "mood": "relaxed", "energy": 0.5, "likes_acoustic": False}
    song = {"title": "T", "genre": "rock", "mood": "intense", "energy": 0.5, "acousticness": 0.5}
    score, _ = score_song(user_prefs, song)
    # Only energy proximity contributes; 1.0 - |0.5-0.5| = 1.0 → 0.20
    assert abs(score - 0.20) < 0.01


def test_recommend_songs_returns_correct_count():
    songs = [
        {"title": f"Song {i}", "genre": "pop", "mood": "happy",
         "energy": 0.8, "acousticness": 0.2}
        for i in range(8)
    ]
    user_prefs = {"genre": "pop", "mood": "happy", "energy": 0.8, "likes_acoustic": False}
    results = recommend_songs(user_prefs, songs, k=5)
    assert len(results) == 5


def test_recommend_songs_empty_catalog():
    results = recommend_songs({"genre": "pop"}, songs=[], k=5)
    assert results == []


# ── Confidence label tests ────────────────────────────────────────────────────

def test_confidence_label_high():
    assert confidence_label(0.70) == "High"
    assert confidence_label(1.00) == "High"


def test_confidence_label_medium():
    assert confidence_label(0.40) == "Medium"
    assert confidence_label(0.69) == "Medium"


def test_confidence_label_low():
    assert confidence_label(0.00) == "Low"
    assert confidence_label(0.39) == "Low"


# ── build_context tests ───────────────────────────────────────────────────────

def test_build_context_contains_song_titles():
    ctx = build_context(_POP_USER, _SCORED_SONGS)
    assert "Test Pop Track" in ctx
    assert "Chill Lofi Loop" in ctx


def test_build_context_contains_scores():
    ctx = build_context(_POP_USER, _SCORED_SONGS)
    assert "0.90" in ctx
    assert "0.20" in ctx


def test_build_context_contains_user_info():
    ctx = build_context(_POP_USER, _SCORED_SONGS)
    assert "pop" in ctx
    assert "happy" in ctx


# ── generate_recommendation edge cases ───────────────────────────────────────

def test_generate_recommendation_empty_songs_no_api_call():
    result = generate_recommendation(_POP_USER, scored_songs=[])
    assert "No songs" in result or "no songs" in result.lower()


def test_generate_recommendation_uses_claude():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great picks just for you!")]
    mock_client.messages.create.return_value = mock_response

    result = generate_recommendation(_POP_USER, _SCORED_SONGS, client=mock_client)
    assert result == "Great picks just for you!"
    mock_client.messages.create.assert_called_once()


def test_generate_recommendation_handles_api_error():
    mock_client = MagicMock()
    mock_client.messages.create.side_effect = Exception("connection refused")

    result = generate_recommendation(_POP_USER, _SCORED_SONGS, client=mock_client)
    assert "unavailable" in result.lower() or "AI generation" in result


def test_generate_recommendation_prompt_contains_song_titles():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Here are your songs.")]
    mock_client.messages.create.return_value = mock_response

    generate_recommendation(_POP_USER, _SCORED_SONGS, client=mock_client)

    call_kwargs = mock_client.messages.create.call_args
    prompt_text = str(call_kwargs)
    assert "Test Pop Track" in prompt_text


# ── AIRecommender.top_confidence ──────────────────────────────────────────────

def test_top_confidence_high():
    score, label = AIRecommender.top_confidence(_SCORED_SONGS)
    assert score == 0.90
    assert label == "High"


def test_top_confidence_medium():
    medium_songs = [(_SCORED_SONGS[0][0], 0.55, "reason")]
    score, label = AIRecommender.top_confidence(medium_songs)
    assert label == "Medium"


def test_top_confidence_empty():
    score, label = AIRecommender.top_confidence([])
    assert score == 0.0
    assert label == "None"


# ── Genre guide retrieval tests ───────────────────────────────────────────────

_SAMPLE_GUIDE = """
## pop

Pop music is built for broad appeal with catchy melodies and polished production.

---

## lofi

Lofi is defined by its imperfections and calm atmosphere.

---
"""


def test_retrieve_genre_context_known_genre():
    ctx = retrieve_genre_context("pop", _SAMPLE_GUIDE)
    assert "broad appeal" in ctx
    assert "catchy melodies" in ctx


def test_retrieve_genre_context_case_insensitive():
    ctx = retrieve_genre_context("POP", _SAMPLE_GUIDE)
    assert "broad appeal" in ctx


def test_retrieve_genre_context_unknown_genre():
    ctx = retrieve_genre_context("classical", _SAMPLE_GUIDE)
    assert ctx == ""


def test_retrieve_genre_context_empty_guide():
    ctx = retrieve_genre_context("pop", "")
    assert ctx == ""


def test_load_genre_guide_missing_file():
    result = load_genre_guide("nonexistent/path/guide.md")
    assert result == ""


def test_build_context_includes_genre_context():
    ctx = build_context(_POP_USER, _SCORED_SONGS, genre_context="Pop is energetic and catchy.")
    assert "Pop is energetic and catchy." in ctx
    assert "Test Pop Track" in ctx


def test_build_context_without_genre_context():
    ctx = build_context(_POP_USER, _SCORED_SONGS, genre_context="")
    assert "genre guide" not in ctx.lower()
    assert "Test Pop Track" in ctx


def test_generate_recommendation_prompt_includes_genre_context():
    mock_client = MagicMock()
    mock_response = MagicMock()
    mock_response.content = [MagicMock(text="Great picks!")]
    mock_client.messages.create.return_value = mock_response

    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as f:
        f.write("## pop\n\nPop is fun and energetic.\n\n---\n")
        tmp_path = f.name

    try:
        generate_recommendation(_POP_USER, _SCORED_SONGS, client=mock_client, genre_guide_path=tmp_path)
        prompt_text = str(mock_client.messages.create.call_args)
        assert "Pop is fun and energetic" in prompt_text
    finally:
        os.unlink(tmp_path)
