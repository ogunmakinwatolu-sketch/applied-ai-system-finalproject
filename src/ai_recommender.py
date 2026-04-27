"""
AI recommendation layer using Claude (multi-source RAG pattern).

Retrieval sources:
    1. Song catalog  — scored songs from recommender.py (data/songs.csv)
    2. Genre guide   — human-readable genre descriptions (docs/genre_guide.md)

Flow:
    Retrieve  — score every song + pull matching genre context from the guide
    Augment   — format both sources into a structured context block
    Generate  — send combined context to Claude for a natural-language reply

Using two sources measurably improves output quality: Claude can speak to
what a genre *feels like*, not just which songs matched a label.
"""

import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import anthropic

from src.recommender import UserProfile, load_songs, recommend_songs

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"
_GENRE_GUIDE_PATH = os.path.join(
    os.path.dirname(__file__), "..", "docs", "genre_guide.md"
)


# ── Confidence helpers ────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    """Map a 0-1 match score to a human-readable confidence tier."""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


# ── Genre guide retrieval ─────────────────────────────────────────────────────

def load_genre_guide(path: str = _GENRE_GUIDE_PATH) -> str:
    """Load the full genre guide document."""
    try:
        with open(path, encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        logger.warning("Genre guide not found at %s", path)
        return ""


def retrieve_genre_context(genre: str, guide_text: str) -> str:
    """
    Extract the section for a specific genre from the guide document.
    Returns an empty string if the genre is not found.
    """
    if not guide_text or not genre:
        return ""

    pattern = rf"## {re.escape(genre.lower())}\n(.*?)(?=\n---|\Z)"
    match = re.search(pattern, guide_text, re.IGNORECASE | re.DOTALL)
    if match:
        context = match.group(1).strip()
        logger.info("Genre context retrieved for '%s' (%d chars)", genre, len(context))
        return context

    logger.warning("No genre guide entry found for '%s'", genre)
    return ""


# ── RAG building blocks ───────────────────────────────────────────────────────

def build_context(
    user: UserProfile,
    scored_songs: List[Tuple[Dict, float, str]],
    genre_context: str = "",
) -> str:
    """
    Serialize both retrieval sources into a single prompt-friendly context block.
    Source 1: scored songs from the catalog
    Source 2: genre description from the guide (if available)
    """
    lines = [
        f"User: genre={user.favorite_genre}, mood={user.favorite_mood}, "
        f"energy={user.target_energy:.2f}, likes_acoustic={user.likes_acoustic}",
        "",
    ]

    if genre_context:
        lines += [
            f"Genre context for '{user.favorite_genre}' (retrieved from genre guide):",
            genre_context,
            "",
        ]

    lines.append("Retrieved songs (ranked by match score):")
    for rank, (song, score, explanation) in enumerate(scored_songs, 1):
        label = confidence_label(score)
        lines.append(
            f"  {rank}. \"{song['title']}\" by {song['artist']}"
            f" [{song['genre']}, {song['mood']}, energy={song['energy']:.2f}]"
            f"  score={score:.2f} ({label})"
        )
        lines.append(f"     Reason: {explanation}")

    return "\n".join(lines)


def generate_recommendation(
    user: UserProfile,
    scored_songs: List[Tuple[Dict, float, str]],
    client: Optional[anthropic.Anthropic] = None,
    model: str = _DEFAULT_MODEL,
    genre_guide_path: str = _GENRE_GUIDE_PATH,
) -> str:
    """
    Multi-source RAG generation: retrieve songs + genre context, then ask
    Claude to generate a grounded, natural-language recommendation.
    """
    if not scored_songs:
        logger.warning("generate_recommendation called with no songs")
        return (
            "No songs matched your profile well. "
            "Try broadening your genre or mood preference."
        )

    if client is None:
        client = anthropic.Anthropic()

    # Retrieve from second source: genre guide
    guide_text = load_genre_guide(genre_guide_path)
    genre_context = retrieve_genre_context(user.favorite_genre, guide_text)

    context = build_context(user, scored_songs, genre_context)

    system_prompt = (
        "You are an enthusiastic, concise music recommendation assistant. "
        "You only recommend songs that appear in the retrieved list — never invent titles. "
        "Use the genre context to make your recommendation feel specific and personal."
    )

    user_prompt = f"""{context}

Based on the retrieved songs and genre context above, write a warm, personal recommendation (under 200 words):
1. Open with one sentence acknowledging the user's taste using the genre description.
2. Highlight the top 3 songs by name, each with one sentence on why it fits.
3. Close with one observation about the user's overall vibe.
Do not use bullet points."""

    logger.info("Sending multi-source context to Claude (%s)", model)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=400,
            system=system_prompt,
            messages=[{"role": "user", "content": user_prompt}],
        )
        text = response.content[0].text.strip()
        logger.info("Claude response received (%d chars)", len(text))
        return text
    except Exception as exc:
        logger.error("Claude API error: %s", exc)
        return f"[AI generation unavailable: {exc}]"


# ── High-level class ──────────────────────────────────────────────────────────

class AIRecommender:
    """
    End-to-end pipeline:
        load catalog → score songs → retrieve genre context → generate with Claude.

    Usage:
        rec = AIRecommender("data/songs.csv")
        scored, reply = rec.recommend(user_profile, k=5)
    """

    def __init__(self, songs_path: str, model: str = _DEFAULT_MODEL):
        self.songs = load_songs(songs_path)
        self.model = model
        self._client = anthropic.Anthropic()

    def recommend(
        self, user: UserProfile, k: int = 5
    ) -> Tuple[List[Tuple[Dict, float, str]], str]:
        """
        Returns (scored_songs, ai_reply).

        scored_songs — [(song_dict, score, explanation), ...] sorted best-first
        ai_reply     — Claude's natural-language recommendation
        """
        scored = recommend_songs(
            user_prefs={
                "genre": user.favorite_genre,
                "mood": user.favorite_mood,
                "energy": user.target_energy,
                "likes_acoustic": user.likes_acoustic,
            },
            songs=self.songs,
            k=k,
        )
        reply = generate_recommendation(
            user=user,
            scored_songs=scored,
            client=self._client,
            model=self.model,
        )
        return scored, reply

    @staticmethod
    def top_confidence(
        scored_songs: List[Tuple[Dict, float, str]],
    ) -> Tuple[float, str]:
        """Return (score, label) for the top-ranked song."""
        if not scored_songs:
            return 0.0, "None"
        best = scored_songs[0][1]
        return best, confidence_label(best)
