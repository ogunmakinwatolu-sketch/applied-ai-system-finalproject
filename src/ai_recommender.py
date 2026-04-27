"""
AI recommendation layer using Claude (RAG pattern).

Flow:
    Retrieve  — score every song with deterministic rules (recommender.py)
    Augment   — format top-k results into a structured context block
    Generate  — send context + user profile to Claude for a natural-language reply

The AI only generates text; all song selection is done by the scoring
engine first, so the output is grounded in real catalog data.
"""

import logging
import os
from typing import Dict, List, Optional, Tuple

import anthropic

from src.recommender import UserProfile, load_songs, recommend_songs

logger = logging.getLogger(__name__)

_DEFAULT_MODEL = "claude-haiku-4-5-20251001"


# ── Confidence helpers ────────────────────────────────────────────────────────

def confidence_label(score: float) -> str:
    """Map a 0-1 match score to a human-readable confidence tier."""
    if score >= 0.70:
        return "High"
    if score >= 0.40:
        return "Medium"
    return "Low"


# ── RAG building blocks ───────────────────────────────────────────────────────

def build_context(
    user: UserProfile,
    scored_songs: List[Tuple[Dict, float, str]],
) -> str:
    """Serialize retrieved songs into a prompt-friendly context block."""
    lines = [
        f"User: genre={user.favorite_genre}, mood={user.favorite_mood}, "
        f"energy={user.target_energy:.2f}, likes_acoustic={user.likes_acoustic}",
        "",
        "Retrieved songs (ranked by match score):",
    ]
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
) -> str:
    """
    RAG generation step: pass retrieved context to Claude and return
    a friendly, grounded natural-language recommendation.
    """
    if not scored_songs:
        logger.warning("generate_recommendation called with no songs")
        return (
            "No songs matched your profile well. "
            "Try broadening your genre or mood preference."
        )

    if client is None:
        client = anthropic.Anthropic()

    context = build_context(user, scored_songs)

    system_prompt = (
        "You are an enthusiastic, concise music recommendation assistant. "
        "You only recommend songs that appear in the retrieved list — never invent titles."
    )

    user_prompt = f"""{context}

Based on the retrieved songs above, write a warm, personal recommendation (under 180 words):
1. Open with one sentence acknowledging the user's taste.
2. Highlight the top 3 songs by name, each with one sentence on why it fits.
3. Close with one observation about the user's overall vibe.
Do not use bullet points."""

    logger.info("Sending retrieval context to Claude (%s)", model)
    try:
        response = client.messages.create(
            model=model,
            max_tokens=350,
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
    End-to-end pipeline: load catalog → score (retrieve) → generate with Claude.

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
