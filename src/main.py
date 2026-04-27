"""
CLI runner for VibeFinder AI 1.0.

Usage (from the project root):
    python -m src.main
"""

import os

from dotenv import load_dotenv
load_dotenv()

from src.recommender import UserProfile
from src.ai_recommender import AIRecommender, confidence_label

_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")


def _divider() -> None:
    print("-" * 62)


def _run_profile(rec: AIRecommender, user: UserProfile, label: str) -> None:
    print(f"\nProfile: {label}")
    print(
        f"  genre={user.favorite_genre}  mood={user.favorite_mood}  "
        f"energy={user.target_energy:.1f}  acoustic={user.likes_acoustic}"
    )
    _divider()

    scored_songs, ai_reply = rec.recommend(user, k=5)
    top_score, top_label = AIRecommender.top_confidence(scored_songs)

    print(f"Top match confidence: {top_label} ({top_score:.2f})\n")
    print("AI Recommendation:")
    print(ai_reply)
    print("\nScored Songs:")
    for song, score, explanation in scored_songs:
        label_str = confidence_label(score)
        print(f"  [{label_str:6s} {score:.2f}] {song['title']} — {song['artist']}")
        print(f"           {explanation}")
    _divider()


def main() -> None:
    print("\n🎵  VibeFinder AI 1.0 — Music Recommender")
    _divider()

    rec = AIRecommender(_SONGS_PATH)

    profiles = [
        (
            UserProfile(
                favorite_genre="pop",
                favorite_mood="happy",
                target_energy=0.8,
                likes_acoustic=False,
            ),
            "Pop / Happy / High Energy",
        ),
        (
            UserProfile(
                favorite_genre="lofi",
                favorite_mood="chill",
                target_energy=0.4,
                likes_acoustic=True,
            ),
            "Lofi / Chill / Acoustic",
        ),
        (
            UserProfile(
                favorite_genre="jazz",
                favorite_mood="relaxed",
                target_energy=0.35,
                likes_acoustic=True,
            ),
            "Jazz / Relaxed / Acoustic",
        ),
    ]

    for user, label in profiles:
        _run_profile(rec, user, label)


if __name__ == "__main__":
    main()
