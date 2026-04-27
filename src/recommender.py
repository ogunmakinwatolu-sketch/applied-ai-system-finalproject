"""
Music Recommender core logic.

Provides:
  - Song and UserProfile dataclasses
  - Recommender class (OOP interface)
  - load_songs()       — load catalog from CSV
  - score_song()       — score one song against user preferences
  - recommend_songs()  — return top-k ranked recommendations
"""

import csv
import logging
import os
from typing import List, Dict, Tuple
from dataclasses import dataclass

# ── Logging setup ─────────────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_LOG_DIR = os.path.join(_PROJECT_ROOT, "logs")
os.makedirs(_LOG_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(_LOG_DIR, "recommender.log")),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger(__name__)


# ── Data classes ──────────────────────────────────────────────────────────────

@dataclass
class Song:
    """Represents a song and its attributes."""
    id: int
    title: str
    artist: str
    genre: str
    mood: str
    energy: float
    tempo_bpm: float
    valence: float
    danceability: float
    acousticness: float


@dataclass
class UserProfile:
    """Represents a user's taste preferences."""
    favorite_genre: str
    favorite_mood: str
    target_energy: float
    likes_acoustic: bool


# ── OOP interface ─────────────────────────────────────────────────────────────

class Recommender:
    """OOP wrapper around the scoring and ranking logic."""

    def __init__(self, songs: List[Song]):
        self.songs = songs

    def recommend(self, user: UserProfile, k: int = 5) -> List[Song]:
        if not self.songs:
            logger.warning("recommend() called on empty catalog")
            return []

        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }

        scored = []
        for song in self.songs:
            song_dict = {
                "title": song.title,
                "genre": song.genre,
                "mood": song.mood,
                "energy": song.energy,
                "acousticness": song.acousticness,
            }
            score, _ = score_song(user_prefs, song_dict)
            scored.append((song, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [song for song, _ in scored[:k]]

    def explain_recommendation(self, user: UserProfile, song: Song) -> str:
        user_prefs = {
            "genre": user.favorite_genre,
            "mood": user.favorite_mood,
            "energy": user.target_energy,
            "likes_acoustic": user.likes_acoustic,
        }
        song_dict = {
            "title": song.title,
            "genre": song.genre,
            "mood": song.mood,
            "energy": song.energy,
            "acousticness": song.acousticness,
        }
        _, reasons = score_song(user_prefs, song_dict)
        if reasons:
            return ". ".join(reasons) + "."
        return f"{song.title} is a broad match based on your profile."


# ── Functional interface ──────────────────────────────────────────────────────

def load_songs(csv_path: str) -> List[Dict]:
    """Load songs from a CSV file. Returns an empty list if the file is missing or invalid."""
    logger.info(f"Loading songs from {csv_path}")
    try:
        songs = []
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                songs.append({
                    "id":           int(row["id"]),
                    "title":        row["title"],
                    "artist":       row["artist"],
                    "genre":        row["genre"],
                    "mood":         row["mood"],
                    "energy":       float(row["energy"]),
                    "tempo_bpm":    float(row["tempo_bpm"]),
                    "valence":      float(row["valence"]),
                    "danceability": float(row["danceability"]),
                    "acousticness": float(row["acousticness"]),
                })
        logger.info(f"Loaded {len(songs)} songs successfully")
        return songs
    except FileNotFoundError:
        logger.error(f"Song file not found: {csv_path}")
        return []
    except Exception as e:
        logger.error(f"Error loading songs: {e}")
        return []


def score_song(user_prefs: Dict, song: Dict) -> Tuple[float, List[str]]:
    """
    Score a single song against user preferences.

    Weights:
      Genre match  → 0.40
      Mood match   → 0.30
      Energy match → 0.20  (continuous proximity, not binary)
      Acoustic fit → 0.10

    Returns (score, reasons) where score is in [0.0, 1.0].
    """
    score = 0.0
    reasons = []

    # Genre match (0.40)
    user_genre = str(user_prefs.get("genre", "")).lower().strip()
    song_genre = str(song.get("genre", "")).lower().strip()
    if user_genre and user_genre == song_genre:
        score += 0.40
        reasons.append(f"Matches your {song_genre} genre preference")

    # Mood match (0.30)
    user_mood = str(user_prefs.get("mood", "")).lower().strip()
    song_mood = str(song.get("mood", "")).lower().strip()
    if user_mood and user_mood == song_mood:
        score += 0.30
        reasons.append(f"Matches your {song_mood} mood")

    # Energy proximity (0.20)
    user_energy = float(user_prefs.get("energy", 0.5))
    song_energy = float(song.get("energy", 0.5))
    energy_proximity = 1.0 - abs(user_energy - song_energy)
    score += 0.20 * energy_proximity
    if energy_proximity >= 0.85:
        reasons.append(
            f"Energy level ({song_energy:.2f}) closely matches yours ({user_energy:.2f})"
        )

    # Acoustic fit (0.10)
    likes_acoustic = bool(user_prefs.get("likes_acoustic", False))
    acousticness = float(song.get("acousticness", 0.5))
    if likes_acoustic and acousticness >= 0.6:
        score += 0.10
        reasons.append(f"High acousticness ({acousticness:.2f}) fits your preference")
    elif not likes_acoustic and acousticness <= 0.4:
        score += 0.10
        reasons.append(f"Low acousticness ({acousticness:.2f}) suits your preference")

    title = song.get("title", "unknown")
    logger.debug(f"  '{title}' -> score={score:.3f} | {reasons}")

    return (score, reasons)


def recommend_songs(
    user_prefs: Dict, songs: List[Dict], k: int = 5
) -> List[Tuple[Dict, float, str]]:
    """
    Score, rank, and return the top-k songs for a user.

    Returns a list of (song_dict, score, explanation) tuples, highest score first.
    """
    if not songs:
        logger.warning("recommend_songs called with empty catalog")
        return []

    k = min(k, len(songs))

    scored = []
    for song in songs:
        score, reasons = score_song(user_prefs, song)
        explanation = (
            ". ".join(reasons) + "."
            if reasons
            else f"{song.get('title', 'This song')} is a broad match based on your profile."
        )
        scored.append((song, score, explanation))

    scored.sort(key=lambda x: x[1], reverse=True)
    top_k = scored[:k]

    logger.info(f"Returning {len(top_k)} recommendations")
    return top_k
