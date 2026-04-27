"""
Streamlit UI for the AI Music Recommender (VibeFinder AI 1.0).

Run from the project root:
    streamlit run src/app.py
"""

import os
import sys

# Ensure the project root is on sys.path when streamlit launches this file directly
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

import streamlit as st

from src.recommender import UserProfile
from src.ai_recommender import AIRecommender, confidence_label

_SONGS_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "songs.csv")

st.set_page_config(
    page_title="VibeFinder AI",
    page_icon="🎵",
    layout="centered",
)

st.title("🎵 VibeFinder AI 1.0")
st.caption("Music recommendations powered by Retrieval-Augmented Generation · Claude")

# ── Sidebar: user profile ─────────────────────────────────────────────────────
with st.sidebar:
    st.header("Your Taste Profile")
    genre = st.selectbox(
        "Favorite Genre",
        ["pop", "lofi", "rock", "jazz", "ambient", "synthwave", "indie pop"],
    )
    mood = st.selectbox(
        "Favorite Mood",
        ["happy", "chill", "intense", "relaxed", "moody", "focused"],
    )
    energy = st.slider(
        "Energy Level",
        0.0, 1.0, 0.7, step=0.05,
        help="0 = very calm · 1 = very energetic",
    )
    likes_acoustic = st.checkbox("I like acoustic music")
    k = st.slider("Songs to retrieve", 3, 10, 5)
    run = st.button("Get Recommendations", type="primary", use_container_width=True)

# ── Main panel ────────────────────────────────────────────────────────────────
if run:
    user = UserProfile(
        favorite_genre=genre,
        favorite_mood=mood,
        target_energy=energy,
        likes_acoustic=likes_acoustic,
    )

    with st.spinner("Retrieving songs and generating your recommendation…"):
        try:
            rec = AIRecommender(_SONGS_PATH)
            scored_songs, ai_reply = rec.recommend(user, k=k)
            top_score, top_label = AIRecommender.top_confidence(scored_songs)
        except Exception as exc:
            st.error(f"Something went wrong: {exc}")
            st.stop()

    st.subheader("Your Personalized Recommendation")
    st.info(ai_reply)

    col1, col2 = st.columns(2)
    col1.metric("Top Match Confidence", top_label)
    col2.metric("Best Match Score", f"{top_score:.2f}")

    st.divider()
    st.subheader(f"Retrieved Songs — {len(scored_songs)} matches")

    for song, score, explanation in scored_songs:
        label = confidence_label(score)
        icon = "⭐" if label == "High" else ("✓" if label == "Medium" else "·")
        with st.expander(
            f"{icon} **{song['title']}** · {song['artist']}  —  {label} confidence ({score:.2f})"
        ):
            c1, c2 = st.columns(2)
            c1.markdown(f"**Genre:** {song['genre']}  \n**Mood:** {song['mood']}")
            c2.markdown(
                f"**Energy:** {song['energy']:.2f}  \n**Acousticness:** {song['acousticness']:.2f}"
            )
            st.caption(f"Why: {explanation}")
else:
    st.info(
        "Set your taste profile in the sidebar and click **Get Recommendations** to begin."
    )

    with st.expander("How does this work?"):
        st.markdown(
            """
**VibeFinder AI uses a two-step RAG pipeline:**

1. **Retrieve** — A deterministic scoring engine reads every song in the catalog
   and ranks them by how well they match your genre, mood, energy, and acoustic preference.

2. **Generate** — The top-ranked songs (with their scores and reasons) are sent to
   Claude as context. Claude writes a warm, personalized recommendation *using only
   those songs* — it never invents titles.

This means the recommendations are both grounded in real data and expressed naturally.
"""
        )
