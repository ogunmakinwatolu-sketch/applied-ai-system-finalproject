# 🎧 Model Card: VibeFinder AI 1.0

## 1. Model Name

**VibeFinder AI 1.0**

---

## 2. Intended Use

VibeFinder AI is designed to suggest 3–5 songs from a small catalog based on a user's
preferred genre, mood, energy level, and acoustic taste.
It is built for classroom exploration of how AI recommendation systems work — not for
production use with real users.
The system demonstrates a RAG (Retrieval-Augmented Generation) pipeline: a deterministic
scoring engine selects candidates, and a language model (Claude) narrates the recommendation.

---

## 3. How the Model Works

**Step 1 — Retrieve:** When you provide your taste profile (favorite genre, mood, energy
level, and whether you like acoustic music), the system reads through every song in the
catalog and gives each one a score from 0 to 1. The score is calculated using four rules:

- Genre match adds 40 % to the score
- Mood match adds 30 %
- How close the song's energy is to yours adds up to 20 %
- Whether the acousticness fits your preference adds 10 %

Songs are then ranked from highest to lowest score.

**Step 2 — Generate:** The top-ranked songs (with their scores and the reasons they matched)
are handed to Claude, an AI language model. Claude reads that list as context and writes a
short, friendly, personalized recommendation — using only the songs it was given.
It cannot invent songs that aren't in the retrieved list.

The score also becomes a **confidence label**: High (≥ 0.70), Medium (0.40–0.69), or Low (< 0.40).
This tells you how well the catalog actually matches your taste.

---

## 4. Data

The catalog (`data/songs.csv`) contains **10 songs** across 7 genres:

| Genre | Songs |
|---|---|
| Pop | 2 (Sunrise City, Gym Hero) |
| Lofi | 3 (Midnight Coding, Library Rain, Focus Flow) |
| Rock | 1 (Storm Runner) |
| Ambient | 1 (Spacewalk Thoughts) |
| Jazz | 1 (Coffee Shop Stories) |
| Synthwave | 1 (Night Drive Loop) |
| Indie Pop | 1 (Rooftop Lights) |

No songs were removed from the starter dataset.
The catalog was created for a classroom simulation and reflects a narrow, primarily
English-language, Western-genre perspective on music. Genres like R&B, hip-hop, classical,
K-pop, Afrobeats, and country are entirely absent.

---

## 5. Strengths

- **Transparent scoring:** Every recommendation comes with a plain-language reason
  ("Matches your pop genre preference. Energy level closely matches yours.").
- **Grounded generation:** Because Claude only sees pre-scored, real songs, it cannot
  hallucinate titles. The AI output is anchored to the retrieval step.
- **Testable without an API key:** All Claude calls can be mocked, so the test suite
  runs fully offline.
- **Works well for pop and lofi profiles:** These two genres have the most catalog
  coverage (5 of 10 songs) and consistently produce High-confidence results.
- **Confidence labels surface real gaps:** When a profile gets all Low scores, it tells
  the user the catalog doesn't serve their taste — which is honest and useful.

---

## 6. Limitations and Bias

- **Tiny catalog (10 songs):** Any genre with one song (jazz, rock, ambient, synthwave)
  will quickly run out of matches. Niche profiles get Low-confidence recommendations by default.
- **Binary genre/mood matching:** If the user says "pop" and a song is labeled "indie pop,"
  there is zero match — even though those are closely related. There is no fuzzy matching.
- **No user history:** The system treats every session as a fresh start. It has no memory
  of songs you've already heard or liked.
- **Western-genre bias:** The catalog entirely reflects Western, English-language popular
  music genres. Users whose taste centers on other traditions (e.g., Afrobeats, K-pop,
  classical) will get poor results.
- **Energy is the only continuous feature used in scoring:** Tempo, valence, and danceability
  are stored but ignored by the scorer, which means two songs with identical genre/mood/energy
  but very different feels could score identically.
- **Over-recommendation of lofi:** With three lofi songs in a 10-song catalog, any user
  with moderate energy tends to get lofi songs in their top results even if that's not their preference.

---

## 7. Evaluation

Three user profiles were tested manually by running `python -m src.main` and reviewing output:

| Profile | Top Score | Confidence | Observations |
|---|---|---|---|
| Pop / Happy / Energy 0.8 | 0.90 | High | Clear winner (Sunrise City), logical runner-up |
| Lofi / Chill / Energy 0.4 / Acoustic | 0.80 | High | Two tied high-scorers; Claude picked well |
| Jazz / Relaxed / Energy 0.35 / Acoustic | 0.80 | High | Only 1 real match; AI correctly noted catalog limits |

**Automated tests:** 22 unit tests cover scoring weights, edge cases (empty catalog, missing
fields), confidence label boundaries, context formatting, and mocked Claude responses.
All 22 pass. The API error test confirmed graceful fallback text is returned without crashing.

**What surprised me:** The jazz profile scored 0.80 — High confidence — yet only one song
was actually jazz. The score was high because *Coffee Shop Stories* matched every criterion
perfectly. The Low-confidence songs (0.26) made it clear most of the catalog was not relevant.
The system behaved honestly.

---

## 8. Future Work

- **Expand the catalog:** Even 50–100 songs would dramatically improve niche profile results.
- **Fuzzy genre/mood matching:** Treat "indie pop" as a partial match for "pop"; use genre
  similarity rather than exact string equality.
- **Use more audio features:** Incorporate tempo range, valence, and danceability into scoring
  for finer-grained matching.
- **User history and feedback loop:** Let users thumbs-up/thumbs-down recommendations so
  the scoring weights adapt over sessions.
- **Group recommendations:** Accept multiple user profiles and find songs that satisfy
  the intersection of several taste preferences.
- **Real catalog integration:** Connect to the Spotify API or MusicBrainz for a live,
  large-scale catalog instead of a static CSV.

---

## 9. Personal Reflection

The most important thing I learned is that RAG is not just a technique — it's a design
philosophy. By separating *what to retrieve* (rule-based, auditable, testable) from *how to
explain it* (generative, natural, expressive), you get a system that is both trustworthy
and pleasant to use. Neither part alone would be as good: pure rules feel robotic, pure
generation hallucinates.

Building this also changed how I think about Spotify and YouTube Music. Those systems must
face the same fundamental tension: how do you keep recommendations grounded in real listening
data while also surfacing new, diverse content? My system cheats — it only picks from 10 songs.
Real recommenders have to balance a catalog of 100 million songs with billions of user signals.
The confidence label idea, though, scales: every real recommender should tell you *why it thinks
you'll like something* and *how confident it is*, not just silently surface a result.
