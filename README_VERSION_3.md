# Movie Recommender System 3.0

This is an upgraded version of your movie recommender project.

## What 3.0 Adds

- Cleaner `src/` architecture
- Streamlit app with tabs
- User profile and movie rating system
- Custom recommendations from user ratings
- Advanced movie search
- Similar movie finder
- Chatbox-style movie recommender
- Metrics dashboard
- Interview talking points
- Dockerfile for deployment

## Files Included

```text
app/app.py
src/config.py
src/data_loader.py
src/recommender.py
src/user_profiles.py
src/search_engine.py
src/chatbot.py
src/poster_utils.py
src/metrics.py
src/utils.py
requirements.txt
Dockerfile
.gitignore
tests/test_imports.py
```

## Required Existing Data

You must already have:

```text
data/processed/train.csv
data/processed/val.csv
data/processed/test.csv
data/processed/movies_clean.csv
data/processed/genre_encoded.csv
```

These should come from your data-cleaning notebook.

## Install

From the root of your project:

```bash
pip install -r requirements.txt
```

## Run

```bash
streamlit run app/app.py
```

## Docker Run

```bash
docker build -t movie-recommender .
docker run -p 8501:8501 movie-recommender
```

## Recommended GitHub Commit

```bash
git add .
git commit -m "Upgrade recommender system to version 3.0"
git push
```

## Portfolio Summary

Built a hybrid movie recommendation system using collaborative filtering, matrix factorization, genre-based personalization, popularity fallback, custom user profiles, chatbot-style search, and ranking evaluation metrics.
