# Movie Recommender 6.0



## Run locally

```bash
pip install -r requirements.txt
streamlit run app/app.py
```

## Add real movie posters

Create a free TMDB API key and add it as an environment variable:

```bash
export TMDB_API_KEY=your_key_here
```

On Render, add `TMDB_API_KEY` in your service Environment settings.

## Deploy on Render

This repo includes `render.yaml` with the correct start command:

```bash
streamlit run app/app.py --server.port $PORT --server.address 0.0.0.0
```
