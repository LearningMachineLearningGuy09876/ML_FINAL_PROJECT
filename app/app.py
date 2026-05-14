import os
import sys
import re
import random
import base64
import requests
from urllib.parse import quote_plus

import pandas as pd
import streamlit as st

# ----------------------------------------------------------
# PATH SETUP
# ----------------------------------------------------------
APP_DIR = os.path.dirname(__file__)
BASE_DIR = os.path.abspath(os.path.join(APP_DIR, ".."))
SRC_DIR = os.path.join(BASE_DIR, "src")

if SRC_DIR not in sys.path:
    sys.path.append(SRC_DIR)

# ----------------------------------------------------------
# IMPORT PROJECT MODULES
# ----------------------------------------------------------
from config import (
    MODEL_PATH,
    USER_RATINGS_PATH,
    RECOMMENDER_PARAMS,
    DEFAULT_TOP_N,
)
from data_loader import load_processed_data
from recommender import HybridRecommender
from user_profiles import UserProfileStore
from search_engine import MovieSearchEngine
from chatbot import MovieChatbot

# ----------------------------------------------------------
# PAGE CONFIG
# ----------------------------------------------------------
st.set_page_config(
    page_title="FlikPik AI",
    page_icon="🎬",
    layout="wide",
)

# ----------------------------------------------------------
# API KEYS
# ----------------------------------------------------------
TMDB_API_KEY = os.getenv("TMDB_API_KEY", "")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")
TMDB_SEARCH_URL = "https://api.themoviedb.org/3/search/movie"
YOUTUBE_SEARCH_URL = "https://www.googleapis.com/youtube/v3/search"

# ----------------------------------------------------------
# BRAND ASSETS
# ----------------------------------------------------------
LOGO_PATH = os.path.join(BASE_DIR, "assets", "flikpik_logo.jpg")


def get_base64_image(image_path):
    """Return a base64 string for local images used in custom HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

# ----------------------------------------------------------
# CUSTOM UI STYLES
# ----------------------------------------------------------
def inject_custom_css():
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

            :root {
                --flik-red: #e50914;
                --flik-red-soft: rgba(229, 9, 20, 0.28);
                --panel: rgba(18, 18, 18, 0.82);
                --panel-strong: rgba(23, 23, 23, 0.96);
                --stroke: rgba(255, 255, 255, 0.10);
                --muted: #a9a9a9;
                --text: #f7f7f7;
            }

            html, body, [class*="css"] {
                font-family: 'Inter', system-ui, -apple-system, BlinkMacSystemFont, sans-serif;
            }

            .stApp {
                background:
                    radial-gradient(circle at 82% 8%, rgba(229, 9, 20, 0.42) 0%, rgba(229, 9, 20, 0.04) 32%, transparent 50%),
                    radial-gradient(circle at 14% 2%, rgba(255, 255, 255, 0.08) 0%, transparent 26%),
                    linear-gradient(145deg, #050505 0%, #0d0d0f 45%, #030303 100%);
                color: var(--text);
            }

            #MainMenu, footer, header {visibility: hidden;}
            .block-container {
                padding-top: 1.15rem !important;
                padding-bottom: 3rem !important;
                max-width: 1500px !important;
            }

            [data-testid="stSidebar"] {
                background: linear-gradient(180deg, rgba(6,6,7,0.98), rgba(13,13,15,0.96));
                border-right: 1px solid var(--stroke);
                box-shadow: 22px 0 60px rgba(0,0,0,0.36);
            }
            [data-testid="stSidebar"] .block-container {
                padding-top: 1.25rem !important;
            }
            [data-testid="stSidebar"] img {
                border-radius: 18px;
                box-shadow: 0 18px 44px rgba(0,0,0,0.42);
            }
            [data-testid="stSidebar"] label, [data-testid="stSidebar"] p, [data-testid="stSidebar"] span {
                color: #d7d7d7 !important;
            }
            [data-testid="stSidebar"] [role="radiogroup"] label {
                background: transparent;
                border-radius: 14px;
                padding: .45rem .55rem;
                margin: .10rem 0;
                transition: all .18s ease;
            }
            [data-testid="stSidebar"] [role="radiogroup"] label:hover {
                background: rgba(255,255,255,0.06);
            }
            [data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {
                background: linear-gradient(90deg, rgba(229,9,20,0.92), rgba(125,8,12,0.72));
                box-shadow: 0 10px 28px rgba(229,9,20,0.18);
            }

            h1, h2, h3, h4 { color: #fff !important; letter-spacing: -0.035em; }
            .stMarkdown, .stTextInput, .stSelectbox, .stSlider, .stRadio { color: var(--text); }

            div[data-testid="stTextInput"] input,
            div[data-testid="stTextArea"] textarea,
            div[data-testid="stSelectbox"] div[data-baseweb="select"] > div {
                background: rgba(255,255,255,0.075) !important;
                border: 1px solid rgba(255,255,255,0.10) !important;
                border-radius: 16px !important;
                color: white !important;
                box-shadow: inset 0 1px 0 rgba(255,255,255,0.05);
            }
            div[data-testid="stTextInput"] input:focus,
            div[data-testid="stTextArea"] textarea:focus {
                border-color: rgba(229,9,20,0.78) !important;
                box-shadow: 0 0 0 3px rgba(229,9,20,0.18) !important;
            }

            .hero-shell {
                position: relative;
                overflow: hidden;
                min-height: 235px;
                padding: 2.1rem 2.25rem;
                margin-bottom: 1.1rem;
                border-radius: 30px;
                border: 1px solid rgba(255,255,255,0.11);
                background:
                    linear-gradient(90deg, rgba(3,3,3,0.98) 0%, rgba(12,12,13,0.92) 44%, rgba(137,7,13,0.85) 100%),
                    radial-gradient(circle at 78% 30%, rgba(229,9,20,0.55), transparent 38%);
                box-shadow: 0 25px 80px rgba(0,0,0,0.48);
            }
            .hero-shell::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(120deg, transparent 0%, rgba(255,255,255,0.05) 35%, transparent 60%);
                pointer-events: none;
            }
            .hero-grid {
                position: relative;
                z-index: 1;
                display: grid;
                grid-template-columns: 170px 1fr;
                gap: 1.6rem;
                align-items: center;
            }
            .hero-logo {
                width: 160px;
                height: 160px;
                object-fit: cover;
                border-radius: 28px;
                box-shadow: 0 18px 60px rgba(0,0,0,0.42);
            }
            .hero-kicker {
                color: #ff4d56;
                font-weight: 800;
                text-transform: uppercase;
                letter-spacing: .16em;
                font-size: .78rem;
                margin-bottom: .15rem;
            }
            .hero-title {
                font-size: clamp(3rem, 7vw, 6rem);
                font-weight: 950;
                line-height: .88;
                margin: 0;
                color: white;
                text-shadow: 0 18px 50px rgba(0,0,0,0.45);
            }
            .hero-tagline {
                font-size: clamp(1.35rem, 2.7vw, 2.35rem);
                font-weight: 800;
                color: white;
                margin-top: .45rem;
            }
            .hero-copy {
                max-width: 760px;
                color: #d8d8d8;
                margin-top: .7rem;
                font-size: 1rem;
            }
            .ai-badge-row { margin-top: 1rem; display: flex; flex-wrap: wrap; gap: .55rem; }
            .ai-badge {
                display: inline-flex;
                align-items: center;
                gap: .35rem;
                padding: .42rem .72rem;
                border-radius: 999px;
                background: rgba(255,255,255,0.08);
                border: 1px solid rgba(255,255,255,0.10);
                color: #f2f2f2;
                font-size: .82rem;
                font-weight: 650;
            }

            .search-panel {
                margin: .3rem 0 1.2rem 0;
                padding: .85rem;
                border-radius: 22px;
                background: rgba(255,255,255,0.055);
                border: 1px solid rgba(255,255,255,0.09);
                box-shadow: 0 16px 45px rgba(0,0,0,0.26);
            }

            .section-head {
                display: flex;
                align-items: center;
                justify-content: space-between;
                gap: 1rem;
                margin: 1.35rem 0 .75rem 0;
            }
            .section-title {
                font-size: 1.55rem;
                font-weight: 900;
                color: #fff;
            }
            .section-link {
                color: #ff313c;
                font-weight: 800;
                font-size: .92rem;
            }
            .section-note { color: var(--muted); margin-top: -.35rem; margin-bottom: 1rem; }

            .movie-card {
                min-height: 398px;
                padding: .72rem;
                border-radius: 18px;
                background: linear-gradient(180deg, rgba(255,255,255,0.075), rgba(255,255,255,0.035));
                border: 1px solid rgba(255,255,255,0.10);
                box-shadow: 0 16px 38px rgba(0,0,0,0.35);
                transition: transform .18s ease, border-color .18s ease, background .18s ease;
                overflow: hidden;
                margin-bottom: 1rem;
            }
            .movie-card:hover {
                transform: translateY(-7px);
                border-color: rgba(229,9,20,0.78);
                background: linear-gradient(180deg, rgba(255,255,255,0.105), rgba(255,255,255,0.052));
            }
            .movie-card img { border-radius: 14px !important; }
            .movie-title {
                font-weight: 850;
                font-size: .96rem;
                line-height: 1.17;
                margin-top: .68rem;
                color: #ffffff;
                min-height: 38px;
            }
            .movie-meta { font-size: .76rem; color: #bdbdbd; min-height: 30px; }
            .pill {
                display: inline-block;
                padding: .19rem .48rem;
                margin: .12rem .10rem .12rem 0;
                border-radius: 999px;
                background: rgba(229,9,20,0.20);
                border: 1px solid rgba(229,9,20,0.42);
                color: #fff;
                font-size: .68rem;
                font-weight: 700;
            }
            .score-line { color: #dedede; font-size: .75rem; margin-top: .32rem; }
            .metric-card {
                padding: 1rem;
                border-radius: 20px;
                background: rgba(255,255,255,0.065);
                border: 1px solid rgba(255,255,255,0.10);
            }
            .status-box {
                padding: .85rem 1rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.065);
                border: 1px solid rgba(255,255,255,0.10);
                margin-bottom: 1rem;
            }
            .small-red { color: #ff313c; font-size: .75rem; font-weight: 900; letter-spacing: .08em; text-transform: uppercase; }
            button[kind="secondary"], button[kind="primary"] { border-radius: 999px !important; font-weight: 800 !important; }

            @media (max-width: 900px) {
                .hero-grid { grid-template-columns: 1fr; }
                .hero-logo { width: 110px; height: 110px; }
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    logo_base64 = get_base64_image(LOGO_PATH)
    logo_html = ""
    if logo_base64:
        logo_html = f'<img class="hero-logo" src="data:image/jpg;base64,{logo_base64}" alt="FlikPik AI logo">'
    else:
        logo_html = '<div class="hero-logo" style="display:grid;place-items:center;background:#111;font-size:4rem;">🎬</div>'

    st.markdown(
        f"""
        <div class="hero-shell">
            <div class="hero-grid">
                <div>{logo_html}</div>
                <div>
                    <div class="hero-kicker">AI entertainment discovery platform</div>
                    <div class="hero-title">FlikPik AI</div>
                    <div class="hero-tagline">Sit tight, we got you!</div>
                    <div class="hero-copy">Find what to watch next with hybrid recommendations, AI search, social buzz, trailers, streaming options, and personalized taste insights.</div>
                    <div class="ai-badge-row">
                        <span class="ai-badge">🤖 Hybrid AI</span>
                        <span class="ai-badge">🔥 Trending intelligence</span>
                        <span class="ai-badge">🎞 Real posters</span>
                        <span class="ai-badge">▶ Trailers</span>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------------------------------------
# UTILS
# ----------------------------------------------------------
def shorten(text, max_len=72):
    text = "" if text is None else str(text)
    return text if len(text) <= max_len else text[: max_len - 3] + "..."


def clean_movie_title(title):
    if not isinstance(title, str):
        return ""
    return re.sub(r"\s*\(\d{4}\)\s*$", "", title).strip()


def extract_year(title):
    if not isinstance(title, str):
        return None
    match = re.search(r"\((\d{4})\)\s*$", title)
    return match.group(1) if match else None


def placeholder_poster(title):
    safe_title = quote_plus(clean_movie_title(title) or "Movie")
    return f"https://placehold.co/342x513/141414/FFFFFF?text={safe_title}"


@st.cache_data(show_spinner=False)
def fetch_tmdb_poster(title):
    """Fetch a poster from TMDB. Cached so cards do not call the API repeatedly."""
    if not TMDB_API_KEY:
        return None

    clean_title = clean_movie_title(title)
    if not clean_title:
        return None

    params = {
        "api_key": TMDB_API_KEY,
        "query": clean_title,
        "include_adult": "false",
    }

    year = extract_year(title)
    if year:
        params["year"] = year

    try:
        response = requests.get(TMDB_SEARCH_URL, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        results = data.get("results", [])

        if not results and year:
            params.pop("year", None)
            response = requests.get(TMDB_SEARCH_URL, params=params, timeout=8)
            response.raise_for_status()
            results = response.json().get("results", [])

        for movie in results:
            poster_path = movie.get("poster_path")
            if poster_path:
                return f"https://image.tmdb.org/t/p/w500{poster_path}"

    except Exception:
        return None

    return None


def get_poster_url(title):
    return fetch_tmdb_poster(title) or placeholder_poster(title)


@st.cache_data(show_spinner=False)
def poster_url_works(url):
    """Validate that a poster URL actually returns a loadable image."""
    if not url:
        return False

    try:
        response = requests.get(url, timeout=6, stream=True)
        content_type = response.headers.get("Content-Type", "")
        response.close()
        return response.status_code == 200 and "image" in content_type.lower()
    except Exception:
        return False


# ----------------------------------------------------------
# CACHE LOADERS
# ----------------------------------------------------------
@st.cache_data
def cached_data():
    return load_processed_data()


@st.cache_resource
def cached_model(train, movies, genres):
    if os.path.exists(MODEL_PATH):
        try:
            return HybridRecommender.load(MODEL_PATH)
        except Exception:
            pass

    model = HybridRecommender(
        train_df=train,
        movies_df=movies,
        genre_df=genres,
        **RECOMMENDER_PARAMS,
    ).fit()

    model.save(MODEL_PATH)
    return model


# ----------------------------------------------------------
# MOCK LIVE FEATURES
# ----------------------------------------------------------
def get_streaming(title):
    platforms = [
        "Netflix",
        "Prime Video",
        "Max",
        "Hulu",
        "Disney+",
        "Peacock",
        "Paramount+",
    ]
    random.seed(abs(hash(title)) % 100000)
    return random.sample(platforms, random.randint(1, 3))


def get_social(title):
    random.seed(abs(hash(title)) % 100000)
    return {
        "X": random.randint(1000, 40000),
        "Reddit": random.randint(500, 12000),
        "Facebook": random.randint(1000, 25000),
        "Instagram": random.randint(2000, 55000),
    }


def get_sentiment(title):
    random.seed(abs(hash(title)) % 100000)
    return random.randint(74, 97)


def get_youtube_search_url(title):
    query = quote_plus(f"{clean_movie_title(title)} movie trailer review")
    return f"https://www.youtube.com/results?search_query={query}"


@st.cache_data(show_spinner=False)
def get_youtube_trailer(title):
    if not YOUTUBE_API_KEY:
        return None

    params = {
        "part": "snippet",
        "q": f"{clean_movie_title(title)} official movie trailer",
        "key": YOUTUBE_API_KEY,
        "type": "video",
        "maxResults": 1,
        "videoEmbeddable": "true",
        "safeSearch": "moderate",
    }

    try:
        response = requests.get(YOUTUBE_SEARCH_URL, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        items = data.get("items", [])
        if not items:
            return None

        item = items[0]
        video_id = item["id"]["videoId"]
        snippet = item["snippet"]

        return {
            "title": snippet.get("title", ""),
            "channel": snippet.get("channelTitle", ""),
            "thumbnail": snippet.get("thumbnails", {}).get("high", {}).get("url", ""),
            "url": f"https://www.youtube.com/watch?v={video_id}",
        }
    except Exception:
        return None


def hype_score(title):
    buzz = get_social(title)
    mentions = sum(buzz.values())
    sentiment = get_sentiment(title)
    score = ((mentions / 130000) * 0.60 + (sentiment / 100) * 0.40) * 100
    return round(score, 2)


# ----------------------------------------------------------
# DISPLAY HELPERS
# ----------------------------------------------------------
def show_movie_table(df, limit=20):
    if df is None or df.empty:
        st.info("No results found.")
        return
    st.dataframe(df.head(limit), use_container_width=True)


def show_movie_cards(df, score_col=None, max_items=24):
    if df is None or df.empty:
        st.info("No movies found.")
        return

    # Remove duplicate titles so the UI does not show the same movie more than once.
    if "title" in df.columns:
        df = df.drop_duplicates(subset=["title"])

    # ------------------------------------------------------
    # FILTER BEFORE RENDERING
    # ------------------------------------------------------
    # Only keep movies that have a real TMDB poster AND whose image URL loads.
    # This prevents broken images from holding empty space in the Streamlit layout.
    filtered_records = []

    for row in df.to_dict("records"):
        title = row.get("title", "Unknown")
        poster = fetch_tmdb_poster(title)

        if poster and poster_url_works(poster):
            row["poster_url"] = poster
            filtered_records.append(row)

        if len(filtered_records) >= max_items:
            break

    if not filtered_records:
        if not TMDB_API_KEY:
            st.warning("TMDB_API_KEY is missing. Add your TMDB key so the app can find movies with real posters.")
        else:
            st.warning("No movies with working posters were found for this section.")
        return

    cols_per_row = 6

    # ------------------------------------------------------
    # DISPLAY ONLY MOVIES WITH WORKING POSTERS
    # ------------------------------------------------------
    for row_start in range(0, len(filtered_records), cols_per_row):
        row_items = filtered_records[row_start : row_start + cols_per_row]

        # Create only as many columns as there are valid poster cards.
        cols = st.columns(len(row_items))

        for col, row in zip(cols, row_items):
            title = row.get("title", "Unknown")
            genres = row.get("genres", "")
            movie_id = row.get("movieId", "")
            poster = row["poster_url"]
            stream = get_streaming(title)
            total_buzz = sum(get_social(title).values())

            score_html = ""
            if score_col and score_col in row:
                try:
                    score_html = f"<div class='score-line'>Score: {float(row[score_col]):.3f}</div>"
                except Exception:
                    score_html = ""

            platform_html = "".join([f"<span class='pill'>{p}</span>" for p in stream[:3]])

            with col:
                st.markdown("<div class='movie-card'>", unsafe_allow_html=True)
                st.image(poster, use_container_width=True)
                st.markdown(
                    f"""
                    <div class='movie-title'>{shorten(title, 58)}</div>
                    <div class='movie-meta'>{shorten(genres, 72)}</div>
                    <div>{platform_html}</div>
                    {score_html}
                    <div class='score-line'>Hype Score: {hype_score(title)} | Mentions: {total_buzz:,}</div>
                    <div class='score-line'>Movie ID: {movie_id}</div>
                    """,
                    unsafe_allow_html=True,
                )
                st.link_button("Trailer / Reviews", get_youtube_search_url(title), use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------------------------------------
# MAIN APP
# ----------------------------------------------------------
def main():
    inject_custom_css()

    try:
        train, val, test, movies, genres = cached_data()
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    model = cached_model(train, movies, genres)
    search_engine = MovieSearchEngine(movies)
    chatbot = MovieChatbot(model=model, movies_df=movies)
    profile_store = UserProfileStore(USER_RATINGS_PATH)

    users = profile_store.get_users()
    default_user = users[0] if users else "brandon"

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)
        else:
            st.markdown("## 🎬 FlikPik AI")
            st.caption("Sit tight, we got you!")

        st.markdown("<div class='small-red'>Main Menu</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigate",
            [
                "🏠 Home",
                "🔎 Ask FlikPik Search",
                "🎭 Actor / Director",
                "🎞 Similar",
                "🎯 My Recs",
                "⭐ Rate Movies",
                "📺 Streaming",
                "📱 Social Buzz",
                "▶️ Reviews & Trailers",
                "🤖 Ask FlikPik",
                "👤 Insights",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("<div class='small-red'>Account</div>", unsafe_allow_html=True)
        username = st.text_input("Profile", default_user)
        top_n = st.slider("Recommendations", 5, 36, DEFAULT_TOP_N)
        display = st.radio("Display", ["Cards", "Table"], horizontal=True)

        st.divider()
        st.markdown("<div class='small-red'>System Status</div>", unsafe_allow_html=True)
        st.write(f"🧠 Recommender: **Online**")
        st.write(f"🎞 TMDB API: **{'Connected' if TMDB_API_KEY else 'Missing'}**")
        st.write(f"▶ YouTube API: **{'Connected' if YOUTUBE_API_KEY else 'Missing'}**")
        st.caption(f"Movies: {movies.shape[0]:,} • Ratings: {train.shape[0]:,}")

        if st.button("Clear poster cache", use_container_width=True):
            st.cache_data.clear()
            st.success("Cache cleared. Rerun the app if needed.")

    render_hero()

    st.markdown("<div class='search-panel'>", unsafe_allow_html=True)
    search_cols = st.columns([5, 1])
    with search_cols[0]:
        global_search = st.text_input(
            "Search across FlikPik",
            placeholder="Search for movies, TV shows, anime...",
            label_visibility="collapsed",
            key="global_search",
        )
    with search_cols[1]:
        run_global_search = st.button("Search", type="primary", use_container_width=True)
    st.markdown("</div>", unsafe_allow_html=True)

    if global_search and (run_global_search or page == "🏠 Home"):
        st.markdown("<div class='section-head'><div class='section-title'>🔎 Search Results</div><div class='section-link'>AI matched titles</div></div>", unsafe_allow_html=True)
        results = search_engine.search(title_query=global_search, genre="All", min_year="", max_year="", limit=100)
        show_movie_cards(results, max_items=18) if display == "Cards" else show_movie_table(results)
        st.divider()

    if page == "🏠 Home":
        trending = model.popularity_df.copy()
        trending["hype"] = trending["title"].apply(hype_score)
        trending = trending.sort_values(["hype", "weighted_score"], ascending=False)

        st.markdown("<div class='section-head'><div class='section-title'>🔥 Trending This Week</div><div class='section-link'>View all →</div></div>", unsafe_allow_html=True)
        show_movie_cards(trending, "hype", max_items=12) if display == "Cards" else show_movie_table(trending)

        st.markdown("<div class='section-head'><div class='section-title'>🍿 Recommended For You</div><div class='section-link'>View all →</div></div>", unsafe_allow_html=True)
        try:
            recs = model.recommend_hybrid(user_id=1, top_n=18)
        except Exception:
            recs = model._popularity_fallback(top_n=18)
        show_movie_cards(recs, "final_score", max_items=12) if display == "Cards" else show_movie_table(recs)

    elif page == "🔎 Ask FlikPik Search":
        st.markdown("<div class='section-head'><div class='section-title'>🔎 Ask FlikPik Search</div><div class='section-link'>Smart catalog search</div></div>", unsafe_allow_html=True)
        q = st.text_input("Title contains", placeholder="Try Matrix, Toy Story, Batman...")
        results = search_engine.search(title_query=q, genre="All", min_year="", max_year="", limit=100)
        show_movie_cards(results, max_items=24) if display == "Cards" else show_movie_table(results)

    elif page == "🎭 Actor / Director":
        st.markdown("<div class='section-head'><div class='section-title'>🎭 Actor / Director Discovery</div><div class='section-link'>TMDB-powered idea zone</div></div>", unsafe_allow_html=True)
        st.info("This section is ready for your next upgrade: connect TMDB person search to discover movies by actor or director.")
        q = st.text_input("Search a movie title for now", placeholder="Example: Inception")
        results = search_engine.search(title_query=q, genre="All", min_year="", max_year="", limit=60)
        show_movie_cards(results, max_items=18) if display == "Cards" else show_movie_table(results)

    elif page == "🎞 Similar":
        st.markdown("<div class='section-head'><div class='section-title'>🎞 Similar Movie Finder</div><div class='section-link'>Nearest-neighbor picks</div></div>", unsafe_allow_html=True)
        q = st.text_input("Search movie", "Toy Story")
        matches = search_engine.title_matches(q, limit=20)
        if not matches.empty:
            chosen = st.selectbox("Choose Movie", matches["title"].tolist())
            movie_id = int(matches.loc[matches["title"] == chosen, "movieId"].iloc[0])
            similar = model.get_similar_movies(movie_id, n_neighbors=top_n)
            show_movie_cards(similar, "similarity", max_items=top_n) if display == "Cards" else show_movie_table(similar)
        else:
            st.info("No matching movies found.")

    elif page == "🎯 My Recs":
        st.markdown("<div class='section-head'><div class='section-title'>🎯 Personalized Recommendations</div><div class='section-link'>Built from your taste profile</div></div>", unsafe_allow_html=True)
        try:
            recs = model.recommend_hybrid(user_id=1, top_n=top_n)
        except Exception:
            recs = model._popularity_fallback(top_n=top_n)
        show_movie_cards(recs, "final_score", max_items=top_n) if display == "Cards" else show_movie_table(recs)

    elif page == "⭐ Rate Movies":
        st.markdown("<div class='section-head'><div class='section-title'>⭐ Build Your Profile</div><div class='section-link'>Train your recommendations</div></div>", unsafe_allow_html=True)
        q = st.text_input("Search movie to rate", "Matrix")
        matches = search_engine.title_matches(q, limit=20)
        if not matches.empty:
            selected = st.selectbox("Choose title", matches["title"].tolist())
            rating = st.slider("Your Rating", 1.0, 5.0, 4.0, 0.5)
            if st.button("Save Rating", type="primary"):
                movie_id = int(matches.loc[matches["title"] == selected, "movieId"].iloc[0])
                profile_store.add_or_update_rating(username=username, movie_id=movie_id, title=selected, rating=rating)
                st.success("Rating saved.")
        else:
            st.info("Search for a movie to rate.")

    elif page == "📺 Streaming":
        st.markdown("<div class='section-head'><div class='section-title'>📺 Where To Watch</div><div class='section-link'>Streaming availability mockup</div></div>", unsafe_allow_html=True)
        movie = st.selectbox("Select Movie", movies["title"].dropna().unique())
        platforms = get_streaming(movie)
        cols = st.columns(len(platforms))
        for col, p in zip(cols, platforms):
            with col:
                st.markdown(f"<div class='metric-card'><h3>{p}</h3><p>Available now</p></div>", unsafe_allow_html=True)

    elif page == "📱 Social Buzz":
        st.markdown("<div class='section-head'><div class='section-title'>📱 Social Buzz</div><div class='section-link'>Audience signal engine</div></div>", unsafe_allow_html=True)
        movie = st.selectbox("Movie", movies["title"].dropna().unique(), key="buzz")
        buzz = get_social(movie)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("X", f"{buzz['X']:,}")
        c2.metric("Reddit", f"{buzz['Reddit']:,}")
        c3.metric("Facebook", f"{buzz['Facebook']:,}")
        c4.metric("Instagram", f"{buzz['Instagram']:,}")
        st.subheader("Audience Sentiment")
        st.progress(get_sentiment(movie) / 100)

    elif page == "▶️ Reviews & Trailers":
        st.markdown("<div class='section-head'><div class='section-title'>▶️ Reviews & Trailers</div><div class='section-link'>YouTube discovery</div></div>", unsafe_allow_html=True)
        yt_search = st.text_input("Search movie title", placeholder="Type a movie name like Avatar, Matrix, Toy Story...", key="yt_search_input")
        if yt_search:
            movie_options = movies[movies["title"].str.contains(yt_search, case=False, na=False)]["title"].dropna().unique()
        else:
            movie_options = movies["title"].dropna().unique()

        if len(movie_options) == 0:
            st.warning("No movies found. Try another search.")
        else:
            movie = st.selectbox("Choose from results", movie_options, key="yt_movie_select")
            trailer = get_youtube_trailer(movie)
            st.markdown(f"### 🎬 {movie}")
            if trailer:
                st.caption(f"{trailer['title']} | Channel: {trailer['channel']}")
                st.video(trailer["url"], autoplay=True, muted=True)
                st.markdown(f"[Open on YouTube]({trailer['url']})")
            else:
                search_url = get_youtube_search_url(movie)
                if not YOUTUBE_API_KEY:
                    st.info("Add a YOUTUBE_API_KEY environment variable to fetch real trailers automatically.")
                else:
                    st.warning("No trailer found from the YouTube API.")
                st.markdown(f"[🔎 Search YouTube for {movie}]({search_url})")

    elif page == "🤖 Ask FlikPik":
        st.markdown("<div class='section-head'><div class='section-title'>🤖 Ask FlikPik</div><div class='section-link'>Conversational AI picks</div></div>", unsafe_allow_html=True)
        st.caption("Ask for movies by mood, genre, decade, year, or similarity. Try: 'scary but not too old', 'like Inception after 2000', or 'funny date night movies'.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        mood = st.selectbox("What mood are you in?", ["Surprise me", "Funny", "Scary", "Romantic", "Action", "Mind-bending", "Family", "Drama", "Thriller", "Sci-Fi"])
        prompt = st.text_area("Tell me what you want to watch", "Recommend exciting sci-fi movies with strong ratings after 2000")

        example_cols = st.columns(4)
        examples = ["Like The Matrix but newer", "Funny date night movies", "Scary movies but not horror", "Mind-bending thrillers from the 2000s"]
        for ex_col, example in zip(example_cols, examples):
            with ex_col:
                if st.button(example, use_container_width=True):
                    prompt = example

        if st.button("Ask Assistant", type="primary"):
            try:
                response, recs, explanations, parsed = chatbot.recommend(prompt, top_n=top_n, mood=mood)
            except ValueError:
                response, recs = chatbot.recommend(prompt, top_n=top_n)
                explanations = []
                parsed = {}

            st.session_state.chat_history.append({"mood": mood, "prompt": prompt, "response": response, "count": len(recs)})
            st.success(response)

            if parsed:
                parsed_cols = st.columns(4)
                parsed_cols[0].metric("Mood", mood)
                parsed_cols[1].metric("Genres", ", ".join(parsed.get("genres", [])) or "Any")
                parsed_cols[2].metric("Exclude", ", ".join(parsed.get("excluded_genres", [])) or "None")
                year_label = parsed.get("year") or parsed.get("decade") or parsed.get("year_min") or "Any"
                parsed_cols[3].metric("Year Filter", year_label)

            if explanations:
                with st.expander("Why these movies?", expanded=True):
                    for item in explanations[:top_n]:
                        st.markdown(f"- {item}")

            st.subheader("Recommended Movies")
            show_movie_cards(recs, "weighted_score", max_items=top_n) if display == "Cards" else show_movie_table(recs)

        with st.expander("Chat History"):
            if not st.session_state.chat_history:
                st.caption("Your assistant conversation will appear here during this session.")
            else:
                for item in reversed(st.session_state.chat_history[-8:]):
                    st.markdown(f"**You ({item['mood']}):** {item['prompt']}")
                    st.markdown(f"**Assistant:** {item['response']}")
                    st.divider()

    elif page == "👤 Insights":
        st.markdown("<div class='section-head'><div class='section-title'>👤 User Insights</div><div class='section-link'>Taste analytics</div></div>", unsafe_allow_html=True)
        ratings = profile_store.get_user_ratings(username)
        if ratings.empty:
            st.info("Rate movies first.")
        else:
            c1, c2, c3 = st.columns(3)
            c1.metric("Movies Rated", len(ratings))
            c2.metric("Average Rating", round(ratings["rating"].mean(), 2))
            c3.metric("Highest", round(ratings["rating"].max(), 2))
            st.dataframe(ratings, use_container_width=True)

    st.markdown("---")
    st.caption("FlikPik AI | AI Streaming-Style Interface | Real Posters | Deploy Ready")


if __name__ == "__main__":
    main()
