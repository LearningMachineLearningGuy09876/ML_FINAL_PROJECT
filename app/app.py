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
TMDB_PERSON_SEARCH_URL = "https://api.themoviedb.org/3/search/person"
TMDB_PERSON_MOVIE_CREDITS_URL = "https://api.themoviedb.org/3/person/{person_id}/movie_credits"

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
            .stApp {
                background: radial-gradient(circle at top left, #451010 0%, #141414 38%, #050505 100%);
                color: #f5f5f5;
            }
            [data-testid="stSidebar"] {
                background: rgba(8, 8, 8, 0.96);
                border-right: 1px solid rgba(255,255,255,0.08);
            }

            .hero-wrapper {
                position: relative;
                overflow: hidden;
                border-radius: 28px;
                margin-bottom: 1.5rem;
                background:
                    linear-gradient(
                        90deg,
                        rgba(0,0,0,0.98) 0%,
                        rgba(15,15,15,0.96) 55%,
                        rgba(120,0,0,0.92) 100%
                    );
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 25px 80px rgba(0,0,0,0.45);
                padding: 1.8rem 2rem 2rem 2rem;
            }
            .brand-logo-wrap {
                text-align: center;
                padding: 0.5rem 0 0.5rem 0;
            }
            .brand-logo {
                width: min(780px, 94%);
                border-radius: 24px;
                box-shadow: 0 22px 70px rgba(0,0,0,0.45);
                border: 1px solid rgba(255,255,255,0.10);
            }
            .hero-subtitle {
                text-align: center;
                font-size: 1.12rem;
                color: #f5f5f5;
                margin-top: 0.7rem;
                margin-bottom: 0.2rem;
                letter-spacing: 0.02rem;
            }

            .hero {
                padding: 2.2rem;
                border-radius: 28px;
                background: linear-gradient(135deg, rgba(229,9,20,0.96), rgba(20,20,20,0.94));
                box-shadow: 0 22px 70px rgba(0,0,0,0.45);
                margin-bottom: 1.3rem;
                border: 1px solid rgba(255,255,255,0.10);
            }
            .hero h1 {
                font-size: 3.2rem;
                margin-bottom: 0.2rem;
                color: white;
            }
            .hero p {
                font-size: 1.08rem;
                color: #f1f1f1;
                max-width: 900px;
            }
            .section-note {
                color: #cfcfcf;
                margin-top: -0.5rem;
                margin-bottom: 1rem;
            }
            .movie-card {
                min-height: 430px;
                padding: 0.8rem;
                border-radius: 24px;
                background: rgba(255,255,255,0.07);
                border: 1px solid rgba(255,255,255,0.11);
                box-shadow: 0 16px 38px rgba(0,0,0,0.35);
                transition: transform .18s ease, border-color .18s ease, background .18s ease;
                overflow: hidden;
                margin-bottom: 1rem;
            }
            .movie-card:hover {
                transform: translateY(-6px) scale(1.04);
                border-color: rgba(229,9,20,0.75);
                background: rgba(255,255,255,0.105);
            }
            .movie-title {
                font-weight: 850;
                font-size: 1.02rem;
                line-height: 1.22;
                margin-top: 0.7rem;
                color: #ffffff;
                min-height: 45px;
            }
            .movie-meta {
                font-size: 0.78rem;
                color: #cfcfcf;
                min-height: 40px;
            }
            .pill {
                display: inline-block;
                padding: 0.23rem 0.55rem;
                margin: 0.14rem 0.14rem 0.14rem 0;
                border-radius: 999px;
                background: rgba(229,9,20,0.22);
                border: 1px solid rgba(229,9,20,0.42);
                color: #fff;
                font-size: 0.72rem;
            }
            .score-line {
                color: #f7d6d8;
                font-size: 0.8rem;
                margin-top: 0.42rem;
            }
            .status-box {
                padding: 0.75rem 1rem;
                border-radius: 18px;
                background: rgba(255,255,255,0.065);
                border: 1px solid rgba(255,255,255,0.10);
                margin-bottom: 1rem;
            }
            button[kind="secondary"] {
                border-radius: 999px !important;
            }

            .block-container {
                padding-top: 1rem;
                padding-bottom: 0rem;
                max-width: 100%;
            }

            .main-menu-label {
                color: #ff2f3d;
                font-size: 0.78rem;
                font-weight: 900;
                letter-spacing: 0.08rem;
                margin-top: 1rem;
                margin-bottom: 0.4rem;
            }

            div[role="radiogroup"] > label {
                background: rgba(255,255,255,0.03);
                padding: 0.65rem 0.9rem;
                border-radius: 14px;
                margin-bottom: 0.35rem;
                transition: all 0.2s ease;
                border: 1px solid rgba(255,255,255,0.03);
            }

            div[role="radiogroup"] > label:hover {
                background: rgba(229,9,20,0.25);
                border-color: rgba(229,9,20,0.42);
            }

            .home-search-wrap {
                margin-top: -0.2rem;
                margin-bottom: 1.5rem;
                padding: 1rem;
                border-radius: 20px;
                background: rgba(255,255,255,0.045);
                border: 1px solid rgba(255,255,255,0.08);
                box-shadow: 0 16px 50px rgba(0,0,0,0.28);
            }

            .row-header {
                display: flex;
                align-items: center;
                justify-content: space-between;
                margin-top: 1rem;
                margin-bottom: 0.7rem;
            }

            .row-header h2 {
                font-size: 1.45rem;
                margin: 0;
                color: #ffffff;
            }

            .view-all {
                color: #ff2f3d;
                font-weight: 800;
                font-size: 0.9rem;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_hero():
    logo_base64 = get_base64_image(LOGO_PATH)

    if logo_base64:
        st.markdown(
            f"""
            <div class="hero-wrapper">
                <div class="brand-logo-wrap">
                    <img class="brand-logo" src="data:image/jpg;base64,{logo_base64}" alt="FlikPik AI logo">
                </div>
                <div class="hero-subtitle">
                    Your AI-powered entertainment discovery platform for movies, TV shows, anime, trailers, streaming, and personalized recommendations.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            """
            <div class="hero">
                <h1>🎬 FlikPik AI</h1>
                <p>Sit tight, we got you! Smart movie discovery, recommendations, trailers, and personalized entertainment intelligence.</p>
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


@st.cache_data(show_spinner=False)
def search_tmdb_person(person_name):
    """Search TMDB for an actor, actress, or director."""
    if not TMDB_API_KEY or not person_name:
        return None

    params = {
        "api_key": TMDB_API_KEY,
        "query": person_name,
        "include_adult": "false",
    }

    try:
        response = requests.get(TMDB_PERSON_SEARCH_URL, params=params, timeout=8)
        response.raise_for_status()
        results = response.json().get("results", [])
        return results[0] if results else None
    except Exception:
        return None


@st.cache_data(show_spinner=False)
def get_person_movies(person_id):
    """Return acting and directing movie credits for a TMDB person ID."""
    if not TMDB_API_KEY or not person_id:
        return {"acted": [], "directed": []}

    url = TMDB_PERSON_MOVIE_CREDITS_URL.format(person_id=person_id)

    try:
        response = requests.get(url, params={"api_key": TMDB_API_KEY}, timeout=8)
        response.raise_for_status()
        data = response.json()

        acted = data.get("cast", [])
        directed = [
            movie for movie in data.get("crew", [])
            if movie.get("job") == "Director"
        ]

        return {"acted": acted, "directed": directed}
    except Exception:
        return {"acted": [], "directed": []}


def tmdb_movies_to_dataframe(tmdb_movies):
    """Convert TMDB credit results to a DataFrame compatible with show_movie_cards()."""
    rows = []

    for movie in tmdb_movies:
        title = movie.get("title") or movie.get("original_title")
        release_date = movie.get("release_date", "")
        year = release_date[:4] if release_date else ""

        if not title:
            continue

        rows.append({
            "title": f"{title} ({year})" if year else title,
            "genres": "TMDB Cast/Crew Result",
            "movieId": movie.get("id", ""),
            "overview": movie.get("overview", ""),
            "popularity": movie.get("popularity", 0),
            "weighted_score": movie.get("vote_average", 0),
        })

    df = pd.DataFrame(rows)

    if not df.empty:
        df = df.drop_duplicates(subset=["title"])
        df = df.sort_values("popularity", ascending=False)

    return df


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

    for row_start in range(0, len(filtered_records), cols_per_row):
        row_items = filtered_records[row_start : row_start + cols_per_row]
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
    render_hero()

    if not TMDB_API_KEY:
        st.warning("TMDB_API_KEY is missing. Real posters, actor search, and director search will not work yet.")
    else:
        st.caption("✅ TMDB key loaded. Real poster, actor, and director fetching is enabled.")

    st.caption("Hybrid AI + Social Buzz + Streaming + Analytics")

    try:
        train, val, test, movies, genres = cached_data()
    except Exception as e:
        st.error(f"Data loading failed: {e}")
        st.stop()

    model = cached_model(train, movies, genres)
    search_engine = MovieSearchEngine(movies)
    chatbot = MovieChatbot(model=model, movies_df=movies)
    profile_store = UserProfileStore(USER_RATINGS_PATH)

    with st.sidebar:
        if os.path.exists(LOGO_PATH):
            st.image(LOGO_PATH, use_container_width=True)

        st.markdown("<div class='main-menu-label'>MAIN MENU</div>", unsafe_allow_html=True)
        page = st.radio(
            "Navigation",
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
        st.markdown("<div class='main-menu-label'>ACCOUNT</div>", unsafe_allow_html=True)
        users = profile_store.get_users()
        default_user = users[0] if users else "brandon"
        username = st.text_input("Username", default_user)

        top_n = st.slider("Recommendations", 5, 25, DEFAULT_TOP_N)
        display = st.radio("Display Mode", ["Cards", "Table"], horizontal=True)

        st.divider()
        st.markdown("<div class='main-menu-label'>SYSTEM STATUS</div>", unsafe_allow_html=True)
        st.write(f"Recommender: **Online**")
        st.write(f"TMDB API: **{'Connected' if TMDB_API_KEY else 'Missing'}**")
        st.write(f"YouTube API: **{'Connected' if YOUTUBE_API_KEY else 'Missing'}**")

        with st.expander("Dataset Stats"):
            st.write(f"Movies: **{movies.shape[0]:,}**")
            st.write(f"Ratings: **{train.shape[0]:,}**")
            st.write(f"Users: **{train['userId'].nunique():,}**")

        if st.button("Clear poster cache"):
            st.cache_data.clear()
            st.success("Cache cleared. Rerun the app if needed.")

    if page == "🏠 Home":
        st.markdown("<div class='home-search-wrap'>", unsafe_allow_html=True)
        global_search = st.text_input(
            "",
            placeholder="Search for movies, TV shows, anime...",
            key="global_search",
        )
        st.markdown("</div>", unsafe_allow_html=True)

        if global_search:
            st.markdown("<div class='row-header'><h2>🔎 Search Results</h2><span class='view-all'>Powered by FlikPik AI</span></div>", unsafe_allow_html=True)
            search_results = search_engine.search(
                title_query=global_search,
                genre="All",
                min_year="",
                max_year="",
                limit=100,
            )
            show_movie_cards(search_results, max_items=18) if display == "Cards" else show_movie_table(search_results)
        else:
            st.markdown("<div class='row-header'><h2>🔥 Trending This Week</h2><span class='view-all'>View all →</span></div>", unsafe_allow_html=True)
            trending = model.popularity_df.copy()
            trending["hype"] = trending["title"].apply(hype_score)
            trending = trending.sort_values(["hype", "weighted_score"], ascending=False)
            show_movie_cards(trending, "hype", max_items=12) if display == "Cards" else show_movie_table(trending)

            st.markdown("<div class='row-header'><h2>🍿 Recommended For You</h2><span class='view-all'>View all →</span></div>", unsafe_allow_html=True)
            try:
                home_recs = model.recommend_hybrid(user_id=1, top_n=12)
            except Exception:
                home_recs = model._popularity_fallback(top_n=12)
            show_movie_cards(home_recs, "final_score", max_items=12) if display == "Cards" else show_movie_table(home_recs)

    elif page == "🔎 Ask FlikPik Search":
        st.header("🔎 Ask FlikPik Search")
        q = st.text_input("Title contains")
        results = search_engine.search(title_query=q, genre="All", min_year="", max_year="", limit=100)
        show_movie_cards(results) if display == "Cards" else show_movie_table(results)

    elif page == "🎭 Actor / Director":
        st.header("🎭 Actor / Director Search")
        st.caption("Search movies by actor, actress, or director using TMDB cast and crew credits.")

        person_query = st.text_input(
            "Search actor or director",
            placeholder="Example: Denzel Washington, Zendaya, Christopher Nolan",
            key="person_search",
        )

        search_type = st.radio(
            "Search by",
            ["Acting Roles", "Directed Movies"],
            horizontal=True,
        )

        if person_query:
            if not TMDB_API_KEY:
                st.warning("Add your TMDB_API_KEY to use actor and director search.")
            else:
                person = search_tmdb_person(person_query)

                if not person:
                    st.warning("No person found. Try another name.")
                else:
                    st.subheader(person.get("name", person_query))
                    credits = get_person_movies(person["id"])

                    tmdb_movies = credits["acted"] if search_type == "Acting Roles" else credits["directed"]
                    actor_director_df = tmdb_movies_to_dataframe(tmdb_movies)

                    if actor_director_df.empty:
                        st.info("No movies found for this search.")
                    else:
                        show_movie_cards(actor_director_df, "weighted_score") if display == "Cards" else show_movie_table(actor_director_df)

    elif page == "🎞 Similar":
        st.header("🎞 Similar Movie Finder")
        q = st.text_input("Search movie", "Toy Story")
        matches = search_engine.title_matches(q, limit=20)
        if not matches.empty:
            chosen = st.selectbox("Choose Movie", matches["title"].tolist())
            movie_id = int(matches.loc[matches["title"] == chosen, "movieId"].iloc[0])
            similar = model.get_similar_movies(movie_id, n_neighbors=top_n)
            show_movie_cards(similar, "similarity") if display == "Cards" else show_movie_table(similar)
        else:
            st.info("No matching movies found.")

    elif page == "🎯 My Recs":
        st.header("🎯 Personalized Recommendations")
        try:
            recs = model.recommend_hybrid(user_id=1, top_n=top_n)
        except Exception:
            recs = model._popularity_fallback(top_n=top_n)
        show_movie_cards(recs, "final_score") if display == "Cards" else show_movie_table(recs)

    elif page == "⭐ Rate Movies":
        st.header("⭐ Build Your Profile")
        q = st.text_input("Search movie to rate", "Matrix")
        matches = search_engine.title_matches(q, limit=20)
        if not matches.empty:
            selected = st.selectbox("Choose title", matches["title"].tolist())
            rating = st.slider("Your Rating", 1.0, 5.0, 4.0, 0.5)
            if st.button("Save Rating"):
                movie_id = int(matches.loc[matches["title"] == selected, "movieId"].iloc[0])
                profile_store.add_or_update_rating(
                    username=username,
                    movie_id=movie_id,
                    title=selected,
                    rating=rating,
                )
                st.success("Rating saved.")
        else:
            st.info("Search for a movie to rate.")

    elif page == "📺 Streaming":
        st.header("📺 Where To Watch")
        movie = st.selectbox("Select Movie", movies["title"].dropna().unique())
        for p in get_streaming(movie):
            st.success(p)

    elif page == "📱 Social Buzz":
        st.header("📱 Social Buzz")
        movie = st.selectbox("Movie", movies["title"].dropna().unique(), key="buzz")
        buzz = get_social(movie)
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("X", buzz["X"])
        c2.metric("Reddit", buzz["Reddit"])
        c3.metric("Facebook", buzz["Facebook"])
        c4.metric("Instagram", buzz["Instagram"])
        st.subheader("Audience Sentiment")
        st.progress(get_sentiment(movie) / 100)

    elif page == "▶️ Reviews & Trailers":
        st.header("▶️ Reviews & Trailers")
        st.subheader("Search or choose a movie")

        yt_search = st.text_input(
            "Search movie title",
            placeholder="Type a movie name like Avatar, Matrix, Toy Story...",
            key="yt_search_input",
        )

        if yt_search:
            movie_options = movies[
                movies["title"].str.contains(yt_search, case=False, na=False)
            ]["title"].dropna().unique()
        else:
            movie_options = movies["title"].dropna().unique()

        if len(movie_options) == 0:
            st.warning("No movies found. Try another search.")
        else:
            movie = st.selectbox(
                "Choose from results",
                movie_options,
                key="yt_movie_select",
            )

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
        st.header("🤖 Ask FlikPik")
        st.caption("Ask for movies by mood, genre, decade, year, similarity, actor, or director. Try: 'movies with Denzel Washington', 'directed by Christopher Nolan', or 'like Inception after 2000'.")

        if "chat_history" not in st.session_state:
            st.session_state.chat_history = []

        mood = st.selectbox(
            "What mood are you in?",
            [
                "Surprise me",
                "Funny",
                "Scary",
                "Romantic",
                "Action",
                "Mind-bending",
                "Family",
                "Drama",
                "Thriller",
                "Sci-Fi",
            ],
        )

        prompt = st.text_area(
            "Tell me what you want to watch",
            "Recommend exciting sci-fi movies with strong ratings after 2000",
        )

        example_cols = st.columns(4)
        examples = [
            "Movies with Denzel Washington",
            "Directed by Christopher Nolan",
            "Funny date night movies",
            "Mind-bending thrillers from the 2000s",
        ]

        for ex_col, example in zip(example_cols, examples):
            with ex_col:
                if st.button(example, use_container_width=True):
                    prompt = example

        if st.button("Ask Assistant", type="primary"):
            try:
                response, recs, explanations, parsed = chatbot.recommend(
                    prompt,
                    top_n=top_n,
                    mood=mood,
                )
            except ValueError:
                response, recs = chatbot.recommend(prompt, top_n=top_n)
                explanations = []
                parsed = {}

            st.session_state.chat_history.append(
                {
                    "mood": mood,
                    "prompt": prompt,
                    "response": response,
                    "count": len(recs),
                }
            )

            st.subheader("Assistant Response")
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
            show_movie_cards(recs, "weighted_score") if display == "Cards" else show_movie_table(recs)

        with st.expander("Chat History"):
            if not st.session_state.chat_history:
                st.caption("Your assistant conversation will appear here during this session.")
            else:
                for item in reversed(st.session_state.chat_history[-8:]):
                    st.markdown(f"**You ({item['mood']}):** {item['prompt']}")
                    st.markdown(f"**Assistant:** {item['response']}")
                    st.divider()

    elif page == "👤 Insights":
        st.header("👤 User Insights")
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
    st.caption("FlikPik AI | Netflix-Style UI | Real Posters | Actor/Director Search | Deploy Ready")


if __name__ == "__main__":
    main()