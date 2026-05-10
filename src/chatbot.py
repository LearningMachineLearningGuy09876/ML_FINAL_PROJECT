import re
import pandas as pd
from utils import extract_year


class MovieChatbot:
    """
    Smarter rules-based movie assistant for the Streamlit movie recommender.

    Features:
    - Understands genres, moods, decades, years, negative preferences, actors, and directors.
    - Handles phrases like "after 2010", "before 2000", "not horror", and "like Toy Story".
    - Handles phrases like "starring Tom Hanks", "with Leonardo DiCaprio",
      and "directed by Christopher Nolan".
    - Produces short explanation bullets for recommendations.
    - Requires no paid AI API key, so it is easy to deploy on Render.
    """

    GENRES = [
        "action",
        "adventure",
        "animation",
        "children",
        "comedy",
        "crime",
        "documentary",
        "drama",
        "fantasy",
        "film-noir",
        "horror",
        "musical",
        "mystery",
        "romance",
        "sci-fi",
        "thriller",
        "war",
        "western",
    ]

    MOOD_TO_GENRES = {
        "funny": ["Comedy"],
        "laugh": ["Comedy"],
        "scary": ["Horror", "Thriller"],
        "creepy": ["Horror", "Thriller"],
        "romantic": ["Romance"],
        "date night": ["Romance", "Comedy"],
        "sad": ["Drama"],
        "emotional": ["Drama"],
        "exciting": ["Action", "Adventure"],
        "intense": ["Action", "Thriller"],
        "family": ["Children", "Animation"],
        "kids": ["Children", "Animation"],
        "mind-bending": ["Mystery", "Sci-Fi", "Thriller"],
        "mind bending": ["Mystery", "Sci-Fi", "Thriller"],
        "smart": ["Mystery", "Sci-Fi", "Drama"],
    }

    QUALITY_WORDS = [
        "best",
        "top",
        "popular",
        "highest",
        "great",
        "good",
        "strong ratings",
        "well rated",
    ]

    # Words that usually introduce an actor name.
    ACTOR_PATTERNS = [
        r"\bstarring\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bwith\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bfeaturing\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bactor\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bacted by\s+([a-zA-Z][a-zA-Z .'\-]+)",
    ]

    # Words that usually introduce a director name.
    DIRECTOR_PATTERNS = [
        r"\bdirected by\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bdirector\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bfrom\s+director\s+([a-zA-Z][a-zA-Z .'\-]+)",
        r"\bby\s+director\s+([a-zA-Z][a-zA-Z .'\-]+)",
    ]

    # These words are used to trim names from natural language prompts.
    PERSON_STOP_WORDS = [
        " after ",
        " before ",
        " since ",
        " from ",
        " newer than ",
        " older than ",
        " in ",
        " but ",
        " and ",
        " not ",
        " no ",
        " without ",
        " avoid ",
        " movies",
        " movie",
        " films",
        " film",
        " recommendations",
        " recommendation",
        " please",
        " that are",
        " with strong ratings",
        " strong ratings",
        " good",
        " great",
        " best",
        " top",
        " popular",
    ]

    def __init__(self, model, movies_df):
        self.model = model
        self.movies_df = movies_df.copy()

    def _title_case_genre(self, genre):
        mapping = {
            "sci-fi": "Sci-Fi",
            "film-noir": "Film-Noir",
        }
        return mapping.get(genre, genre.title())

    def _clean_person_name(self, name):
        """Clean actor/director text captured from the user's prompt."""
        if not name:
            return None

        cleaned = str(name).strip().strip(".?!,;:")

        lowered = f" {cleaned.lower()} "
        cut_at = len(cleaned)

        for stop_word in self.PERSON_STOP_WORDS:
            idx = lowered.find(stop_word)
            if idx != -1:
                # subtract the added leading space
                cut_at = min(cut_at, max(idx - 1, 0))

        cleaned = cleaned[:cut_at].strip().strip(".?!,;:")

        # Avoid capturing overly long chunks by keeping the first few name parts.
        parts = cleaned.split()
        if len(parts) > 4:
            cleaned = " ".join(parts[:4])

        return cleaned if len(cleaned) >= 2 else None

    def _extract_person_from_patterns(self, query, patterns):
        for pattern in patterns:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                person = self._clean_person_name(match.group(1))
                if person:
                    return person
        return None

    def _base_movies(self):
        if hasattr(self.model, "popularity_df") and self.model.popularity_df is not None:
            df = self.model.popularity_df.copy()
        else:
            df = self.movies_df.copy()

        if "year" not in df.columns:
            df["year"] = df["title"].apply(extract_year)

        if "weighted_score" not in df.columns:
            df["weighted_score"] = 0.0

        if "genres" not in df.columns:
            df["genres"] = ""

        # These columns may be added by app.py through TMDB enrichment.
        # If they do not exist yet, create empty columns so filtering is safe.
        if "cast" not in df.columns:
            df["cast"] = ""

        if "director" not in df.columns:
            df["director"] = ""

        return df

    def parse_query(self, query):
        query = query or ""
        query_lower = query.lower()

        selected_genres = []
        excluded_genres = []

        for genre in self.GENRES:
            genre_name = self._title_case_genre(genre)
            if genre in query_lower:
                selected_genres.append(genre_name)

            negative_patterns = [
                f"not {genre}",
                f"no {genre}",
                f"without {genre}",
                f"avoid {genre}",
            ]
            if any(pattern in query_lower for pattern in negative_patterns):
                excluded_genres.append(genre_name)
                if genre_name in selected_genres:
                    selected_genres.remove(genre_name)

        for keyword, genres in self.MOOD_TO_GENRES.items():
            if keyword in query_lower:
                selected_genres.extend(genres)

        year_exact = None
        year_match = re.search(r"\b(19\d{2}|20\d{2})\b", query_lower)
        if year_match:
            year_exact = int(year_match.group(1))

        year_min = None
        year_max = None

        after_match = re.search(r"(?:after|since|newer than|from)\s+(19\d{2}|20\d{2})", query_lower)
        if after_match:
            year_min = int(after_match.group(1))
            year_exact = None

        before_match = re.search(r"(?:before|older than|pre-)\s*(19\d{2}|20\d{2})", query_lower)
        if before_match:
            year_max = int(before_match.group(1))
            year_exact = None

        if any(word in query_lower for word in ["new", "newer", "recent", "modern"]):
            year_min = year_min or 2000
            year_exact = None

        if any(word in query_lower for word in ["classic", "old school", "older"]):
            year_max = year_max or 1999
            year_exact = None

        decade = None
        decade_match = re.search(r"\b(\d{2})s\b", query_lower)
        if decade_match:
            val = int(decade_match.group(1))
            decade = 1900 + val if val >= 30 else 2000 + val
            year_exact = None

        similar_to = None
        like_match = re.search(r"(?:like|similar to)\s+([a-zA-Z0-9:'!?,.&\- ]+)", query, re.IGNORECASE)
        if like_match:
            similar_to = like_match.group(1).strip().rstrip(".?!")

        actor = self._extract_person_from_patterns(query, self.ACTOR_PATTERNS)
        director = self._extract_person_from_patterns(query, self.DIRECTOR_PATTERNS)

        return {
            "raw_query": query,
            "genres": sorted(set(selected_genres)),
            "excluded_genres": sorted(set(excluded_genres)),
            "year": year_exact,
            "year_min": year_min,
            "year_max": year_max,
            "decade": decade,
            "similar_to": similar_to,
            "actor": actor,
            "director": director,
            "wants_quality": any(word in query_lower for word in self.QUALITY_WORDS),
        }

    def _filter_movies(self, parsed):
        recs = self._base_movies()

        for genre in parsed["genres"]:
            recs = recs[recs["genres"].str.contains(genre, case=False, na=False)]

        for genre in parsed["excluded_genres"]:
            recs = recs[~recs["genres"].str.contains(genre, case=False, na=False)]

        if parsed["year"]:
            recs = recs[recs["year"] == parsed["year"]]

        if parsed["year_min"]:
            recs = recs[recs["year"] >= parsed["year_min"]]

        if parsed["year_max"]:
            recs = recs[recs["year"] <= parsed["year_max"]]

        if parsed["decade"]:
            recs = recs[
                (recs["year"] >= parsed["decade"])
                & (recs["year"] <= parsed["decade"] + 9)
            ]

        if parsed.get("actor"):
            recs = recs[
                recs["cast"].str.contains(
                    parsed["actor"],
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        if parsed.get("director"):
            recs = recs[
                recs["director"].str.contains(
                    parsed["director"],
                    case=False,
                    na=False,
                    regex=False,
                )
            ]

        return recs

    def _similar_recommendations(self, parsed, top_n):
        if not parsed.get("similar_to"):
            return None

        target = parsed["similar_to"].lower()
        matches = self.movies_df[self.movies_df["title"].str.lower().str.contains(target, na=False)]
        if matches.empty:
            return None

        movie_id = int(matches.iloc[0]["movieId"])
        if not hasattr(self.model, "get_similar_movies"):
            return None

        try:
            similar = self.model.get_similar_movies(movie_id, n_neighbors=max(top_n, 10))
            if similar is not None and not similar.empty:
                return similar
        except Exception:
            return None

        return None

    def explain_movie(self, row, parsed):
        reasons = []
        genres = str(row.get("genres", ""))
        title = str(row.get("title", "This movie"))
        year = row.get("year", None)
        cast = str(row.get("cast", ""))
        director = str(row.get("director", ""))

        matched_genres = [genre for genre in parsed["genres"] if genre.lower() in genres.lower()]
        if matched_genres:
            reasons.append("matches your " + ", ".join(matched_genres) + " preference")

        if parsed.get("actor") and parsed["actor"].lower() in cast.lower():
            reasons.append(f"features {parsed['actor']}")

        if parsed.get("director") and parsed["director"].lower() in director.lower():
            reasons.append(f"was directed by {parsed['director']}")

        if parsed.get("similar_to"):
            reasons.append(f"fits the vibe of {parsed['similar_to']}")

        if parsed.get("year_min") and pd.notna(year):
            reasons.append(f"released after {parsed['year_min']}")

        if parsed.get("year_max") and pd.notna(year):
            reasons.append("fits your older/classic preference")

        if parsed.get("decade") and pd.notna(year):
            reasons.append(f"from the {parsed['decade']}s")

        if "weighted_score" in row and pd.notna(row.get("weighted_score")):
            reasons.append("has a strong recommendation score")

        if not reasons:
            reasons.append("is a strong popular pick from the dataset")

        return f"**{title}** — " + "; ".join(reasons[:3]) + "."

    def explain_response(self, parsed, count):
        pieces = []
        if parsed["genres"]:
            pieces.append("genres: " + ", ".join(parsed["genres"]))
        if parsed["excluded_genres"]:
            pieces.append("excluding: " + ", ".join(parsed["excluded_genres"]))
        if parsed.get("actor"):
            pieces.append(f"actor: {parsed['actor']}")
        if parsed.get("director"):
            pieces.append(f"director: {parsed['director']}")
        if parsed["year"]:
            pieces.append(f"year: {parsed['year']}")
        if parsed["year_min"]:
            pieces.append(f"after: {parsed['year_min']}")
        if parsed["year_max"]:
            pieces.append(f"before: {parsed['year_max']}")
        if parsed["decade"]:
            pieces.append(f"decade: {parsed['decade']}s")
        if parsed["similar_to"]:
            pieces.append(f"similar to: {parsed['similar_to']}")

        if pieces:
            return f"I found {count} movies matching " + " | ".join(pieces) + "."
        return f"I found {count} strong movie picks based on your request."

    def recommend(self, query, top_n=10, mood=None):
        full_query = query or ""
        if mood and mood.lower() != "surprise me":
            full_query = f"{mood}. {full_query}"

        parsed = self.parse_query(full_query)

        similar = self._similar_recommendations(parsed, top_n)
        if similar is not None:
            recs = similar.copy()
            filtered = self._filter_movies(parsed)
            has_filters = (
                parsed["genres"]
                or parsed["excluded_genres"]
                or parsed["year"]
                or parsed["year_min"]
                or parsed["year_max"]
                or parsed["decade"]
                or parsed.get("actor")
                or parsed.get("director")
            )

            if has_filters:
                keep_ids = set(filtered["movieId"].tolist()) if "movieId" in filtered.columns else set()
                if keep_ids and "movieId" in recs.columns:
                    recs = recs[recs["movieId"].isin(keep_ids)]

            if recs.empty:
                recs = similar.copy()
        else:
            recs = self._filter_movies(parsed)

        sort_cols = [
            col
            for col in ["weighted_score", "similarity", "rating_count", "mean_rating"]
            if col in recs.columns
        ]
        if sort_cols:
            recs = recs.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        recs = recs.head(top_n).reset_index(drop=True)
        response = self.explain_response(parsed, len(recs))
        explanations = [self.explain_movie(row, parsed) for _, row in recs.iterrows()]

        return response, recs, explanations, parsed
