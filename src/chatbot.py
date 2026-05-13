import re
import pandas as pd
from utils import extract_year


class MovieChatbot:
    """
    Upgraded rules-based AI movie assistant for FlikPik.

    Works without a paid AI API. It understands:
    - moods, genres, decades, exact years, after/before filters
    - negative preferences like "not horror" or "no romance"
    - similar movie requests like "like The Matrix but newer"
    - actor/director/cast searches when those columns exist in your dataset
    - fallback keyword searches across title, genres, tags, overview, cast, and crew
    """

    GENRES = [
        "action", "adventure", "animation", "children", "comedy", "crime",
        "documentary", "drama", "fantasy", "film-noir", "horror", "musical",
        "mystery", "romance", "sci-fi", "thriller", "war", "western", "imax"
    ]

    MOOD_TO_GENRES = {
        "funny": ["Comedy"],
        "laugh": ["Comedy"],
        "comedy": ["Comedy"],
        "scary": ["Horror", "Thriller"],
        "creepy": ["Horror", "Thriller"],
        "dark": ["Thriller", "Drama", "Crime"],
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
        "suspense": ["Thriller", "Mystery"],
        "superhero": ["Action", "Adventure", "Sci-Fi"],
        "animated": ["Animation", "Children"],
    }

    QUALITY_WORDS = [
        "best", "top", "popular", "highest", "great", "good", "strong ratings",
        "well rated", "highly rated", "classic", "must watch"
    ]

    SEARCH_COLUMNS = [
        "title", "title_clean", "genres", "tags", "tag", "keywords", "overview",
        "cast", "actors", "actor", "crew", "director", "directors"
    ]

    PERSON_PATTERNS = [
        ("director", r"(?:directed by|director|films? by|movies? by)\s+([a-zA-Z .'-]{3,60})"),
        ("actor", r"(?:with|starring|actor|actors|cast|featuring)\s+([a-zA-Z .'-]{3,60})"),
    ]

    def __init__(self, model, movies_df, person_lookup_func=None):
        self.model = model
        self.movies_df = movies_df.copy()
        # Optional app-level lookup function, usually backed by TMDB.
        # Signature: person_lookup_func(person_name, role="actor", limit=50) -> DataFrame
        self.person_lookup_func = person_lookup_func

    def _title_case_genre(self, genre):
        mapping = {"sci-fi": "Sci-Fi", "film-noir": "Film-Noir", "imax": "IMAX"}
        return mapping.get(genre, genre.title())

    def _safe_text(self, value):
        if pd.isna(value):
            return ""
        return str(value)

    def _base_movies(self):
        if hasattr(self.model, "popularity_df") and self.model.popularity_df is not None:
            df = self.model.popularity_df.copy()
        else:
            df = self.movies_df.copy()

        if "year" not in df.columns:
            df["year"] = df["title"].apply(extract_year)

        # Important: make year numeric so filters like after 2000 work correctly.
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

        if "weighted_score" not in df.columns:
            df["weighted_score"] = 0.0

        if "genres" not in df.columns:
            df["genres"] = ""

        return df

    def _available_search_columns(self, df):
        return [col for col in self.SEARCH_COLUMNS if col in df.columns]

    def _combined_text(self, df):
        """Return one lowercase searchable text string per movie row.

        This version is intentionally defensive. Some movie datasets contain
        duplicate column names or list/dict values, which can make pandas return
        a DataFrame instead of a Series and crash on `.str.lower()`.
        """
        if df is None or df.empty:
            return pd.Series([], dtype=str)

        cols = self._available_search_columns(df)
        if not cols:
            return pd.Series([""] * len(df), index=df.index, dtype=str)

        # Remove duplicate column names while preserving order.
        cols = list(dict.fromkeys(cols))

        combined = []
        for _, row in df.iterrows():
            parts = []
            for col in cols:
                value = row.get(col, "")
                if isinstance(value, (list, tuple, set)):
                    value = " ".join(str(v) for v in value)
                elif isinstance(value, dict):
                    value = " ".join(str(v) for v in value.values())
                elif pd.isna(value):
                    value = ""
                parts.append(str(value))
            combined.append(" ".join(parts).lower())

        return pd.Series(combined, index=df.index, dtype=str)

    def _clean_person_query(self, name):
        name = re.sub(r"\b(after|before|from|since|newer|older|movies?|films?|shows?)\b.*$", "", name, flags=re.I)
        return name.strip(" .?!,:")

    def parse_query(self, query):
        query = query or ""
        query_lower = query.lower()

        selected_genres = []
        excluded_genres = []

        for genre in self.GENRES:
            genre_name = self._title_case_genre(genre)
            negative_patterns = [f"not {genre}", f"no {genre}", f"without {genre}", f"avoid {genre}"]

            if any(pattern in query_lower for pattern in negative_patterns):
                excluded_genres.append(genre_name)
            elif genre in query_lower:
                selected_genres.append(genre_name)

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
        decade_match = re.search(r"\b(1930|1940|1950|1960|1970|1980|1990|2000|2010|2020)s\b|\b(\d{2})s\b", query_lower)
        if decade_match:
            if decade_match.group(1):
                decade = int(decade_match.group(1))
            else:
                val = int(decade_match.group(2))
                decade = 1900 + val if val >= 30 else 2000 + val
            year_exact = None

        similar_to = None
        like_match = re.search(r"(?:like|similar to)\s+([a-zA-Z0-9:'!?,.&\- ]+)", query, re.IGNORECASE)
        if like_match:
            similar_to = like_match.group(1).strip().rstrip(".?!")

        person = None
        person_role = None
        for role, pattern in self.PERSON_PATTERNS:
            match = re.search(pattern, query, re.IGNORECASE)
            if match:
                person = self._clean_person_query(match.group(1))
                person_role = role
                break

        # General keyword fallback: useful for actor names typed without "with".
        keyword = query.strip()
        keyword = re.sub(r"\b(recommend|show me|find|movies?|films?|please|best|top)\b", "", keyword, flags=re.I).strip()
        if similar_to:
            keyword = ""

        return {
            "raw_query": query,
            "genres": sorted(set(selected_genres)),
            "excluded_genres": sorted(set(excluded_genres)),
            "year": year_exact,
            "year_min": year_min,
            "year_max": year_max,
            "decade": decade,
            "similar_to": similar_to,
            "person": person,
            "person_role": person_role,
            "keyword": keyword,
            "wants_quality": any(word in query_lower for word in self.QUALITY_WORDS),
        }

    def _filter_movies(self, parsed):
        recs = self._base_movies()

        for genre in parsed["genres"]:
            recs = recs[recs["genres"].str.contains(re.escape(genre), case=False, na=False)]

        for genre in parsed["excluded_genres"]:
            recs = recs[~recs["genres"].str.contains(re.escape(genre), case=False, na=False)]

        if parsed["year"]:
            recs = recs[recs["year"] == parsed["year"]]

        if parsed["year_min"]:
            recs = recs[recs["year"] >= parsed["year_min"]]

        if parsed["year_max"]:
            recs = recs[recs["year"] <= parsed["year_max"]]

        if parsed["decade"]:
            recs = recs[(recs["year"] >= parsed["decade"]) & (recs["year"] <= parsed["decade"] + 9)]

        search_text = self._combined_text(recs)

        if parsed.get("person"):
            person = parsed["person"].lower()
            recs = recs[search_text.str.contains(re.escape(person), na=False)]
            search_text = self._combined_text(recs)

        # Only use generic keyword filtering when it seems useful and does not wipe out genre-only searches.
        keyword = parsed.get("keyword", "").strip().lower()
        if keyword and not parsed.get("person"):
            cleaned_keyword = re.sub(r"\b(after|before|since|from|newer|older|than|not|no|without|avoid)\b.*", "", keyword).strip()
            if len(cleaned_keyword) >= 3 and not any(g.lower() in cleaned_keyword for g in parsed["genres"]):
                keyword_matches = recs[search_text.str.contains(re.escape(cleaned_keyword), na=False)]
                if not keyword_matches.empty:
                    recs = keyword_matches

        return recs

    def _similar_recommendations(self, parsed, top_n):
        if not parsed.get("similar_to"):
            return None

        target = parsed["similar_to"].lower()
        matches = self.movies_df[self.movies_df["title"].str.lower().str.contains(re.escape(target), na=False)]
        if matches.empty and "title_clean" in self.movies_df.columns:
            matches = self.movies_df[self.movies_df["title_clean"].str.lower().str.contains(re.escape(target), na=False)]
        if matches.empty:
            return None

        movie_id = int(matches.iloc[0]["movieId"])
        if not hasattr(self.model, "get_similar_movies"):
            return None

        try:
            similar = self.model.get_similar_movies(movie_id, n_neighbors=max(top_n, 20))
            if similar is not None and not similar.empty:
                return similar
        except Exception:
            return None

        return None

    def explain_movie(self, row, parsed):
        reasons = []
        genres = self._safe_text(row.get("genres", ""))
        title = self._safe_text(row.get("title", "This movie"))
        year = row.get("year", None)

        matched_genres = [genre for genre in parsed["genres"] if genre.lower() in genres.lower()]
        if matched_genres:
            reasons.append("matches your " + ", ".join(matched_genres) + " preference")

        if parsed.get("person"):
            role = parsed.get("person_role")
            if role == "director":
                reasons.append(f"was directed by or connected to {parsed['person']}")
            elif role == "actor":
                reasons.append(f"features or is connected to {parsed['person']}")
            else:
                reasons.append(f"matches your search for {parsed['person']}")

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
            reasons.append("is a strong match from the dataset")

        return f"**{title}** — " + "; ".join(reasons[:3]) + "."

    def explain_response(self, parsed, count):
        pieces = []
        if parsed["genres"]:
            pieces.append("genres: " + ", ".join(parsed["genres"]))
        if parsed["excluded_genres"]:
            pieces.append("excluding: " + ", ".join(parsed["excluded_genres"]))
        if parsed.get("person"):
            role = parsed.get("person_role") or "cast/crew"
            pieces.append(f"{role}: " + parsed["person"])
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

        if count == 0:
            return "I could not find a strong match for that request, so try a broader genre, actor/director name, or year range."
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
            has_extra_filters = any([
                parsed["genres"], parsed["excluded_genres"], parsed["year"], parsed["year_min"],
                parsed["year_max"], parsed["decade"], parsed.get("person")
            ])
            if has_extra_filters and "movieId" in filtered.columns and "movieId" in recs.columns:
                keep_ids = set(filtered["movieId"].tolist())
                narrowed = recs[recs["movieId"].isin(keep_ids)]
                if not narrowed.empty:
                    recs = narrowed
        else:
            recs = self._filter_movies(parsed)

        # If the local dataset does not have actor/director fields, fall back to
        # the optional app-level TMDB lookup for cast/crew searches.
        if (
            (recs is None or recs.empty)
            and parsed.get("person")
            and self.person_lookup_func is not None
        ):
            try:
                role = parsed.get("person_role") or "any"
                recs = self.person_lookup_func(parsed["person"], role=role, limit=max(top_n * 4, 30))
            except Exception:
                recs = pd.DataFrame()

        sort_cols = [col for col in ["weighted_score", "similarity", "rating_count", "mean_rating", "popularity"] if col in recs.columns]
        if sort_cols:
            recs = recs.sort_values(sort_cols, ascending=[False] * len(sort_cols))

        recs = recs.drop_duplicates(subset=["title"]) if "title" in recs.columns else recs
        recs = recs.head(top_n).reset_index(drop=True)
        response = self.explain_response(parsed, len(recs))
        explanations = [self.explain_movie(row, parsed) for _, row in recs.iterrows()]

        return response, recs, explanations, parsed
