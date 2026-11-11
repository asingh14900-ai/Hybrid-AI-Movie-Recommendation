import os
import re
import requests
from dotenv import load_dotenv

# --------------------------------------------------
# LOAD ENVIRONMENT VARIABLES
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

load_dotenv(os.path.join(BASE_DIR, ".env"))

TMDB_TOKEN = os.getenv("TMDB_BEARER_TOKEN")


# --------------------------------------------------
# TMDB CONFIGURATION
# --------------------------------------------------

BASE_URL = "https://api.themoviedb.org/3"

IMAGE_BASE_URL = "https://image.tmdb.org/t/p/w500"


# --------------------------------------------------
# SEARCH MOVIE USING TMDB API
# --------------------------------------------------

def search_movie(movie_title):

    if not TMDB_TOKEN:
        print("TMDB token not found!")
        return None

    # Extract release year
    year_match = re.search(
        r"\((\d{4})\)\s*$",
        movie_title
    )

    release_year = year_match.group(1) if year_match else None

    # Remove year from title
    clean_title = re.sub(
        r"\s*\(\d{4}\)\s*$",
        "",
        movie_title
    ).strip()

    url = f"{BASE_URL}/search/movie"

    headers = {
        "accept": "application/json",
        "Authorization": f"Bearer {TMDB_TOKEN}"
    }

    # --------------------------------------------------
    # FIRST SEARCH: WITH RELEASE YEAR
    # SECOND SEARCH: WITHOUT RELEASE YEAR
    # --------------------------------------------------

    search_attempts = []

    first_params = {
        "query": clean_title,
        "include_adult": "false"
    }

    if release_year:
        first_params["year"] = release_year

    search_attempts.append(first_params)

    # Retry without year if the first search is unsuccessful
    if release_year:

        second_params = {
            "query": clean_title,
            "include_adult": "false"
        }

        search_attempts.append(second_params)

    try:

        results = []

        for params in search_attempts:

            response = requests.get(
                url,
                headers=headers,
                params=params,
                timeout=15
            )

            response.raise_for_status()

            results = response.json().get("results", [])

            # Continue searching if no results or no posters
            poster_results = [
                movie for movie in results
                if movie.get("poster_path")
            ]

            if poster_results:
                break

        # No matching movie with a poster
        if not poster_results:
            print(f"No poster found for: {movie_title}")
            return None

        # --------------------------------------------------
        # PREFER EXACT RELEASE YEAR
        # --------------------------------------------------

        if release_year:

            matching_year = [
                movie for movie in poster_results
                if movie.get("release_date", "").startswith(
                    release_year
                )
            ]

            if matching_year:
                poster_results = matching_year

        # --------------------------------------------------
        # SELECT MOST POPULAR MATCHING MOVIE
        # --------------------------------------------------

        movie = max(
            poster_results,
            key=lambda item: item.get("popularity", 0)
        )

        poster_path = movie.get("poster_path")

        poster_url = (
            IMAGE_BASE_URL + poster_path
            if poster_path else None
        )

        # --------------------------------------------------
        # RETURN MOVIE INFORMATION
        # --------------------------------------------------

        return {
            "tmdb_id": movie.get("id"),
            "title": movie.get("title"),
            "overview": movie.get("overview"),
            "release_date": movie.get("release_date"),
            "poster_url": poster_url
        }

    except requests.RequestException as error:

        print("TMDB API Error:", error)

        return None

    except Exception as error:

        print("Unexpected TMDB Error:", error)

        return None