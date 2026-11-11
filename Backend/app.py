
import os
import sys

from flask import Flask, jsonify, request
from flask_cors import CORS

# --------------------------------------------------
# ADD PROJECT ROOT TO PYTHON PATH
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# --------------------------------------------------
# IMPORT PROJECT MODULES
# --------------------------------------------------

from tmdb import search_movie
from recommendation.hybrid_recommendation import (
    recommend_movies,
    movies
)
from db import get_db_connection


# --------------------------------------------------
# INITIALIZE FLASK APPLICATION
# --------------------------------------------------

app = Flask(__name__)
CORS(app)


# --------------------------------------------------
# TMDB POSTER CACHING
# --------------------------------------------------

# Store only successfully retrieved poster URLs.
POSTER_CACHE = {}


def get_movie_poster(movie_title):

    if not movie_title:
        return None

    # Return previously cached poster.
    if movie_title in POSTER_CACHE:
        return POSTER_CACHE[movie_title]

    try:

        movie_data = search_movie(movie_title)

        if movie_data:

            poster_url = movie_data.get("poster_url")

            # Cache only valid poster URLs.
            if poster_url:
                POSTER_CACHE[movie_title] = poster_url
                return poster_url

        # Do not cache None.
        # Failed searches can be retried later.
        return None

    except Exception as error:

        print(f"Poster Error for {movie_title}: {error}")
        return None


# --------------------------------------------------
# ADD POSTER URL TO MOVIE DATA
# --------------------------------------------------

def add_movie_posters(movie_list):

    for movie in movie_list:

        movie["poster_url"] = get_movie_poster(
            movie.get("title", "")
        )

    return movie_list


# --------------------------------------------------
# 1. HOME API
# --------------------------------------------------

@app.route("/", methods=["GET"])
def home():

    return jsonify({
        "message": "Hybrid AI-Based Personalized Movie Recommendation Engine",
        "status": "Backend is running successfully"
    })


# --------------------------------------------------
# 2. HEALTH CHECK API
# --------------------------------------------------

@app.route("/health", methods=["GET"])
def health():

    return jsonify({
        "status": "healthy",
        "backend": "Flask",
        "project": "Movie Recommendation System"
    })


# --------------------------------------------------
# 3. PERSONALIZED MOVIE RECOMMENDATION API
# --------------------------------------------------

@app.route("/api/recommendations/<int:user_id>", methods=["GET"])
def get_recommendations(user_id):

    try:

        limit = request.args.get("limit", default=10, type=int)

        if limit < 1 or limit > 50:
            return jsonify({
                "error": "Limit must be between 1 and 50"
            }), 400

        results = recommend_movies(
            user_id,
            number_of_recommendations=limit
        )

        if results.empty:
            return jsonify({
                "error": "User not found or no recommendations available"
            }), 404

        recommendations = results.to_dict(orient="records")

        # Attach TMDB posters.
        recommendations = add_movie_posters(recommendations)

        return jsonify({
            "user_id": user_id,
            "total_recommendations": len(recommendations),
            "recommendations": recommendations
        })

    except Exception as error:

        print("Recommendation API Error:", error)

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# 4. MOVIE SEARCH API
# --------------------------------------------------

@app.route("/api/movies/search", methods=["GET"])
def search_movies():

    try:

        query = request.args.get("q", "").strip()

        if not query:
            return jsonify({
                "error": "Please enter a movie name"
            }), 400

        limit = request.args.get("limit", default=10, type=int)

        if limit < 1 or limit > 50:
            return jsonify({
                "error": "Limit must be between 1 and 50"
            }), 400

        search_results = movies[
            movies["title"].str.contains(
                query,
                case=False,
                na=False,
                regex=False
            )
        ]

        search_results = search_results.head(limit)

        movie_data = search_results[
            ["movieId", "title", "genres"]
        ].to_dict(orient="records")

        # Attach TMDB posters.
        movie_data = add_movie_posters(movie_data)

        return jsonify({
            "query": query,
            "total_results": len(movie_data),
            "movies": movie_data
        })

    except Exception as error:

        print("Movie Search API Error:", error)

        return jsonify({
            "error": str(error)
        }), 500


# --------------------------------------------------
# 5. MYSQL DATABASE STATISTICS API
# --------------------------------------------------

@app.route("/api/db/stats", methods=["GET"])
def database_stats():

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        cursor.execute(
            "SELECT COUNT(*) AS total_users FROM users"
        )
        users = cursor.fetchone()

        cursor.execute(
            "SELECT COUNT(*) AS total_movies FROM movies"
        )
        movie_count = cursor.fetchone()

        cursor.execute(
            "SELECT COUNT(*) AS total_ratings FROM ratings"
        )
        rating_count = cursor.fetchone()

        return jsonify({
            "database": "Connected Successfully",
            "total_users": users["total_users"],
            "total_movies": movie_count["total_movies"],
            "total_ratings": rating_count["total_ratings"]
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# --------------------------------------------------
# 6. SAVE RECOMMENDATIONS TO MYSQL
# --------------------------------------------------

@app.route(
    "/api/recommendations/<int:user_id>/save",
    methods=["POST"]
)
def save_recommendations(user_id):

    connection = None
    cursor = None

    try:

        limit = request.args.get("limit", default=10, type=int)

        if limit < 1 or limit > 50:
            return jsonify({
                "error": "Limit must be between 1 and 50"
            }), 400

        results = recommend_movies(
            user_id,
            number_of_recommendations=limit
        )

        if results.empty:
            return jsonify({
                "error": "User not found or no recommendations available"
            }), 404

        connection = get_db_connection()
        cursor = connection.cursor()

        query = """
        INSERT INTO recommendations
        (user_id, movie_id, recommendation_score,
         recommendation_rank, model_name)
        VALUES (%s, %s, %s, %s, %s)
        """

        recommendation_data = []

        for rank, row in enumerate(
            results.itertuples(index=False), start=1
        ):

            recommendation_data.append((
                user_id,
                int(row.movieId),
                float(row.final_score),
                rank,
                "Hybrid AI Recommendation Model"
            ))

        cursor.executemany(query, recommendation_data)

        connection.commit()

        return jsonify({
            "message": "Recommendations saved successfully!",
            "user_id": user_id,
            "total_saved": cursor.rowcount
        })

    except Exception as error:

        if connection:
            connection.rollback()

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# --------------------------------------------------
# 7. GET SAVED RECOMMENDATIONS FROM MYSQL
# --------------------------------------------------

@app.route(
    "/api/recommendations/<int:user_id>/saved",
    methods=["GET"]
)
def get_saved_recommendations(user_id):

    connection = None
    cursor = None

    try:

        connection = get_db_connection()
        cursor = connection.cursor(dictionary=True)

        query = """
        SELECT
            r.recommendation_id,
            r.user_id,
            r.movie_id,
            m.title,
            m.genres,
            r.recommendation_score,
            r.recommendation_rank,
            r.model_name,
            r.generated_at
        FROM recommendations r
        JOIN movies m ON r.movie_id = m.movie_id
        WHERE r.user_id = %s
        ORDER BY r.generated_at DESC, r.recommendation_rank ASC
        """

        cursor.execute(query, (user_id,))

        results = cursor.fetchall()

        # Attach TMDB posters.
        results = add_movie_posters(results)

        return jsonify({
            "user_id": user_id,
            "total_recommendations": len(results),
            "recommendations": results
        })

    except Exception as error:

        return jsonify({
            "error": str(error)
        }), 500

    finally:

        if cursor:
            cursor.close()

        if connection and connection.is_connected():
            connection.close()


# --------------------------------------------------
# 8. TMDB MOVIE SEARCH API
# --------------------------------------------------

@app.route("/api/tmdb/search", methods=["GET"])
def tmdb_search():

    movie_title = request.args.get("query", "").strip()

    if not movie_title:
        return jsonify({
            "error": "Movie title is required"
        }), 400

    try:

        movie = search_movie(movie_title)

        if movie:
            return jsonify(movie), 200

        return jsonify({
            "error": "Movie not found"
        }), 404

    except Exception as error:

        print("TMDB Search Error:", error)

        return jsonify({
            "error": "Unable to retrieve movie information"
        }), 500


# --------------------------------------------------
# REGISTERED ROUTES
# --------------------------------------------------

print("\nRegistered Routes:")
print(app.url_map)


# --------------------------------------------------
# RUN FLASK APPLICATION
# --------------------------------------------------

if __name__ == "__main__":

    print("\nHybrid Recommendation Model Connected!")
    print("Starting Flask Backend...")

    app.run(
        debug=True,
        use_reloader=False,
        port=5001
    )
