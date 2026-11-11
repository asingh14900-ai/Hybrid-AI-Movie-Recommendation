
import pandas as pd
import numpy as np
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from surprise import Dataset, Reader, SVD


# --------------------------------
# STEP 1: Load Dataset
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))
movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Datasets loaded successfully!")


# --------------------------------
# STEP 2: Content-Based Model
# --------------------------------

movies["genres_clean"] = movies["genres"].str.replace(
    "|", " ", regex=False
)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["genres_clean"].fillna("")
)

movie_id_to_index = pd.Series(
    movies.index,
    index=movies["movieId"]
).to_dict()

print("Content-Based Model Ready!")


# --------------------------------
# STEP 3: Collaborative Filtering
# --------------------------------

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

print("Collaborative Model Ready!")


# --------------------------------
# STEP 4: SVD Model
# --------------------------------

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

trainset = data.build_full_trainset()

svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("Training SVD Model...")

svd_model.fit(trainset)

print("SVD Model Ready!")


# --------------------------------
# STEP 5: Popularity Model
# --------------------------------

movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

C = ratings["rating"].mean()

m = movie_stats["rating_count"].quantile(0.70)

movie_stats["popularity_score"] = (
    (movie_stats["rating_count"] /
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m /
     (movie_stats["rating_count"] + m))
    * C
)

popularity_dict = dict(
    zip(movie_stats["movieId"], movie_stats["popularity_score"])
)

print("Popularity Model Ready!")


# --------------------------------
# STEP 6: Normalize Scores
# --------------------------------

def normalize_scores(scores):

    scores = np.array(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros(len(scores))

    return (scores - minimum) / (maximum - minimum)


# --------------------------------
# STEP 7: Hybrid Recommendation
# --------------------------------

def recommend_movies(user_id, number_of_recommendations=10, k=20):

    # Check whether user exists
    if user_id not in user_movie_matrix.index:
        return pd.DataFrame()

    # Get user index and ratings
    user_index = user_movie_matrix.index.get_loc(user_id)

    user_ratings = user_movie_matrix.iloc[user_index]

    rated_movies = user_ratings[user_ratings > 0].index.tolist()

    # Get all unrated movies
    candidate_movies = [
        movie_id for movie_id in movies["movieId"]
        if movie_id not in rated_movies
    ]

    if len(candidate_movies) == 0:
        return pd.DataFrame()

    # --------------------------------
    # A. Content-Based Scores
    # --------------------------------

    user_mean = user_ratings[user_ratings > 0].mean()

    liked_movies = user_ratings[
        user_ratings > user_mean
    ]

    if len(liked_movies) == 0:
        liked_movies = user_ratings[user_ratings > 0]

    liked_indices = [
        movie_id_to_index[mid]
        for mid in liked_movies.index
        if mid in movie_id_to_index
    ]

    liked_weights = np.array([
        liked_movies[mid] - user_mean
        for mid in liked_movies.index
        if mid in movie_id_to_index
    ])

    if len(liked_weights) > 0:

        liked_weights = np.maximum(liked_weights, 0.1)

        content_similarity = cosine_similarity(
            tfidf_matrix,
            tfidf_matrix[liked_indices]
        )

        content_scores = np.average(
            content_similarity,
            axis=1,
            weights=liked_weights
        )

    else:
        content_scores = np.zeros(len(movies))

    content_dict = dict(
        zip(movies["movieId"], content_scores)
    )

    # --------------------------------
    # B. Collaborative Filtering Scores
    # --------------------------------

    similarity_scores = user_similarity[user_index].copy()

    similarity_scores[user_index] = -1

    similar_users = np.argsort(similarity_scores)[::-1][:k]

    similar_users_scores = similarity_scores[similar_users]

    collaborative_dict = {}

    for movie_id in candidate_movies:

        if movie_id not in user_movie_matrix.columns:
            collaborative_dict[movie_id] = C
            continue

        other_ratings = user_movie_matrix.iloc[
            similar_users
        ][movie_id].values

        mask = other_ratings > 0

        weights = similar_users_scores[mask]
        values = other_ratings[mask]

        positive_mask = weights > 0

        weights = weights[positive_mask]
        values = values[positive_mask]

        if len(weights) > 0:

            collaborative_dict[movie_id] = (
                np.dot(weights, values) / weights.sum()
            )

        else:
            collaborative_dict[movie_id] = C

    # --------------------------------
    # C. SVD Scores
    # --------------------------------

    svd_dict = {}

    for movie_id in candidate_movies:

        svd_dict[movie_id] = svd_model.predict(
            user_id,
            movie_id
        ).est

    # --------------------------------
    # D. Popularity Scores
    # --------------------------------

    popularity_scores = [
        popularity_dict.get(movie_id, C)
        for movie_id in candidate_movies
    ]

    # --------------------------------
    # STEP 8: Normalize All Scores
    # --------------------------------

    content_values = [
        content_dict.get(movie_id, 0)
        for movie_id in candidate_movies
    ]

    collaborative_values = [
        collaborative_dict[movie_id]
        for movie_id in candidate_movies
    ]

    svd_values = [
        svd_dict[movie_id]
        for movie_id in candidate_movies
    ]

    normalized_content = normalize_scores(content_values)

    normalized_collaborative = normalize_scores(
        collaborative_values
    )

    normalized_svd = normalize_scores(svd_values)

    normalized_popularity = normalize_scores(
        popularity_scores
    )

    # --------------------------------
    # STEP 9: Calculate Final Score
    # --------------------------------

    final_scores = (
        0.20 * normalized_content
        + 0.20 * normalized_collaborative
        + 0.50 * normalized_svd
        + 0.10 * normalized_popularity
    )

    # --------------------------------
    # STEP 10: Generate Recommendations
    # --------------------------------

    result = pd.DataFrame({
        "movieId": candidate_movies,
        "content_score": normalized_content,
        "collaborative_score": normalized_collaborative,
        "svd_score": normalized_svd,
        "popularity_score": normalized_popularity,
        "final_score": final_scores
    })

    result = result.sort_values(
        by="final_score",
        ascending=False
    )

    result = result.head(number_of_recommendations)

    result = result.merge(
        movies[["movieId", "title", "genres"]],
        on="movieId"
    )

    # Return recommendations to Flask
    return result[
        [
            "movieId",
            "title",
            "genres",
            "content_score",
            "collaborative_score",
            "svd_score",
            "popularity_score",
            "final_score"
        ]
    ]


# --------------------------------
# STEP 11: Models Ready
# --------------------------------

print("\nAll Recommendation Models Ready!")
