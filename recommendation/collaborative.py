import pandas as pd
import numpy as np
import os

from sklearn.metrics.pairwise import cosine_similarity


# --------------------------------
# STEP 1: Load Dataset
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))
movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Datasets loaded successfully!")


# --------------------------------
# STEP 2: Create User-Movie Matrix
# --------------------------------

user_movie_matrix = ratings.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
)

user_movie_matrix = user_movie_matrix.fillna(0)

print("\nUser-Movie Matrix Shape:")
print(user_movie_matrix.shape)


# --------------------------------
# STEP 3: Calculate User Similarity
# --------------------------------

user_similarity = cosine_similarity(user_movie_matrix)

print("\nUser Similarity Matrix Shape:")
print(user_similarity.shape)


# --------------------------------
# STEP 4: Recommendation Function
# --------------------------------

def recommend_movies(user_id, number_of_recommendations=10, k=20):

    if user_id not in user_movie_matrix.index:
        print("User not found!")
        return

    # Get user's position in matrix
    user_index = user_movie_matrix.index.get_loc(user_id)

    # Get similarity scores of other users
    similarity_scores = user_similarity[user_index].copy()

    # Exclude the current user
    similarity_scores[user_index] = -1

    # Find top similar users
    similar_users = np.argsort(similarity_scores)[::-1][:k]

    similar_users_scores = similarity_scores[similar_users]

    # Get movies already rated by current user
    user_ratings = user_movie_matrix.iloc[user_index]

    # Find unrated movies
    recommendations = []

    for movie_id in user_movie_matrix.columns:

        if user_ratings[movie_id] == 0:

            # Ratings given by similar users
            other_ratings = user_movie_matrix.iloc[
                similar_users
            ][movie_id].values

            # Consider only users who rated this movie
            mask = other_ratings > 0

            if mask.sum() == 0:
                continue

            weights = similar_users_scores[mask]
            values = other_ratings[mask]

            # Ignore negative and zero similarities
            positive_mask = weights > 0

            weights = weights[positive_mask]
            values = values[positive_mask]

            if len(weights) == 0:
                continue

            # Calculate weighted predicted rating
            predicted_rating = np.dot(weights, values) / weights.sum()

            # Keep predictions within rating range
            predicted_rating = max(0.5, min(5.0, predicted_rating))

            recommendations.append(
                (movie_id, predicted_rating)
            )

    # --------------------------------
    # STEP 5: Sort Recommendations
    # --------------------------------

    recommendations.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Display top recommendations
    top_recommendations = recommendations[
        :number_of_recommendations
    ]

    recommended_movies = pd.DataFrame(
        top_recommendations,
        columns=["movieId", "predicted_rating"]
    )

    recommended_movies = recommended_movies.merge(
        movies,
        on="movieId"
    )

    print("\nRecommendations for User:", user_id)

    print(
        recommended_movies[
            ["title", "genres", "predicted_rating"]
        ].to_string(index=False)
    )


# --------------------------------
# STEP 6: Test Recommendation
# --------------------------------

recommend_movies(1, 10)