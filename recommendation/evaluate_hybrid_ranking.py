import pandas as pd
import numpy as np
import os

from sklearn.model_selection import train_test_split
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

print("Dataset Loaded Successfully!", flush=True)


# --------------------------------
# STEP 2: Split Dataset
# --------------------------------

train_df, test_df = train_test_split(
    ratings,
    test_size=0.20,
    random_state=42
)

print("\nTraining Ratings:", len(train_df))
print("Testing Ratings:", len(test_df))


# --------------------------------
# STEP 3: Content-Based Model
# --------------------------------

movies["genres_clean"] = movies["genres"].str.replace(
    "|", " ", regex=False
)

tfidf = TfidfVectorizer()

tfidf_matrix = tfidf.fit_transform(
    movies["genres_clean"].fillna("")
)

movie_id_to_index = dict(
    zip(movies["movieId"], movies.index)
)

print("\nContent-Based Model Ready!")


# --------------------------------
# STEP 4: Collaborative Filtering
# --------------------------------

user_movie_matrix = train_df.pivot_table(
    index="userId",
    columns="movieId",
    values="rating"
).fillna(0)

user_similarity = cosine_similarity(user_movie_matrix)

print("Collaborative Model Ready!", flush=True)


# --------------------------------
# STEP 5: Train SVD Model
# --------------------------------

print("\\nStep 5 Started: Initializing SVD...", flush=True)

reader = Reader(rating_scale=(0.5, 5.0))
print("Reader Created!", flush=True)

train_data = Dataset.load_from_df(
    train_df[["userId", "movieId", "rating"]],
    reader
)
print("Training Dataset Loaded!", flush=True)

trainset = train_data.build_full_trainset()
print("Training Set Created!", flush=True)

svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("Training SVD Model...", flush=True)
svd_model.fit(trainset)
print("SVD Model Ready!", flush=True)


# STEP 6: Popularity Model
# --------------------------------

movie_stats = train_df.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()

C = train_df["rating"].mean()

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

print("Popularity Model Ready!", flush=True)


# --------------------------------
# STEP 7: Normalize Scores
# --------------------------------

def normalize_scores(scores):

    scores = np.array(scores, dtype=float)

    minimum = scores.min()
    maximum = scores.max()

    if maximum == minimum:
        return np.zeros(len(scores))

    return (scores - minimum) / (maximum - minimum)


# --------------------------------
# STEP 8: Evaluation Functions
# --------------------------------

def precision_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    if len(recommended) == 0:
        return 0

    hits = len(set(recommended) & set(relevant))

    return hits / k


def recall_at_k(recommended, relevant, k=10):

    if len(relevant) == 0:
        return 0

    recommended = recommended[:k]

    hits = len(set(recommended) & set(relevant))

    return hits / len(relevant)


def ndcg_at_k(recommended, relevant, k=10):

    recommended = recommended[:k]

    dcg = 0

    for i, movie_id in enumerate(recommended):

        if movie_id in relevant:
            dcg += 1 / np.log2(i + 2)

    ideal_hits = min(len(relevant), k)

    idcg = sum(
        1 / np.log2(i + 2)
        for i in range(ideal_hits)
    )

    if idcg == 0:
        return 0

    return dcg / idcg


# --------------------------------
# STEP 9: Generate Hybrid Recommendations
# --------------------------------

def recommend_for_evaluation(user_id, test_movies, number_of_recommendations=10):

    user_index = user_movie_matrix.index.get_loc(user_id)

    user_ratings = user_movie_matrix.iloc[user_index]

    rated_movies = set(
        user_ratings[user_ratings > 0].index
    )

    test_movie_ids = set(test_movies)

    # Sample 100 movies not rated in training or testing
    negative_candidates = list(
        set(movies["movieId"])
        - rated_movies
        - test_movie_ids
    )

    rng = np.random.default_rng(user_id)

    sample_size = min(100, len(negative_candidates))

    sampled_movies = rng.choice(
        negative_candidates,
        size=sample_size,
        replace=False
    ).tolist()

    candidate_movies = sorted(test_movie_ids) + sampled_movies

    # A. Content-Based Scores

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

    if len(liked_indices) > 0:

        liked_weights = np.maximum(liked_weights, 0.1)

        content_profile = (
            tfidf_matrix[liked_indices].T @ liked_weights
        )

        # Convert the result to a numeric 2D array.
        if hasattr(content_profile, "toarray"):
            content_profile = content_profile.toarray()
        else:
            content_profile = np.asarray(content_profile)

        content_profile = content_profile.reshape(1, -1)

        content_scores = cosine_similarity(
            tfidf_matrix,
            content_profile
        ).ravel()

        content_dict = dict(
            zip(movies["movieId"], content_scores)
        )

    else:
        content_dict = {}

    # B. Collaborative Filtering Scores

    similarity_scores = user_similarity[user_index].copy()

    similarity_scores[user_index] = -1

    similar_users = np.argsort(similarity_scores)[::-1][:20]

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

    # C. SVD Scores

    svd_scores = [
        svd_model.predict(user_id, movie_id).est
        for movie_id in candidate_movies
    ]

    # D. Popularity Scores

    popularity_scores = [
        popularity_dict.get(movie_id, C)
        for movie_id in candidate_movies
    ]

    # E. Collect Scores

    content_values = [
        content_dict.get(movie_id, 0)
        for movie_id in candidate_movies
    ]

    collaborative_values = [
        collaborative_dict[movie_id]
        for movie_id in candidate_movies
    ]

    # F. Normalize Scores

    normalized_content = normalize_scores(content_values)

    normalized_collaborative = normalize_scores(
        collaborative_values
    )

    normalized_svd = normalize_scores(svd_scores)

    normalized_popularity = normalize_scores(
        popularity_scores
    )

    # G. Calculate Hybrid Score

    final_scores = (
        0.20 * normalized_content
        + 0.20 * normalized_collaborative
        + 0.50 * normalized_svd
        + 0.10 * normalized_popularity
    )

    result = pd.DataFrame({
        "movieId": candidate_movies,
        "final_score": final_scores
    })

    result = result.sort_values(
        by="final_score",
        ascending=False
    )

    return result["movieId"].head(
        number_of_recommendations
    ).tolist()


# --------------------------------
# STEP 10: Evaluate All Users
# --------------------------------

precision_scores = []
recall_scores = []
ndcg_scores = []

test_groups = test_df.groupby("userId")

print("\nEvaluating Hybrid Recommendation System...", flush=True)

for user_number, (user_id, user_test) in enumerate(test_groups, start=1):

    if user_number % 50 == 0:
        print(f"Processed {user_number} test users...", flush=True)

    # Only evaluate users present in training data
    if user_id not in user_movie_matrix.index:
        continue

    relevant_movies = set(
        user_test[
            user_test["rating"] >= 4.0
        ]["movieId"]
    )

    if len(relevant_movies) == 0:
        continue

    test_movie_ids = user_test["movieId"].tolist()

    recommendations = recommend_for_evaluation(
        user_id,
        test_movie_ids,
        10
    )

    precision_scores.append(
        precision_at_k(recommendations, relevant_movies)
    )

    recall_scores.append(
        recall_at_k(recommendations, relevant_movies)
    )

    ndcg_scores.append(
        ndcg_at_k(recommendations, relevant_movies)
    )


# --------------------------------
# STEP 11: Display Results
# --------------------------------

print("\n========== HYBRID MODEL EVALUATION ==========")

print("Users Evaluated:", len(precision_scores))

print(
    "Precision@10:",
    round(np.mean(precision_scores), 4)
)

print(
    "Recall@10:",
    round(np.mean(recall_scores), 4)
)

print(
    "NDCG@10:",
    round(np.mean(ndcg_scores), 4)
)

print("=============================================")