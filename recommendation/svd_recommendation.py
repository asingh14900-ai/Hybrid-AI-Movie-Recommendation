import pandas as pd
import os

from surprise import Dataset, Reader, SVD, accuracy
from surprise.model_selection import train_test_split


# --------------------------------
# STEP 1: Load Dataset
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Datasets loaded successfully!")


# --------------------------------
# STEP 2: Prepare Dataset for SVD
# --------------------------------

reader = Reader(rating_scale=(0.5, 5.0))

data = Dataset.load_from_df(
    ratings[["userId", "movieId", "rating"]],
    reader
)

print("\nDataset prepared successfully!")


# --------------------------------
# STEP 3: Split Dataset
# --------------------------------

trainset, testset = train_test_split(
    data,
    test_size=0.2,
    random_state=42
)

print("\nTraining data:", trainset.n_ratings)
print("Testing data:", len(testset))


# --------------------------------
# STEP 4: Train SVD Model
# --------------------------------

svd_model = SVD(
    n_factors=50,
    n_epochs=20,
    lr_all=0.005,
    reg_all=0.02,
    random_state=42
)

print("\nTraining SVD Model...")

svd_model.fit(trainset)

print("SVD Model trained successfully!")


# --------------------------------
# STEP 5: Evaluate Model
# --------------------------------

predictions = svd_model.test(testset)

print("\nModel Evaluation:")

rmse = accuracy.rmse(predictions)

mae = accuracy.mae(predictions)

print("\nRMSE:", rmse)
print("MAE:", mae)


# --------------------------------
# STEP 6: Recommendation Function
# --------------------------------

def recommend_movies(user_id, number_of_recommendations=10):

    # Check whether user exists
    if user_id not in ratings["userId"].values:
        print("User not found!")
        return

    # Get movies already rated by user
    rated_movies = ratings[
        ratings["userId"] == user_id
    ]["movieId"].tolist()

    # Get all available movie IDs
    all_movies = movies["movieId"].tolist()

    # Find movies not rated by user
    unrated_movies = [
        movie_id for movie_id in all_movies
        if movie_id not in rated_movies
    ]

    # Predict ratings for unrated movies
    predictions = []

    for movie_id in unrated_movies:

        predicted_rating = svd_model.predict(
            user_id,
            movie_id
        ).est

        predictions.append(
            (movie_id, predicted_rating)
        )

    # Sort predictions in descending order
    predictions.sort(
        key=lambda x: x[1],
        reverse=True
    )

    # Select top recommendations
    top_recommendations = predictions[
        :number_of_recommendations
    ]

    # Convert recommendations into DataFrame
    recommended_movies = pd.DataFrame(
        top_recommendations,
        columns=["movieId", "predicted_rating"]
    )

    # Add movie titles and genres
    recommended_movies = recommended_movies.merge(
        movies,
        on="movieId"
    )

    print("\nSVD Recommendations for User:", user_id)

    print(
        recommended_movies[
            ["title", "genres", "predicted_rating"]
        ].to_string(index=False)
    )


# --------------------------------
# STEP 7: Test Recommendation
# --------------------------------

recommend_movies(1, 10)