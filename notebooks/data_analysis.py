import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os

# --------------------------------
# STEP 1: Dataset Path
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")


# --------------------------------
# STEP 2: Load Datasets
# --------------------------------

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Datasets loaded successfully!")


# --------------------------------
# STEP 3: Dataset Dimensions
# --------------------------------

print("\nRatings Dataset Shape:")
print(ratings.shape)

print("\nMovies Dataset Shape:")
print(movies.shape)


# --------------------------------
# STEP 4: Display First Five Rows
# --------------------------------

print("\nRatings Dataset:")
print(ratings.head())

print("\nMovies Dataset:")
print(movies.head())


# --------------------------------
# STEP 5: Dataset Information
# --------------------------------

print("\nRatings Information:")
ratings.info()

print("\nMovies Information:")
movies.info()


# --------------------------------
# STEP 6: Check Missing Values
# --------------------------------

print("\nMissing Values in Ratings:")
print(ratings.isnull().sum())

print("\nMissing Values in Movies:")
print(movies.isnull().sum())


# --------------------------------
# STEP 7: Rating Analysis
# --------------------------------

print("\nRating Distribution:")

print(ratings["rating"].value_counts().sort_index())

print("\nAverage Rating:")
print(ratings["rating"].mean())

print("\nMaximum Rating:")
print(ratings["rating"].max())

print("\nMinimum Rating:")
print(ratings["rating"].min())


# --------------------------------
# STEP 8: Rating Visualization
# --------------------------------

ratings["rating"].value_counts().sort_index().plot(
    kind="bar",
    title="Movie Rating Distribution"
)

plt.xlabel("Rating")
plt.ylabel("Number of Ratings")

plt.show()


# --------------------------------
# STEP 9: Most Popular Movies
# --------------------------------

movie_rating_count = ratings.groupby("movieId").size()

popular_movies = movie_rating_count.sort_values(
    ascending=False
).head(10)

popular_movies = popular_movies.reset_index(name="total_ratings")

popular_movies = popular_movies.merge(
    movies[["movieId", "title"]],
    on="movieId"
)

print("\nTop 10 Most Popular Movies:")

print(popular_movies[["title", "total_ratings"]])


# --------------------------------
# STEP 10: Average Movie Ratings
# --------------------------------

average_ratings = ratings.groupby("movieId")["rating"].mean()

rating_counts = ratings.groupby("movieId").size()

movie_statistics = pd.DataFrame({
    "average_rating": average_ratings,
    "total_ratings": rating_counts
})

movie_statistics = movie_statistics.reset_index()

movie_statistics = movie_statistics.merge(
    movies[["movieId", "title"]],
    on="movieId"
)

# Select movies with at least 50 ratings

top_rated_movies = movie_statistics[
    movie_statistics["total_ratings"] >= 50
]

top_rated_movies = top_rated_movies.sort_values(
    by="average_rating",
    ascending=False
)

print("\nTop 10 Highest Rated Movies:")

print(
    top_rated_movies[
        ["title", "average_rating", "total_ratings"]
    ].head(10)
)


# --------------------------------
# STEP 11: Genre Analysis
# --------------------------------

genre_counts = movies["genres"].str.split("|").explode()

genre_counts = genre_counts[genre_counts != "(no genres listed)"]

genre_counts = genre_counts.value_counts()

print("\nMovie Genre Distribution:")

print(genre_counts)

genre_counts.plot(
    kind="bar",
    title="Number of Movies by Genre"
)

plt.xlabel("Genre")
plt.ylabel("Number of Movies")

plt.xticks(rotation=45)

plt.tight_layout()

plt.show()


# --------------------------------
# STEP 12: Unique Users and Movies
# --------------------------------

print("\nTotal Unique Users:")
print(ratings["userId"].nunique())

print("\nTotal Unique Movies:")
print(movies["movieId"].nunique())

print("\nEDA COMPLETED SUCCESSFULLY!")