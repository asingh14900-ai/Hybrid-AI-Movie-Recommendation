import pandas as pd
import os


# --------------------------------
# STEP 1: Load Dataset
# --------------------------------

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Datasets loaded successfully!")


# --------------------------------
# STEP 2: Calculate Movie Statistics
# --------------------------------

movie_stats = ratings.groupby("movieId").agg(
    average_rating=("rating", "mean"),
    rating_count=("rating", "count")
).reset_index()


# --------------------------------
# STEP 3: Calculate Global Average
# --------------------------------

C = ratings["rating"].mean()

# Minimum number of ratings required
m = movie_stats["rating_count"].quantile(0.70)

print("\nGlobal Average Rating:", C)
print("Minimum Ratings Required:", m)


# --------------------------------
# STEP 4: Calculate Weighted Rating
# --------------------------------

movie_stats["weighted_rating"] = (
    (movie_stats["rating_count"] / 
     (movie_stats["rating_count"] + m))
    * movie_stats["average_rating"]
    +
    (m / 
     (movie_stats["rating_count"] + m))
    * C
)


# --------------------------------
# STEP 5: Merge Movie Information
# --------------------------------

popular_movies = movie_stats.merge(
    movies,
    on="movieId"
)


# --------------------------------
# STEP 6: Sort Movies
# --------------------------------

popular_movies = popular_movies.sort_values(
    by="weighted_rating",
    ascending=False
)


# --------------------------------
# STEP 7: Display Top 10 Movies
# --------------------------------

print("\nTop 10 Popular Movies:\n")

print(
    popular_movies[
        ["title", "genres", "average_rating",
         "rating_count", "weighted_rating"]
    ].head(10).to_string(index=False)
)