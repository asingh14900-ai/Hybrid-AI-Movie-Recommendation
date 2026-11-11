import pandas as pd
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
# Get project directory

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset path

dataset_path = os.path.join(BASE_DIR, "dataset", "ml-latest-small")

# Load movies

movies = pd.read_csv(os.path.join(dataset_path, "movies.csv"))

print("Movies dataset loaded successfully!")

print(movies.head())

# Replace | with spaces

movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)

print(movies[["title", "genres"]].head())

# Create TF-IDF object

tfidf = TfidfVectorizer()

# Convert genres into numerical vectors

tfidf_matrix = tfidf.fit_transform(movies["genres"])

print("\nTF-IDF Matrix Shape:")
print(tfidf_matrix.shape)

print("\nGenre Vocabulary:")

print(tfidf.get_feature_names_out())

# Calculate cosine similarity

similarity_matrix = cosine_similarity(tfidf_matrix)

print("\nSimilarity Matrix Shape:")
print(similarity_matrix.shape)

def recommend_movies(movie_title, number_of_recommendations=10):

    # Find movie index

    movie_index = movies[
        movies["title"].str.lower() == movie_title.lower()
    ].index

    if len(movie_index) == 0:
        print("Movie not found!")
        return

    movie_index = movie_index[0]

    # Get similarity scores

    similarity_scores = list(enumerate(similarity_matrix[movie_index]))

    # Sort movies by similarity

    similarity_scores = sorted(
        similarity_scores,
        key=lambda x: x[1],
        reverse=True
    )

    # Remove the movie itself

    similarity_scores = similarity_scores[1:number_of_recommendations + 1]

    print("\nMovies Recommended for:", movie_title)

    for index, score in similarity_scores:

        print(
            movies.iloc[index]["title"],
            "| Similarity:",
            round(score, 3)
        )
        
        # Test recommendation system

recommend_movies("Toy Story (1995)", 10)