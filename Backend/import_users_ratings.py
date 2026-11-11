
import os
import pandas as pd
import mysql.connector

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dataset paths
dataset_path = os.path.join(
    BASE_DIR,
    "dataset",
    "ml-latest-small"
)

ratings = pd.read_csv(os.path.join(dataset_path, "ratings.csv"))

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Mysqlankit2005**",
    database="movie_recommendation_db"
)

cursor = connection.cursor()

try:
    # STEP 1: Extract unique users
    user_ids = ratings["userId"].unique()

    user_data = [
        (int(user_id), f"movielens_user_{user_id}")
        for user_id in user_ids
    ]

    # Insert users
    user_query = """
    INSERT IGNORE INTO users (user_id, username)
    VALUES (%s, %s)
    """

    cursor.executemany(user_query, user_data)
    print("Users imported:", cursor.rowcount)

    # STEP 2: Prepare ratings
    rating_data = [
        (
            int(row.userId),
            int(row.movieId),
            float(row.rating),
            int(row.timestamp)
        )
        for row in ratings.itertuples(index=False)
    ]

    # Insert ratings
    rating_query = """
    INSERT IGNORE INTO ratings
    (user_id, movie_id, rating, rating_timestamp)
    VALUES (%s, %s, %s, %s)
    """

    cursor.executemany(rating_query, rating_data)

    connection.commit()

    print("Ratings imported:", cursor.rowcount)
    print("Users and ratings imported successfully!")

except Exception as e:
    connection.rollback()
    print("Import failed:", e)

finally:
    cursor.close()
    connection.close()
