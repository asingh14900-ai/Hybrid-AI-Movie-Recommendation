import os
import pandas as pd
import mysql.connector

# Project directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Load MovieLens movies.csv
csv_path = os.path.join(
    BASE_DIR,
    "dataset",
    "ml-latest-small",
    "movies.csv"
)

movies = pd.read_csv(csv_path)

# Connect to MySQL
connection = mysql.connector.connect(
    host="localhost",
    user="root",
    password="Mysqlankit2005**",
    database="movie_recommendation_db"
)

cursor = connection.cursor()

# Insert movies into MySQL
query = """
INSERT INTO movies (movie_id, title, genres)
VALUES (%s, %s, %s)
"""

movie_data = [
    (int(row.movieId), row.title, row.genres)
    for row in movies.itertuples(index=False)
]

cursor.executemany(query, movie_data)

connection.commit()

print("Movies imported successfully!")
print("Total movies imported:", cursor.rowcount)

cursor.close()
connection.close()