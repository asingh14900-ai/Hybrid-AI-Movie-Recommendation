
import os
import mysql.connector
from dotenv import load_dotenv

# Load environment variables from Backend/.env
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
load_dotenv(os.path.join(BASE_DIR, ".env"))


def get_db_connection():
    connection = mysql.connector.connect(
        host=os.getenv("MYSQL_HOST"),
        user=os.getenv("MYSQL_USER"),
        password=os.getenv("MYSQL_PASSWORD"),
        database=os.getenv("MYSQL_DATABASE")
    )

    return connection


# Test database connection
if __name__ == "__main__":
    try:
        connection = get_db_connection()

        if connection.is_connected():
            print("MySQL Database Connected Successfully!")

        connection.close()

    except mysql.connector.Error as error:
        print("Database Connection Failed:", error)
