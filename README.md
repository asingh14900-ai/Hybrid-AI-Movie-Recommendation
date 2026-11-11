# 🎬 Hybrid AI-Based Personalized Movie Recommendation Engine

A personalized movie recommendation system that combines **Content-Based Filtering, Collaborative Filtering, Singular Value Decomposition (SVD), and Popularity-Based Filtering** to recommend movies based on user preferences and movie characteristics.

The project integrates machine learning, a Flask backend, a React frontend, and MySQL database management to provide an interactive movie recommendation experience.

---

## 📌 Project Overview

Traditional movie recommendation systems often rely on a single recommendation technique, which can limit recommendation quality.

This project implements a hybrid recommendation approach that combines multiple algorithms to generate more relevant and personalized movie recommendations.

The system uses movie genres, user rating patterns, predicted ratings, and movie popularity to calculate a final recommendation score.

## ✨ Key Features

- **Hybrid Recommendation System:** Combines four recommendation techniques.
- **Content-Based Filtering:** Recommends movies based on genre similarity using TF-IDF and cosine similarity.
- **Collaborative Filtering:** Uses similarities between users to generate recommendations.
- **SVD-Based Recommendation:** Uses matrix factorization to predict user ratings.
- **Popularity-Based Filtering:** Identifies movies based on weighted ratings and rating counts.
- **Movie Search:** Allows users to search for movies.
- **Movie Posters:** Integrates The Movie Database (TMDB) API to retrieve movie posters and details.
- **Database Integration:** Uses MySQL to store users, movies, ratings, and saved recommendations.
- **Interactive Frontend:** Provides a React-based interface for exploring recommendations.

## 🧠 Recommendation System Architecture

The hybrid recommendation engine combines four components:

| Algorithm | Technique | Weight |
|---|---|---:|
| Content-Based Filtering | TF-IDF and Cosine Similarity | 20% |
| Collaborative Filtering | User-Based Collaborative Filtering | 20% |
| SVD | Matrix Factorization | 50% |
| Popularity-Based Filtering | Weighted Rating | 10% |

### Recommendation Workflow

1. Load movie and rating datasets.
2. Preprocess movie information and user ratings.
3. Generate content-based similarity scores.
4. Calculate collaborative filtering scores.
5. Predict ratings using SVD.
6. Calculate popularity scores.
7. Normalize individual recommendation scores.
8. Combine the scores using weighted hybrid scoring.
9. Rank movies and return personalized recommendations.

The final recommendation score is a ranking score, not a probability.

## 🛠️ Technologies Used

### Machine Learning and Data Analysis
- Python
- Pandas
- NumPy
- Scikit-learn
- Scikit-Surprise
- TF-IDF
- Cosine Similarity
- Singular Value Decomposition (SVD)

### Backend
- Flask
- Flask-CORS
- Python
- MySQL Connector

### Frontend
- React.js
- JavaScript
- HTML5
- CSS3
- Vite

### External API
- TMDB API

### Database
- MySQL

### Development Tools
- VS Code
- Git
- GitHub
- MySQL Workbench

## 📂 Project Structure

```text
Hybrid-AI-Movie-Recommendation/
│
├── Backend/
│   ├── app.py
│   ├── db.py
│   ├── tmdb.py
│   ├── test_tmdb.py
│   ├── import_movies.py
│   └── import_users_ratings.py
│
├── Frontend/
│   ├── public/
│   ├── src/
│   │   ├── assets/
│   │   ├── App.jsx
│   │   ├── App.css
│   │   ├── index.css
│   │   └── main.jsx
│   ├── package.json
│   └── vite.config.js
│
├── dataset/
│   └── ml-latest-small/
│       ├── movies.csv
│       ├── ratings.csv
│       ├── links.csv
│       └── tags.csv
│
├── notebooks/
│   └── data_analysis.py
│
├── recommendation/
│   ├── content_based.py
│   ├── collaborative.py
│   ├── svd_recommendation.py
│   ├── popularity.py
│   ├── hybrid_recommendation.py
│   ├── evaluate_hybrid_ranking.py
│   └── evaluate_hybrid.py
│
├── .gitignore
└── README.md
```

## 📊 Dataset

This project uses the **MovieLens Latest Small Dataset**.

Dataset statistics:

| Parameter | Value |
|---|---:|
| Number of Movies | 9,742 |
| Number of Users | 610 |
| Number of Ratings | 100,836 |
| Average Rating | 3.50 |
| Rating Scale | 0.5 – 5.0 |

Dataset features include:
- Movie IDs
- Movie titles
- Genres
- User IDs
- User ratings
- Timestamps
- Movie tags

## 📈 Model Evaluation

The recommendation system was evaluated using rating prediction and ranking metrics.

### Rating Prediction

| Metric | Result |
|---|---:|
| RMSE | 0.8775 |
| MAE | 0.6742 |

### Recommendation Ranking

| Metric | Result |
|---|---:|
| Precision@10 | 0.4047 |
| Recall@10 | 0.4235 |
| NDCG@10 | 0.5544 |

The ranking evaluation uses a sampled-negative evaluation approach with 100 sampled negative movies per evaluated user and ratings of 4 or higher treated as relevant.

These results represent the current experimental baseline.

## ⚙️ Installation and Setup

### 1. Clone the Repository

```bash
git clone https://github.com/asingh14900-ai/Hybrid-AI-Movie-Recommendation.git
```

Navigate to the project directory:

```bash
cd Hybrid-AI-Movie-Recommendation
```

### 2. Install Python Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn scikit-surprise Flask flask-cors mysql-connector-python python-dotenv requests
```

### 3. Configure Environment Variables

Create a `.env` file inside the `Backend` directory.

Add the following configuration:

```env
TMDB_BEARER_TOKEN=your_tmdb_bearer_token

MYSQL_HOST=localhost
MYSQL_USER=root
MYSQL_PASSWORD=your_mysql_password
MYSQL_DATABASE=movie_recommendation_db
```

Replace the example values with your own credentials.

**Important:** Never upload your actual `.env` file or API credentials to GitHub.

### 4. Configure MySQL

1. Install and start MySQL Server.
2. Create the database:

```sql
CREATE DATABASE movie_recommendation_db;
```

3. Configure your database credentials in `Backend/.env`.
4. Execute your project database table creation SQL.
5. Import the movie and user-rating data using the provided import scripts.

Run the import scripts from the project root:

```bash
python Backend/import_movies.py
python Backend/import_users_ratings.py
```

### 5. Start the Flask Backend

Open a terminal from the project root:

```bash
cd Backend
python app.py
```

Backend server:

```text
http://127.0.0.1:5001
```

### 6. Start the React Frontend

Open another terminal:

```bash
cd Frontend
npm install
npm run dev
```

Open the local address displayed by Vite, usually:

```text
http://localhost:5173
```

## 🔗 API Endpoints

The Flask backend provides the following endpoints:

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | Backend home route |
| GET | `/health` | Health check |
| GET | `/api/recommendations/<user_id>` | Get personalized recommendations |
| GET | `/api/movies/search?q=` | Search movies |
| GET | `/api/db/stats` | Retrieve database statistics |
| POST | `/api/recommendations/<user_id>/save` | Save recommendations |
| GET | `/api/recommendations/<user_id>/saved` | Retrieve saved recommendations |
| GET | `/api/tmdb/search?query=` | Search movies using TMDB |

## 🚀 Future Enhancements

- Improve personalization using explicit user feedback.
- Integrate deep learning-based recommendation models.
- Improve cold-start recommendations for new users.
- Incorporate movie descriptions, cast, directors, and tags into content-based filtering.
- Improve recommendation diversity and ranking.
- Deploy the application to a cloud platform.
- Add user authentication and profile management.

## 👨‍💻 Author

**Ankit Kumar Singh**

B.Tech – Computer Science and Engineering (AI & ML)

ABES Engineering College

GitHub: [asingh14900-ai](https://github.com/asingh14900-ai)

---

⭐ If you find this project interesting, consider giving the repository a star!
