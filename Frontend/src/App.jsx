import { useState } from "react";
import "./App.css";

const API_URL = "http://127.0.0.1:5001";

function App() {
  const [userId, setUserId] = useState(1);
  const [recommendations, setRecommendations] = useState([]);
  const [searchQuery, setSearchQuery] = useState("");
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState("");

  // Fetch personalized recommendations
  const getRecommendations = async () => {
    try {
      setLoading(true);
      setMessage("");

      const response = await fetch(
        `${API_URL}/api/recommendations/${userId}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Unable to fetch recommendations");
      }

      setRecommendations(data.recommendations);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

  // Search movies
  const searchMovies = async (e) => {
    e.preventDefault();

    if (!searchQuery.trim()) {
      setSearchResults([]);
      return;
    }

    try {
      setLoading(true);
      setMessage("");

      const response = await fetch(
        `${API_URL}/api/movies/search?q=${encodeURIComponent(searchQuery)}`
      );

      const data = await response.json();

      if (!response.ok) {
        throw new Error(data.error || "Movie search failed");
      }

      setSearchResults(data.movies);
    } catch (error) {
      setMessage(error.message);
    } finally {
      setLoading(false);
    }
  };

 
const MovieCard = ({ movie, index, isRecommendation = false }) => (
  <div className="movie-card">

    <div className="movie-poster">

      {movie.poster_url && (
        <img
          src={movie.poster_url}
          alt={movie.title}
          loading="lazy"
          onError={(e) => {
            e.currentTarget.style.display = "none";

            const placeholder =
              e.currentTarget.parentElement.querySelector(
                ".poster-placeholder"
              );

            if (placeholder) {
              placeholder.style.display = "flex";
            }
          }}
        />
      )}

      <div
        className="poster-placeholder"
        style={{
          display: movie.poster_url ? "none" : "flex"
        }}
      >
        🎬
      </div>

      <div className="movie-number">
        #{index + 1}
      </div>

    </div>

    <div className="movie-info">

      <h3>{movie.title}</h3>

      <p className="genres">
        {movie.genres?.replaceAll("|", " • ")}
      </p>

      {isRecommendation && (
        <div className="score">
          ⭐ Score: {Number(movie.final_score).toFixed(3)}
        </div>
      )}

    </div>

  </div>
);

  return (
    <div className="app">

      {/* Navbar */}
      <nav className="navbar">
        <div className="logo">
          🎬 Cine<span>Match</span>
        </div>

        <div className="nav-links">
          <a href="#home">Home</a>
          <a href="#search">Search Movies</a>
          <a href="#recommendations">Recommendations</a>
        </div>
      </nav>

      {/* Hero Section */}
      <section className="hero" id="home">
        <div className="hero-content">
          <p className="tagline">AI-POWERED MOVIE DISCOVERY</p>

          <h1>
            Discover Movies
            <br />
            <span>Made For You</span>
          </h1>

          <p className="hero-description">
            Explore movies and discover personalized recommendations
            powered by hybrid artificial intelligence.
          </p>

          <a href="#recommendations" className="hero-button">
            Explore Recommendations
          </a>
        </div>
      </section>

      {/* Movie Search */}
      <section className="section" id="search">
        <div className="section-heading">
          <p className="tagline">EXPLORE OUR COLLECTION</p>
          <h2>Find Your Movie</h2>
        </div>

        <form className="search-box" onSubmit={searchMovies}>
          <input
            type="text"
            placeholder="Search movies... e.g. Batman"
            value={searchQuery}
            onChange={(e) => setSearchQuery(e.target.value)}
          />

          <button type="submit">Search</button>
        </form>

        {searchResults.length > 0 && (
          <div className="movie-grid">
            {searchResults.map((movie, index) => (
              <MovieCard
                key={movie.movieId}
                movie={movie}
                index={index}
              />
            ))}
          </div>
        )}

        {searchQuery && searchResults.length === 0 && !loading && (
          <p className="empty-message">
            Search for a movie to see matching results.
          </p>
        )}
      </section>

      {/* Personalized Recommendations */}
      <section className="section recommendation-section" id="recommendations">
        <div className="section-heading">
          <p className="tagline">POWERED BY HYBRID AI</p>
          <h2>Recommended For You</h2>
          <p>Discover movies selected according to your interests.</p>
        </div>

        <div className="user-controls">
          <input
            type="number"
            min="1"
            value={userId}
            onChange={(e) => setUserId(e.target.value)}
            placeholder="Enter User ID"
          />

          <button onClick={getRecommendations}>
            Get Recommendations
          </button>
        </div>

        {loading && <p className="status-message">Loading movies...</p>}

        {message && <p className="error-message">{message}</p>}

        {recommendations.length > 0 && (
          <div className="movie-grid">
            {recommendations.map((movie, index) => (
              <MovieCard
                key={movie.movieId}
                movie={movie}
                index={index}
                isRecommendation={true}
              />
            ))}
          </div>
        )}

        {!loading && recommendations.length === 0 && (
          <p className="empty-message">
            Enter a user ID and click Get Recommendations.
          </p>
        )}
      </section>

      {/* Footer */}
      <footer>
        <h3>CineMatch 🎬</h3>
        <p>
          Hybrid AI-Based Personalized Movie Recommendation Engine
        </p>
        <p>Powered by Machine Learning and Artificial Intelligence</p>
      </footer>

    </div>
  );
}

export default App;