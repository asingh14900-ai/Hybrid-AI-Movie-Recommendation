from tmdb import search_movie

movie = search_movie("Batman")

if movie:
    print("\nTMDB Connected Successfully!")
    print("\nMovie Details:")

    for key, value in movie.items():
        print(f"{key}: {value}")

else:
    print("Movie not found or API request failed.")