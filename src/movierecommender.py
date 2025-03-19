import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer

#############################
# 1. Load and Preprocess Data
#############################

# Load historical movies and ratings data
movies_df = pd.read_csv("data/movies.csv")
ratings_df = pd.read_csv("data/ratings.csv")

# Convert the genres string into a list for each movie (split on the pipe '|')
movies_df['genres'] = movies_df['genres'].apply(lambda x: x.split('|'))

# One-Hot Encode the genres using MultiLabelBinarizer
mlb = MultiLabelBinarizer()
genres_onehot = mlb.fit_transform(movies_df['genres'])
genres_df = pd.DataFrame(genres_onehot, columns=mlb.classes_)
movies_df = pd.concat([movies_df, genres_df], axis=1)

print("Movies DataFrame with One-Hot Encoded Genres:")
print(movies_df.head())

# Merge the movies and ratings DataFrames on 'movieId'
merged_df = pd.merge(ratings_df, movies_df, on='movieId')
print("\nMerged DataFrame:")
print(merged_df.head())

#######################################
# 2. Build Optimized User Profiles
#######################################

# For each genre, compute the average rating for users who rated movies in that genre.
user_profiles = {}

for genre in mlb.classes_:
    print(f"Processing genre: {genre}")
    # Filter rows where the movie belongs to the current genre
    genre_ratings = merged_df[merged_df[genre] == 1]
    # Group by userId and compute the average rating for this genre
    avg_ratings = genre_ratings.groupby('userId')['rating'].mean()
    user_profiles[genre] = avg_ratings

# Convert the dictionary of Series into a DataFrame and fill missing values with 0.
user_profiles = pd.DataFrame(user_profiles).fillna(0)
print("\nOptimized User Profiles (Average Rating per Genre):")
print(user_profiles.head())

########################################
# 3. Define the Recommender Functions
########################################

# Cosine similarity function
def cosine_similarity(a, b):
    if np.linalg.norm(a) == 0 or np.linalg.norm(b) == 0:
        return 0
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Recommender function: Given a user_id, candidate movies DataFrame, and user profiles,
# return the top_n movies that best match the user's genre preferences.
def recommend_movies(user_id, candidate_movies_df, user_profiles, genre_columns, top_n=5):
    # Retrieve the user's genre preference vector
    user_vector = user_profiles.loc[user_id].values
    
    # Calculate similarity score for each movie using its one-hot encoded genre vector.
    scores = []
    for idx, row in candidate_movies_df.iterrows():
        movie_vector = row[genre_columns].values
        score = cosine_similarity(user_vector, movie_vector)
        scores.append(score)
    
    # Attach the computed scores to the candidate movies DataFrame
    candidate_movies_df = candidate_movies_df.copy()  # avoid modifying the original DataFrame
    candidate_movies_df['score'] = scores

    # Sort movies by the score in descending order and return the top recommendations.
    recommendations = candidate_movies_df.sort_values(by='score', ascending=False)
    return recommendations[['movieId', 'title', 'score']].head(top_n)

########################################
# 4. Load and Preprocess Current Airing Movies
########################################

# Load current airing movies (assumed to have columns similar to movies.csv)
current_airing_movies = pd.read_csv("data/current_airing_movies.csv")

# Convert the genres string into a list
current_airing_movies['genres'] = current_airing_movies['genres'].apply(lambda x: x.split('|'))

# One-Hot Encode using the same MultiLabelBinarizer (to ensure same genre columns)
genres_onehot_current = mlb.transform(current_airing_movies['genres'])
genres_df_current = pd.DataFrame(genres_onehot_current, columns=mlb.classes_)
current_airing_movies = pd.concat([current_airing_movies, genres_df_current], axis=1)

print("\nCurrent Airing Movies DataFrame with One-Hot Encoded Genres:")
print(current_airing_movies.head())

########################################
# 5. Example Usage: Recommend Current Airing Movies for an Example User
########################################

# Choose an example user (using the first user in our user profiles)
user_id_example = user_profiles.index[0]
genre_columns = list(mlb.classes_)  # The one-hot encoded genre columns

recommended_movies = recommend_movies(user_id_example, current_airing_movies, user_profiles, genre_columns, top_n=5)

print("\nTop recommended current airing movies for user", user_id_example)
print(recommended_movies)
