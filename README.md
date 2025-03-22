
# ML Movie Recommender: DBTT Cathay Project Movie Recommender

Welcome to the ML Movie Recommender project! This project aims to build a content-based recommender system that leverages movie genres and user ratings to suggest movies that users are likely to enjoy. The project was developed as part of the DBTT Cathay Project.

## Table of Contents
- [Project Overview](#project-overview)
- [Data Description](#data-description)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Methodology](#model-methodology)
- [Implementation Details](#implementation-details)
- [How to Run](#how-to-run)
- [Future Work](#future-work)
- [License](#license)

## Project Overview

The goal of this project is to create a movie recommender system that:
- Analyzes historical user ratings and movie metadata (especially genres).
- Builds user profiles based on the average rating given per genre.
- Uses cosine similarity to recommend movies that match the user's taste.
- Incorporates current airing movies into the recommendation process.

## Data Description

The project uses three primary datasets:
- **movies.csv**: Contains movie details with columns such as `movieId`, `title`, and `genres`. The `genres` field is a pipe-separated string (e.g., "Action|Adventure").
- **ratings.csv**: Contains user ratings for movies with columns `userId`, `movieId`, `rating`, and `timestamp`.
- **current_airing_movies.csv**: Contains details of currently airing movies with a similar structure to `movies.csv` (for generating real-time recommendations).

## Exploratory Data Analysis (EDA)

The EDA phase was done for  understanding the data and making informed decisions for the recommendation model:

1. **Data Quality and Preprocessing:**
   - Loaded the datasets and checked for missing values.
   - Converted the `genres` field from a pipe-separated string into a list of genres for each movie.
   - Applied one-hot encoding to transform genre information into binary features.

2. **Rating Distribution:**
   - A sample of 1,000 ratings was visualized using a histogram.
   - The distribution revealed that ratings tend to cluster around certain values, highlighting overall user satisfaction trends.

   

3. **Average Rating per Genre:**
   - By merging the movies and ratings datasets and “exploding” the genres column, the average rating per genre was computed.
   - A bar chart was created to illustrate which genres tend to receive higher or lower average ratings, suggesting that genre is a strong signal in user preferences.

   

4. **User Profiles:**
   - User profiles were generated by grouping ratings by `userId` and averaging the ratings for each genre.
   - These profiles provide an interpretable vector representation of each user’s preferences.

   

## Model Methodology

Based on the EDA, the following decisions were made for the recommendation model:
- **Content-Based Approach:**  
  Using movie genres as features and building user profiles from average genre ratings provides a transparent and interpretable recommendation method.
- **Cosine Similarity:**  
  The similarity between a user’s preference vector and a movie’s one-hot encoded genre vector is computed using cosine similarity. This similarity score is used to rank movies for recommendation.
- **Current Airing Movies Integration:**  
  The model also preprocesses a current list of movies (from `current_airing_movies.csv`) and applies the same recommendation logic to suggest new content.

## Implementation Details

The complete implementation is divided into the following steps:
1. **Data Loading and Preprocessing:**  
   - Load and clean the movies and ratings datasets.
   - Convert genre strings to lists and apply one-hot encoding.
2. **Merging and User Profile Creation:**  
   - Merge the datasets on `movieId` and compute average ratings per genre for each user.
3. **Cosine Similarity Recommender:**  
   - Define a cosine similarity function.
   - Create a function to recommend movies based on the similarity score.
4. **Current Movies Processing:**  
   - Load current airing movies, preprocess them similarly, and generate recommendations.

Refer to the source code for detailed implementation.

## How to Run

1. **Install Dependencies:**
   Ensure you have the required packages installed:
   ```bash
   pip install pandas numpy matplotlib scikit-learn
   ```
2. **Prepare the Data:**
   Place the datasets (`movies.csv`, `ratings.csv`, `current_airing_movies.csv`) in a folder named `data/`.
3. **Run the Script:**
   Execute the main Python script to generate user profiles, perform EDA, and produce recommendations:
   ```bash
   python movierecommender.py
   ```
4. **View the Visualizations:**
   The script will display graphs that summarize the rating distribution, average ratings per genre, and user profile snapshots.

## Future Work

- **Model Enhancements:**  
  Explore collaborative filtering or hybrid models to further improve recommendation accuracy.
- **Dynamic Updates:**  
  Integrate real-time data streaming to update user profiles and movie ratings.
- **User Interface:**  
  Develop a web interface or API for interactive movie recommendations.

## License

This project is open-source and available under the [MIT License](LICENSE).



