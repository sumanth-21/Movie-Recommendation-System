# 1-a) Plot a histogram distribution of the movie ratings.
import matplotlib.pyplot as plt
import pandas as pd

# load the ratings data from an Excel file
ratings_df = pd.read_csv('ratings.csv')

movie_ratings = ratings_df['rating']

# Set the number of bins
num_bins = 10

# Plot the histogram
plt.hist(movie_ratings, bins=num_bins, rwidth=0.8)

# Set the axis label and title
plt.xlabel('Movie Ratings')
plt.ylabel('Frequency')
plt.title('Histogram of Movie Ratings')

# Show the plot
plt.show()

# --------------------------------------------------------------#

# 1-b) Plot a histogram graph of movies published per year (year is given in the movie title)

# Load the ratings data from Excel file
df = pd.read_csv("movies.csv")

# Extract the year from the movie title and create a new column
df["year"] = df["title"].str.extract("\((\d{4})\)", expand=True)

# Count the number of movies per year
year_count = df["year"].value_counts().sort_index()

# Set the bins to be the year values, with an extra bin for the last year
bins = year_count.index.astype(int).tolist() + [year_count.index.astype(int)[-1] + 1]

# Plot the histogram
plt.hist(df["year"].fillna(0).astype(int), bins=bins, rwidth=0.8)

start_year = int(year_count.index[0]) - int(year_count.index[0]) % 5
plt.xticks(range(start_year, int(year_count.index[-1])+4, 5), rotation=90)

# Set the axis label and title
plt.xlabel('Year')
plt.ylabel('Number of movies')
plt.title('Movies published per year')

# Show the plot
plt.show()


# --------------------------------------------------------- #

# 1-c) Plot a histogram of count of genres showing the most popular movie genres.

df = pd.read_csv("movies.csv")

# Split genres into separate columns
df['genres'] = df['genres'].str.split('|')

# Create a list of all genres
all_genres = [genre for genres in df['genres'] for genre in genres]

# Count the occurrence of each genre and create a dataframe
genre_counts = pd.DataFrame(all_genres, columns=['genre']).value_counts().reset_index(name='count')

# Plot histogram of genre counts
plt.figure(figsize=(10,6))
plt.bar(genre_counts['genre'], genre_counts['count'])
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Count')
plt.title('Histogram of Movie Genre Counts')
plt.show()

# --------------------------------------------------------- #

# 1-d) Plot a histogram of the average ratings for each genre.

# Load the data from csv files
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Join the dataframes on movie_id column
joined_df = pd.merge(ratings_df, movies_df, on='movieId')

# Split the genres string and create a new row for each genre
genre_df = joined_df['genres'].str.split('|', expand=True).stack().reset_index(level=1, drop=True).rename('genre')

# Merge the new genre_df with the original joined_df
joined_genre_df = joined_df.drop('genres', axis=1).join(genre_df)

# Calculate the average rating for each genre
genre_rating_df = joined_genre_df.groupby('genre')['rating'].mean().sort_values(ascending=False)

# Plot the histogram
plt.figure(figsize=(10,6))
plt.bar(genre_rating_df.index, genre_rating_df.values)
plt.xticks(rotation=90)
plt.xlabel('Genre')
plt.ylabel('Average Rating')
plt.title('Histogram of Average Ratings for Each Genre')
plt.show()


# --------------------------------------------------------- #

# 1-e) Using pearson's R correlation on the ratings, recommend top 10 movies similar to Pulp Fiction (1994).

# Load the data from csv files
ratings_df = pd.read_csv('ratings.csv')
movies_df = pd.read_csv('movies.csv')

# Merge the datasets on the 'movieId' column
movies_ratings = pd.merge(movies_df, ratings_df, on='movieId', how='inner')

# Pivot the table to create a user-item matrix
ratings_pivot = pd.pivot_table(movies_ratings, values='rating', index='userId', columns='title', fill_value=0)

# Select the movie to find similar movies for
movie_name = 'Pulp Fiction (1994)'
pulp_fiction_ratings = ratings_pivot[movie_name]

# Compute the correlation between Pulp Fiction and all other movies
similarities = ratings_pivot.corrwith(pulp_fiction_ratings)

# Drop movies with missing values and sort the correlations in descending order
similarities.dropna(inplace=True)
similarities.sort_values(ascending=False, inplace=True)

# Recommend the top 10 movies with the highest correlation values
top_10_movies = similarities.drop(movie_name).head(10)

print(f"Top 10 movies similar to {movie_name}:\n")
print(top_10_movies)



# --------------------------------------------------------- #

# 1-f) Using K-Nearest Neighbor algorithm and cosine distance similarity, recommend the top-10 movies similar to 'jumanji'.

import pandas as pd
from sklearn.neighbors import NearestNeighbors
from scipy.sparse import csr_matrix

# Load the data from csv files
movies_df = pd.read_csv('movies.csv')
ratings_df = pd.read_csv('ratings.csv')

# Filter movies that are not 'Jumanji'
jumanji_movie_id = movies_df[movies_df['title'] == 'Jumanji (1995)']['movieId'].values[0]
jumanji_ratings = ratings_df[ratings_df['movieId'] == jumanji_movie_id]

# Create a pivot table of movie ratings
movie_ratings_pivot = ratings_df.pivot_table(index='userId', columns='movieId', values='rating')

# Convert the pivot table to a sparse matrix
movie_ratings_matrix = csr_matrix(movie_ratings_pivot.fillna(0).values)

# Create a NearestNeighbors model with cosine similarity
knn_model = NearestNeighbors(metric='cosine', algorithm='brute')
knn_model.fit(movie_ratings_matrix)

# Find the index of 'Jumanji' in the pivot table
jumanji_index = movie_ratings_pivot.columns.get_loc(jumanji_movie_id)

# Get the nearest neighbors of 'Jumanji' based on cosine similarity
distances, indices = knn_model.kneighbors(movie_ratings_matrix[jumanji_index], n_neighbors=11)

# Get the movieIds of the nearest neighbors
similar_movie_ids = movie_ratings_pivot.columns[indices.flatten()].values[1:]
similarity_scores = distances.flatten()[1:]

# Sort the similar movies based on their similarity score
similar_movies_df = pd.DataFrame({'movieId': similar_movie_ids, 'similarity': similarity_scores})
similar_movies_df = similar_movies_df.merge(movies_df[['movieId', 'title']], on='movieId')
similar_movies_df = similar_movies_df.sort_values('similarity')

# Print the top-10 similar movies to 'Jumanji'
print("\nTop-10 similar movies to 'Jumanji':")
for i, (_,row) in enumerate(similar_movies_df[:10].iterrows(), 1):
    print(f"{i}. {row['title']:50} (similarity score: {1 - row['similarity']:.4f})")



# --------------------------------------------------------- #

# 2-a) Using the TF-IDF on the 'overview' text, give recommendation of top-10 movies similar to 'Avatar'.

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the data from CSV files
movies_df = pd.read_csv('G:/DataMining/Final Project/IMDB_dataset_2/tmdb_5000_movies.csv')
credits_df = pd.read_csv('G:/DataMining/Final Project/IMDB_dataset_2/tmdb_5000_credits.csv')

# Extract the 'id' and 'overview' columns from movies_df
movies_overview = movies_df[['id', 'overview']]

# Merge the 'title' column from credits_df with movies_overview
movies_overview = pd.concat([movies_overview, credits_df['title']], axis=1)

# Drop any NaN values in the 'overview' column and reset the index
movies_overview.dropna(subset=['overview'], inplace=True)
movies_overview.reset_index(drop=True, inplace=True)

# Filter movies_overview for 'Avatar'
avatar_overview = movies_overview[movies_overview['title'] == 'Avatar']

# print(movies_overview[movies_overview['overview'].isnull() | (movies_overview['overview'] == '')])

# Initialize the TfidfVectorizer
tfidf_vectorizer = TfidfVectorizer(stop_words='english')

# Apply the TfidfVectorizer on the 'overview' column
tfidf_matrix = tfidf_vectorizer.fit_transform(movies_overview['overview'])

# Calculate the TF-IDF matrix for 'Avatar' movie
avatar_tfidf_matrix = tfidf_vectorizer.transform(avatar_overview['overview'])

# Calculate the cosine similarity between 'Avatar' movie and all other movies
cosine_similarities = cosine_similarity(avatar_tfidf_matrix, tfidf_matrix, dense_output=True)

# Get the indices of the top-10 movies with the highest cosine similarity values
similar_movie_indices = cosine_similarities.argsort()[0][-11:-1][::-1]

# Get the titles of the top-10 similar movies
similar_movie_titles = movies_overview.iloc[similar_movie_indices]['title']

# Print the top-10 similar movie titles
print("\n Top-10 Movies Similar to 'Avatar':")
for i, title in enumerate(similar_movie_titles, 1):
    print(f"{i}. {title}")
