import streamlit as st
import pandas as pd
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv(url)

@st.cache(allow_output_mutation=True)
def calculate_cosine_similarity(data):
    features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
    features_scaled_ml = scaler_ml.transform(features_ml)
    cosine_sim = cosine_similarity(features_scaled_ml, features_scaled_ml)
    return cosine_sim

# Load data
data = load_data()

# Check if cosine similarity matrix is cached
if 'cosine_sim' not in st.session_state:
    st.session_state.cosine_sim = calculate_cosine_similarity(data)

# Mapping categorical data to numeric values
data['start_season_season'].fillna('unknown', inplace=True)
data['rating'].fillna('r', inplace=True)
data['source'].fillna('original', inplace=True)

status_mapping = {'finished_airing': 1, 'currently_airing': 2, 'not_yet_aired': 3}
data['status'] = data['status'].map(status_mapping)

media_type_mapping = {'tv': 1, 'movie': 2, 'ova': 3, 'ona': 4, 'music': 5, 'special': 6, 'unknown': 7}
data['media_type'] = data['media_type'].map(media_type_mapping)

source_mapping = {'manga': 1, 'visual_novel': 2, 'original': 3, 'light_novel': 4,
                  'web_manga': 5, 'novel': 6, '4_koma_manga': 7, 'game': 8,
                  'other': 9, 'web_novel': 10, 'mixed_media': 11, 'music': 12,
                  'card_game': 13, 'book': 14, 'picture_book': 15, 'radio': 16}
data['source'] = data['source'].map(source_mapping)

season_mapping = {'spring': 1, 'fall': 2, 'winter': 3, 'summer': 4}
data['start_season_season'] = data['start_season_season'].map(season_mapping)

rating_mapping = {'r': 1, 'pg_13': 2, 'r+': 3, 'pg': 4, 'g': 5, 'rx': 6}
data['rating'] = data['rating'].map(rating_mapping)

# List of all genres
all_genres = ['Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 'Comedy', 'Drama', 'Fantasy', 
              'Girls Love', 'Gourmet', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports', 
              'Supernatural', 'Suspense', 'Ecchi', 'Erotica', 'Hentai', 'Adult Cast', 'Anthropomorphic', 'CGDCT', 
              'Childcare', 'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective', 'Educational', 'Gag Humor', 
              'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei', 
              'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military', 
              'Music', 'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts', 'Pets', 
              'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext', 'Samurai', 'School', 
              'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival', 'Team Sports', 'Time Travel', 'Vampire', 
              'Video Game', 'Visual Arts', 'Workplace', 'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen']

# Add columns for each genre
for genre in all_genres:
    data[genre] = data['genres'].apply(lambda x: 1 if genre in x else 0)

# Normalize feature scales using StandardScaler for content-based recommendation
scaler_ml = StandardScaler()
features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
features_scaled_ml = scaler_ml.fit_transform(features_ml)

# Function to get content-based recommendations
def content_based_recommendation(title, num_recommendations=5, genre_weight=2):
    # Ensure the input is a string
    title = str(title)

    # Check if the user input exists in the dataset
    if title not in data['title'].values:
        st.warning(f"No information found for the anime: {title}")
        return pd.DataFrame()

    # Calculate cosine similarity matrix
    cosine_sim = calculate_cosine_similarity(data)

    # Get features for machine learning model
    features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]

    # Normalize feature scales using StandardScaler
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)

    # Get features of the user's selected similar anime
    user_features = features_ml[data['title'].isin([title])]
    if user_features.empty:
        st.warning(f"No information found for the anime: {title}")
        return pd.DataFrame()

    user_features_scaled = scaler_ml.transform(user_features)

    # Get genres of the user's input anime
    user_genres = data[data['title'] == title]['genres'].iloc[0].split(',')

    # Calculate cosine similarity between the user's preferred anime and all others
    sim_scores = cosine_similarity(user_features_scaled, features_scaled_ml)

    # Modify the scoring to give higher weight to genre similarity
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: (x[1] + genre_weight * sum(g in user_genres for g in data['genres'].iloc[x[0]].split(','))), reverse=True)

    sim_scores = sim_scores[1:(num_recommendations + 1)]
    film_indices = [i[0] for i in sim_scores]

    # Filter recommended films based on improved genre matching
    recommended_films = data.iloc[film_indices]

    return recommended_films

# Function to get similar titles based on a simple string match
def search_similar_titles(user_input, num_similar_titles=5):
    # Check if the user input exists in the dataset
    if user_input not in data['title'].values:
        st.warning(f"No information found for the anime: {user_input}")
        return []

    # Get similar titles based on a simple string match
    similar_titles = data[data['title'].str.contains(user_input, case=False)]['title'].tolist()

    return similar_titles[:num_similar_titles]

# Streamlit app
st.title("Anime Recommendation App")

# User input for anime title
user_input = st.text_input("Enter the name of an anime:")

# Button to trigger recommendations
if st.button("Get Recommendations"):
    if user_input:
        # Display similar titles
        similar_titles = search_similar_titles(user_input)
        if similar_titles:
            selected_title = st.selectbox("Select a similar title:", similar_titles)

            if st.button("Get Recommendations for Similar Title"):
                # Display information for the selected title
                user_likes_info = data[data['title'] == selected_title]
                if not user_likes_info.empty:
                    st.subheader(f"Information for {selected_title}:")
                    st.table(user_likes_info)

                    # Get and display recommendations
                    recommendations = content_based_recommendation(selected_title)
                    if not recommendations.empty:
                        st.subheader(f"Recommended Anime for {selected_title}:")
                        st.table(recommendations[['title', 'genres', 'media_type', 'mean', 'rating', 'start_season_year']])
                    else:
                        st.warning(f"No recommendations found for {selected_title}")
                else:
                    st.warning(f"No information found for the anime: {selected_title}")
        else:
            st.warning(f"No similar titles found for the anime: {user_input}")
    else:
        st.warning("Please enter the name of an anime.")
