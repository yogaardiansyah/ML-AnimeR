import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"
data = pd.read_csv(url)

# Load the trained machine learning model
model_filename = 'animeR.sav'
with open(model_filename, 'rb') as model_file:
    model_ml = pickle.load(model_file)

# Normalize feature scales using StandardScaler for content-based recommendation
scaler_ml = StandardScaler()
features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
features_scaled_ml = scaler_ml.fit_transform(features_ml)

# Calculate cosine similarity matrix for content-based recommendation
cosine_sim = cosine_similarity(features_scaled_ml, features_scaled_ml)

# Function to get content-based recommendations
def content_based_recommendation(title, cosine_sim, df, num_recommendations=5, genre_weight=2):
    # Get features for machine learning model
    features_ml = df[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]

    # Normalize feature scales using StandardScaler
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)

    # Calculate cosine similarity between the user's preferred anime and all others
    user_features = features_ml[df['title'] == title]
    if user_features.empty:
        return pd.DataFrame()

    user_features_scaled = scaler_ml.transform(user_features)

    sim_scores = cosine_similarity(user_features_scaled, features_scaled_ml)

    # Modify the scoring to give higher weight to genre similarity
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: (x[1] + genre_weight * sum(g in user_genres for g in df['genres'].iloc[x[0]].split(','))), reverse=True)

    sim_scores = sim_scores[1:(num_recommendations + 1)]
    film_indices = [i[0] for i in sim_scores]

    # Filter recommended films based on improved genre matching
    recommended_films = df.iloc[film_indices]

    return recommended_films

# Streamlit app
st.title("Anime Recommendation App")

# User input for anime title
user_input = st.text_input("Enter the name of an anime:")

# Button to trigger recommendations
if st.button("Get Recommendations"):
    if user_input:
        recommendations = content_based_recommendation(user_input, cosine_sim, data)
        if not recommendations.empty:
            st.subheader("Recommended Anime:")
            st.table(recommendations[['title', 'genres', 'media_type', 'mean', 'rating', 'start_season_year']])
        else:
            st.warning(f"No information found for the anime: {user_input}")
    else:
        st.warning("Please enter the name of an anime.")