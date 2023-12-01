# anime_recommendation_app.py

import pandas as pd
import streamlit as st
import joblib
from anime_recommendation import load_original_data, make_prediction

# Load model and scaler
model_ml = joblib.load('anime_recommendation_model.joblib')
scaler_ml = joblib.load('anime_scaler.joblib')

# Function to get the app's state
@st.cache(allow_output_mutation=True)
def get_state():
    return {'user_likes_info': None, 'recommendations': None}

state = get_state()

# App title
st.title("Anime Recommendation App")

# Read DataFrame from an external source with caching
original_data = load_original_data()

# Create a list of all genres
all_genres = [
    'Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love',
    'Comedy', 'Drama', 'Fantasy', 'Girls Love', 'Gourmet', 'Horror',
    'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports',
    'Supernatural', 'Suspense', 'Ecchi', 'Erotica', 'Hentai',
    'Adult Cast', 'Anthropomorphic', 'CGDCT', 'Childcare',
    'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective',
    'Educational', 'Gag Humor', 'Gore', 'Harem', 'High Stakes Game',
    'Historical', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei',
    'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts',
    'Mecha', 'Medical', 'Military', 'Music', 'Mythology', 'Organized Crime',
    'Otaku Culture', 'Parody', 'Performing Arts', 'Pets', 'Psychological',
    'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext', 'Samurai',
    'School', 'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival',
    'Team Sports', 'Time Travel', 'Vampire', 'Video Game', 'Visual Arts',
    'Workplace', 'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen'
]

# Create DataFrame with new columns for each genre
genre_dummies = original_data['genres'].str.get_dummies(sep=', ')
original_data = pd.concat([original_data, genre_dummies], axis=1)

# User input
user_likes_input = st.text_input("Masukkan Judul Anime yang Anda Cari:")

# Button to make predictions
if st.button("Cari dan Dapatkan Rekomendasi") and user_likes_input:
    state['user_likes_info'], state['recommendations'] = make_prediction(original_data, user_likes_input, all_genres, scaler_ml, model_ml)

# Display user likes info
if state['user_likes_info'] is not None:
    st.subheader(f"Informasi Anime yang Dicari: {user_likes_input}")
    st.write(state['user_likes_info'])

# Display recommendations
if state['recommendations'] is not None:
    st.subheader("Rekomendasi Anime untuk Pengguna:")
    st.write(state['recommendations'])

    # Display the genre columns
    st.subheader("Genre Columns:")
    genre_columns = [genre for genre in all_genres if genre in original_data.columns]
    st.write(original_data[genre_columns])
    
# Button to get recommendations for a random title
if st.button("Dapatkan Rekomendasi Berdasarkan Judul Acak"):
    random_title = original_data['title'].sample().iloc[0]  # Select a random title from the dataset
    st.subheader(f"Rekomendasi untuk Judul Acak: {random_title}")

    # Call the make_prediction function for the random title
    state['user_likes_info'], state['recommendations'] = make_prediction(original_data, random_title, all_genres, scaler_ml, state['X_resampled'])

    if state['user_likes_info'] is not None:
        st.write(state['user_likes_info'])

    if state['recommendations'] is not None:
        st.write(state['recommendations'])

        # Display the genre columns
        st.subheader("Genre Columns:")
        genre_columns = [genre for genre in all_genres if genre in original_data.columns]
        st.write(original_data[genre_columns])
