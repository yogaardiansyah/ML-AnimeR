import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"

@st.cache
def load_data():
    return pd.read_csv(url)

data = load_data()

# Load the trained machine learning model
model_filename = 'animeR.sav'
with open(model_filename, 'rb') as model_file:
    model_ml = pickle.load(model_file)

# Daftar semua genre yang ada
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

# Menambahkan kolom baru untuk setiap genre
for genre in all_genres:
    data[genre] = data['genres'].apply(lambda x: 1 if genre in x else 0)

# Normalize feature scales using StandardScaler for content-based recommendation
scaler_ml = StandardScaler()
features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
features_scaled_ml = scaler_ml.fit_transform(features_ml)

# Define a cache for the cosine similarity matrix
@st.cache
def calculate_cosine_similarity(features_scaled_ml):
    return cosine_similarity(features_scaled_ml, features_scaled_ml)

# Function to get content-based recommendations
def content_based_recommendation(title, cosine_sim, df, num_recommendations=5, genre_weight=2):
    # Get features for machine learning model
    features_ml = df[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]

    # Normalize feature scales using StandardScaler
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)

    # Calculate cosine similarity matrix using the cached function
    cosine_sim = calculate_cosine_similarity(features_scaled_ml)

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
        recommendations = content_based_recommendation(user_input, None, data)
        if not recommendations.empty:
            st.subheader("Recommended Anime:")
            st.table(recommendations[['title', 'genres', 'media_type', 'mean', 'rating', 'start_season_year']])
        else:
            st.warning(f"No information found for the anime: {user_input}")
    else:
        st.warning("Please enter the name of an anime.")
