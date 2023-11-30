import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv(url)

data = load_data()

# Load the trained machine learning model
model_filename = 'animeR.sav'
with open(model_filename, 'rb') as model_file:
    model_ml = pickle.load(model_file)

# Mengganti nilai kosong 'start_season_season' dengan label 'unknown'
data['start_season_season'].fillna('unknown', inplace=True)

# Mengisi nilai kosong pada 'rating' dengan 'r'
data['rating'].fillna('r', inplace=True)

# Mengisi nilai kosong pada 'source' dengan 'original'
data['source'].fillna('original', inplace=True)

# Mapping data 'status' menjadi numeric
# Membuat peta untuk mengganti nilai string dengan integer
status_mapping = {'finished_airing': 1, 'currently_airing': 2, 'not_yet_aired': 3}

# Mengganti nilai string dengan integer
data['status'] = data['status'].map(status_mapping)

# Membuat peta untuk mengganti nilai string dengan integer
media_type_mapping = {'tv': 1, 'movie': 2, 'ova': 3, 'ona': 4, 'music': 5, 'special': 6, 'unknown': 7}

# Mengganti nilai string dengan integer
data['media_type'] = data['media_type'].map(media_type_mapping)

# Membuat peta untuk mengganti nilai string dengan integer
source_mapping = {
    'manga': 1, 'visual_novel': 2, 'original': 3, 'light_novel': 4,
    'web_manga': 5, 'novel': 6, '4_koma_manga': 7, 'game': 8,
    'other': 9, 'web_novel': 10, 'mixed_media': 11, 'music': 12,
    'card_game': 13, 'book': 14, 'picture_book': 15, 'radio': 16
}

# Mengganti nilai string dengan integer
data['source'] = data['source'].map(source_mapping)

# Membuat peta untuk mengganti nilai string dengan integer
season_mapping = {'spring': 1, 'fall': 2, 'winter': 3, 'summer': 4}

# Mengganti nilai string dengan integer
data['start_season_season'] = data['start_season_season'].map(season_mapping)

# Membuat peta untuk mengganti nilai string dengan integer
rating_mapping = {'r': 1, 'pg_13': 2, 'r+': 3, 'pg': 4, 'g': 5, 'rx': 6}

# Mengganti nilai string dengan integer
data['rating'] = data['rating'].map(rating_mapping)

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

# Calculate cosine similarity matrix for content-based recommendation
cosine_sim = calculate_cosine_similarity(features_scaled_ml)

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
