# anime_recommendation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Fungsi untuk memuat data asli
def load_original_data():
    return pd.read_csv("https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv")

original_data = load_original_data()
print(original_data.columns)

# Membuat peta untuk mengganti nilai string dengan integer
status_mapping = {'finished_airing': 1, 'currently_airing': 2, 'not_yet_aired': 3}

# Membuat peta untuk mengganti nilai string dengan integer
media_type_mapping = {'tv': 1, 'movie': 2, 'ova': 3, 'ona': 4, 'music': 5, 'special': 6, 'unknown': 7}

# Membuat peta untuk mengganti nilai string dengan integer
source_mapping = {
    'manga': 1, 'visual_novel': 2, 'original': 3, 'light_novel': 4,
    'web_manga': 5, 'novel': 6, '4_koma_manga': 7, 'game': 8,
    'other': 9, 'web_novel': 10, 'mixed_media': 11, 'music': 12,
    'card_game': 13, 'book': 14, 'picture_book': 15, 'radio': 16
}

# Membuat peta untuk mengganti nilai string dengan integer
season_mapping = {'spring': 1, 'fall': 2, 'winter': 3, 'summer': 4}

# Membuat peta untuk mengganti nilai string dengan integer
rating_mapping = {'r': 1, 'pg_13': 2, 'r+': 3, 'pg': 4, 'g': 5, 'rx': 6}

# Fungsi untuk melakukan pemetaan data pengguna
def map_user_data(user_data):
    user_data.loc[:, 'status'] = user_data['status'].map(status_mapping)
    user_data.loc[:, 'media_type'] = user_data['media_type'].map(media_type_mapping)
    user_data.loc[:, 'source'] = user_data['source'].map(source_mapping)
    user_data.loc[:, 'rating'] = user_data['rating'].map(rating_mapping)
    
    # Example mapping in map_user_data function
    user_data.loc[:, 'start_season_season'] = user_data['start_season_season'].map(season_mapping)
    
    # Ensure that only existing columns are selected
    selected_columns = all_genres + ['media_type', 'mean', 'rating', 'start_season_year', 'status']
    user_data = user_data[selected_columns]

    # Menampilkan informasi kolom user data ke Streamlit
    st.write("User Data:")
    st.write(user_data)

    return user_data

# Fungsi untuk membuat prediksi berdasarkan input pengguna
def make_prediction(original_data, title, all_genres, scaler_ml, model_ml):
    # Ambil data untuk judul anime yang dicari
    user_anime_info = original_data[original_data['title'] == title]

    if user_anime_info.empty:
        st.warning(f"Tidak ditemukan informasi untuk anime: {title}")
        return None, None

    # Gunakan fungsi pemetaan
    user_features = map_user_data(user_anime_info.loc[:, all_genres + ['media_type', 'mean', 'rating', 'start_season_year', 'status']])

    # Normalisasi fitur
    user_features_scaled = scaler_ml.transform(user_features)

    # Prediksi
    sim_scores = cosine_similarity(user_features_scaled, model_ml)
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: x[1], reverse=True)

    # Ambil indeks film teratas
    top_indices = [i[0] for i in sim_scores[:5]]

    # Ambil informasi film rekomendasi dari data asli
    recommended_films = original_data.iloc[top_indices]

    return user_anime_info, recommended_films
