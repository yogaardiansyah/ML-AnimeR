# anime_recommendation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_original_data():
    return pd.read_csv("https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv")

def map_user_data(user_data):
    # Lakukan pemetaan data pengguna
    user_data['status'] = user_data['status'].map(status_mapping)
    user_data['media_type'] = user_data['media_type'].map(media_type_mapping)
    user_data['source'] = user_data['source'].map(source_mapping)
    user_data['start_season_season'] = user_data['start_season_season'].map(season_mapping)
    user_data['rating'] = user_data['rating'].map(rating_mapping)
    
    return user_data

# Fungsi untuk membuat prediksi berdasarkan input pengguna
def make_prediction(original_data, title, all_genres, scaler_ml, model_ml):
    # Ambil data untuk judul anime yang dicari
    user_anime_info = original_data[original_data['title'] == title]

    if user_anime_info.empty:
        st.warning(f"Tidak ditemukan informasi untuk anime: {title}")
        return None, None

    # Gunakan fungsi pemetaan
    user_features = map_user_data(user_anime_info[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']])

    # Normalisasi fitur
    user_features_scaled = scaler_ml.transform(user_features)

    # Prediksi
    sim_scores = cosine_similarity(user_features_scaled, X_resampled)
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: x[1], reverse=True)

    # Ambil indeks film teratas
    top_indices = [i[0] for i in sim_scores[:5]]

    # Ambil informasi film rekomendasi dari data asli
    recommended_films = original_data.iloc[top_indices]

    return user_anime_info, recommended_films
