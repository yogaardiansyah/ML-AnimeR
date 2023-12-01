# anime_recommendation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_original_data():
    return pd.read_csv("https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv")

# Fungsi untuk membuat prediksi berdasarkan input pengguna
def make_prediction(original_data, title, all_genres, scaler_ml, X_resampled):
    # Ambil data untuk judul anime yang dicari
    user_anime_info = original_data[original_data['title'] == title]

    if user_anime_info.empty:
        st.warning(f"Tidak ditemukan informasi untuk anime: {title}")
        return None, None

    # Normalisasi fitur
    user_features = user_anime_info[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
    user_features_scaled = scaler_ml.transform(user_features)

    # Prediksi
    sim_scores = cosine_similarity(user_features_scaled, X_resampled)
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: x[1], reverse=True)
    top_indices = [i[0] for i in sim_scores[:5]]
    
    # Ambil informasi film rekomendasi dari data asli
    recommended_films = original_data.iloc[top_indices]

    return user_anime_info, recommended_films
