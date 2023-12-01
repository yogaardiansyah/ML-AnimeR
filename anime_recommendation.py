# anime_recommendation.py

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def load_original_data():
    return pd.read_csv("https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv")

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

# Membuat DataFrame dengan kolom-kolom baru untuk setiap genre
for genre in all_genres:
    original_data[genre] = original_data['genres'].apply(lambda x: 1 if genre in x else 0)

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
