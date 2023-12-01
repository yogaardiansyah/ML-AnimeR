# app.py

import streamlit as st
import joblib
from anime_recommendation import load_original_data, make_prediction

# Load model dan scaler
model_ml = joblib.load('anime_recommendation_model.joblib')
scaler_ml = joblib.load('anime_scaler.joblib')

# Fungsi state aplikasi untuk menyimpan data yang bisa berubah selama sesi aplikasi
@st.cache(allow_output_mutation=True)
def get_state():
    return {
        'user_likes_info': None,
        'recommendations': None,
        'X_resampled': None,
    }

state = get_state()

# Judul aplikasi
st.title("Anime Recommendation App")

# Baca DataFrame dari sumber eksternal dengan cache
original_data = load_original_data()

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
    
# Tombol untuk membuat prediksi
if st.button("Cari dan Dapatkan Rekomendasi") and user_likes_input:
    # Pastikan X_resampled diakses sebelum pemanggilan fungsi make_prediction
    X_resampled = state['X_resampled']
    
    state['user_likes_info'], state['recommendations'] = make_prediction(original_data, user_likes_input, all_genres, scaler_ml, X_resampled)

if state['user_likes_info'] is not None:
    st.subheader(f"Informasi Anime yang Dicari: {user_likes_input}")
    st.write(state['user_likes_info'])

if state['recommendations'] is not None:
    st.subheader("Rekomendasi Anime untuk Pengguna:")
    st.write(state['recommendations'])
