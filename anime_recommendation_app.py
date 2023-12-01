# app.py (versi diperbaiki dengan state aplikasi, st.cache, dan DataFrame cache)

import streamlit as st
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load model dan scaler
model_ml = joblib.load('anime_recommendation_model.joblib')
scaler_ml = joblib.load('anime_scaler.joblib')

# Fungsi state aplikasi untuk menyimpan data yang bisa berubah selama sesi aplikasi
@st.cache(allow_output_mutation=True)
def get_state():
    return {
        'user_likes_info': None,
        'recommendations': None,
    }

state = get_state()

# Fungsi cache untuk membaca DataFrame dari sumber eksternal
@st.cache
def load_original_data():
    return pd.read_csv("https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv")

# Fungsi untuk membuat prediksi berdasarkan input pengguna
def make_prediction(original_data, title,all_genres):
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

    # Ambil indeks film teratas
    top_indices = [i[0] for i in sim_scores[:5]]

    # Ambil informasi film rekomendasi dari data asli
    recommended_films = original_data.iloc[top_indices]

    return user_anime_info, recommended_films

# Judul aplikasi
st.title("Anime Recommendation App")

# Baca DataFrame dari sumber eksternal dengan cache
original_data = load_original_data()

# Input pengguna
user_likes_input = st.text_input("Masukkan Judul Anime yang Anda Cari:")

# Tombol untuk membuat prediksi
if st.button("Cari dan Dapatkan Rekomendasi") and user_likes_input:
    state['user_likes_info'], state['recommendations'] = make_prediction(original_data, user_likes_input)

if state['user_likes_info'] is not None:
    st.subheader(f"Informasi Anime yang Dicari: {user_likes_input}")
    st.write(state['user_likes_info'])

if state['recommendations'] is not None:
    st.subheader("Rekomendasi Anime untuk Pengguna:")
    st.write(state['recommendations'])
