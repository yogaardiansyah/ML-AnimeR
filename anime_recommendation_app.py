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
    }

state = get_state()

# Judul aplikasi
st.title("Anime Recommendation App")

# Baca DataFrame dari sumber eksternal dengan cache
original_data = load_original_data()

# Input pengguna
user_likes_input = st.text_input("Masukkan Judul Anime yang Anda Cari:")

# Tombol untuk membuat prediksi
if st.button("Cari dan Dapatkan Rekomendasi") and user_likes_input:
    state['user_likes_info'], state['recommendations'] = make_prediction(original_data, user_likes_input, all_genres, scaler_ml, X_resampled)

if state['user_likes_info'] is not None:
    st.subheader(f"Informasi Anime yang Dicari: {user_likes_input}")
    st.write(state['user_likes_info'])

if state['recommendations'] is not None:
    st.subheader("Rekomendasi Anime untuk Pengguna:")
    st.write(state['recommendations'])
