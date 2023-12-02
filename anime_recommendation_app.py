# anime_recommendation_app.py
import streamlit as st
from recommendation_model import content_based_recommendation, load_model_and_scaler, load_data, preprocess_data

@st.cache(allow_output_mutation=True)
def load_cached_data():
    model_ml, scaler_ml = load_model_and_scaler()
    data = load_data()
    all_genres = get_all_genres()  # Assuming get_all_genres is defined in anime_data.py
    cosine_sim = preprocess_data(data, all_genres)
    return model_ml, scaler_ml, data, cosine_sim, all_genres

def main():
    st.title("Anime Recommendation App")

    model_ml, scaler_ml, data, cosine_sim, all_genres = load_cached_data()

    st.sidebar.header("User Input")
    user_likes = st.sidebar.selectbox("Select an anime you like:", data['title'].unique())

    user_likes_info, recommendations = content_based_recommendation(
        user_likes, cosine_sim, data, user_likes, all_genres, num_recommendations=5, genre_weight=2
    )

    st.subheader(f"Anime Information: {user_likes}")
    st.table(user_likes_info)

    st.subheader("Recommended Anime:")
    st.table(recommendations)

if __name__ == '__main__':
    main()
