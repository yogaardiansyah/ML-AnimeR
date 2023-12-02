# app.py
import streamlit as st
from recommendation_model import content_based_recommendation, load_model_and_scaler, load_data, preprocess_data, get_user_genres
from anime_data import display_tabulated_dataframe, get_all_genres

@st.cache(allow_output_mutation=True)
def load_cached_data():
    model_ml, scaler_ml = load_model_and_scaler()
    data = load_data()
    all_genres = get_all_genres()  # Use the function to get the list of genres
    cosine_sim = preprocess_data(data, all_genres)
    return model_ml, scaler_ml, data, cosine_sim, all_genres

def main():
    st.title("Anime Recommendation App")

    # Load model, scaler, and data
    model_ml, scaler_ml, data, cosine_sim, all_genres = load_cached_data()

    # Sidebar
    st.sidebar.header("User Input")
    user_likes = st.sidebar.selectbox("Select an anime you like:", data['title'].unique())

    # Get user genres
    user_genres = get_user_genres(user_likes, data)

    # Get recommendations
    user_likes_info, recommendations = content_based_recommendation(
        user_likes, cosine_sim, data, user_likes, num_recommendations=5, genre_weight=2
    )

    # Display user input and recommendations
    st.subheader(f"Anime Information: {user_likes}")
    st.table(user_likes_info)

    st.subheader("Recommended Anime:")
    st.table(recommendations)

if __name__ == '__main__':
    main()
