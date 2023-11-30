import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

# Load the anime dataset
url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"

@st.cache(allow_output_mutation=True)
def load_data():
    return pd.read_csv(url)

data = load_data()

# Streamlit app
st.title("Anime Recommendation App")

# User input for anime title
user_input = st.text_input("Enter the name of an anime:")

# Display similar titles
similar_titles = search_similar_titles(user_input)
if similar_titles:
    # Form to select a similar title
    with st.form("select_similar_title_form"):
        selected_title = st.selectbox("Select a similar title:", similar_titles, key="similar_titles")
        st.form_submit_button("Get Recommendations for Similar Title")

    # Display information for the selected title (Original Data)
    user_likes_info_original = data[data['title'] == selected_title]
    if not user_likes_info_original.empty:
        st.subheader(f"Information for {selected_title} (Original Data):")
        st.table(user_likes_info_original)

        # Remap user data for recommendation processing
        user_likes_info_remapped = remap_user_data(user_likes_info_original.copy())

        # Get and display recommendations using remapped data
        if st.button("Get Recommendations"):
            recommendations = content_based_recommendation_original(user_likes_info_remapped, selected_title)
            if not recommendations.empty:
                st.subheader(f"Recommended Anime for {selected_title}:")
                st.table(recommendations[['title', 'genres', 'media_type', 'mean', 'rating', 'start_season_year']])
            else:
                st.warning(f"No recommendations found for {selected_title}")
else:
    st.warning("Please enter the name of an anime.")

# Function to get content-based recommendations using remapped data
def content_based_recommendation_original(user_data, title, num_recommendations=5, genre_weight=2):
    # Calculate cosine similarity matrix using original data
    cosine_sim = calculate_cosine_similarity(data)

    # Get features for machine learning model using remapped data
    features_ml_remapped = user_data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]

    # Map categorical data to numeric values for user input (remapped)
    user_data_remapped = remap_user_data(user_data.copy())

    # Get features of the user's selected similar anime (remapped)
    user_features_remapped = user_data_remapped[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]

    # Concatenate user features with the original data for similarity calculation
    combined_features_remapped = pd.concat([features_ml_remapped, user_features_remapped])

    # Normalize feature scales using StandardScaler
    scaler_ml_remapped = StandardScaler()
    features_scaled_ml_remapped = scaler_ml_remapped.fit_transform(combined_features_remapped)

    # Get features of the user's selected similar anime (remapped)
    user_features_scaled_remapped = features_scaled_ml_remapped[-1:]  # Last row corresponds to user input

    # Get genres of the user's input anime (remapped)
    user_genres_remapped = data[data['title'] == title]['genres'].iloc[0].split(',')

    # Calculate cosine similarity between the user's preferred anime and all others
    sim_scores_remapped = cosine_similarity(user_features_scaled_remapped, features_scaled_ml_remapped[:-1])

    # Modify the scoring to give higher weight to genre similarity
    sim_scores_remapped = sorted(enumerate(sim_scores_remapped[0]), key=lambda x: (x[1] + genre_weight * sum(g in user_genres_remapped for g in data['genres'].iloc[x[0]].split(','))), reverse=True)

    sim_scores_remapped = sim_scores_remapped[:num_recommendations]
    film_indices_remapped = [i[0] for i in sim_scores_remapped]

    # Filter recommended films based on improved genre matching
    recommended_films_remapped = data.iloc[film_indices_remapped]

    return recommended_films_remapped

# Function to remap user data
def remap_user_data(user_data):
    # Mapping categorical data to numeric values
    user_data['start_season_season'].fillna('unknown', inplace=True)
    user_data['rating'].fillna('r', inplace=True)
    user_data['source'].fillna('original', inplace=True)

    status_mapping = {'finished_airing': 1, 'currently_airing': 2, 'not_yet_aired': 3}
    user_data['status'] = user_data['status'].map(status_mapping)

    media_type_mapping = {'tv': 1, 'movie': 2, 'ova': 3, 'ona': 4, 'music': 5, 'special': 6, 'unknown': 7}
    user_data['media_type'] = user_data['media_type'].map(media_type_mapping)

    source_mapping = {'manga': 1, 'visual_novel': 2, 'original': 3, 'light_novel': 4,
                      'web_manga': 5, 'novel': 6, '4_koma_manga': 7, 'game': 8,
                      'other': 9, 'web_novel': 10, 'mixed_media': 11, 'music': 12,
                      'card_game': 13, 'book': 14, 'picture_book': 15, 'radio': 16}
    user_data['source'] = user_data['source'].map(source_mapping)

    season_mapping = {'spring': 1, 'fall': 2, 'winter': 3, 'summer': 4}
    user_data['start_season_season'] = user_data['start_season_season'].map(season_mapping)

    rating_mapping = {'r': 1, 'pg_13': 2, 'r+': 3, 'pg': 4, 'g': 5, 'rx': 6}
    user_data['rating'] = user_data['rating'].map(rating_mapping)

    # List of all genres
    all_genres = ['Action', 'Adventure', 'Avant Garde', 'Award Winning', 'Boys Love', 'Comedy', 'Drama', 'Fantasy',
                  'Girls Love', 'Gourmet', 'Horror', 'Mystery', 'Romance', 'Sci-Fi', 'Slice of Life', 'Sports',
                  'Supernatural', 'Suspense', 'Ecchi', 'Erotica', 'Hentai', 'Adult Cast', 'Anthropomorphic', 'CGDCT',
                  'Childcare', 'Combat Sports', 'Crossdressing', 'Delinquents', 'Detective', 'Educational', 'Gag Humor',
                  'Gore', 'Harem', 'High Stakes Game', 'Historical', 'Idols (Female)', 'Idols (Male)', 'Isekai', 'Iyashikei',
                  'Love Polygon', 'Magical Sex Shift', 'Mahou Shoujo', 'Martial Arts', 'Mecha', 'Medical', 'Military',
                  'Music', 'Mythology', 'Organized Crime', 'Otaku Culture', 'Parody', 'Performing Arts', 'Pets',
                  'Psychological', 'Racing', 'Reincarnation', 'Reverse Harem', 'Romantic Subtext', 'Samurai', 'School',
                  'Showbiz', 'Space', 'Strategy Game', 'Super Power', 'Survival', 'Team Sports', 'Time Travel', 'Vampire',
                  'Video Game', 'Visual Arts', 'Workplace', 'Josei', 'Kids', 'Seinen', 'Shoujo', 'Shounen']

    # Add columns for each genre
    for genre in all_genres:
        user_data[genre] = user_data['genres'].apply(lambda x: 1 if genre in x else 0)

    return user_data

# Function to get similar titles based on a simple string match
def search_similar_titles(user_input, num_similar_titles=5):
    # Check if the user input exists in the dataset
    if user_input not in data['title'].values:
        st.warning(f"No information found for the anime: {user_input}")
        return []

    # Get similar titles based on a simple string match
    similar_titles = data[data['title'].str.contains(user_input, case=False)]['title'].tolist()

    return similar_titles[:num_similar_titles]
