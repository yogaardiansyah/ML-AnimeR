# recommendation_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity

def content_based_recommendation(title, cosine_sim, df, user_preferences, num_recommendations=5, genre_weight=2):
    # Get features for machine learning model
    features_ml = df[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
  
    # Normalize feature scales using StandardScaler
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)

    # Calculate cosine similarity between the user's preferred anime and all others
    user_features = features_ml[df['title'] == user_preferences]
    if user_features.empty:
        print(f"No information found for the anime: {user_preferences}")
        return pd.DataFrame(), pd.DataFrame()    user_features_scaled = scaler_ml.transform(user_features)

    sim_scores = cosine_similarity(user_features_scaled, features_scaled_ml)

    # Modify the scoring to give higher weight to genre similarity
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: (x[1] + genre_weight * sum(g in user_genres for g in df['genres'].iloc[x[0]].split(','))), reverse=True)
    sim_scores = sim_scores[1:(num_recommendations + 1)]
    film_indices = [i[0] for i in sim_scores]

    # Filter recommended films based on improved genre matching
    recommended_films = df.iloc[film_indices]

    return user_features, recommended_films
  
def load_model_and_scaler():
    model_ml = joblib.load('anime_recommendation_model.joblib')
    scaler_ml = joblib.load('anime_scaler.joblib')
    return model_ml, scaler_ml

def load_data():
    url = "https://raw.githubusercontent.com/yogaardiansyah/ML-AnimeR/main/anime.csv_exported.csv"
    data = pd.read_csv(url)
    return data

def preprocess_data(data, all_genres):
    features_ml = data[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)
    cosine_sim = cosine_similarity(features_scaled_ml, features_scaled_ml)
    return cosine_sim

def get_user_genres(user_likes, data):
    user_genres = data[data['title'] == user_likes]['genres'].values[0].split(',')
    return user_genres
