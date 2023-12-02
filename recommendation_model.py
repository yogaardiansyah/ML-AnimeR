# recommendation_model.py
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import cosine_similarity
# anime_recommendation_app.py
from recommendation_model import content_based_recommendation, load_model_and_scaler, load_data, preprocess_data, get_user_features


def content_based_recommendation(title, cosine_sim, df, user_preferences, all_genres, num_recommendations=5, genre_weight=2):
    features_ml = df[all_genres + ['media_type', 'mean', 'rating', 'start_season_year']]
  
    scaler_ml = StandardScaler()
    features_scaled_ml = scaler_ml.fit_transform(features_ml)

    user_features = get_user_features(df, user_preferences, features_ml, scaler_ml)

    if user_features.empty:
        print(f"No information found for the anime: {user_preferences}")
        return pd.DataFrame(), pd.DataFrame()  

    user_features_scaled = scaler_ml.transform(user_features)

    sim_scores = cosine_similarity(user_features_scaled, features_scaled_ml)

    sim_scores = sort_and_filter_recommendations(sim_scores, df, all_genres, num_recommendations, genre_weight)

    recommended_films = df.iloc[sim_scores]

    return user_features, recommended_films

def get_user_features(df, user_preferences, features_ml, scaler_ml):
    user_features = features_ml[df['title'] == user_preferences]
    return user_features if not user_features.empty else pd.DataFrame()

def sort_and_filter_recommendations(sim_scores, df, all_genres, num_recommendations, genre_weight):
    sim_scores = sorted(enumerate(sim_scores[0]), key=lambda x: (x[1] + genre_weight * sum(g in all_genres for g in df['genres'].iloc[x[0]].split(','))), reverse=True)
    film_indices = [i[0] for i in sim_scores[1:(num_recommendations + 1)]]
    return film_indices

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