from flask import Flask, redirect, redirect, request, session, jsonify, render_template
from datetime import datetime
import requests
from dotenv import load_dotenv
import os
import urllib.parse
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestRegressor


load_dotenv()

app = Flask(__name__)
app.secret_key = "12d34fr-213f-123d-asd9-1234f56g7h8j"

CLIENT_ID = os.getenv("CLIENT_ID")
CLIENT_SECRET = os.getenv("CLIENT_SECRET")
REDIRECT_URI = 'http://localhost:5000/callback'

AUTH_URL = 'https://accounts.spotify.com/authorize'
TOKEN_URL = 'https://accounts.spotify.com/api/token'
API_BASE_URL = 'https://api.spotify.com/v1'


@app.route('/')
def index():
    return "Welcome to the Datify <a href='/login'>Login with Spotify</a>"

@app.route('/login')
def login():
    scope = 'user-read-private user-read-email user-top-read'
    
    params = {
        'client_id': CLIENT_ID,
        'response_type': 'code',
        'scope': scope,
        'redirect_uri': REDIRECT_URI,
        'show_dialog': False
    }

    auth_url = f"{AUTH_URL}?{urllib.parse.urlencode(params)}"
    return redirect(auth_url)

@app.route('/callback')
def callback():
    if 'error' in request.args:
        return jsonify({"error": request.args['error']})
    if 'code' in request.args:
        req_body = {
            'code': request.args['code'],
            'grant_type': 'authorization_code',
            'redirect_uri': REDIRECT_URI,
            'client_id': CLIENT_ID,
            'client_secret': CLIENT_SECRET
        }
        response = requests.post(TOKEN_URL, data=req_body)
        token_info = response.json()
        session['access_token'] = token_info['access_token']
        session['refresh_token'] = token_info['refresh_token']
        session['expires_at'] = datetime.now().timestamp() + token_info['expires_in']

        return redirect('/datify') #changed from /playlists

@app.route('/playlists') #not used
def get_playlists():
    if 'access_token' not in session:
        return redirect('/login')

    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')

    headers = {
        'Authorization' : f"Bearer {session['access_token']}"
    }
    response = requests.get(API_BASE_URL + '/me/playlists', headers= headers)
    playlists = response.json()
    return jsonify(playlists)

@app.route('/refresh-token')
def refresh_token():
    if 'refresh_token' not in session:
        return redirect('/login')
    
    if datetime.now().timestamp() > session['expires_at']:
        req_body = {
            'grant_type':'refresh_token',
            'refresh_token': session['refresh_token'],
            'client-id':CLIENT_ID,
            'client-secret':CLIENT_SECRET
        }

        response = requests.post(TOKEN_URL, data= req_body)
        new_token_info = response.json
        session['access_token'] = new_token_info['access_token']
        session['expires_at'] = datetime.now().timestamp() + new_token_info['expires_in']

        return redirect('/datify')
    
@app.route('/datify')
def get_top_artist():
    if 'access_token' not in session:
        return redirect('/login')

    if datetime.now().timestamp() > session['expires_at']:
        return redirect('/refresh-token')

    headers = {
        'Authorization' : f"Bearer {session['access_token']}"
    }
    try:
        params = {'limit': 50, 'time_range': 'long_term'} #max limit is 50
        #long_term - several years, medium_term - 6mo, short_term- 4 weeks
        response_top_artists = requests.get(API_BASE_URL + '/me/top/artists', headers=headers, params=params)
        response_top_artists.raise_for_status() 

        response_top_tracks = requests.get(API_BASE_URL + '/me/top/tracks', headers=headers, params=params)
        response_top_tracks.raise_for_status() 

        top_artists_json = response_top_artists.json()
        top_tracks_json = response_top_tracks.json()

        top_artists_df = pd.DataFrame(top_artists_json['items'])[['name', 'id']]
        top_tracks_df = pd.DataFrame(top_tracks_json['items'])[['name', 'id']]


        features = ['acousticness', 'danceability', 'energy', 'instrumentalness',
                    'key', 'liveness', 'loudness','speechiness', 'tempo', 'valence'] #removed duration, mode, time_signature, popularity, explicit, id, uri, track_href, analysis_url, type
        
        
        top_tracks_features = []
        for track_id in top_tracks_df['id']:
            response = requests.get(API_BASE_URL + f'/audio-features/{track_id}', headers=headers)
            response.raise_for_status()  # Raise an exception for HTTP errors
            top_tracks_features.append(response.json())
            #gets fearures for each track
        
        top_tracks_features_df = pd.DataFrame(top_tracks_features)[features]
        top_tracks_df = pd.concat([top_tracks_df, top_tracks_features_df], axis=1)
        top_tracks_df['rank'] = range(len(top_tracks_df), 0, -1)
 
        #print("Top Tracks Features")
        #print(top_tracks_df)

        top_artist_genres_data = []

        for artist_id in top_artists_df['id']:
            response = requests.get(API_BASE_URL + f'/artists/{artist_id}', headers=headers)
            #gets artist info for each top artist

            try:
                response.raise_for_status()
                artist_data = response.json()
                artist_genres = artist_data.get('genres', [])

                for genre in artist_genres:
                    for genre_entry in top_artist_genres_data:
                        if genre == genre_entry[0]:
                            genre_entry[1] += 1
                            genre_entry[2].append(artist_data['name'])
                            break
                    else:
                        top_artist_genres_data.append([genre, 1, [artist_data['name']]])

            except requests.exceptions.RequestException as e:
                print(f"Error fetching artist details: {e}")
                print(f"Response content: {response.content}")

        bar_chart_df = pd.DataFrame(top_artist_genres_data, columns=['Genre', 'Count', 'Artists'])

        # Convert the "Artists" list to a string for hover data
        bar_chart_df['Artists'] = bar_chart_df['Artists'].apply(lambda x: ', '.join(x))

        # Create a bar chart using Plotly Express with a custom hover template
        bar_chart_fig = px.bar(
            bar_chart_df,
            x='Genre',
            y='Count',
            color='Genre',
            hover_name='Genre',
            hover_data={'Artists': True},
            title='Bar Chart of Song Count per Genre',
            labels={'Genre': 'Genre', 'Count': 'Count'},
            template='plotly',
        )

        bar_chart_plot_html = bar_chart_fig.to_html(full_html=False)

        target = 'rank'  # Replace 'rank' with the actual column representing the rank

        # Split data into train and test sets
        X_train, X_test, y_train, y_test = train_test_split(top_tracks_df[features], top_tracks_df[target], test_size=0.2, random_state=42)

        # Train a random forest regressor
        rf_regressor = RandomForestRegressor(random_state=42)
        rf_regressor.fit(X_train, y_train)

        # Get feature importances
        feature_importances = pd.Series(rf_regressor.feature_importances_, index=features).sort_values(ascending=False)

        # Select the top 3 features
        top_3_features = feature_importances.head(3).index.tolist()

        kmeans_data = top_tracks_df[features].dropna()

        # Find optimal k using silhouette score
        silhouette_scores = []
        for i in range(2, 11): 
            kmeans = KMeans(n_clusters=i, random_state=42)
            cluster_labels = kmeans.fit_predict(kmeans_data)
            silhouette_avg = silhouette_score(kmeans_data, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Choose the optimal k based on the highest silhouette score
        optimal_k = silhouette_scores.index(max(silhouette_scores)) + 2  # Add 2 to get the actual k value

        # Perform KMeans clustering with the optimal k
        kmeans = KMeans(n_clusters=optimal_k, random_state=42)
        top_tracks_df['cluster'] = kmeans.fit_predict(kmeans_data)

        # Create a 3D scatter plot using the top 3 features
        fig = px.scatter_3d(
            top_tracks_df,
            x=top_3_features[0],
            y=top_3_features[1],
            z=top_3_features[2],
            color='cluster',
            hover_data=['name', 'cluster'],
            labels={'cluster': 'Cluster'},
            title='KMeans Clustering of Top Tracks (3D)',
        )
        #plot may not display clusters well since we are representing 10 features in 3-feature space

        plot_html = fig.to_html(full_html=False)

        num_features_for_recommendation = 10  # Change this number as needed

 
        X_train, y_train = top_tracks_df[features], top_tracks_df['rank']

        # Train a new random forest regressor
        rf_regressor_new = RandomForestRegressor(random_state=42)
        rf_regressor_new.fit(X_train, y_train)

        # Get feature importances for the new model
        feature_importances_new = pd.Series(rf_regressor_new.feature_importances_, index=features).sort_values(ascending=False)

        # Select the top features for the new model
        top_features_new = feature_importances_new.head(num_features_for_recommendation).index.tolist()

        # Update the user features with the new top features as a dictionary
        user_features_new = {f"target_{feature}": top_tracks_df[feature].mean() for feature in top_features_new}
        genres = []
        for genre_entry in top_artist_genres_data:
            genres.append(genre_entry[0])

        params1 = { 'seed_genres': genres,
                    'seed_tracks': None,
                    'seed_artists': None,
                    }
        
        params2 = {'limit': 50}
        user_features_new['target_key'] = int(user_features_new['target_key'])
        combined_params = {**params1, **params2, **user_features_new}


        response_top_rec = requests.get(API_BASE_URL + '/recommendations', headers=headers, params=combined_params)
        response_top_rec.raise_for_status()  # Raise an exception for HTTP errors
        top_rec_json = response_top_rec.json()
        #print(top_rec_json)
        top_rec_df = pd.DataFrame(top_rec_json['tracks'])[['name', 'external_urls','id']]# ,'preview_url' , 'id']]
        #print(top_rec_df)
        #print("Top Recommendations\n\n")

        #print("\n\nLINKS:\n")
        #for item in top_rec_df['external_urls']:
            #print(item['spotify'])

        #remove top tracks from recommendations
        for track_id in top_tracks_df['id']:
            if track_id in top_rec_df['id']:
                top_rec_df = top_rec_df.drop(track_id)
        #print("\n\n\n\n")
        #print(top_rec_df)
        top_recommendations_dict = top_rec_df.to_dict(orient='records')
        #print(top_recommendations_dict)
        
        return render_template('index.html', plot_html=plot_html, plot_bar_chart_html=bar_chart_plot_html, top_recommendations=top_recommendations_dict)


    except requests.exceptions.RequestException as e:


        return jsonify({'error': f"Failed to fetch top artists: {str(e)}"}), 500
    

    
if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)
