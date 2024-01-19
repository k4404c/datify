# Datify: Spotify Data Analysis and Recommendation System

Datify is a web application built with Flask that leverages the Spotify API to analyze a user's top tracks and artists, perform clustering, and provide personalized music recommendations. The application integrates data visualization using Plotly Express for showcasing the distribution of top tracks across genres and clustering results in a 3D scatter plot (keep in mind clusters may not look like clusters as 10-dimensional clusters are being represended in 3-dimensions). Additionally, it employs a Random Forest Regressor model to identify important features and generate recommendations based on user preferences.

## Features:

- **Spotify Integration:** Allows users to log in with their Spotify account and retrieve their top tracks and artists.
- **Data Analysis:** Utilizes various Spotify API endpoints to gather information on top tracks, artists, and audio features.
- **Clustering:** Applies KMeans clustering to identify patterns and group similar tracks together.
- **Data Visualization:** Presents the analysis results through interactive visualizations, including a 3D scatter plot and a bar chart displaying song counts per genre utilizing Plotly Express.
- **Recommendation System:** Generates personalized music recommendations based on user top genres and top features identified by the Random Forest Regressor.

## Usage:

1. Clone the repository.
2. Set up a Spotify Developer account and obtain your `CLIENT_ID` and `CLIENT_SECRET`.
3. Create a `.env` file with your Spotify credentials:

```
CLIENT_ID=your_client_id
CLIENT_SECRET=your_client_secret
```

4. Run the Flask application:

```
python app.py
```

5. Access the application at [http://localhost:5000](http://localhost:5000) and log in with your Spotify account.

6. Explore your top tracks, cluster analysis, and receive personalized music recommendations!

Feel free to customize and enhance the application based on your preferences and requirements.
