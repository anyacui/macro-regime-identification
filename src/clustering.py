from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.features import engineer_features, standardise_features
from config import N_CLUSTERS


# determining number of clusters to use


def elbow_method(df_scaled, k_range=range(2, 10)):
    inertias = []
    for k in k_range:
        # create k means object
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        # will run 10 times with different random starts
        # run the k-means on the data
        kmeans.fit(df_scaled)
        inertias.append(kmeans.inertia_)
        # record the final inertia
    return k_range, inertias


def compute_silhouette(df_scaled, k_range=range(2, 10)):
    silhouette_scores = []
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        # will run 10 times with different random starts
        labels = kmeans.fit_predict(df_scaled)
        # fit_predict fit the kmeans and then gets the predicted cluster labels (what cluster each row of data belongs to)
        score = silhouette_score(df_scaled, labels)
        silhouette_scores.append(score)
    return k_range, silhouette_scores


# Determined optimal number of clusters is 4
def fit_clusters(df, df_scaled, k=N_CLUSTERS):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    df['regime'] = kmeans.fit_predict(df_scaled)
    return df, kmeans


if __name__ == '__main__':
    fit_clusters()
