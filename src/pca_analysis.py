from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from src.features import engineer_features, standardise_features
from config import N_CLUSTERS


def run_pca(df_scaled):
    pca = PCA()
    pca.fit(df_scaled)
    explained_variance = pca.explained_variance_ratio_.cumsum()
    n_components = (explained_variance >= 0.95).argmax() + 1

    return pca, n_components, explained_variance


def pca_clustering(components, df_scaled, df, k=N_CLUSTERS):
    # Fit PCA with 6 components
    pca_final = PCA(n_components=components)
    df_pca = pca_final.fit_transform(df_scaled)

    # Rerun k-means on PCA components
    kmeans_pca = KMeans(n_clusters=k, random_state=42, n_init=10)
    pca_labels = kmeans_pca.fit_predict(df_pca)

    # Compare assignments so add new column of these regimes
    df['regime_pca'] = pca_labels

    return df


if __name__ == '__main__':
    df = engineer_features()
    df_scaled = standardise_features(df)
    pca, n_components = run_pca(df_scaled)
    pca_clustering(n_components, df_scaled, df)
