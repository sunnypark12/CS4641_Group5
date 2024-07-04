import pandas as pd
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def kmeans(pca_file: str, n_clusters: int):
    # Load PCA output
    pc_df = pd.read_csv(pca_file)
    
    # Perform KMeans clustering using all principal components
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    kmeans.fit(pc_df)
    labels = kmeans.labels_
    
    # Add the cluster labels to the DataFrame
    pc_df['Cluster'] = labels
    
    return pc_df, kmeans

if __name__ == "__main__":
    pca_file = 'Data/pca_output.csv'
    n_clusters = 3  # Replace with the number of clusters you want to create
    
    # Run KMeans clustering on the PCA output
    clustered_df, kmeans_model = kmeans(pca_file, n_clusters)
    
    # Display the first few rows of the clustered DataFrame
    print(clustered_df.head())
    
    # Plot the clusters using the first two principal components
    plt.figure(figsize=(10, 6))
    scatter = plt.scatter(clustered_df['PC1'], clustered_df['PC2'], c=clustered_df['Cluster'], cmap='viridis', alpha=0.6)
    
    # Adding legend
    legend1 = plt.legend(*scatter.legend_elements(), title="Clusters")
    plt.gca().add_artist(legend1)
    
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('KMeans Clustering on PCA Output')
    plt.grid(True)
    plt.show()
