import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def run_pca(csv_file: str, n_components: int):
    # Load dataset
    data = pd.read_csv(csv_file)
    
    # Check if the dataset contains non-numeric columns and drop them
    numeric_data = data.select_dtypes(include=[float, int])
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(numeric_data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Print explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot the explained variance ratio
    plt.figure(figsize=(8, 5))
    plt.bar(range(n_components), pca.explained_variance_ratio_, alpha=0.5, align='center')
    plt.step(range(n_components), np.cumsum(pca.explained_variance_ratio_), where='mid')
    plt.xticks(range(n_components), numeric_data.columns[:n_components], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio by Principal Components')
    plt.show()
    
    return pc_df

if __name__ == "__main__":
    # Replace 'your_dataset.csv' with the path to your actual CSV file
    csv_file = 'Data/cleaned_heart.csv'
    n_components = 5  # Replace with the number of principal components you want to keep
    pc_df = run_pca(csv_file, n_components)
    
    # Display the first few rows of the principal components DataFrame
    print(pc_df.head())