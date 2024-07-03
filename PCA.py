import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

def pca(csv_file: str, n_components: int):
    # Load dataset
    data = pd.read_csv(csv_file)
    
    # Handle non-numeric data
    data = pd.get_dummies(data)
    
    # Standardize the data
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(data)
    
    # Perform PCA
    pca = PCA(n_components=n_components)
    principal_components = pca.fit_transform(scaled_data)
    
    # Create a DataFrame with the principal components
    pc_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(n_components)])
    
    # Print explained variance ratio
    print(f"Explained variance ratio: {pca.explained_variance_ratio_}")
    
    # Plot the explained variance ratio and cumulative explained variance
    plt.figure(figsize=(10, 6))
    explained_variance_ratio = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance_ratio)
    components = np.arange(1, n_components + 1)
    
    # Bar plot for explained variance ratio
    plt.bar(components, explained_variance_ratio, alpha=0.6, label='Explained Variance Ratio')
    
    # Line plot for cumulative explained variance
    plt.plot(components, cumulative_variance, marker='o', color='red', label='Cumulative Explained Variance')
    
    # Adding labels and title
    plt.xlabel('Principal Components')
    plt.ylabel('Explained Variance Ratio')
    plt.title('Explained Variance Ratio and Cumulative Explained Variance by Principal Components')
    plt.legend()
    plt.grid(True)
    plt.xticks(components)
    
    plt.show()
    
    return pc_df

if __name__ == "__main__":
    # Replace 'your_dataset.csv' with the path to your actual CSV file
    csv_file = 'Data/cleaned_heart.csv'
    n_components = 8 # Replace with the number of principal components you want to keep
    pc_df = pca(csv_file, n_components)
    
    # Display the first few rows of the principal components DataFrame
    print(pc_df.head())
    # Save output of PCA to .csv file
    pc_df.to_csv('Data/pca_output.csv', index=False)