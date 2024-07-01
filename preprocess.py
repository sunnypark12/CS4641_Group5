import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, KBinsDiscretizer
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv('./Data/heart.csv')

# Print first few rows to understand the data structure
print(df.head())

# Separate features and target
# If 'heart_disease' is your target variable, replace 'target' with 'heart_disease'
X = df.drop('HeartDisease', axis=1)  # Drop the target column from features
y = df['HeartDisease']  # Target column (e.g., heart disease)

# Separate numeric and categorical columns
numeric_features = X.select_dtypes(include=['int64', 'float64']).columns
categorical_features = X.select_dtypes(include=['object']).columns

# Create pipelines for numeric and categorical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),  # Fill missing values with mean

    # normal distribution
    # StandardScaler standardizes a feature by subtracting the mean and then scaling to unit variance. 
    # Unit variance means dividing all the values by the standard deviation
    ('scaler', StandardScaler())  # Normalize numeric features
])

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),  # Fill missing values with most frequent value

    # OneHotEncoder represent categorical variables as numerical values in a machine learning model
    ('onehot', OneHotEncoder(handle_unknown='ignore'))  # Apply one-hot encoding to categorical features
])

# Combine preprocessing steps for numeric and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),  # Apply numeric transformer to numeric features
        ('cat', categorical_transformer, categorical_features)  # Apply categorical transformer to categorical features
    ])

# Apply preprocessing to the feature set
X_preprocessed = preprocessor.fit_transform(X)



# # Determine the optimal number of clusters using the Elbow Method
# wcss = []
# for i in range(1, 11):
#     kmeans = KMeans(n_clusters=i, random_state=42)
#     kmeans.fit(X_preprocessed)
#     wcss.append(kmeans.inertia_)

# # Plot the WCSS to visualize the elbow
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, 11), wcss, marker='o')
# plt.title('Elbow Method for Optimal Number of Clusters')
# plt.xlabel('Number of Clusters')
# plt.ylabel('WCSS')
# plt.show()

# After visualizing the elbow plot, choose the optimal number of clusters
optimal_clusters = 3  # based on my elbow plot

# Apply KMeans with the optimal number of clusters
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
X_clustered = kmeans.fit_predict(X_preprocessed)


# # Binning Method (Discretization) for continuous features
# binning_transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform')
# X_binned = binning_transformer.fit_transform(X_preprocessed)

# # Combine the results (example of using binned data)
# # Here we're using binned data as an example; in practice, you might combine multiple processed datasets
# X_cleaned = X_binned  # Placeholder for combining binned and cleaned data

# Dimensionality Reduction with PCA
# Reducing the number of features while retaining 95% of the variance
pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_preprocessed)

# Combine preprocessed features and target into a DataFrame for saving
df_processed = pd.DataFrame(X_reduced)
df_processed['HeartDisease'] = y.values  # Add the target column back to the processed DataFrame

# Save the processed data for model training or further analysis
df_processed.to_csv('./processed_heart.csv', index=False)

print("Data preprocessing completed successfully.")
