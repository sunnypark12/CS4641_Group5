import pandas as pd
from sklearn.preprocessing import OneHotEncoder

import setup as su
import explotaryDataAnalysis as eda
import dataPreprocessing as dp


# main:

def main():
    file_path = './Data/heart.csv'
    
    # Load the dataset
    heart_df = su.load_dataset(file_path)

    # convert string data to appropriate type
    heart_df = su.convert_string_columns(heart_df)

    categorical_columns = su.get_categorical_columns(heart_df)  # Get categorical columns
    numerical_columns = su.get_numerical_columns(heart_df, None)  # Get numerical columns excluding "HeartDisease"

    print("Categorical columns:", categorical_columns)
    print("Numerical columns:", numerical_columns)

    # Create correlation matrix for numerical columns
    eda.create_correlation_matrix(heart_df, columns=numerical_columns, column_type='numerical', title="Numerical Correlation Matrix")

    # Create correlation matrix for categorical columns
    eda.create_correlation_matrix(heart_df, columns=categorical_columns, column_type='categorical', title="Categorical Correlation Matrix")

    # Create histogram:
    eda.draw_histogram(heart_df, "HeartDisease", "Sex", "Distribution of Heart Diseases by Sex")
    # eda.draw_histogram(heart_df, "ChestPainType", "Sex", "Types of Chest Pain by Sex")
    # eda.draw_histogram(heart_df, "Sex", None, "Sex Ratio in Data")
    # eda.draw_histogram(heart_df, "RestingECG", None, "Distribution of ECG")

    # # Create clustermap for numerical columns
    # eda.create_clustermap(heart_df, columns=numerical_columns, column_type='numerical', title="Numerical Clustermap")

    # # Create clustermap for categorical columns
    # eda.create_clustermap(heart_df, columns=categorical_columns, column_type='categorical', title="Categorical Clustermap")

    # Create pairplot
    eda.create_pairplot(heart_df, hue_column="HeartDisease", title="Looking for Insights in Data")

    # Plot distributions with KDE
    eda.plot_distributions(heart_df, title="Distribution of Heart Data")

    # Create box plot (finding outliers)
    eda.create_box_plot(heart_df, x_column="HeartDisease", y_column="Age", title="Distribution of Age by Heart Disease")

    
if __name__ == "__main__":
    main()