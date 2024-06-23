import pandas as pd

# Load the dataset
# heart_2020_cleansed.csv
file_path = './Data/heart_2020_cleaned.csv'
heart_2020_df = pd.read_csv(file_path)

# heart_failure_clinical_records_dataset.csv
file_path = './Data/heart_failure_clinical_records_dataset.csv'
clinical_records = pd.read_csv(file_path)

# heart.csv
file_path = './Data/heart.csv'
heart = pd.read_csv(file_path)

# Display basic information about the dataset
heart_2020_df.info()
clinical_records.info()
heart.info()

# Calculate the percentage of missing values for each column
missing_percentage1 = (heart_2020_df.isnull().sum() / len(heart_2020_df)) * 100
missing_percentage2 = (clinical_records.isnull().sum() / len(heart_2020_df)) * 100
missing_percentage3 = (heart.isnull().sum() / len(heart_2020_df)) * 100

# Display the result
print(missing_percentage1) # 0.0 -> no missing data
print(missing_percentage2) # 0.0
print(missing_percentage3) # 0.0


# Display the first few rows of the dataset
# print(heart_2020_df.head())


