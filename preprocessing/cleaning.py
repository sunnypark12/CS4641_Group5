import pandas as pd

# Load the dataset
file_path = '../Data/heart.csv'
df = pd.read_csv(file_path)

# Remove rows where RestingBP or Cholesterol columns have zero values
cleaned_df = df[(df['RestingBP'] != 0) & (df['Cholesterol'] != 0)]

# Save the cleaned dataset to a new CSV file
cleaned_file_path = 'cleaned_heart.csv'
cleaned_df.to_csv(cleaned_file_path, index=False)

# Display the first few rows of the cleaned dataframe
print(cleaned_df.head())

print("Cleaned dataset saved to:", cleaned_file_path)





