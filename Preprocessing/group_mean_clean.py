import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

df = pd.read_csv("../Data/heart.csv")
df.head()

string_col = df.select_dtypes(include="object").columns
df[string_col] = df[string_col].astype("string")
print(df.dtypes)  # check if converted properly

# distribution of categorical values:
df[string_col].head()
for col in string_col:
    print(f"The distribution of categorical valeus in the {col} is : ")
    print(df[col].value_counts())

# label encoding of categorical variables (this is fine for random forest)
# Initialize a dictionary to store label encoders
label_encoders = {}

# Label encoding of categorical variables
for col in string_col:
    le = LabelEncoder()  # initialize label encoder
    df[col] = le.fit_transform(df[col])
    # fit: During the fitting process, LabelEncoder learns the unique classes present in the data and assigns each class a unique integer.
    # transform: After learning the classes, it then transforms the original categorical values into their corresponding integer codes.
    label_encoders[col] = le

# Print the label encoding mapping for each categorical feature
for col in string_col:
    le = label_encoders[col]
    print(f"Label encoding for {col}:")
    for class_, label in zip(le.classes_, le.transform(le.classes_)):
        print(f"{class_} -> {label}")
    print()

label_column = "HeartDisease"

df_chol_not_missing = df[df["Cholesterol"] != 0]
df_resting_not_missing = df[df["RestingBP"] != 0]

chol_means = df_chol_not_missing.groupby(label_column)["Cholesterol"].transform("mean")
resting_means = df_resting_not_missing.groupby(label_column)["RestingBP"].transform(
    "mean"
)

# Print the group means for verification
print("Cholesterol group means:\n", chol_means)
print("\nRestingBP group means:\n", resting_means)

# Fill missing values using the calculated group means
df["Cholesterol"] = df.apply(
    lambda row: (
        chol_means[row[label_column]] if row["Cholesterol"] == 0 else row["Cholesterol"]
    ),
    axis=1,
)

df["RestingBP"] = df.apply(
    lambda row: (
        resting_means[row[label_column]] if row["RestingBP"] == 0 else row["RestingBP"]
    ),
    axis=1,
)

# Check the DataFrame after filling missing values
print("\nDataFrame after filling missing values:\n", df.head())

# Save the cleaned dataset to a new CSV file
cleaned_file_path = "../Data/group_cleaned_mean_heart.csv"
df.to_csv(cleaned_file_path, index=False)
