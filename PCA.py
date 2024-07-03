import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

# Load the datasets
df1 = pd.read_csv('./Data/heart_2020_cleaned.csv')
df2 = pd.read_csv('./Data/heart_failure_clinical_records_dataset.csv')
df3 = pd.read_csv('./Data/heart.csv')

# Inspect the column names
print("Dataset 1 Columns:", df1.columns)
print("Dataset 2 Columns:", df2.columns)
print("Dataset 3 Columns:", df3.columns)

# Standardize column names to lowercase and replace spaces with underscores
df1.columns = df1.columns.str.lower().str.replace(' ', '_')
df2.columns = df2.columns.str.lower().str.replace(' ', '_')
df3.columns = df3.columns.str.lower().str.replace(' ', '_')

# Rename columns for consistency
df1 = df1.rename(columns={'heartdisease': 'target', 'alcoholdrinking': 'alcohol_drinking', 
                          'physicalhealth': 'physical_health', 'mentalhealth': 'mental_health', 
                          'diffwalking': 'diff_walking', 'agecategory': 'age', 
                          'physicalactivity': 'physical_activity', 'genhealth': 'gen_health', 
                          'sleeptime': 'sleep_time', 'kidneydisease': 'kidney_disease', 'skincancer': 'skin_cancer'})

df2 = df2.rename(columns={'death_event': 'target', 'creatinine_phosphokinase': 'creatinine_phosphokinase', 
                          'ejection_fraction': 'ejection_fraction', 'high_blood_pressure': 'high_blood_pressure', 
                          'serum_creatinine': 'serum_creatinine', 'serum_sodium': 'serum_sodium'})

df3 = df3.rename(columns={'heartdisease': 'target', 'chestpaintype': 'chest_pain_type', 'restingbp': 'resting_bp', 
                          'fastingbs': 'fasting_bs', 'restingecg': 'resting_ecg', 'exerciseangina': 'exercise_angina', 
                          'oldpeak': 'oldpeak', 'st_slope': 'st_slope'})

# Define a function to convert age categories to numeric
def convert_age(age):
    if '-' in age:
        return int(age.split('-')[0])
    elif '+' in age:
        return int(age.split('+')[0]) + 5
    elif age == '80 or older':
        return 80
    else:
        return int(age)

# Convert 'age' column in df1
df1['age'] = df1['age'].apply(convert_age)

# Ensure the 'age' column is present in all datasets
df2['age'] = df2['age']
df3['age'] = df3['age']

# Add missing 'smoking' column to df3 with default value
if 'smoking' not in df3.columns:
    df3['smoking'] = 'Unknown'  # or set to a default value, e.g., 'No'

# Common columns for PCA
common_cols = ['target', 'age', 'sex', 'smoking']

# Select common columns and align datasets
df1_common = df1[common_cols + ['bmi', 'physical_health', 'mental_health', 'diff_walking', 'physical_activity', 'gen_health', 'sleep_time']]
df2_common = df2[common_cols + ['anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium']]
df3_common = df3[common_cols + ['chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 'resting_ecg', 'maxhr', 'exercise_angina', 'oldpeak', 'st_slope']]

# Fill missing columns with NaN for alignment
df1_common = df1_common.reindex(columns=common_cols + ['bmi', 'physical_health', 'mental_health', 'diff_walking', 'physical_activity', 'gen_health', 'sleep_time', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 'resting_ecg', 'maxhr', 'exercise_angina', 'oldpeak', 'st_slope'])
df2_common = df2_common.reindex(columns=common_cols + ['bmi', 'physical_health', 'mental_health', 'diff_walking', 'physical_activity', 'gen_health', 'sleep_time', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 'resting_ecg', 'maxhr', 'exercise_angina', 'oldpeak', 'st_slope'])
df3_common = df3_common.reindex(columns=common_cols + ['bmi', 'physical_health', 'mental_health', 'diff_walking', 'physical_activity', 'gen_health', 'sleep_time', 'anaemia', 'creatinine_phosphokinase', 'diabetes', 'ejection_fraction', 'high_blood_pressure', 'platelets', 'serum_creatinine', 'serum_sodium', 'chest_pain_type', 'resting_bp', 'cholesterol', 'fasting_bs', 'resting_ecg', 'maxhr', 'exercise_angina', 'oldpeak', 'st_slope'])

# Combine datasets
combined_df = pd.concat([df1_common, df2_common, df3_common], ignore_index=True)

# Encode categorical variables
categorical_cols = ['sex', 'smoking', 'diff_walking', 'physical_activity', 'gen_health', 'chest_pain_type', 'resting_ecg', 'exercise_angina', 'st_slope']
for col in categorical_cols:
    combined_df[col] = combined_df[col].astype(str).fillna('Unknown')
    le = LabelEncoder()
    combined_df[col] = le.fit_transform(combined_df[col])

# Drop rows with missing target variable
combined_df = combined_df.dropna(subset=['target'])

# Convert target to numeric
combined_df['target'] = pd.to_numeric(combined_df['target'], errors='coerce')

# Separate features and target
X = combined_df.drop('target', axis=1)
y = combined_df['target']

# Ensure all columns are numeric
X = X.apply(pd.to_numeric, errors='coerce')

# Fill remaining NaNs with the median of each column
X = X.fillna(X.median())

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA
pca = PCA(n_components=6)
principal_components = pca.fit_transform(X_scaled)
principal_df = pd.DataFrame(data=principal_components, columns=[f'PC{i+1}' for i in range(6)])
principal_df['Label'] = y.values

# Plot explained variance
plt.figure(figsize=(10, 6))
plt.plot(range(1, len(pca.explained_variance_ratio_) + 1), pca.explained_variance_ratio_.cumsum(), marker='o')
plt.title('Cumulative Explained Variance by PCA Components')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid(True)
plt.show()
