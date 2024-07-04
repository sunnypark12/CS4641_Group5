import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import numpy as np
import matplotlib.pyplot as plt

# Load the dataset
file_path = './Data/heart.csv'
df = pd.read_csv(file_path)

# Convert string data to appropriate type; object -> string
string_col = df.select_dtypes(include="object").columns
df[string_col] = df[string_col].astype("string")
print(df.dtypes)  # check if converted properly

df[string_col].head()
for col in string_col:
    print(f"The distribution of categorical values in the {col} is : ")
    print(df[col].value_counts())

# Label encoding of categorical variables (this is fine for random forest)
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

# Split the data into features and target
X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

# Scale numerical features
scaler = StandardScaler()
X = scaler.fit_transform(X)

rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Define the stratified k-fold cross-validation procedure
kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Perform cross-validation
cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='roc_auc')
# Print cross-validation results
print(f"Cross-Validation ROC-AUC Scores: {cv_scores}")
print(f"Mean ROC-AUC Score: {cv_scores.mean()}")

n_estimators_range = range(1, 201, 10)  # Define a range of n_estimators
# List to store accuracy for each value of n_estimators
accuracy_scores = []
# Loop over the n_estimators_range
for n in n_estimators_range:
    # Define the Random Forest model
    rf_model = RandomForestClassifier(n_estimators=n, random_state=90)
    
    # Perform cross-validation and calculate accuracy
    cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring='accuracy')
    
    # Append the mean accuracy to the list
    accuracy_scores.append(np.mean(cv_scores))

# Plot the results
plt.figure(figsize=(10, 6))
plt.plot(n_estimators_range, accuracy_scores, marker='o')
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Mean Accuracy')
plt.title('Random Forest Model Accuracy vs. Number of Trees')
plt.grid(True)
plt.show()

# Define the Random Forest model
rf_model = RandomForestClassifier(n_estimators=152, random_state=90) 

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the Random Forest model on the training set
rf_model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = rf_model.predict(X_test)
y_proba = rf_model.predict_proba(X_test)[:, 1]

# Evaluate the model
print(classification_report(y_test, y_pred))
print("ROC-AUC Score:", roc_auc_score(y_test, y_proba))

# Calculate and print the accuracy
accuracy = accuracy_score(y_test, y_pred)
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Plot feature importance
feature_importances = rf_model.feature_importances_
features = df.drop('HeartDisease', axis=1).columns

plt.figure(figsize=(12, 8))
plt.barh(features, feature_importances, color='skyblue')
plt.xlabel('Feature Importance')
plt.title('Feature Importance in Random Forest Model')
plt.show()

new_data = pd.DataFrame({
    'Age': [45],
    'Sex': ['Male'],
    'ChestPainType': ['ASY'],
    'RestingBP': [130],
    'Cholesterol': [233],
    'FastingBS': [1],
    'RestingECG': ['Normal'],
    'MaxHR': [150],
    'ExerciseAngina': ['N'],
    'Oldpeak': [0.2],
    'ST_Slope': ['Up']
})

# Encode new data using the same label encoders
for col in new_data.columns:
    if col in string_col:
        le = label_encoders[col]
        new_data[col] = new_data[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

# Scale new data
new_data_scaled = scaler.transform(new_data)

# Predict the risk
risk_percentage = rf_model.predict_proba(new_data_scaled)[:, 1] * 100
print("Predicted risk of heart failure: {:.2f}%".format(risk_percentage[0]))
