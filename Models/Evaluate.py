import joblib

# Load the trained Random Forest model
# rf_classifier = joblib.load('./random_forest_model.pkl')
# knn_classifier = joblib.load('./knn_model.pkl')

# Example of preprocessing new data
new_data = pd.DataFrame({
    'Age': [45],
    'Sex': ['M'],
    'ChestPainType': ['ATA'],
    'RestingBP': [140],
    'Cholesterol': [289],
    'FastingBS': [0],
    'RestingECG': ['Normal'],
    'MaxHR': [172],
    'ExerciseAngina': ['N'],
    'Oldpeak': [0.0],
    'ST_Slope': ['Up']
})

# Apply the same preprocessing steps
X_new_preprocessed = preprocessor.transform(new_data)

# Make a prediction for Random Forest
prediction = rf_classifier.predict(X_new_preprocessed)
prediction_proba = rf_classifier.predict_proba(X_new_preprocessed)

# Make a prediction for KNN Model
prediction = knn_classifier.predict(X_new_preprocessed)
prediction_proba = knn_classifier.predict_proba(X_new_preprocessed)

print("Prediction:", prediction)
print("Prediction Probability:", prediction_proba)
