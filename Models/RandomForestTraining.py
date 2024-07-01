import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, precision_score, recall_score, f1_score
import joblib

# Load the processed dataset
df_processed = pd.read_csv('./processed_heart.csv')

# Separate features and target
X = df_processed.drop('HeartDisease', axis=1)  # Drop the target column from features
y = df_processed['HeartDisease']  # Target column

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Random Forest Classifier
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
rf_classifier.fit(X_train, y_train)

# Make predictions
y_pred = rf_classifier.predict(X_test)

### Evaluate the model ###

# The ratio of correctly predicted positive observations to the total predicted positives.
# Precision = TP / (TP + FP)
# High precision means that the model has a low false positive rate.
precision = precision_score(y_test, y_pred)

# The ratio of correctly predicted positive observations to the all observations in the actual class.
# Recall(Sensitivity) = TP / (TP + FN)
# High recall means that the model has a low false negative rate.
recall = recall_score(y_test, y_pred)

# The weighted average of Precision and Recall. 
# It considers both false positives and false negatives.
# F1 Score = 2 * (Precision * Recall) / (Precision + Recall)
# High F1-score means the model has a good balance between precision and recall.
f1 = f1_score(y_test, y_pred)

# Print evaluation metrics
print("Classification Report:\n", classification_report(y_test, y_pred))
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1 Score: {f1:.2f}")

# Feature Importance
feature_importances = pd.DataFrame(rf_classifier.feature_importances_, index=X.columns, columns=['importance']).sort_values('importance', ascending=False)
print("\nFeature Importances:\n", feature_importances)

# Save the model if needed
joblib.dump(rf_classifier, './random_forest_model.pkl')

print("Random Forest model training and evaluation completed successfully.")
