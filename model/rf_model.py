import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score


class RandomForest:
    def __init__(self, df, config):
        label_encoders = {}

        # Label encoding of categorical variables
        for col in config["discrete_col"]:
            le = LabelEncoder()  # initialize label encoder
            self.df[col] = le.fit_transform(self.df[col])
            label_encoders[col] = le

        self.n_estimator = config["n_estimator"]
        self.depth = config["depth"]
        self.seed = config["seed"]
        self.kf_split = config["kf_split"]

    def fit(self):
        X = self.df.drop("HeartDisease", axis=1)
        y = self.df["HeartDisease"]

        # Scale numerical features (optional but recommended)
        # scaler = StandardScaler()
        # X = scaler.fit_transform(X)

        # Define the Random Forest model
        rf_model = RandomForestClassifier(
            n_estimators=self.n_estimator, max_depth=self.depth, random_state=self.seed
        )

        # Define the stratified k-fold cross-validation procedure
        kf = StratifiedKFold(
            n_splits=self.kf_split, shuffle=True, random_state=self.seed
        )

        # Perform cross-validation
        cv_scores = cross_val_score(rf_model, X, y, cv=kf, scoring="roc_auc")

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        # Train the Random Forest model on the training set
        rf_model.fit(X_train, y_train)
        # Make predictions on the test set
        y_pred = rf_model.predict(X_test)

        # Evaluate the model
        return classification_report(y_test, y_pred), accuracy_score(y_test, y_pred)
