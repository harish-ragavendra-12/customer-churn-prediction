import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from preprocessing import load_and_preprocess_data


def train_models():

    # Load data
    df = load_and_preprocess_data(
        "../data/Telco-Customer-Churn.csv"
    )

    # Features and Target
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    # Logistic Regression
    lr_model = LogisticRegression(
        max_iter=5000
    )

    lr_model.fit(
        X_train,
        y_train
    )

    # Random Forest
    rf_model = RandomForestClassifier(
        n_estimators=100,
        random_state=42
    )

    rf_model.fit(
        X_train,
        y_train
    )

    # Save Random Forest Model
    joblib.dump(
        rf_model,
        "../models/churn_model.pkl"
    )

    print("Model training completed.")
    print("Model saved as models/churn_model.pkl")

    return (
        lr_model,
        rf_model,
        X_test,
        y_test
    )


if __name__ == "__main__":
    train_models()