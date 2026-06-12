import pandas as pd


def load_and_preprocess_data(file_path):
    """
    Load and preprocess the Telco Customer Churn dataset.
    """

    # Load dataset
    df = pd.read_csv(file_path)

    # Remove customerID
    df.drop("customerID", axis=1, inplace=True)

    # Convert TotalCharges to numeric
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"], errors="coerce"
    )

    # Fill missing values
    df["TotalCharges"] = df["TotalCharges"].fillna(
        df["TotalCharges"].median()
    )

    # Convert target variable
    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map(
            {"Yes": 1, "No": 0}
        )

    # One-Hot Encoding
    df = pd.get_dummies(
        df,
        drop_first=True
    )


    return df