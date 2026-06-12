import streamlit as st
import pandas as pd
import joblib

from src.preprocessing import load_and_preprocess_data

# -----------------------------
# Load Model
# -----------------------------
model = joblib.load("models/churn_model.pkl")

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(
    page_title="Customer Churn Prediction",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Churn Prediction")
st.write(
    "Upload customer data and predict whether customers are likely to churn."
)

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # Read uploaded file
        df = pd.read_csv(uploaded_file)

        st.subheader("Uploaded Data")
        st.dataframe(df.head())

        # Save temporary file
        temp_path = "temp_customer_data.csv"
        df.to_csv(temp_path, index=False)

        # Preprocess
        processed_df = load_and_preprocess_data(temp_path)

        # Remove target column if present
        if "Churn" in processed_df.columns:
            X = processed_df.drop("Churn", axis=1)
        else:
            X = processed_df

        # Prediction
        predictions = model.predict(X)

        #Prediction probability
        probabilities = model.predict_proba(X)[:, 1]

        # Add Prediction Column
        result_df = df.copy()
        result_df["Predicted_Churn"] = predictions

        result_df["Predicted_Churn"] = result_df[
            "Predicted_Churn"
        ].map({
            1: "Yes",
            0: "No"
        })

        # Add Churn Probability
        result_df["Churn_Probability"] = (
                probabilities * 100
        ).round(2).astype(str) + " %"

        st.subheader("Prediction Results")
        st.dataframe(result_df)

        # Download Results
        csv = result_df.to_csv(index=False)

        st.download_button(
            label="Download Predictions",
            data=csv,
            file_name="churn_predictions.csv",
            mime="text/csv"
        )

    except Exception as e:
        st.error(f"Error: {e}")