# Customer Churn Prediction System

## Overview

The Customer Churn Prediction System is a Machine Learning project that predicts whether a customer is likely to leave a subscription-based service. The project uses customer demographic and service-related information to identify customers at risk of churn, helping businesses take proactive retention measures.

The application includes data analysis, machine learning model training, model evaluation, feature importance analysis, and a Streamlit-based web application for batch churn prediction.

---

## Problem Statement

Customer churn is a major challenge for subscription-based businesses. Losing existing customers can significantly impact revenue and growth.

This project aims to:

* Analyze customer behavior and service usage patterns.
* Identify factors contributing to customer churn.
* Build machine learning models to predict customer churn.
* Provide a user-friendly dashboard for bulk customer churn prediction.

---

## Tech Stack

### Programming Language

* Python

### Libraries & Frameworks

* Pandas
* NumPy
* Scikit-Learn
* Matplotlib
* Seaborn
* Joblib
* Streamlit

### Machine Learning Algorithms

* Logistic Regression
* Random Forest Classifier

---

## Project Structure

```text
customer-churn-prediction/

│
├── data/
│   └── Telco-Customer-Churn.csv
│
├── models/
│   └── churn_model.pkl
│
├── notebooks/
│   └── churn_analysis.ipynb
│
├── src/
│   ├── preprocessing.py
│   ├── model_training.py
│   ├── evaluation.py
│   └── feature_importance.py
│
├── app.py
│
├── requirements.txt
│
└── README.md
```

---

## Dataset

Dataset: Telco Customer Churn Dataset

The dataset contains customer information such as:

* Customer demographics
* Contract details
* Internet services
* Payment methods
* Monthly charges
* Total charges
* Churn status

Target Variable:

```text
Churn
```

* Yes = Customer left the service
* No = Customer retained the service

---

## Data Analysis & EDA

Performed exploratory data analysis to understand customer behavior and churn patterns.

### Analysis Performed

* Dataset inspection
* Missing value analysis
* Data type verification
* Churn distribution analysis
* Numerical feature analysis
* Categorical feature analysis
* Service usage analysis
* Contract type analysis
* Payment method analysis

### Visualizations

* Count plots
* Histograms
* Box plots
* Bar charts
* Correlation heatmap

---

## Data Preprocessing

The following preprocessing steps were performed:

* Removed unnecessary columns
* Handled missing values
* Converted data types
* Encoded categorical variables using One-Hot Encoding
* Feature selection
* Train-test split

---

## Correlation Analysis

Performed correlation analysis to identify relationships between features and customer churn.

Techniques used:

* Correlation matrix
* Heatmap visualization

This helped identify the most influential features affecting churn behavior.

---

## Machine Learning Models

### Logistic Regression

Used as a baseline classification model for churn prediction.

### Random Forest Classifier

Used to improve prediction performance and capture complex relationships within the data.

---

## Model Evaluation

The models were evaluated using:

* Accuracy Score
* Precision
* Recall
* F1-Score
* Confusion Matrix

The evaluation results were compared to select the best-performing model.

---

## Feature Importance Analysis

Feature importance analysis was performed using the Random Forest model.

This helped identify the most important factors contributing to customer churn, such as:

* Contract type
* Monthly charges
* Tenure
* Internet service
* Payment method

---

## Streamlit Dashboard

An interactive Streamlit application was developed for customer churn prediction.

### Features

* Upload customer CSV file
* Automatic data preprocessing
* Batch churn prediction
* Churn probability calculation
* Display prediction results
* Download prediction results as CSV

---

## Workflow

```text
Customer Data
      │
      ▼
Data Preprocessing
      │
      ▼
Exploratory Data Analysis
      │
      ▼
Correlation Analysis
      │
      ▼
Model Training
      │
      ├── Logistic Regression
      │
      └── Random Forest
      │
      ▼
Model Evaluation
      │
      ▼
Feature Importance
      │
      ▼
Streamlit Application
      │
      ▼
Customer Churn Predictions
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/harish-ragavendra-12/customer-churn-prediction.git
```

Navigate to project folder:

```bash
cd customer-churn-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run the Application

```bash
streamlit run app.py
```

---

## Project Outcomes

* Built an end-to-end Machine Learning pipeline.
* Performed exploratory data analysis and correlation analysis.
* Developed and evaluated classification models.
* Implemented feature importance analysis.
* Created a Streamlit web application for churn prediction.
* Enabled batch customer prediction with downloadable results.

---

## Future Enhancements

* PostgreSQL database integration
* XGBoost model implementation
* Real-time customer prediction form
* Model deployment on cloud platforms
* Automated model retraining pipeline

---

## Author

Harish Ragavendra

Aspiring Data Analyst | Data Science Enthusiast | Machine Learning Learner
