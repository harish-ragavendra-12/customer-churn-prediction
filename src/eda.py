import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv("../data/Telco-Customer-Churn.csv")

#Basic Info
print("=" * 50)
print("DATASET SHAPE")
print("=" * 50)
print(df.shape)

print("\n" + "=" * 50)
print("DATASET INFO")
print("=" * 50)
print(df.info())

print("\n" + "=" * 50)
print("MISSING VALUES")
print("=" * 50)
print(df.isnull().sum)

print("\n" + "=" * 50)
print("DUPLICATES")
print("=" * 50)
print(df.duplicated().sum())

#Data Cleaning

df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")

df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

print("\n" + "=" * 50)
print("DATASET INFO after changing Total charges to numeric datatype")
print("=" * 50)
print(df.info())

#Churn Distribution
plt.figure(figsize=(6,4))
sns.countplot(x="Churn",data=df)
plt.title("Churn Distribution")
plt.show()
plt.savefig("../output/churn_distribution.png")
plt.close()

#Gender vs Churn
plt.figure(figsize=(6,4))
sns.countplot(x="gender",hue="Churn",data=df)
plt.title("Gender vs Churn")
plt.show()
plt.savefig("../output/gender_vs_churn.png")
plt.close()

#Contract vs Churn
plt.figure(figsize=(8,5))
sns.countplot(x="Contract",hue="Churn",data=df)
plt.title("Contract Type vs Churn")
plt.show()
plt.savefig("../output/contract_vs_churn.png")
plt.close()

#Tenure vs Churn
plt.figure(figsize=(7,5))
sns.boxplot(x="Churn",y="tenure",data=df)
plt.title("Tenure vs Churn")
plt.show()
plt.savefig("../output/tenure_vs_churn.png")
plt.close()

#Monthly Charges vs Churn
plt.figure(figsize=(7,5))
sns.boxplot(x="Churn",y="MonthlyCharges",data=df)
plt.title("Monthly Charges vs Churn")
plt.show()
plt.savefig("../output/monthlycharges_vs_churn.png")
plt.close()

#Payment Method vs Churn
plt.figure(figsize=(10,5))
sns.countplot(
    x="PaymentMethod",
    hue='Churn',
    data=df
)
plt.title("Payment Method vs Churn")
plt.show()
plt.savefig("../output/paymentmethod_vs_churn.png")
plt.close()

#Correlation Heatmap
numeric_cols = [
    "tenure",
    "MonthlyCharges",
    "TotalCharges"
]

corr_matrix = df[numeric_cols].corr()

plt.figure(figsize=(6,4))
sns.heatmap(
    corr_matrix,
    annot=True,
    cmap="coolwarm"
)
plt.title("Correlation Heatmap")
plt.show()
plt.savefig("../output/correlation_heatmap.png")
plt.close()