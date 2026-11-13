from pathlib import Path
import pandas as pd

DATA_PATH = (
    
    Path(__file__).resolve().parents[1]
    / "data"
    / "raw"
    / "telco-churn.csv"
)

df = pd.read_csv(DATA_PATH)

print("Dataset shape:", df.shape)
print("\nFirst five rows:")
print(df.head())

print("\mColumn data types:")
print(df.dtypes)

print("\nMissing values:")
print(df.isna().sum().sort_values(ascending=False).head(10))

print("\nChurn distributions :")
print(df["Churn"].value_counts(dropna=False))

print("\nChurn distribution :")
print(df["Churn"].value_counts(normalize=True, dropna=False).round(3))

print("\nDuplicate customer IDs:",
      df["customerID"].duplicated().sum())