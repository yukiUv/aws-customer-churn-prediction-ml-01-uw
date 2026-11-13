
from pathlib import Path
import json

import boto3
import joblib
import pandas as pd


# 1. Project configuration


ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"
INPUT_PATH = ROOT / "data" / "input" / "new_customer.csv"

MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
METRICS_PATH = REPORT_DIR / "metrics.json"
OUTPUT_PATH = REPORT_DIR / "churn_predictions.csv"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR.mkdir(parents=True, exist_ok=True)



# 2. AWS S3 configuration


BUCKET_NAME = "aws-ml-churn-945461161707"
REGION = "ap-southeast-2"
PROFILE = "churn-project"

session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)

s3 = session.client("s3")



# 3. Download model and metrics from S3


s3.download_file(
    BUCKET_NAME,
    "models/churn_pipeline.joblib",
    str(MODEL_PATH),
)

s3.download_file(
    BUCKET_NAME,
    "reports/metrics.json",
    str(METRICS_PATH),
)

print("Model and metrics downloaded successfully from S3.")



# 4. Load the model and decision threshold


model = joblib.load(MODEL_PATH)

with open(METRICS_PATH, encoding="utf-8") as file:
    metrics = json.load(file)

threshold = float(metrics["selected_threshold"])



# 5. Load customer data from CSV


if not INPUT_PATH.exists():
    raise FileNotFoundError(
        f"Customer input file not found: {INPUT_PATH}"
    )

customer = pd.read_csv(INPUT_PATH)

if customer.empty:
    raise ValueError("The customer CSV file contains no records.")

# Validate the input features against the trained model.

expected_features = list(model.feature_names_in_)
missing_features = [
    column for column in expected_features
    if column not in customer.columns
]

if missing_features:
    raise ValueError(
        f"Missing required customer features: {missing_features}"
    )

# Keep the exact feature order used during model training.
customer = customer[expected_features]

print("Customer data loaded successfully.")
print(f"Number of customers: {len(customer)}")



# 6. Predict churn for every customer


churn_probabilities = model.predict_proba(customer)[:, 1]

predictions = (churn_probabilities >= threshold).astype(int)



# 7. Create and save the prediction report


results = customer.copy()

results["ChurnProbability"] = churn_probabilities
results["PredictedChurn"] = [
    "Churn" if prediction == 1 else "No churn"
    for prediction in predictions
]

results.to_csv(OUTPUT_PATH, index=False)



# 8. Display the results


print(f"Decision threshold: {threshold:.2f}")
print(f"Predictions saved to: {OUTPUT_PATH}")

print(
    results[
        ["ChurnProbability", "PredictedChurn"]
    ].to_string(index=False, formatters={
        "ChurnProbability": "{:.1%}".format
    })
)
