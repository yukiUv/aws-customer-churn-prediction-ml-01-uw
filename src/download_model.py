from pathlib import Path

import boto3

ROOT = Path(__file__).resolve().parent.parent

BUCKET_NAME = "aws-ml-churn-945461161707"
REGION = "ap-southeast-2"
PROFILE = "churn-project"

MODEL_KEY = "models/churn_pipeline.joblib"
METRICS_KEY= "reports/metrics.json"


session = boto3.Session(
    profile_name=PROFILE,
    region_name=REGION,
)


s3 = session.client("s3")

(ROOT / "models").mkdir(parents=True,exist_ok=True)
(ROOT / "reports").mkdir(parents=True,exist_ok=True)

s3.download_file(
    BUCKET_NAME,
    MODEL_KEY,
    str(ROOT / "models" / "churn_pipeline.joblib"),
)

s3.download_file(
    BUCKET_NAME,
    METRICS_KEY,
    str(ROOT / "reports" / "metrics.json"),
)

print("Model and metrics downloaded successfully from S#.")