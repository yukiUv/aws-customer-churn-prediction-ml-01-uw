# Customer Churn Prediction

An end-to-end machine learning project that predicts customer churn using Python and scikit-learn, with AWS S3 artifact storage and a prepared Amazon SageMaker training workflow.

> **Project status:** The local scikit-learn training and prediction workflow is implemented. AWS S3 integration is implemented. The SageMaker training request was prepared and inspected with a dry run; a managed SageMaker training job has **not** been confirmed as successfully completed.

## Overview

Customer churn occurs when a customer stops using a service. Identifying customers who may churn can help a business prioritize retention efforts.

This project builds a binary classification pipeline using the Telco Customer Churn dataset format. It includes:

- Data inspection and validation
- Numeric and categorical preprocessing
- Logistic Regression with class weighting
- Train / validation / test splitting
- Decision-threshold selection using validation F1-score
- Evaluation on a held-out test set
- Model serialization with Joblib
- Model and metrics download from Amazon S3
- Batch predictions for new customer records
- A prepared SageMaker training configuration

## Repository structure

```text
.
├── data/
│   └── input/
│       └── new_customer.csv       # Safe sample input for batch prediction
├── docs/
│   └── architecture.md            # System architecture
├── reports/
│   └── metrics.json               # Committed evaluation snapshot
├── src/
│   ├── download_model.py           # Download model and metrics from S3
│   ├── inspect_data.py             # Inspect the source dataset
│   ├── predict.py                 # Batch churn prediction using S3 artifacts
│   ├── run_sagemaker.py            # Prepare and inspect a SageMaker request
│   ├── sagemaker_train.py          # SageMaker-compatible training entry point
│   └── train.py                    # Local training and evaluation
├── .env.example                    # Environment-variable template
├── .gitignore
└── requirements.txt
```

The raw training dataset and trained model artifacts are intentionally not committed. The training scripts expect the raw CSV at `data/raw/telco-churn.csv`.

## Machine learning workflow

1. Load the Telco churn CSV and validate required columns.
2. Convert `TotalCharges` to numeric values and handle missing values through preprocessing pipelines.
3. Remove `customerID` from the model features.
4. Encode the target: `No = 0`, `Yes = 1`.
5. Split data into training, validation, and test sets using stratification.
6. Impute numeric features with the median and standardize them.
7. Impute categorical features with the most frequent value and one-hot encode them.
8. Train a class-weighted Logistic Regression model.
9. Select a probability threshold using validation-set F1-score.
10. Evaluate once on the held-out test set and save the model and metrics.

The default local training split is 60% training, 20% validation, and 20% test. The selected threshold is a decision-policy choice; it should be reviewed against business costs such as the cost of missing a churner versus the cost of contacting a customer who would not churn.

## Evaluation snapshot

The currently committed `reports/metrics.json` contains the following snapshot:

| Metric    | Value  |
|---|---:   |        |
| Accuracy  | 76.44% |
| Precision | 54.22% |
| Recall    | 72.19% |
| F1-score  | 61.93% |
| ROC-AUC   | 83.21% |
| Selected  |        |
threshold   | 0.60 |

These values describe the committed metrics file, not a guarantee that a fresh training run will reproduce identical results. Re-run the training script to generate a new report for your local dataset and environment.

## Requirements

- Python 3.10 or later (the project was developed with Python 3.12)
- pip
- Git
- For S3 workflows: an AWS account, AWS CLI, and permission to access the configured S3 bucket
- For SageMaker workflows: a SageMaker execution role and the required account-level permissions

Install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Prepare the dataset

Obtain a lawful copy of the Telco Customer Churn dataset and place it at:

```text
data/raw/telco-churn.csv
```

Do not commit raw customer-level data to a public repository. Confirm that you have the right to use and redistribute any dataset before sharing it.

Inspect the data:

```powershell
python src/inspect_data.py
```

## Train locally

```powershell
python src/train.py
```

Expected generated artifacts:

```text
models/churn_pipeline.joblib
reports/metrics.json
```

The `models/` directory and local model artifacts are excluded by `.gitignore`.

## Run batch predictions using S3 artifacts

The prediction scripts read AWS settings from environment variables. Copy `.env.example` to `.env` for your own reference, but note that PowerShell does not automatically load `.env` files. Set the variables in your current PowerShell session:

```powershell
$env:S3_BUCKET = "your-s3-bucket-name"
$env:AWS_REGION = "ap-southeast-2"
$env:AWS_PROFILE = "your-aws-cli-profile"
```

Ensure your AWS CLI profile is configured and has permission to download the model and metrics objects. Then run:

```powershell
python src/download_model.py
python src/predict.py
```

`predict.py` reads `data/input/new_customer.csv` and writes predictions to:

```text
reports/churn_predictions.csv
```

The output contains the input features, estimated churn probability, and predicted class. Do not use real personal customer information in a public sample file.

## SageMaker workflow

The repository includes:

- `src/sagemaker_train.py`: a SageMaker-compatible training entry point.
- `src/run_sagemaker.py`: builds and inspects a SageMaker training request.

Set the required environment variables before preparing the request:

```powershell
$env:S3_BUCKET = "your-s3-bucket-name"
$env:AWS_REGION = "ap-southeast-2"
$env:AWS_PROFILE = "your-aws-cli-profile"
$env:SAGEMAKER_ROLE_ARN = "arn:aws:iam::YOUR_ACCOUNT_ID:role/YOUR_SAGEMAKER_ROLE"
```

Then run:

```powershell
python src/run_sagemaker.py
```

This script inspects the training configuration; it does not submit a training job. A successful dry run is not proof that a managed training job can be launched. Actual execution requires the relevant IAM, service-role, EC2 networking, ECR, S3, and SageMaker permissions. Verify costs and permissions before submitting a managed training job.

## Configuration and security

- Never commit AWS access keys, secret keys, session tokens, private keys, `.env` files, or personal AWS configuration.
- Use named AWS CLI profiles or IAM roles rather than embedding credentials in source code.
- Keep raw customer datasets and generated prediction outputs out of public Git history.
- Use `.env.example` only for placeholder variable names and example values.
- If a credential has ever been committed, rotate or revoke it immediately. Removing it from the latest commit does not remove it from Git history.

## Limitations and next improvements

- The current model is a Logistic Regression baseline, not a production churn decision system.
- Threshold selection currently optimizes validation F1-score; a business-specific cost function may be more appropriate.
- Validate performance across time-based splits and customer segments before deployment.
- Add automated tests, experiment tracking, model versioning, drift monitoring, and a deployment endpoint before production use.
- The SageMaker workflow remains subject to AWS account permissions and has not been represented as a completed managed training run.

## License

No license is currently specified. Add a license only after deciding how you want others to use, modify, and redistribute this project.
