from pathlib import Path
import json

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
)
from  sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "telco-churn.csv"
MODEL_DIR = ROOT / "models"
REPORT_DIR = ROOT / "reports"

MODEL_DIR.mkdir(parents=True,exist_ok=True)
REPORT_DIR.mkdir(parents=True,exist_ok=True)

# load and validate the dataset
df= pd.read_csv(DATA_PATH)

required_columns = {"customerID","TotalCharges","Churn"}
if not required_columns.issubset(df.columns):
    raise ValueError(
        f"Dataset is misssing reqired columns: "
        f"{required_columns - set(df.columns)}"
    )
    
# clean the numeric column
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"],
                                   errors="coerce")

# remove rows with missing target labels, if any 
df = df.dropna(subset=["Churn"]).copy()

# seperate features and target labels, if any
X = df.drop(columns=["Churn","customerID"])
y = df["Churn"].map({"No": 0, "Yes": 1})

if y.isna().any():
    raise ValueError("Unexpected values found in Churn column.")

# Identify numeric and categorical columns
numeric_columns = X.select_dtypes(
    include=["number"]
).columns.tolist()

categorical_columns = X.select_dtypes(
    exclude=["number"]
).columns.tolist()

# prepare numeric and categorical data seperately
numeric_pipline = Pipeline([
    ("Imputer",
     SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipline = Pipeline([
    ("imputer",
     SimpleImputer(strategy="most_frequent")),
    ("encoder",
     OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("numeric",numeric_pipline,
     numeric_columns),
    ("categorical",categorical_pipline,categorical_columns)
])

# create a leakage-safe train/ test split
X_train_val,X_test,y_train_val,y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state= 42,
    stratify= y,
)

# reserve 25% of the remaining 80% for validatetion.
# final proporations : 60% train, 20% validation, 20% test.
X_train, X_val ,y_train, y_val = train_test_split(
    X_train_val,
    y_train_val,
    test_size=0.25,
    random_state=42,
    stratify= y_train_val,
)


# train using the training set only
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000,
                                      class_weight="balanced",
                                      )),
])

model.fit(X_train,y_train)

# select a threshold using validation data
val_probabilities =model.predict_proba(X_val) [:,1]

threshold_results = []

for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]:
    val_predictions = (
        val_probabilities >= threshold
    ).astype(int)
    
    threshold_results.append({
        "threshold": threshold,
        "precision":precision_score(
            y_val, val_predictions,zero_division=0
        ),
        "recall": recall_score(
            y_val, val_predictions,zero_division=0
        ),
        "f1": f1_score(
            y_val, val_predictions,zero_division=0
        ),
    })

print("\nVALIDATION THRESHOLD COMPARISON")
print(
    f"{'Threshold':>10} {'Precision':>10}"
    f"{'Recall':>10}  {'F1':>10}"
)

for result in threshold_results:
    print(
        f"{result['threshold']:>10.2f}"
        f"{result['precision']:>10.3f}"
        f"{result['recall']:>10.3f}"
        f"{result['f1']:>10.3f}"
    )
# use f1 as our initial selection objective.
# This is a teaching baseline, not necessary the 
# correct business objective for every  problem
# save model and evaluation results.
best_result = max(
    threshold_results,
    key=lambda result: result['f1'],
)
best_threshold = best_result['threshold']

print("\nSelected threshold:", best_threshold)

# evaluate once on the untouched test set

test_probabilities = model.predict_proba(X_test)[:,1]
test_predictions = (
    test_probabilities >= best_threshold
).astype(int)

metrics = {
    "train_rows": int(len(X_train)),
    "validation_rows": int(len(X_val)),
    "test_rows":int(len(X_test)),
    "selected_threshold":float(best_threshold),
    "test_accuracy":float(accuracy_score(y_test, test_predictions)),
    
    "test_precision": float(
        precision_score(y_test, test_predictions, zero_division=0)
    ),
    "test_recall": float(
        recall_score(y_test,test_predictions, zero_division=0)
    ),
    "test_f1": float(
        f1_score(y_test,test_predictions,zero_division=0)
    ),
    "test_roc_auc": float(
        roc_auc_score(y_test,test_probabilities)
    ),
    "test_confusion_matrix": confusion_matrix(
        y_test, test_predictions
    ).tolist(),
    "validation_threshold_results": threshold_results,
}

# save model and metrics

joblib.dump(model, MODEL_DIR / "churn_pipeline.joblib")

with open(
    REPORT_DIR / "metrics.json", "w", encoding="utf-8"
) as f:
    json.dump(metrics, f , indent=2)
    
    
print("\nFinal Test Results")
for name , value in metrics.items():
    if name not in {
        "validation_threshold_results",
        "test_confusion_matrix",
    }:
        print(f"{name}: {value}")

print("\nTest confusion matrix:")
print(metrics["test_confusion_matrix"])

print("\nTest classification report:")
print(classification_report(
    y_test,
    test_predictions,
    target_names=["No churn" , "Churn"],
    zero_division=0,
))

print("\nSaved model:", MODEL_DIR / "churn_pipeline.joblib")
print("Saved metrics:", REPORT_DIR / "metrics.json")

