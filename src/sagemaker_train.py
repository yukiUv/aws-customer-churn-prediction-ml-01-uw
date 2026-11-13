import os
import json
from pathlib import Path

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

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def main():
    
    ROOT = Path(__file__).resolve().parents[1]

    TRAIN_DIR = Path(
        os.environ.get(
            "SM_CHANNEL_TRAIN",
            str(ROOT / "data" / "raw"),
        )
    )

    MODEL_DIR = Path(
        os.environ.get(
            "SM_MODEL_DIR",
            str(ROOT / "models"),
        )
    )

    OUTPUT_DIR = Path(
        os.environ.get(
            "SM_OUTPUT_DATA_DIR",
            str(ROOT / "reports"),
        )
    )

    DATA_PATH = TRAIN_DIR / "telco-churn.csv"
    MODEL_PATH = MODEL_DIR / "churn_pipeline.joblib"
    METRICS_PATH = OUTPUT_DIR / "metrics.json"

    # create output directories if they do  not exist.
    Path(MODEL_DIR).mkdir(parents=True,exist_ok=True)
    Path(OUTPUT_DIR).mkdir(parents=True,exist_ok=True)

    # load the training directories if they do not exist.
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset load successfully: {df.shape[0]} rows, {df.shape[1]} columns")

    # validate required columns
    required_columns = ["customerID", "TotalCharges","Churn"]

    missing_colums = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing_colums:
        raise ValueError(
            f"Missing required columns: {missing_colums}"
        )
        
    # prepare the data
    df["TotalCharges"] = pd.to_numeric(
        df["TotalCharges"],
        errors="coerce"
    )

    df = df.dropna(subset=["Churn"])

    # separate feature and target
    X = df.drop(columns=["Churn", "customerID"])
    y = df["Churn"].map({"No":0, "Yes":1})

    if y.isna().any():
        raise ValueError("Unexpected values found in the Churn column.")

    print(f"Feature prepared: {X.shape[1]} columns")
    print(f"Churn distribution:\n{y.value_counts()}")

    # Split the dataset into training , validation, and testing sets.

    X_train,X_temp,y_train,y_temp = train_test_split(
        X,
        y,
        test_size= 0.40,
        random_state= 42,
        stratify= y,
    )

    X_val,X_test,y_val,y_test = train_test_split(
        X_temp,
        y_temp,
        test_size= 0.50,
        random_state = 42,
        stratify = y_temp,
    )

    print(f"Training rows: {len(X_train)}")
    print(f"Validation rows: {len(X_val)}")
    print(f"Testsing rows: {len(X_test)}")

    # identify numeric and categorical columns.

    numeric_features = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_features = X.select_dtypes(
        include=["object", "category", "bool"]
    ).columns.tolist()

    print("Numeric features:", numeric_features)
    print("Categorical features:", categorical_features)

    # preprocess numeric features.

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer",SimpleImputer(strategy="median")),
            ("scaler",StandardScaler()),
        ]
    )

    # preprocess categorical features.
    categorical_pipline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
            
        ]
    )

    # combine both preprocessing pipelines.
    preprocessor =ColumnTransformer(
        transformers=[
            ("numeric",numeric_pipeline,numeric_features),
            ("categorical",categorical_pipline,categorical_features),
        ]
    )

    print("Data spliting and preprocessing setup completed.")

    # build the machine learning pipeline.

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight= "balanced",
                    ),
            ),
        ]
    )

    # train the model

    print("Starting model training....")

    model.fit(X_train,y_train)

    print("Model training completed successfullty.")

    # Get churn probabilities for validation data.
    val_probabilites = model.predict_proba(X_val)[:,1]

    # find churn probabilities for validation data
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70]

    best_threshold = 0.50
    best_f1 = 0.0

    for threshold in thresholds:
        val_predictions =(
            val_probabilites >= threshold
        ).astype(int)
        
        current_f1 = f1_score(
            y_val,
            val_predictions,
        )
        
        print(
            f"Threshold: {threshold:.2f} | "
            f"Validations F1: {current_f1:.4f}"
        )
        
        if current_f1 > best_f1:
            best_f1 = current_f1
            best_threshold = threshold
            
    print(f"Selected threshold: {best_threshold:.2f}")
    print(f"Best validation F!-score: {best_f1:.4f}")

    # evaluate the model using the test dataset.
    test_probabilities = model.predict_proba(X_test)[:,1]

    test_predictions = (
        test_probabilities >= best_threshold
    ).astype(int)

    # calculate evaluation metrics.
    accuracy = accuracy_score(y_test,test_predictions)
    precision = precision_score(y_test,test_predictions,zero_division=0)
    recall = recall_score(y_test,test_predictions,zero_division=0)
    f1 = f1_score(y_test,test_predictions)
    roc_auc = roc_auc_score(y_test,test_probabilities)

    conf_matrix = confusion_matrix(y_test,test_predictions)

    # display evaluation results.
    print("\nFinal Test Results:")
    print("---------------------")
    print(f"Accuracy: {accuracy:.6f}")
    print(f"Precision: {precision:.6f}")
    print(f"Recall: {recall:.6f}")
    print(f"F1: {f1:.6f}")
    print(f"ROC_AUC: {roc_auc:.6f}")


    print("\nConf Matrix:")
    print(conf_matrix)

    print("\nClassification Report: ")
    print(classification_report(y_test,test_predictions))

    # prepare metrics for saving.
    metrics ={
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "roc_auc": float(roc_auc),
        "selected_threshold": float(best_threshold),
        "validation_f1": float(best_f1),
        "confusion_matrix": conf_matrix.tolist(),
        "train_rows": int(len(X_train)),
        "validation_rows": int(len(X_val)),
        "test_rows": int(len(X_test)),
    }

    # save the trained model.
    joblib.dump(model,MODEL_PATH)

    # save evaluation metrics.
    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics,file, indent=2)
        
        
    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Metrics saved to: {METRICS_PATH}")
    

if __name__ == "__main__":
    main()