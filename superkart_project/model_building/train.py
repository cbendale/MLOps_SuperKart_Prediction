# for data manipulation
import os, json, math
from pathlib import Path
import numpy as np
import pandas as pd

# Scikit-learn
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# ── Model
import xgboost as xgb
import joblib
import mlflow

#   Hugging Face Hub
from huggingface_hub import HfApi, create_repo, hf_hub_download
from huggingface_hub.utils import RepositoryNotFoundError

# MLflow tracking
mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MLOps_SuperKart_PROD_experiment_1")

api = HfApi()

# -----------------------
# Load train/test from HF Datasets repo
# -----------------------

Xtrain_path = "hf://datasets/cbendale10/MLOps-SuperKart-Prediction/Xtrain.csv"
Xtest_path = "hf://datasets/cbendale10/MLOps-SuperKart-Prediction/Xtest.csv"
ytrain_path = "hf://datasets/cbendale10/MLOps-SuperKart-Prediction/ytrain.csv"
ytest_path = "hf://datasets/cbendale10/MLOps-SuperKart-Prediction/ytest.csv"

X_train = pd.read_csv(Xtrain_path)
X_test = pd.read_csv(Xtest_path)
y_train = pd.read_csv(ytrain_path)
y_test = pd.read_csv(ytest_path)


# -----------------------
# Column groups
# -----------------------
NUMERIC_COLS = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age']
CATEGORICAL_COLS = ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

# Restrict to columns present post-split (safety)
NUM_COLS = [c for c in NUMERIC_COLS if c in X_train.columns]
CAT_COLS = [c for c in CATEGORICAL_COLS if c in X_train.columns]

# -----------------------
# Preprocessor
# -----------------------
preprocessor = make_column_transformer(
    (make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    ), NUM_COLS),
    (make_pipeline(
        SimpleImputer(strategy="most_frequent"),
        OneHotEncoder(handle_unknown="ignore")
    ), CAT_COLS),
)

# -----------------------
# Regressor (XGBoost)
# -----------------------
xgb_model = xgb.XGBRegressor(
    objective="reg:squarederror",
    random_state=42,
    n_jobs=-1,
    tree_method="hist"
)

model_pipeline = make_pipeline(
    preprocessor,
    xgb_model,
)

#  prod — tune as needed
param_grid = {
    'xgbregressor__n_estimators': [100, 200, 300],
    'xgbregressor__max_depth': [3, 5, 7],
    'xgbregressor__learning_rate': [0.02, 0.05, 0.1],
    'xgbregressor__subsample': [0.8, 1.0],
    'xgbregressor__colsample_bytree': [0.8, 1.0],
    'xgbregressor__reg_lambda': [1.0, 2.0],
}

# -----------------------
# TRAIN + TUNE + LOG
# -----------------------
def rmse(y_true, y_pred):
    return float(np.sqrt(mean_squared_error(y_true, y_pred)))

with mlflow.start_run():
    gs = GridSearchCV(
        estimator=model_pipeline,
        param_grid=param_grid,
        cv=5,
        n_jobs=-1,
        scoring="neg_root_mean_squared_error",
        verbose=0
    )
    gs.fit(X_train, y_train)

    best_model = gs.best_estimator_
    mlflow.log_params(gs.best_params_)

    # Evaluate on train/test
    y_pred_tr = best_model.predict(X_train)
    y_pred_te = best_model.predict(X_test)

    metrics = {
        "train_rmse": rmse(y_train, y_pred_tr),
        "train_mae": float(mean_absolute_error(y_train, y_pred_tr)),
        "train_r2": float(r2_score(y_train, y_pred_tr)),
        "test_rmse": rmse(y_test, y_pred_te),
        "test_mae": float(mean_absolute_error(y_test, y_pred_te)),
        "test_r2": float(r2_score(y_test, y_pred_te)),
    }

    for k, v in metrics.items():
        if v is not None and not (math.isnan(v) or math.isinf(v)):
            mlflow.log_metric(k, v)

    # Save the model
    model_path = "best_superkart_prediction_model_v1.joblib"
    joblib.dump(best_model, model_path)

    # Log the model artifact
    mlflow.log_artifact(model_path, artifact_path="model")
    print(f"Model saved as artifact at: {model_path}")

    # Upload to Hugging Face Model Hub
    repo_id = "cbendale10/MLOps-SuperKart-Prediction-model"
    repo_type = "model"

    try:
        api.repo_info(repo_id=repo_id, repo_type=repo_type)
        print(f"Model repo '{repo_id}' already exists. Using it.")
    except RepositoryNotFoundError:
        print(f"Model repo '{repo_id}' not found. Creating new repo...")
        create_repo(repo_id=repo_id, repo_type=repo_type, private=False)
        print(f"Repo '{repo_id}' created.")

    api.upload_file(
        path_or_fileobj=model_path,
        path_in_repo=model_path,
        repo_id=repo_id,
        repo_type=repo_type,
    )
