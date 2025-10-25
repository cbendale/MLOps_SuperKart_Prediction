# for data manipulation
import os
import pandas as pd
from sklearn.model_selection import train_test_split

# for Hugging Face dataset access & uploads
from huggingface_hub import HfApi, hf_hub_download

# Config (env-driven) get HF_TOKEN from env variables
HF_TOKEN = os.getenv("HF_TOKEN")
if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN not found in environment. ")

api = HfApi(token=HF_TOKEN)

# Load dataset from HF

HF_DATASET_URL = f"hf://datasets/cbendale10/MLOps-SuperKart-Prediction/SuperKart.csv"
df = pd.read_csv(HF_DATASET_URL)

print("Original shape:", df.shape)

# Minimal cleaning per schema

TARGET_COL = "Product_Store_Sales_Total"
ID_COLS = ["Product_Id"] 


# 1) Drop identifier not used for modeling
if "Product_Id" in df.columns:
    df.drop(columns=["Product_Id"], inplace=True, errors="ignore")


# 2) fix Product_Sugar_Content typo
if "Product_Sugar_Content" in df.columns:
    df.replace({"Product_Sugar_Content": "reg"}, "Regular", inplace=True)

# 3) fix Store_Establishment_Year by transformed into Store_Age = Current Year - Establishment Year.
from datetime import date
df['Store_Age'] = date.today().year - df['Store_Establishment_Year']
df.drop(['Store_Establishment_Year'], axis=1, inplace=True)

# Drop duplicates
before = len(df)
df.drop_duplicates(inplace=True)
print(f"Dropped {before - len(df)} duplicate rows.")


NUMERIC_COLS = ['Product_Weight', 'Product_Allocated_Area', 'Product_MRP', 'Store_Age', 'Product_Store_Sales_Total']
CATEGORICAL_COLS = ['Product_Sugar_Content', 'Product_Type', 'Store_Size', 'Store_Location_City_Type', 'Store_Type']

# ---------------------------
# Split into X/y and train/test
# ---------------------------
X = df.drop(columns=[TARGET_COL])
y = df[TARGET_COL]


Xtrain, Xtest, ytrain, ytest = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print("\nShapes after split:")
print("Xtrain:", Xtrain.shape, "Xtest:", Xtest.shape, "ytrain:", ytrain.shape, "ytest:", ytest.shape)


# Save locally (CSV)

out_dir = "superkart_project/data"

Xtrain_path = os.path.join(out_dir, "Xtrain.csv")
Xtest_path  = os.path.join(out_dir, "Xtest.csv")
ytrain_path = os.path.join(out_dir, "ytrain.csv")
ytest_path  = os.path.join(out_dir, "ytest.csv")

Xtrain.to_csv(Xtrain_path, index=False)
Xtest.to_csv(Xtest_path, index=False)
ytrain.to_csv(ytrain_path, index=False)
ytest.to_csv(ytest_path, index=False)

print("\nSaved CSVs:")
for p in [Xtrain_path, Xtest_path, ytrain_path, ytest_path]:
    print(" -", p)

# ---------------------------
# Upload split files back to HF dataset repo
# (mirroring your sample: upload with just the filename at repo root)
# ---------------------------
files_to_upload = [Xtrain_path, Xtest_path, ytrain_path, ytest_path]

for file_path in files_to_upload:
    api.upload_file(
        path_or_fileobj=file_path,
        path_in_repo=os.path.basename(file_path),  # just the filename
        repo_id="cbendale10/MLOps-SuperKart-Prediction",
        repo_type="dataset",
    )

print(f"\nUploaded splits to dataset repo: cbendale10/MLOps-SuperKart-Prediction")
