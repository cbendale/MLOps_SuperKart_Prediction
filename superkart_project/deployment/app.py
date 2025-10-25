import os
import joblib
import streamlit as st
import pandas as pd
from datetime import date
from huggingface_hub import hf_hub_download

# ================================
# App Title & Description
# ================================
st.set_page_config(page_title="SuperKart Sales Prediction", layout="centered")

st.title("SuperKart Sales Prediction App")
st.markdown("Provide store and product details below to predict expected sales.")

# ================================
# Load Model from Hugging Face Hub
# ================================
@st.cache_resource(show_spinner=True)
def load_model():
    model_path = hf_hub_download(
        repo_id="cbendale10/MLOps-SuperKart-Prediction-model",
        filename="best_superkart_prediction_model_v1.joblib",
        repo_type="model"
    )
    return joblib.load(model_path)

model = load_model()

# ==================================================
# Input Form : Collect input features from the user
# ==================================================
product_weight = st.number_input("Product Weight", min_value=0.0, step=0.1, value=12.5)
product_sugar_content = st.selectbox("Product Sugar Content", ["low sugar", "no sugar", "regular"])
product_allocated_area = st.number_input("Product Allocated Area", min_value=0.0, step=0.01, value=0.05)
product_type = st.selectbox(
    "Product Type",
    [
        "fruits and vegetables", "dairy", "baking goods", "bread", "breakfast", "canned", "meat",
        "household", "frozen foods", "snack foods", "soft drinks", "hard drinks",
        "health and hygiene", "others", "seafood", "starchy foods"
    ]
)
product_mrp = st.number_input("Product MRP", min_value=0.0, step=1.0, value=150.0)
store_id = st.selectbox("Store ID", ["OUT001", "OUT002", "OUT003", "OUT004"])
store_establishment_year = st.selectbox("Store Establishment Year", list(range(1987, 2020)))
store_size = st.selectbox("Store Size", ["Low", "Medium", "High"])
store_location_city_type = st.selectbox("Store Location Type", ["Tier 1", "Tier 2", "Tier 3"])
store_type = st.selectbox("Store Type", ["Departmental Store", "Supermarket Type 1", "Supermarket Type 2", "Food Mart"])

# Package into a dictionary and convert to DataFrame
input_data = pd.DataFrame([{
    "Product_Weight": product_weight,
    "Product_Sugar_Content": product_sugar_content,
    "Product_Allocated_Area": product_allocated_area,
    "Product_Type": product_type,
    "Product_MRP": product_mrp,
    "Store_Id": store_id,
    "Store_Establishment_Year": store_establishment_year,
    "Store_Size": store_size,
    "Store_Location_City_Type": store_location_city_type,
    "Store_Type": store_type
}])

# Compute Store_Age (the model expects this numeric feature)
input_data["Store_Age"] = date.today().year - input_data["Store_Establishment_Year"]

# Predict button
if st.button("Predict"):
    try:
        # The saved object is a Pipeline(preprocessor -> XGBRegressor), so we can call predict directly
        y_pred = model.predict(input_data)[0]
        st.success(f"Predicted Sales: {y_pred:,.2f}")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
