# app.py
import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import OneHotEncoder
import joblib

st.set_page_config(page_title="House Price Predictor", layout="centered")

st.title("House Price Prediction using Smart Regression Techniques")

# Load dataset
@st.cache_data
def load_data():
    df = pd.read_csv("https://raw.githubusercontent.com/selva86/datasets/master/AmesHousing.csv")
    df = df.select_dtypes(include=[np.number]).dropna()
    return df

df = load_data()

# Feature and target selection
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train XGBoost model
@st.cache_resource
def train_model():
    model = xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=5, random_state=42)
    model.fit(X_train, y_train)
    return model

model = train_model()

# Evaluation (optional display)
rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
r2 = r2_score(y_test, model.predict(X_test))

with st.expander("📊 Model Performance"):
    st.write(f"**RMSE**: {rmse:.2f}")
    st.write(f"**R² Score**: {r2:.2f}")

# Create input UI based on top features
top_features = ["GrLivArea", "OverallQual", "TotalBsmtSF", "GarageCars", "YearBuilt"]

st.markdown("### 🔧 Enter House Features")
input_data = {}
for feature in top_features:
    min_val = int(X[feature].min())
    max_val = int(X[feature].max())
    default = int(X[feature].median())
    input_data[feature] = st.slider(feature, min_val, max_val, default)

# Predict button
if st.button("Predict Sale Price"):
    input_df = pd.DataFrame([input_data])
    prediction = model.predict(input_df)[0]
    st.success(f Estimated Sale Price: ${int(prediction):,}")

# Footer
st.caption("Built using Streamlit and XGBoost")
poi
