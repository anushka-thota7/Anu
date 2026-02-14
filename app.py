import streamlit as st
import pandas as pd
import numpy as np
import joblib

# ===============================
# 🎯 PAGE CONFIG
# ===============================
st.set_page_config(page_title="🏠 House Rent Prediction", layout="centered")
st.title("🏠 House Rent Prediction App")
st.markdown("Use this app to predict **House Rent Price** based on Hyderabad data.")

# ===============================
# 📦 LOAD MODEL & SCALER
# ===============================
try:
    model = joblib.load("house_rent_linear_model.pkl")
    scaler = joblib.load("scaler.pkl")
    st.success("✅ Model and Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading model/scaler: {e}")
    st.stop()

# ===============================
# 📚 LOAD TRAINING DATA TO MATCH STRUCTURE
# ===============================
try:
    data = pd.read_csv("Hyderabad_House_Data.csv")

    # Clean columns same as your training file
    data["Washrooms"] = pd.to_numeric(data["Washrooms"], errors="coerce")
    data["Washrooms"].fillna(data["Washrooms"].median(), inplace=True)

    data["Area"] = data["Area"].astype(str).str.extract(r"(\d+)").astype(float)
    data["Area"].fillna(data["Area"].median(), inplace=True)

    data["Price"] = (
        data["Price"]
        .astype(str)
        .str.replace(r"[^\d]", "", regex=True)
        .astype(float)
    )

    X = data.drop("Price", axis=1)
    X = pd.get_dummies(X, drop_first=True)
    training_columns = X.columns.tolist()

except Exception as e:
    st.error(f"⚠ Could not load dataset or recreate training columns: {e}")
    st.stop()

# ===============================
# 🧩 USER INPUTS
# ===============================
st.subheader("Enter House Details")

area = st.number_input("Area (in sq. ft.)", min_value=100, max_value=10000, value=1400)
washrooms = st.number_input("Number of Washrooms", min_value=1, max_value=10, value=2)

# Extract all unique bedroom types from your dataset
bedroom_options = sorted(data["Bedrooms"].dropna().unique().tolist())
bedrooms = st.selectbox("Select Bedroom Type", bedroom_options)

# ===============================
# 🔍 PREDICTION SECTION
# ===============================
if st.button("🔮 Predict Rent"):
    try:
        # Create input as DataFrame
        new_data = pd.DataFrame({
            "Area": [area],
            "Bedrooms": [bedrooms],
            "Washrooms": [washrooms]
        })

        # Apply same dummies and column alignment
        new_data = pd.get_dummies(new_data, drop_first=True)
        new_data = new_data.reindex(columns=training_columns, fill_value=0)

        # Scale and predict
        new_scaled = scaler.transform(new_data)
        predicted_price = model.predict(new_scaled)[0]

        st.success(f"💰 **Predicted House Rent Price: ₹{predicted_price:,.2f}**")

    except Exception as e:
        st.error(f"❌ Prediction failed: {e}")

# ===============================
# 📈 FOOTER
# ===============================
st.markdown("---")
st.markdown("""
**Developed by:** Thota Anushka  
**Model:** Linear Regression  
**Tools Used:** Python, Pandas, Scikit-learn, Streamlit
""")
