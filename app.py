import streamlit as st
import pandas as pd
import joblib
import numpy as np
# from tensorflow.keras.models import load_model


# Load data for visualization
data1 = pd.read_csv("hehe.csv")
data2 = pd.read_csv("hehe2.csv")
data = pd.concat([data1, data2], ignore_index=True)

data.fillna(0, inplace=True)

ml_data = data.copy()
ml_data.rename(columns={
    'Start Year': 'Year'
}, inplace=True)

# -----------------------------
# LOAD MODELS
# -----------------------------
rf_model = joblib.load("rf_model.pkl")
xgb_model = joblib.load("xgb_model.pkl")
scaler = joblib.load("scaler.pkl")

le1 = joblib.load("le_continent.pkl")
le2 = joblib.load("le_type.pkl")
le3 = joblib.load("le_subtype.pkl")

# lstm_model = load_model("lstm_model.h5")
# lstm_scaler = joblib.load("lstm_scaler.pkl")

# -----------------------------
# UI
# -----------------------------
st.title("🌍 Natural Disaster Prediction System")

# -----------------------------
# USER INPUT
# -----------------------------
year = st.number_input("Year", 1900, 2100, 2024)
month = st.number_input("Month", 1, 12, 6)

continent = st.selectbox("Continent", le1.classes_)
dtype = st.selectbox("Disaster Type", le2.classes_)
subtype = st.selectbox("Disaster Subtype", le3.classes_)

lat = st.number_input("Latitude", value=0.0)
lon = st.number_input("Longitude", value=0.0)

mag = st.number_input("Magnitude", value=1.0)
affected = st.number_input("Total Affected", value=1000)
damage = st.number_input("Total Damage", value=1000)

# -----------------------------
# PREDICT
# -----------------------------
if st.button("Predict Risk"):

    input_data = pd.DataFrame([{
        'Year': year,
        'Month': month,
        'Continent': le1.transform([continent])[0],
        'Disaster Type': le2.transform([dtype])[0],
        'Disaster Subtype': le3.transform([subtype])[0],
        'Latitude': lat,
        'Longitude': lon,
        'Dis Mag Value': mag,
        'Total Affected': affected,
        'Total Damage': damage
    }])

    input_scaled = scaler.transform(input_data)

    rf_pred = rf_model.predict(input_scaled)[0]
    xgb_pred = xgb_model.predict(input_scaled)[0]

    labels = ["Low", "Medium", "High"]

    st.success(f"🌳 RF Prediction: {labels[rf_pred]}")
    st.success(f"⚡ XGB Prediction: {labels[xgb_pred]}")

    # ==============================
# 🌍 MAP VISUALIZATION
# ==============================
st.header("🌍 Disaster Location Map")

map_data = pd.DataFrame({
    'lat': [lat],
    'lon': [lon]
})

st.map(map_data)


# ==============================
# 📊 YEARLY TREND GRAPH
# ==============================
st.header("📊 Disaster Trend Over Years")

yearly = ml_data.groupby("Year").size()

st.line_chart(yearly)


# ==============================
# 🌪️ TOP DISASTER TYPES
# ==============================
st.header("🌪️ Most Common Disaster Types")

type_counts = data['Disaster Type'].value_counts().head(10)

st.bar_chart(type_counts)
