import streamlit as st
import pandas as pd
from get_prediction import (
    load_gzipped_model,
    predict_fire_with_usa_check,
    clamp_to_usa,
    is_usa_coordinate,
)

# -----------------------------
# Cached model loading
# -----------------------------
@st.cache_data
def load_model_cached(path: str):
    return load_gzipped_model(path)

MODEL_PATH = "models/wildfire_rf_balanced_classes_best_model.pkl.gz"
model = load_model_cached(MODEL_PATH)

# -----------------------------
# Page title & description
# -----------------------------
st.title("🌲 USA Wildfire Risk Predictor")
st.markdown(
    """
    This project predicts the likelihood of wildfires based on location and environmental features. 
    Enter latitude, longitude, and environmental data below, then click **Predict Wildfire Risk**. 
    The map shows the selected location (clamped to the USA if out of bounds).  
    Predictions include a risk class and probability, with the risk level displayed below in color.
    """
)

# -----------------------------
# Map section
# -----------------------------
# Default coordinates
default_lat, default_lon = 40.0, -100.0
location_df = pd.DataFrame({"lat": [default_lat], "lon": [default_lon]})
st.subheader("Selected Location on Map")
st.map(location_df, zoom=4, use_container_width=True)

# -----------------------------
# Input fields below the map
# -----------------------------
st.subheader("Input Features")
lat_input = st.number_input("Latitude", min_value=-90.0, max_value=90.0, value=default_lat)
lon_input = st.number_input("Longitude", min_value=-180.0, max_value=180.0, value=default_lon)
max_humidity = st.number_input("Max Humidity (%)", value=50)
max_temp = st.number_input("Max Temperature (°C)", value=30)
vapor_pressure = st.number_input("Vapor Pressure", value=10)
fuel_moisture_100h = st.number_input("Fuel Moisture 100h", value=5)
year = st.number_input("Year", min_value=2000, max_value=2030, value=2025)
month = st.number_input("Month", min_value=1, max_value=12, value=10)
day = st.number_input("Day", min_value=1, max_value=31, value=21)

features = [lat_input, lon_input, max_humidity, max_temp, vapor_pressure,
            fuel_moisture_100h, year, month, day]

# -----------------------------
# Predict button
# -----------------------------
if st.button("Predict Wildfire Risk"):
    # Clamp coordinates to USA
    lat, lon = features[0], features[1]
    if not is_usa_coordinate(lat, lon):
        lat, lon = clamp_to_usa(lat, lon)
        features[0], features[1] = lat, lon

    # Update map with clamped coordinates
    location_df = pd.DataFrame({"lat": [lat], "lon": [lon]})
    st.map(location_df, zoom=4, use_container_width=True)

    # Make prediction
    pred_class, pred_proba = predict_fire_with_usa_check(model, features)

    # Determine risk color and message
    if pred_class == 1:
        risk_color = "#FF0000"  # Red = high risk
        risk_message = "🔥 High Wildfire Risk"
    else:
        risk_color = "#00AA00"  # Green = low risk
        risk_message = "✅ Low Wildfire Risk"

    # Display risk message in large font
    st.markdown(f"<h1 style='color:{risk_color};'>{risk_message}</h1>", unsafe_allow_html=True)

    # Show probabilities and coordinates
    st.subheader("Prediction Details")
    st.write(f"Latitude: {lat}, Longitude: {lon}")
    st.write(f"Class 0 Probability: {pred_proba[0]:.2f}")
    st.write(f"Class 1 Probability: {pred_proba[1]:.2f}")
