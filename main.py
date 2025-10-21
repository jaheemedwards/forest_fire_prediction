import streamlit as st
from get_prediction import load_gzipped_model, predict_fire_with_usa_check

# -----------------------------
# Cache model loading
# -----------------------------
@st.cache_data  # Use @st.cache_resource if Streamlit version >=1.18
def load_model_cached(path: str):
    return load_gzipped_model(path)

MODEL_PATH = "models/wildfire_rf_balanced_classes_best_model.pkl.gz"
model = load_gzipped_model(MODEL_PATH)

# -----------------------------
# Streamlit App
# -----------------------------
st.title("🌲 USA Wildfire Risk Predictor")

# Sidebar for input features
st.sidebar.header("Input Features")
lat = st.sidebar.number_input("Latitude", min_value=-90.0, max_value=90.0, value=40.0)
lon = st.sidebar.number_input("Longitude", min_value=-180.0, max_value=180.0, value=-100.0)
max_humidity = st.sidebar.number_input("Max Humidity (%)", value=50)
max_temp = st.sidebar.number_input("Max Temperature (°C)", value=30)
vapor_pressure = st.sidebar.number_input("Vapor Pressure", value=10)
fuel_moisture_100h = st.sidebar.number_input("Fuel Moisture 100h", value=5)
year = st.sidebar.number_input("Year", min_value=2000, max_value=2030, value=2025)
month = st.sidebar.number_input("Month", min_value=1, max_value=12, value=10)
day = st.sidebar.number_input("Day", min_value=1, max_value=31, value=21)

features = [lat, lon, max_humidity, max_temp, vapor_pressure, fuel_moisture_100h, year, month, day]

# Predict button
if st.button("Predict Wildfire Risk"):
    pred_class, pred_proba = predict_fire_with_usa_check(model, features)

    st.subheader("Input Coordinates (clamped to USA if needed)")
    st.write(f"Latitude: {features[0]}, Longitude: {features[1]}")

    st.subheader("Prediction")
    st.write(f"Predicted Class: {pred_class}")

    st.subheader("Prediction Probabilities")
    st.bar_chart(pred_proba, use_container_width=True)
    st.write(f"Class 0 Probability: {pred_proba[0]:.2f}")
    st.write(f"Class 1 Probability: {pred_proba[1]:.2f}")
