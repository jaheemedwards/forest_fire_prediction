import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
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
    Use the sidebar to select latitude, longitude, environmental data, and date.  
    The prediction message appears above the map, which shows the location in the USA.
    """
)

# -----------------------------
# Sidebar inputs with sliders
# -----------------------------
st.sidebar.header("Input Features")

# Latitude and Longitude sliders limited to USA bounds
lat_input = st.sidebar.slider(
    "Latitude (°)", min_value=24.396308, max_value=49.384358, value=40.0, step=0.01
)
lon_input = st.sidebar.slider(
    "Longitude (°)", min_value=-125.0, max_value=-66.93457, value=-100.0, step=0.01
)

# Environmental feature sliders using dataset min/max/range
max_humidity = st.sidebar.slider(
    "Max Humidity", min_value=5.0, max_value=32767.0, value=50.0, step=1.0
)
max_temp = st.sidebar.slider(
    "Max Temperature", min_value=241.9, max_value=32767.0, value=300.0, step=1.0
)
vapor_pressure = st.sidebar.slider(
    "Vapor Pressure", min_value=0.3, max_value=32767.0, value=10.0, step=0.1
)
fuel_moisture_100h = st.sidebar.slider(
    "Fuel Moisture 100h", min_value=1.1, max_value=32767.0, value=5.0, step=0.1
)

# Date selector
selected_date = st.sidebar.date_input("Select Date", value=datetime(2025, 10, 21))
year, month, day = selected_date.year, selected_date.month, selected_date.day

features = [
    lat_input, lon_input, max_humidity, max_temp, vapor_pressure,
    fuel_moisture_100h, year, month, day
]

# -----------------------------
# Map settings
# -----------------------------
usa_center = dict(lat=39.5, lon=-98.35)
zoom_level = 3.1  # shows continental USA

# -----------------------------
# Predict button
# -----------------------------
if st.sidebar.button("Predict Wildfire Risk"):

    # Clamp coordinates to USA if out of bounds
    lat, lon = features[0], features[1]
    if not is_usa_coordinate(lat, lon):
        lat, lon = clamp_to_usa(lat, lon)
        features[0], features[1] = lat, lon

    # Make prediction
    pred_class, pred_proba = predict_fire_with_usa_check(model, features)

    # Determine risk message and color
    if pred_class == 1:
        risk_color = "red"
        risk_message = "🔥 High Wildfire Risk"
    else:
        risk_color = "green"
        risk_message = "✅ Low Wildfire Risk"

    # -----------------------------
    # Show prediction message above map
    # -----------------------------
    st.markdown(f"<h1 style='color:{risk_color};'>{risk_message}</h1>", unsafe_allow_html=True)

    st.subheader("Prediction Details")
    st.write(f"Latitude: {lat}, Longitude: {lon}")
    st.write(f"Max Humidity: {max_humidity}")
    st.write(f"Max Temperature: {max_temp}")
    st.write(f"Vapor Pressure: {vapor_pressure}")
    st.write(f"Fuel Moisture 100h: {fuel_moisture_100h}")
    st.write(f"Date: {year}-{month}-{day}")
    st.write(f"Class 0 Probability: {pred_proba[0]:.2f}")
    st.write(f"Class 1 Probability: {pred_proba[1]:.2f}")

    # -----------------------------
    # Display map under prediction
    # -----------------------------
    df_map = pd.DataFrame({"lat": [lat], "lon": [lon], "risk": [risk_message]})

    fig = px.scatter_map(
        df_map,
        lat="lat",
        lon="lon",
        color_discrete_sequence=[risk_color],
        hover_name="risk",
        zoom=zoom_level,
        height=400
    )

    fig.update_traces(marker=dict(size=25, sizemode='diameter', opacity=0.8))
    fig.update_layout(
        mapbox_style="open-street-map",
        mapbox=dict(center=usa_center, pitch=0, bearing=0, zoom=zoom_level),
        margin={"r":0,"t":0,"l":0,"b":0}
    )

    # Disable zooming/interactions
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
