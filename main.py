import streamlit as st
import pandas as pd
import plotly.express as px
from datetime import datetime
from inference import (
    load_gzipped_model,
    get_wildfire_prediction,
)
from shapely.geometry import Point
from shapely.ops import nearest_points
import geopandas as gpd

# -----------------------------
# 0. Load USA Land Polygon for snapping
# -----------------------------
world = gpd.read_file("map_stuff/ne_50m_admin_0_countries.shp")
usa_land = world[world['ADMIN'] == 'United States of America'].copy()
usa_land['geometry'] = usa_land['geometry'].simplify(0.01)

def snap_to_usa_land(lat: float, lon: float):
    """Snap coordinates to nearest USA land point if not on land."""
    point = Point(lon, lat)
    if usa_land.contains(point).any():
        return lat, lon
    nearest_geom = nearest_points(point, usa_land.unary_union)[1]
    return nearest_geom.y, nearest_geom.x

# -----------------------------
# Cached model loading
# -----------------------------
@st.cache_data
def load_model_cached(path: str):
    return load_gzipped_model(path)

MODEL_PATH = "models/year_and_month.pkl.gz"
model = load_model_cached(MODEL_PATH)

# -----------------------------
# Page title & description
# -----------------------------
st.title("🌲 USA Wildfire Risk Predictor")
st.markdown(
    """
    Predict the likelihood of wildfires based on location, environmental features, and date.
    Coordinates are automatically snapped to USA land.
    """
)

# -----------------------------
# Sidebar inputs with sliders
# -----------------------------
st.sidebar.header("Input Features")

lat_input = st.sidebar.slider("Latitude (°)", 24.396308, 49.384358, 40.0, 0.01)
lon_input = st.sidebar.slider("Longitude (°)", -125.0, -66.93457, -100.0, 0.01)

max_humidity = st.sidebar.slider("Max Humidity (%)", 45.3, 99.9, 50.0, 1.0)
max_temp = st.sidebar.slider("Max Temperature (°C)", 241.9, 313.7, 300.0, 0.1)
wind_speed = st.sidebar.slider("Wind Speed (m/s)", 0.3, 9.1, 5.0, 0.1)
fuel_moisture_100h = st.sidebar.slider("Fuel Moisture 100h (%)", 1.3, 23.7, 5.0, 0.1)

selected_date = st.sidebar.date_input("Select Date", value=datetime(2025, 10, 21))
year, month, day = selected_date.year, selected_date.month, selected_date.day

# -----------------------------
# Predict button
# -----------------------------
if st.sidebar.button("Predict Wildfire Risk"):

    # Snap coordinates to USA land
    lat_input, lon_input = snap_to_usa_land(lat_input, lon_input)

    input_data = {
        "latitude": lat_input,
        "longitude": lon_input,
        "max_humidity": max_humidity,
        "max_temp": max_temp,
        "wind_speed": wind_speed,
        "fuel_moisture_100h": fuel_moisture_100h,
        "year": year,
        "month": month,
        "day": day
    }

    # Call unified prediction
    result = get_wildfire_prediction(model, input_data)

    lat, lon = result["adjusted_location"]
    pred_class = result["prediction"]
    prob_class1 = result["probability"]
    prob_all_classes = result["probabilities"]
    humidity_label = result["humidity_label"]

    # Risk message and color
    if pred_class == 1:
        risk_color = "red"
        risk_message = "🔥 High Wildfire Risk"
    else:
        risk_color = "green"
        risk_message = "✅ Low Wildfire Risk"

    st.markdown(f"<h1 style='color:{risk_color};'>{risk_message}</h1>", unsafe_allow_html=True)

    st.subheader("Prediction Details")
    st.write(f"Latitude: {lat}, Longitude: {lon}")
    st.write(f"Max Humidity: {max_humidity} ({humidity_label})")
    st.write(f"Max Temperature: {max_temp}")
    st.write(f"Wind Speed: {wind_speed}")
    st.write(f"Fuel Moisture 100h: {fuel_moisture_100h}")
    st.write(f"Date: {year}-{month}-{day}")
    st.write(f"Class 0 Probability: {prob_all_classes[0]:.2f}")
    st.write(f"Class 1 Probability: {prob_all_classes[1]:.2f}")

    # -----------------------------
    # Map display
    # -----------------------------
    df_map = pd.DataFrame({"lat": [lat], "lon": [lon], "risk": [risk_message]})
    fig = px.scatter_map(df_map, lat="lat", lon="lon", color_discrete_sequence=[risk_color],
                         hover_name="risk", zoom=3.1, height=400)
    fig.update_traces(marker=dict(size=25, sizemode='diameter', opacity=0.8))
    fig.update_layout(mapbox_style="open-street-map",
                      mapbox=dict(center=dict(lat=39.5, lon=-98.35), pitch=0, bearing=0, zoom=3.1),
                      margin={"r":0,"t":0,"l":0,"b":0})
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})
