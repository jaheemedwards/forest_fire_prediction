import gzip
import joblib
import pandas as pd
from typing import List, Tuple, Union
from datetime import datetime
from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from custom_transformers import TimeFeaturesAdder

from shapely.geometry import Point
from shapely.ops import nearest_points
import geopandas as gpd

# -----------------------------
# 0. Load USA Land Polygon
# -----------------------------
world = gpd.read_file("map_stuff/ne_50m_admin_0_countries.shp")
usa_land = world[world['ADMIN'] == 'United States of America'].copy()
# Simplify geometry for speed (optional)
usa_land['geometry'] = usa_land['geometry'].simplify(0.01)

def is_on_usa_land(lat: float, lon: float) -> bool:
    """Check if a coordinate is on USA land."""
    point = Point(lon, lat)
    return usa_land.contains(point).any()

def snap_to_usa_land(lat: float, lon: float) -> Tuple[float, float]:
    """If a coordinate is in water, snap it to the nearest land point in the USA."""
    point = Point(lon, lat)
    if is_on_usa_land(lat, lon):
        return lat, lon
    # Snap to nearest point on USA land
    nearest_geom = nearest_points(point, usa_land.unary_union)[1]
    return nearest_geom.y, nearest_geom.x

# -----------------------------
# 1. Model Loading
# -----------------------------
def load_gzipped_model(path: str):
    """Load a gzipped joblib (.pkl.gz) model."""
    with gzip.open(path, 'rb') as f:
        model = joblib.load(f)
    return model

# -----------------------------
# 2. Humidity Label Generator
# -----------------------------
def generate_max_humidity_label(max_humidity: float) -> str:
    bin_edges = [45.299, 64.4, 77.3, 87.9, 99.9, 100.0]
    labels = ['Very Low', 'Low', 'Moderate', 'High', 'Very High']
    return pd.cut([max_humidity], bins=bin_edges, labels=labels, include_lowest=True)[0]

# -----------------------------
# 3. Unified Prediction Function
# -----------------------------
def get_wildfire_prediction(model, input_data: Union[dict, List[float]]):
    """
    input_data: dict with keys or list in order:
        [latitude, longitude, max_humidity, max_temp, wind_speed, fuel_moisture_100h, year, month, day]
    """
    if isinstance(input_data, list):
        df = pd.DataFrame([input_data], columns=[
            'latitude', 'longitude', 'max_humidity', 'max_temp',
            'wind_speed', 'fuel_moisture_100h', 'year', 'month', 'day'
        ])
    else:
        df = pd.DataFrame([input_data])

    # Snap coordinates to USA land
    lat, lon = snap_to_usa_land(df['latitude'][0], df['longitude'][0])
    df['latitude'][0], df['longitude'][0] = lat, lon

    # Add rmax_label from humidity if missing
    if 'rmax_label' not in df.columns and 'max_humidity' in df.columns:
        df['rmax_label'] = df['max_humidity'].apply(generate_max_humidity_label)

    # Add 'date' column for pipeline
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df[['year', 'month', 'day']])

    # Make prediction
    prediction = int(model.predict(df)[0])
    prob_class1 = float(model.predict_proba(df)[0][1])
    prob_all_classes = model.predict_proba(df)[0].tolist()

    return {
        "prediction": prediction,
        "probability": prob_class1,
        "probabilities": prob_all_classes,
        "adjusted_location": (lat, lon),
        "humidity_label": df['rmax_label'][0]
    }

# -----------------------------
# 4. Dataset Summary for Slider Ranges
# # -----------------------------
# df = pd.read_parquet("data/wildfire_data.parquet")
# print("First 5 rows:")
# print(df.head())

# print("\nNumeric column summary:")
# print(df.describe())

# print("\nMin and Max per column:")
# for col in df.select_dtypes(include="number").columns:
#     print(f"{col}: min={df[col].min()}, max={df[col].max()}")
