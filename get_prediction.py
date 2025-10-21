import gzip
import joblib
from typing import List, Tuple
from pprint import pprint
import pandas as pd

# -----------------------------
# Constants
# -----------------------------
COLUMN_NAMES = ['lat', 'lon', 'max_humidity', 'max_temp', 'vapor_pressure',
                'fuel_moisture_100h', 'year', 'month', 'day']

# -----------------------------
# Model Loading
# -----------------------------
def load_gzipped_model(path: str):
    """
    Loads a gzipped scikit-learn model file (.pkl.gz) and returns the model object.
    """
    with gzip.open(path, 'rb') as f:
        model = joblib.load(f)
    return model

# -----------------------------
# USA Coordinate Helpers
# -----------------------------
def is_usa_coordinate(lat: float, lon: float) -> bool:
    """
    Returns True if the latitude and longitude are within approximate bounds of the continental USA.
    """
    min_lat, max_lat = 24.396308, 49.384358
    min_lon, max_lon = -125.0, -66.93457
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon

def clamp_to_usa(lat: float, lon: float) -> Tuple[float, float]:
    """
    Clamps latitude and longitude to the USA bounding box.
    """
    lat = max(24.396308, min(49.384358, lat))
    lon = max(-125.0, min(-66.93457, lon))
    return lat, lon

# -----------------------------
# Prediction Functions
# -----------------------------
def make_forest_fire_prediction(model, features: List[float]) -> Tuple[int, List[float]]:
    """
    Makes a prediction using a pre-loaded Random Forest pipeline.
    Returns both predicted class and probabilities.
    """
    # Convert features to DataFrame (required for ColumnTransformer)
    df = pd.DataFrame([features], columns=COLUMN_NAMES)
    prediction = model.predict(df)[0]
    probability = model.predict_proba(df)[0].tolist()
    return prediction, probability

def predict_fire_with_usa_check(model, features: List[float]) -> Tuple[int, List[float]]:
    """
    Checks if coordinates are in the USA and clamps if needed, then predicts wildfire risk.
    """
    lat, lon = features[0], features[1]
    if not is_usa_coordinate(lat, lon):
        lat, lon = clamp_to_usa(lat, lon)
        features[0], features[1] = lat, lon
    return make_forest_fire_prediction(model, features)

# -----------------------------
# Main Script
# -----------------------------

# Load model
model_path = "models/wildfire_rf_balanced_classes_best_model.pkl.gz"
model = load_gzipped_model(model_path)

# Print pipeline steps
print("Pipeline structure:")
pprint(model.named_steps)

# Example input features (lat, lon, humidity, temp, etc.)
sample_input = [50.0, -130.0, 85, 32, 10, 5, 2025, 10, 21]  # intentionally outside USA

# Predict wildfire
pred_class, pred_proba = predict_fire_with_usa_check(model, sample_input)

print("\nInput coordinates after USA check/clamp:", sample_input[0], sample_input[1])
print("Predicted class:", pred_class)
print("Predicted probabilities:", pred_proba)
