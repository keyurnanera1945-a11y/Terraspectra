from pathlib import Path

import numpy as np
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_CSV = PROJECT_ROOT / "outputs" / "tilesplit_predictions.csv"
OUTPUT_CSV = PROJECT_ROOT / "outputs" / "geospatial_predictions.csv"


# ---------------------------------------------------------
# Demo field configuration
# IMPORTANT:
# These coordinates are simulated because the dataset
# does not contain real GPS/geospatial coordinates.
# ---------------------------------------------------------
FIELD_CENTER_LAT = 21.1702
FIELD_CENTER_LON = 72.8311

GRID_SIZE = 10
LAT_SPACING = 0.0005
LON_SPACING = 0.0005

MAX_SAMPLES = 100


# ---------------------------------------------------------
# Class information
# ---------------------------------------------------------
CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}


# ---------------------------------------------------------
# Load prediction results
# ---------------------------------------------------------
if not INPUT_CSV.exists():
    raise FileNotFoundError(
        f"Prediction file not found:\n{INPUT_CSV}"
    )

df = pd.read_csv(INPUT_CSV)

required_columns = {"actual_class", "predicted_class"}

missing_columns = required_columns - set(df.columns)

if missing_columns:
    raise ValueError(
        f"Missing required columns: {sorted(missing_columns)}"
    )

print(f"Loaded {len(df):,} prediction records.")


# ---------------------------------------------------------
# Limit samples for demo field visualization
# ---------------------------------------------------------
geo_df = df.head(MAX_SAMPLES).copy().reset_index(drop=True)


# ---------------------------------------------------------
# Create simulated coordinates
# ---------------------------------------------------------
coordinates = []

half_grid = GRID_SIZE // 2

for index in range(len(geo_df)):

    row = index // GRID_SIZE
    col = index % GRID_SIZE

    latitude = (
        FIELD_CENTER_LAT
        + (row - half_grid) * LAT_SPACING
    )

    longitude = (
        FIELD_CENTER_LON
        + (col - half_grid) * LON_SPACING
    )

    coordinates.append((latitude, longitude))


geo_df["latitude"] = [coord[0] for coord in coordinates]
geo_df["longitude"] = [coord[1] for coord in coordinates]


# ---------------------------------------------------------
# Prediction class
# ---------------------------------------------------------
geo_df["predicted_class"] = (
    geo_df["predicted_class"]
    .astype(int)
)

geo_df["actual_class"] = (
    geo_df["actual_class"]
    .astype(int)
)

geo_df["class_name"] = (
    geo_df["predicted_class"]
    .map(CLASS_NAMES)
    .fillna("Unknown")
)


# ---------------------------------------------------------
# Risk level
# ---------------------------------------------------------
def get_risk_level(predicted_class):
    if predicted_class == 0:
        return "Low"

    if predicted_class == 1:
        return "Moderate"

    if predicted_class == 2:
        return "High"

    return "Unknown"


geo_df["risk_level"] = (
    geo_df["predicted_class"]
    .apply(get_risk_level)
)


# ---------------------------------------------------------
# Confidence
# ---------------------------------------------------------
# The current tilesplit_predictions.csv does NOT contain
# probability/confidence information.
#
# Therefore, do NOT invent confidence values.
geo_df["confidence"] = np.nan


# ---------------------------------------------------------
# Metadata
# ---------------------------------------------------------
geo_df["confidence_status"] = "Not available from evaluation CSV"

geo_df["coordinate_source"] = (
    "SIMULATED DEMO COORDINATES"
)

geo_df["field_id"] = "DEMO_FIELD_001"


# ---------------------------------------------------------
# Arrange columns
# ---------------------------------------------------------
geo_df = geo_df[
    [
        "field_id",
        "latitude",
        "longitude",
        "actual_class",
        "predicted_class",
        "class_name",
        "confidence",
        "confidence_status",
        "risk_level",
        "coordinate_source",
    ]
]


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------
OUTPUT_CSV.parent.mkdir(
    parents=True,
    exist_ok=True
)

geo_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ---------------------------------------------------------
# Display result
# ---------------------------------------------------------
print()
print("Saved geospatial dataset to:")
print(OUTPUT_CSV)

print()
print(geo_df.head(10).to_string(index=False))

print()
print("Prediction distribution:")
print(
    geo_df["class_name"]
    .value_counts()
    .to_string()
)

print()
print("Step 17 geospatial dataset created successfully.")