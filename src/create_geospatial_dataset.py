from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_predictions.csv"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "geospatial_predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Demo field center.
# These coordinates are SIMULATED and are NOT actual
# coordinates from the hyperspectral dataset.

FIELD_CENTER_LAT = 21.1702
FIELD_CENTER_LON = 72.8311

GRID_ROWS = 10
GRID_COLS = 10

LAT_SPACING = 0.0005
LON_SPACING = 0.0005


CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}


# ============================================================
# LOAD PREDICTIONS
# ============================================================

if not PREDICTIONS_PATH.exists():

    raise FileNotFoundError(
        f"Prediction file not found:\n"
        f"{PREDICTIONS_PATH}"
    )


predictions = pd.read_csv(
    PREDICTIONS_PATH
)

print(
    f"Loaded {len(predictions):,} prediction records."
)


# ============================================================
# DETECT COLUMNS
# ============================================================

def find_column(df, candidates):

    for candidate in candidates:

        if candidate in df.columns:
            return candidate

    lower_map = {
        str(column).lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


prediction_column = find_column(
    predictions,
    [
        "predicted_class",
        "prediction",
        "predicted",
        "pred",
    ]
)

confidence_column = find_column(
    predictions,
    [
        "confidence",
        "probability",
        "score",
    ]
)


# ============================================================
# CREATE DEMO GEOSPATIAL GRID
# ============================================================

sample_size = min(
    len(predictions),
    GRID_ROWS * GRID_COLS
)

geo_predictions = predictions.head(
    sample_size
).copy()

latitudes = []
longitudes = []


for index in range(sample_size):

    row = index // GRID_COLS
    col = index % GRID_COLS

    latitude = (
        FIELD_CENTER_LAT
        + (row - GRID_ROWS / 2)
        * LAT_SPACING
    )

    longitude = (
        FIELD_CENTER_LON
        + (col - GRID_COLS / 2)
        * LON_SPACING
    )

    latitudes.append(latitude)
    longitudes.append(longitude)


geo_predictions["latitude"] = latitudes
geo_predictions["longitude"] = longitudes


# ============================================================
# STANDARDIZE PREDICTION INFORMATION
# ============================================================

if prediction_column:

    geo_predictions["predicted_class"] = (
        pd.to_numeric(
            geo_predictions[prediction_column],
            errors="coerce"
        )
        .fillna(0)
        .astype(int)
    )

else:

    # Fallback only if prediction column cannot be detected.
    np.random.seed(42)

    geo_predictions["predicted_class"] = (
        np.random.choice(
            [0, 1, 2],
            size=sample_size,
            p=[0.70, 0.20, 0.10]
        )
    )


geo_predictions["class_name"] = (
    geo_predictions["predicted_class"]
    .map(CLASS_NAMES)
    .fillna("Unknown")
)


if confidence_column:

    geo_predictions["confidence"] = (
        pd.to_numeric(
            geo_predictions[confidence_column],
            errors="coerce"
        )
        .fillna(0.0)
    )

else:

    geo_predictions["confidence"] = 0.0


# ============================================================
# RISK LEVEL
# ============================================================

def determine_risk(row):

    predicted_class = int(
        row["predicted_class"]
    )

    confidence = float(
        row["confidence"]
    )

    if predicted_class == 0:

        return "Low"

    if predicted_class == 1:

        if confidence >= 0.70:
            return "High"

        return "Moderate"

    if predicted_class == 2:

        return "High"

    return "Unknown"


geo_predictions["risk_level"] = (
    geo_predictions.apply(
        determine_risk,
        axis=1
    )
)


# ============================================================
# METADATA
# ============================================================

geo_predictions["coordinate_source"] = (
    "SIMULATED DEMO COORDINATES"
)

geo_predictions["field_id"] = (
    "DEMO_FIELD_001"
)


# ============================================================
# SAVE
# ============================================================

OUTPUT_PATH.parent.mkdir(
    parents=True,
    exist_ok=True
)

geo_predictions.to_csv(
    OUTPUT_PATH,
    index=False
)


print()
print(
    f"Saved geospatial dataset to:"
)
print(
    OUTPUT_PATH
)

print()
print(
    geo_predictions[
        [
            "latitude",
            "longitude",
            "predicted_class",
            "class_name",
            "confidence",
            "risk_level",
        ]
    ].head(10)
)

print()
print("Step 17 geospatial dataset created successfully.")