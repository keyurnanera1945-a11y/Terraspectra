from pathlib import Path
import numpy as np
import pandas as pd
import requests
import streamlit as st


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

PATCH_CSV_PATH = PROJECT_ROOT / "outputs" / "hyperspectral_patches.csv"

TILESPLIT_CONFUSION_PATH = (
    PROJECT_ROOT / "outputs" / "tilesplit_confusion_matrix.csv"
)

TILESPLIT_REPORT_PATH = (
    PROJECT_ROOT / "outputs" / "tilesplit_classification_report.csv"
)

TILESPLIT_PREDICTIONS_PATH = (
    PROJECT_ROOT / "outputs" / "tilesplit_predictions.csv"
)

TILESPLIT_HISTORY_PATH = (
    PROJECT_ROOT / "outputs" / "tilesplit_training_history.csv"
)


# ============================================================
# FASTAPI
# ============================================================

API_URL = "http://127.0.0.1:8000"

HEALTH_ENDPOINT = f"{API_URL}/health"
PREDICT_ENDPOINT = f"{API_URL}/predict"


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_BANDS = 20

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

WAVELENGTHS = [
    420, 440, 500, 520, 540,
    560, 580, 600, 620, 640,
    660, 680, 700, 720, 740,
    760, 770, 800, 850, 900
]


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="TerraSpectra AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# HELPER FUNCTIONS
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


def normalize_band_for_display(band):
    band = np.asarray(band, dtype=np.float32)

    if not np.isfinite(band).all():
        raise ValueError(
            "Spectral band contains NaN or infinite values."
        )

    minimum = float(band.min())
    maximum = float(band.max())

    if maximum > minimum:
        band = (band - minimum) / (maximum - minimum)
    else:
        band = np.zeros_like(band)

    return np.clip(band, 0.0, 1.0)


# ============================================================
# LOAD DATA
# ============================================================

@st.cache_data
def load_patch_data():
    if not PATCH_CSV_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(PATCH_CSV_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_predictions():
    if not TILESPLIT_PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(TILESPLIT_PREDICTIONS_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_report():
    if not TILESPLIT_REPORT_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            TILESPLIT_REPORT_PATH,
            index_col=0
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_history():
    if not TILESPLIT_HISTORY_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(TILESPLIT_HISTORY_PATH)
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_confusion_matrix():
    if not TILESPLIT_CONFUSION_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            TILESPLIT_CONFUSION_PATH,
            index_col=0
        )
    except Exception:
        return pd.DataFrame()


# ============================================================
# FASTAPI HEALTH CHECK
# ============================================================

def check_api_status():
    try:
        response = requests.get(
            HEALTH_ENDPOINT,
            timeout=3
        )

        if response.status_code == 200:
            return True, response.json()

        return False, None

    except requests.exceptions.RequestException:
        return False, None


# ============================================================
# FASTAPI PREDICTION
# ============================================================

def predict_with_api(uploaded_file):
    try:
        response = requests.post(
            PREDICT_ENDPOINT,
            files={
                "file": (
                    uploaded_file.name,
                    uploaded_file.getvalue(),
                    "application/octet-stream",
                )
            },
            timeout=120,
        )

    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Could not connect to FastAPI.\n\n"
            "Start the backend using:\n"
            "python -m uvicorn api.main:app --reload"
        )

    except requests.exceptions.Timeout:
        raise RuntimeError(
            "FastAPI prediction request timed out."
        )

    except requests.exceptions.RequestException as exc:
        raise RuntimeError(
            f"FastAPI request failed: {exc}"
        )

    if response.status_code != 200:
        try:
            detail = response.json().get(
                "detail",
                response.text
            )
        except Exception:
            detail = response.text

        raise RuntimeError(
            f"FastAPI returned HTTP "
            f"{response.status_code}: {detail}"
        )

    try:
        result = response.json()
    except Exception:
        raise RuntimeError(
            "FastAPI returned invalid JSON."
        )

    if not result.get("success", False):
        raise RuntimeError(
            "FastAPI prediction was not successful."
        )

    return result


# ============================================================
# LOAD DASHBOARD DATA
# ============================================================

patch_df = load_patch_data()
prediction_df = load_predictions()
report_df = load_report()
history_df = load_history()
confusion_df = load_confusion_matrix()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 TerraSpectra AI")

st.sidebar.caption(
    "Hyperspectral Potato Disease Monitoring"
)

st.sidebar.divider()

page = st.sidebar.radio(
    "Navigation",
    [
        "Overview",
        "Data Explorer",
        "Spectral Analysis",
        "Model Performance",
        "Prediction",
        "Monitoring",
    ]
)

api_online, api_data = check_api_status()

if api_online:
    st.sidebar.success(
        "🟢 FastAPI Backend: Online"
    )
else:
    st.sidebar.error(
        "🔴 FastAPI Backend: Offline"
    )

st.sidebar.divider()

st.sidebar.caption(
    "3D-CNN • Hyperspectral Imaging • FastAPI"
)


# ============================================================
# MAIN HEADER
# ============================================================

st.title("🌱 TerraSpectra AI")

st.caption(
    "Hyperspectral potato disease monitoring "
    "using a tile-level 3D Convolutional Neural Network."
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.header("System Overview")

    total_tiles = 1115

    total_patches = (
        len(patch_df)
        if not patch_df.empty
        else 8688
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "HSI Tiles",
            f"{total_tiles:,}",
            "Hyperspectral image tiles"
        )

    with col2:
        st.metric(
            "Spectral Bands",
            EXPECTED_BANDS,
            "Bands per HSI tile"
        )

    with col3:
        st.metric(
            "Extracted Patches",
            f"{total_patches:,}",
            "Training-ready patches"
        )

    with col4:
        st.metric(
            "Validation Accuracy",
            "90.78%",
            "Tile-level 3D-CNN"
        )

    st.divider()

    st.subheader("Current AI Model")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Architecture:** Tile-Level 3D-CNN")
        st.write("**Input:** 20-band hyperspectral tile")
        st.write("**Model Input:** 20 × 32 × 32")
        st.write("**Output Classes:** 3")

    with col2:
        st.write("**Validation Samples:** 1,725")
        st.write("**Best Validation Accuracy:** 90.78%")
        st.write("**Parameters:** 18,243")
        st.write("**Backend:** FastAPI")

    st.divider()

    if api_online:
        st.success(
            "Dashboard and FastAPI backend are connected."
        )
    else:
        st.warning(
            "Dashboard is running, but the FastAPI backend "
            "is currently offline."
        )


# ============================================================
# PAGE 2 — DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header("Hyperspectral Dataset Explorer")

    if patch_df.empty:

        st.warning(
            "Patch dataset CSV was not found."
        )

    else:

        st.success(
            f"Loaded {len(patch_df):,} extracted patches."
        )

        st.subheader("Dataset Preview")

        st.dataframe(
            patch_df.head(100),
            use_container_width=True
        )

        label_column = find_column(
            patch_df,
            [
                "label",
                "class",
                "target",
                "disease",
            ]
        )

        if label_column:

            st.subheader("Class Distribution")

            distribution = (
                patch_df[label_column]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(distribution)

        else:

            st.info(
                "No label/class column was detected."
            )


# ============================================================
# PAGE 3 — SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.header("Spectral Analysis")

    uploaded_file = st.file_uploader(
        "Upload a hyperspectral .npz file",
        type=["npz"],
        key="spectral_upload"
    )

    if uploaded_file is None:

        st.info(
            "Upload an NPZ file containing an "
            "'im' array to inspect spectral bands."
        )

    else:

        try:

            data = np.load(uploaded_file)

            if "im" not in data:

                st.error(
                    "The uploaded NPZ file does not "
                    "contain an 'im' array."
                )

            else:

                hsi = data["im"]

                if hsi.ndim != 3:

                    st.error(
                        "Expected H × W × Bands "
                        "hyperspectral data."
                    )

                else:

                    height, width, bands = hsi.shape

                    st.success(
                        f"Loaded HSI: "
                        f"{height} × {width} × {bands}"
                    )

                    band_number = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=bands - 1,
                        value=0,
                        key="spectral_band"
                    )

                    display_band = normalize_band_for_display(
                        hsi[:, :, band_number]
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        if band_number < len(WAVELENGTHS):

                            wavelength = WAVELENGTHS[band_number]

                            caption = (
                                f"Band {band_number + 1} "
                                f"• {wavelength} nm"
                            )

                        else:

                            caption = (
                                f"Band {band_number + 1}"
                            )

                        st.image(
                            display_band,
                            caption=caption,
                            use_container_width=True
                        )

                    with col2:

                        spectral_profile = hsi.mean(
                            axis=(0, 1)
                        )

                        if bands == EXPECTED_BANDS:

                            x_values = WAVELENGTHS

                        else:

                            x_values = list(
                                range(1, bands + 1)
                            )

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength / Band": x_values,
                                "Mean Intensity": spectral_profile,
                            }
                        )

                        st.line_chart(
                            spectral_df.set_index(
                                "Wavelength / Band"
                            )
                        )

                    st.subheader("Spectral Statistics")

                    col1, col2, col3 = st.columns(3)

                    with col1:
                        st.metric(
                            "Minimum",
                            f"{float(hsi.min()):.2f}"
                        )

                    with col2:
                        st.metric(
                            "Maximum",
                            f"{float(hsi.max()):.2f}"
                        )

                    with col3:
                        st.metric(
                            "Mean",
                            f"{float(hsi.mean()):.2f}"
                        )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# PAGE 4 — MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.header("Tile-Level 3D-CNN Performance")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Accuracy",
            "90.78%",
            "Validation"
        )

    with col2:
        st.metric(
            "Healthy Recall",
            "96.48%",
            "Class 0"
        )

    with col3:
        st.metric(
            "Disease 1 Recall",
            "44.29%",
            "Class 1"
        )

    with col4:
        st.metric(
            "Disease 2 Recall",
            "0.00%",
            "Class 2"
        )

    st.divider()

    st.subheader("Classification Report")

    if not report_df.empty:

        st.dataframe(
            report_df,
            use_container_width=True
        )

    else:

        st.info(
            "Classification report was not found."
        )

    st.subheader("Confusion Matrix")

    if not confusion_df.empty:

        st.dataframe(
            confusion_df,
            use_container_width=True
        )

    else:

        st.info(
            "Confusion matrix was not found."
        )

    st.warning(
        "Important: Overall validation accuracy is 90.78%, "
        "but Disease Class 2 currently has 0% recall. "
        "The current model should therefore not be considered "
        "reliable for detecting Disease Class 2."
    )

    st.subheader("Training History")

    if not history_df.empty:

        numeric_columns = (
            history_df
            .select_dtypes(include=np.number)
            .columns
            .tolist()
        )

        if numeric_columns:

            st.line_chart(
                history_df[numeric_columns]
            )

        else:

            st.info(
                "No numeric training-history columns found."
            )

    else:

        st.info(
            "Training history file was not found."
        )


# ============================================================
# PAGE 5 — PREDICTION
# ============================================================

elif page == "Prediction":

    st.header("🔬 AI Hyperspectral Prediction")

    st.write(
        "Upload a hyperspectral NPZ tile and send it "
        "to the FastAPI backend for 3D-CNN inference."
    )

    if api_online:

        st.success(
            "FastAPI inference backend is online."
        )

    else:

        st.error(
            "FastAPI backend is offline."
        )

        st.code(
            "python -m uvicorn api.main:app --reload"
        )

    st.info(
        "Expected input: NPZ file containing an "
        "'im' array with exactly 20 spectral bands."
    )

    uploaded_file = st.file_uploader(
        "Upload hyperspectral NPZ tile",
        type=["npz"],
        key="prediction_upload"
    )

    if uploaded_file is None:

        st.subheader("Example Input")

        st.code(
            "data/raw/hyperspectral/0/0001.npz"
        )

    else:

        try:

            data = np.load(uploaded_file)

            if "im" not in data:

                st.error(
                    "The uploaded NPZ file does not "
                    "contain an 'im' array."
                )

            else:

                hsi = data["im"]

                if hsi.ndim != 3:

                    st.error(
                        "Invalid HSI shape. "
                        "Expected H × W × Bands."
                    )

                elif hsi.shape[2] != EXPECTED_BANDS:

                    st.error(
                        f"This model expects "
                        f"{EXPECTED_BANDS} bands, "
                        f"but the uploaded file contains "
                        f"{hsi.shape[2]} bands."
                    )

                else:

                    height, width, bands = hsi.shape

                    st.success(
                        f"HSI tile loaded successfully: "
                        f"{height} × {width} × {bands}"
                    )

                    # =================================================
                    # INPUT VISUALIZATION
                    # =================================================

                    st.subheader("1. Input Visualization")

                    selected_band = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=EXPECTED_BANDS - 1,
                        value=0,
                        key="prediction_band"
                    )

                    selected_wavelength = WAVELENGTHS[
                        selected_band
                    ]

                    original_band = hsi[
                        :,
                        :,
                        selected_band
                    ]

                    display_band = normalize_band_for_display(
                        original_band
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        st.markdown(
                            "#### Original Spectral Band"
                        )

                        st.image(
                            display_band,
                            caption=(
                                f"Band {selected_band + 1} "
                                f"• {selected_wavelength} nm "
                                f"• {height} × {width}"
                            ),
                            use_container_width=True
                        )

                    with col2:

                        st.markdown(
                            "#### Model Processing"
                        )

                        st.write(
                            f"Original HSI: "
                            f"{height} × {width} × {bands}"
                        )

                        st.write(
                            "Model input: 20 × 32 × 32"
                        )

                        st.write(
                            "Backend: FastAPI"
                        )

                        st.write(
                            "Model: Tile-Level 3D-CNN"
                        )

                    # =================================================
                    # RUN PREDICTION
                    # =================================================

                    st.subheader("2. Run AI Prediction")

                    if st.button(
                        "🔍 Run Tile-Level AI Prediction",
                        type="primary",
                        use_container_width=True,
                        key="run_prediction"
                    ):

                        try:

                            if not api_online:

                                raise RuntimeError(
                                    "FastAPI backend is offline. "
                                    "Start it using:\n\n"
                                    "python -m uvicorn "
                                    "api.main:app --reload"
                                )

                            with st.spinner(
                                "Running prediction through FastAPI..."
                            ):

                                api_result = predict_with_api(
                                    uploaded_file
                                )

                            st.session_state[
                                "prediction_result"
                            ] = api_result

                            st.success(
                                "Prediction completed successfully "
                                "through the FastAPI backend."
                            )

                        except Exception as exc:

                            st.error(
                                f"Prediction failed: {exc}"
                            )

                    # =================================================
                    # DISPLAY RESULT
                    # =================================================

                    result = st.session_state.get(
                        "prediction_result"
                    )

                    if result is not None:

                        predicted_class = int(
                            result["predicted_class"]
                        )

                        predicted_name = result[
                            "class_name"
                        ]

                        confidence = float(
                            result["confidence"]
                        )

                        probability_dict = result[
                            "probabilities"
                        ]

                        probabilities = np.array(
                            [
                                float(
                                    probability_dict[
                                        CLASS_NAMES[0]
                                    ]
                                ),
                                float(
                                    probability_dict[
                                        CLASS_NAMES[1]
                                    ]
                                ),
                                float(
                                    probability_dict[
                                        CLASS_NAMES[2]
                                    ]
                                ),
                            ]
                        )

                        input_shape = result.get(
                            "input_shape",
                            list(hsi.shape)
                        )

                        st.divider()

                        st.subheader(
                            "3. Prediction Result"
                        )

                        col1, col2, col3 = st.columns(3)

                        with col1:

                            st.metric(
                                "Predicted Class",
                                predicted_name
                            )

                        with col2:

                            st.metric(
                                "Model Confidence",
                                f"{confidence * 100:.2f}%"
                            )

                        with col3:

                            st.metric(
                                "Spectral Bands",
                                str(bands)
                            )

                        # =================================================
                        # PROBABILITY DISTRIBUTION
                        # =================================================

                        st.subheader(
                            "4. Class Probability Distribution"
                        )

                        probability_df = pd.DataFrame(
                            {
                                "Class": [
                                    CLASS_NAMES[0],
                                    CLASS_NAMES[1],
                                    CLASS_NAMES[2],
                                ],
                                "Probability (%)": (
                                    probabilities * 100
                                ),
                            }
                        )

                        st.bar_chart(
                            probability_df.set_index(
                                "Class"
                            )
                        )

                        display_probability_df = (
                            probability_df.copy()
                        )

                        display_probability_df[
                            "Probability (%)"
                        ] = (
                            display_probability_df[
                                "Probability (%)"
                            ].map(
                                lambda value:
                                f"{value:.2f}%"
                            )
                        )

                        st.dataframe(
                            display_probability_df,
                            hide_index=True,
                            use_container_width=True
                        )

                        # =================================================
                        # SPECTRAL SIGNATURE
                        # =================================================

                        st.subheader(
                            "5. Mean Spectral Signature"
                        )

                        spectral_profile = hsi.mean(
                            axis=(0, 1)
                        )

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength (nm)": WAVELENGTHS,
                                "Mean Intensity": spectral_profile,
                            }
                        )

                        st.line_chart(
                            spectral_df.set_index(
                                "Wavelength (nm)"
                            )
                        )

                        # =================================================
                        # INTERPRETATION
                        # =================================================

                        st.subheader(
                            "6. Prediction Interpretation"
                        )

                        if predicted_class == 0:

                            st.success(
                                "The tile was classified as "
                                "'Healthy / Normal' by the current "
                                "3D-CNN model."
                            )

                        elif predicted_class == 1:

                            st.warning(
                                "The tile was classified as "
                                "'Disease Class 1'. This is an "
                                "automated AI screening result and "
                                "should be validated separately."
                            )

                        else:

                            st.warning(
                                "The tile was classified as "
                                "'Disease Class 2'. The current "
                                "validation results show very weak "
                                "recall for this class, so this "
                                "prediction requires additional validation."
                            )

                        # =================================================
                        # CONFIDENCE
                        # =================================================

                        st.subheader(
                            "7. Confidence Assessment"
                        )

                        if confidence >= 0.80:

                            st.success(
                                f"High model confidence: "
                                f"{confidence * 100:.2f}%"
                            )

                        elif confidence >= 0.50:

                            st.warning(
                                f"Moderate model confidence: "
                                f"{confidence * 100:.2f}%"
                            )

                        else:

                            st.warning(
                                f"Low model confidence: "
                                f"{confidence * 100:.2f}%. "
                                "Treat the result cautiously."
                            )

                        # =================================================
                        # INFERENCE DETAILS
                        # =================================================

                        st.subheader(
                            "8. Inference Details"
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "**Architecture:** 3D-CNN"
                            )

                            st.write(
                                f"**Input Shape:** "
                                f"{input_shape}"
                            )

                            st.write(
                                "**Model Input:** 20 × 32 × 32"
                            )

                            st.write(
                                "**Output Classes:** 3"
                            )

                        with col2:

                            st.write(
                                "**Backend:** FastAPI"
                            )

                            st.write(
                                "**Endpoint:** POST /predict"
                            )

                            st.write(
                                "**Inference:** CPU compatible"
                            )

                            st.write(
                                "**Prediction Type:** "
                                "Tile-level classification"
                            )

                        st.info(
                            "This model performs tile-level "
                            "classification. It does not produce "
                            "pixel-level disease segmentation or "
                            "a true disease heatmap."
                        )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# =========================================================
# PAGE 6 — MONITORING
# =========================================================

elif page == "Monitoring":

    import folium
    from streamlit_folium import st_folium

    st.header("🗺️ Agricultural GIS Monitoring")

    st.write(
        "Interactive GIS visualization of AI-based hyperspectral "
        "potato disease predictions."
    )

    # -----------------------------------------------------
    # Coordinate warning
    # -----------------------------------------------------

    st.warning(
        "⚠️ DEMO MODE: The current dataset does not contain "
        "real GPS coordinates. The map locations shown here "
        "are simulated coordinates for prototype demonstration."
    )

    # -----------------------------------------------------
    # Load geospatial dataset
    # -----------------------------------------------------

    geo_path = (
        PROJECT_ROOT
        / "outputs"
        / "geospatial_predictions.csv"
    )

    if not geo_path.exists():

        st.error(
            "Geospatial prediction dataset not found."
        )

        st.info(
            "Run the following command first:"
        )

        st.code(
            "python src\\create_geospatial_dataset.py",
            language="powershell"
        )

        st.stop()

    try:

        geo_df = pd.read_csv(geo_path)

    except Exception as e:

        st.error(
            f"Unable to load geospatial dataset: {e}"
        )

        st.stop()

    # -----------------------------------------------------
    # Validate required columns
    # -----------------------------------------------------

    required_geo_columns = [
        "latitude",
        "longitude",
        "predicted_class",
        "class_name",
        "risk_level",
    ]

    missing_columns = [
        column
        for column in required_geo_columns
        if column not in geo_df.columns
    ]

    if missing_columns:

        st.error(
            "Missing required GIS columns: "
            + ", ".join(missing_columns)
        )

        st.stop()

    # -----------------------------------------------------
    # Convert coordinates
    # -----------------------------------------------------

    geo_df["latitude"] = pd.to_numeric(
        geo_df["latitude"],
        errors="coerce"
    )

    geo_df["longitude"] = pd.to_numeric(
        geo_df["longitude"],
        errors="coerce"
    )

    geo_df = geo_df.dropna(
        subset=["latitude", "longitude"]
    )

    if geo_df.empty:

        st.error(
            "No valid geographic coordinates available."
        )

        st.stop()

    # -----------------------------------------------------
    # Dashboard metrics
    # -----------------------------------------------------

    total_locations = len(geo_df)

    healthy_count = int(
        (geo_df["predicted_class"] == 0).sum()
    )

    disease1_count = int(
        (geo_df["predicted_class"] == 1).sum()
    )

    disease2_count = int(
        (geo_df["predicted_class"] == 2).sum()
    )

    disease_count = (
        disease1_count + disease2_count
    )

    disease_percentage = (
        disease_count / total_locations * 100
        if total_locations > 0
        else 0
    )

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Mapped Locations",
            f"{total_locations:,}"
        )

    with col2:
        st.metric(
            "Healthy",
            f"{healthy_count:,}"
        )

    with col3:
        st.metric(
            "Disease Locations",
            f"{disease_count:,}"
        )

    with col4:
        st.metric(
            "Disease Rate",
            f"{disease_percentage:.1f}%"
        )

    st.divider()

    # -----------------------------------------------------
    # Risk summary
    # -----------------------------------------------------

    st.subheader("📊 Field Risk Summary")

    risk_counts = (
        geo_df["risk_level"]
        .value_counts()
        .reindex(
            ["Low", "Moderate", "High"],
            fill_value=0
        )
    )

    risk_col1, risk_col2, risk_col3 = st.columns(3)

    with risk_col1:

        st.metric(
            "🟢 Low Risk",
            int(risk_counts["Low"])
        )

    with risk_col2:

        st.metric(
            "🟠 Moderate Risk",
            int(risk_counts["Moderate"])
        )

    with risk_col3:

        st.metric(
            "🔴 High Risk",
            int(risk_counts["High"])
        )

    st.divider()

    # -----------------------------------------------------
    # Create map
    # -----------------------------------------------------

    center_lat = geo_df["latitude"].mean()
    center_lon = geo_df["longitude"].mean()

    field_map = folium.Map(
        location=[
            center_lat,
            center_lon
        ],
        zoom_start=16,
        control_scale=True,
    )

    # -----------------------------------------------------
    # Field boundary
    # -----------------------------------------------------

    min_lat = geo_df["latitude"].min()
    max_lat = geo_df["latitude"].max()

    min_lon = geo_df["longitude"].min()
    max_lon = geo_df["longitude"].max()

    folium.Rectangle(
        bounds=[
            [min_lat, min_lon],
            [max_lat, max_lon],
        ],
        tooltip="Demo Monitoring Field",
        fill=False,
    ).add_to(field_map)

    # -----------------------------------------------------
    # Marker colors
    # -----------------------------------------------------

    marker_colors = {
        0: "green",
        1: "orange",
        2: "red",
    }

    # -----------------------------------------------------
    # Add prediction markers
    # -----------------------------------------------------

    for _, row in geo_df.iterrows():

        predicted_class = int(
            row["predicted_class"]
        )

        color = marker_colors.get(
            predicted_class,
            "gray"
        )

        confidence_text = "Not available"

        if (
            "confidence" in geo_df.columns
            and pd.notna(row["confidence"])
        ):
            confidence_text = (
                f"{float(row['confidence']) * 100:.2f}%"
            )

        popup_html = f"""
        <b>TerraSpectra AI Prediction</b><br><br>

        <b>Field:</b>
        {row.get("field_id", "DEMO_FIELD_001")}<br>

        <b>Predicted Class:</b>
        {row["class_name"]}<br>

        <b>Risk Level:</b>
        {row["risk_level"]}<br>

        <b>Confidence:</b>
        {confidence_text}<br>

        <b>Latitude:</b>
        {row["latitude"]:.6f}<br>

        <b>Longitude:</b>
        {row["longitude"]:.6f}<br>

        <b>Coordinate Source:</b>
        Simulated Demo Coordinates
        """

        folium.CircleMarker(
            location=[
                row["latitude"],
                row["longitude"],
            ],
            radius=7,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.75,
            popup=folium.Popup(
                popup_html,
                max_width=350,
            ),
            tooltip=row["class_name"],
        ).add_to(field_map)

    # -----------------------------------------------------
    # Map title
    # -----------------------------------------------------

    st.subheader("🗺️ AI Disease Prediction Map")

    st.caption(
        "Green = Healthy | "
        "Orange = Disease Class 1 | "
        "Red = Disease Class 2"
    )

    # -----------------------------------------------------
    # Display map
    # -----------------------------------------------------

    st_folium(
        field_map,
        width=None,
        height=600,
        returned_objects=[],
    )

    st.divider()

    # -----------------------------------------------------
    # Prediction distribution
    # -----------------------------------------------------

    st.subheader("📈 Prediction Distribution")

    distribution_df = (
        geo_df["class_name"]
        .value_counts()
        .rename_axis("Class")
        .reset_index(name="Locations")
    )

    st.bar_chart(
        distribution_df.set_index("Class")
    )

    st.divider()

    # -----------------------------------------------------
    # GIS prediction data
    # -----------------------------------------------------

    st.subheader("📋 Geospatial Prediction Data")

    display_columns = [
        "field_id",
        "latitude",
        "longitude",
        "actual_class",
        "predicted_class",
        "class_name",
        "risk_level",
    ]

    if "confidence" in geo_df.columns:
        display_columns.append(
            "confidence"
        )

    available_display_columns = [
        column
        for column in display_columns
        if column in geo_df.columns
    ]

    st.dataframe(
        geo_df[available_display_columns],
        use_container_width=True,
        hide_index=True,
    )

    # -----------------------------------------------------
    # Coordinate information
    # -----------------------------------------------------

    st.info(
        "📍 Coordinate source: SIMULATED DEMO COORDINATES. "
        "For real agricultural deployment, each hyperspectral "
        "tile should be associated with GPS latitude/longitude "
        "from the UAV or field acquisition system."
    )