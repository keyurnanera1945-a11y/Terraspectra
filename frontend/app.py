from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# PYTORCH
# ============================================================

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_3dcnn import TerraSpectra3DCNN


PATCH_CSV_PATH = (
    PROJECT_ROOT / "outputs" / "hyperspectral_patches.csv"
)

TILESPLIT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_tilesplit_best.pt"
)

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
# CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

EXPECTED_BANDS = 20
PATCH_SIZE = 32

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
# CSS
# ============================================================

st.markdown(
    """
    <style>

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    [data-testid="stMetric"] {
        background-color: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 12px;
    }

    </style>
    """,
    unsafe_allow_html=True,
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
    """
    Convert raw hyperspectral values to 0-1
    for safe Streamlit image display.
    """

    band = np.asarray(
        band,
        dtype=np.float32
    )

    if not np.isfinite(band).all():
        raise ValueError(
            "Spectral band contains NaN or infinite values."
        )

    min_value = float(band.min())
    max_value = float(band.max())

    if max_value > min_value:

        band = (
            band - min_value
        ) / (
            max_value - min_value
        )

    else:

        band = np.zeros_like(band)

    return np.clip(
        band,
        0.0,
        1.0
    )


@st.cache_data
def load_patch_data():

    if not PATCH_CSV_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            PATCH_CSV_PATH
        )
    except Exception:
        return pd.DataFrame()


@st.cache_data
def load_predictions():

    if not TILESPLIT_PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            TILESPLIT_PREDICTIONS_PATH
        )
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
        return pd.read_csv(
            TILESPLIT_HISTORY_PATH
        )
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


def prepare_hyperspectral_tensor(hsi):

    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is not installed."
        )

    hsi = np.asarray(
        hsi,
        dtype=np.float32
    )

    if hsi.ndim != 3:
        raise ValueError(
            "Expected H × W × Bands hyperspectral data."
        )

    height, width, bands = hsi.shape

    if bands != EXPECTED_BANDS:
        raise ValueError(
            f"Expected {EXPECTED_BANDS} bands, "
            f"but received {bands}."
        )

    if not np.isfinite(hsi).all():
        raise ValueError(
            "Hyperspectral data contains NaN or infinite values."
        )

    # Normalize raw uint16 HSI.
    hsi = hsi / 65535.0

    hsi = np.clip(
        hsi,
        0.0,
        1.0
    )

    # HWC -> CHW
    hsi = np.transpose(
        hsi,
        (2, 0, 1)
    )

    tensor = torch.from_numpy(
        hsi
    ).float()

    # [Bands, H, W]
    tensor = tensor.unsqueeze(0)

    # [1, Bands, H, W]
    tensor = tensor.unsqueeze(0)

    # [1, 1, Bands, H, W]
    tensor = F.interpolate(
        tensor,
        size=(
            EXPECTED_BANDS,
            PATCH_SIZE,
            PATCH_SIZE
        ),
        mode="trilinear",
        align_corners=False
    )

    return tensor


def create_terraspectra_model():

    signature = inspect.signature(
        TerraSpectra3DCNN
    )

    parameters = signature.parameters

    kwargs = {}

    if "num_classes" in parameters:
        kwargs["num_classes"] = 3

    if "in_channels" in parameters:
        kwargs["in_channels"] = 1

    if "input_channels" in parameters:
        kwargs["input_channels"] = 1

    if "channels" in parameters:
        kwargs["channels"] = 1

    try:

        return TerraSpectra3DCNN(
            **kwargs
        )

    except TypeError:

        return TerraSpectra3DCNN()


@st.cache_resource
def load_prediction_model():

    if torch is None:
        raise RuntimeError(
            "PyTorch is not available."
        )

    if not TILESPLIT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Model file not found:\n"
            f"{TILESPLIT_MODEL_PATH}"
        )

    model = create_terraspectra_model()

    try:

        checkpoint = torch.load(
            TILESPLIT_MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            TILESPLIT_MODEL_PATH,
            map_location="cpu"
        )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint[
                "model_state_dict"
            ]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint[
                "state_dict"
            ]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        new_key = key

        if new_key.startswith("module."):
            new_key = new_key[
                len("module.") :
            ]

        cleaned_state_dict[
            new_key
        ] = value

    model.load_state_dict(
        cleaned_state_dict,
        strict=True
    )

    model.eval()

    return model


def run_prediction(hsi):

    tensor = prepare_hyperspectral_tensor(
        hsi
    )

    model = load_prediction_model()

    with torch.no_grad():

        logits = model(
            tensor
        )

        probabilities = torch.softmax(
            logits,
            dim=1
        )[0]

        predicted_class = int(
            torch.argmax(
                probabilities
            ).item()
        )

        confidence = float(
            probabilities[
                predicted_class
            ].item()
        )

    probability_values = (
        probabilities
        .cpu()
        .numpy()
    )

    return (
        predicted_class,
        confidence,
        probability_values,
        tensor
    )


# ============================================================
# LOAD DATA
# ============================================================

patch_df = load_patch_data()
prediction_df = load_predictions()
report_df = load_report()
history_df = load_history()
confusion_df = load_confusion_matrix()


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌱 TerraSpectra AI"
)

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
        "Monitoring"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "3D-CNN • Hyperspectral Imaging • AI"
)


# ============================================================
# MAIN TITLE
# ============================================================

st.title(
    "🌱 TerraSpectra AI"
)

st.caption(
    "Hyperspectral potato disease monitoring "
    "using a tile-level 3D Convolutional Neural Network."
)


# ============================================================
# PAGE 1 — OVERVIEW
# ============================================================

if page == "Overview":

    st.header(
        "System Overview"
    )

    total_tiles = 1115

    total_bands = EXPECTED_BANDS

    total_patches = (
        len(patch_df)
        if not patch_df.empty
        else 8688
    )

    validation_samples = 1725

    accuracy = 0.9078

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
            total_bands,
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
            f"{accuracy * 100:.2f}%",
            "Tile-level 3D-CNN"
        )

    st.divider()

    st.subheader(
        "Current AI Model"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.write(
            "**Architecture:** Tile-Level 3D-CNN"
        )

        st.write(
            "**Input:** 20-band hyperspectral tile"
        )

        st.write(
            "**Model Input:** 20 × 32 × 32"
        )

        st.write(
            "**Output Classes:** 3"
        )

    with col2:

        st.write(
            "**Validation Samples:** 1,725"
        )

        st.write(
            "**Best Validation Accuracy:** 90.78%"
        )

        st.write(
            "**Parameters:** 18,243"
        )

        st.write(
            "**Checkpoint:** "
            "terraspectra_3dcnn_tilesplit_best.pt"
        )

    st.divider()

    st.subheader(
        "Project Status"
    )

    st.success(
        "Tile-level training, evaluation, dashboard "
        "integration, and prediction visualization are implemented."
    )


# ============================================================
# PAGE 2 — DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.header(
        "Hyperspectral Dataset Explorer"
    )

    if patch_df.empty:

        st.warning(
            "Patch dataset CSV was not found."
        )

    else:

        st.success(
            f"Loaded {len(patch_df):,} extracted patches."
        )

        st.subheader(
            "Dataset Preview"
        )

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
                "disease"
            ]
        )

        if label_column:

            st.subheader(
                "Class Distribution"
            )

            distribution = (
                patch_df[
                    label_column
                ]
                .value_counts()
                .sort_index()
            )

            st.bar_chart(
                distribution
            )


# ============================================================
# PAGE 3 — SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.header(
        "Spectral Analysis"
    )

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

            data = np.load(
                uploaded_file
            )

            if "im" not in data:

                st.error(
                    "The uploaded NPZ file does not "
                    "contain an 'im' array."
                )

            else:

                hsi = data["im"]

                if hsi.ndim != 3:

                    st.error(
                        "Expected H × W × Bands hyperspectral data."
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
                        value=0
                    )

                    display_band = (
                        normalize_band_for_display(
                            hsi[
                                :,
                                :,
                                band_number
                            ]
                        )
                    )

                    col1, col2 = st.columns(2)

                    with col1:

                        if band_number < len(
                            WAVELENGTHS
                        ):

                            wavelength = (
                                WAVELENGTHS[
                                    band_number
                                ]
                            )

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

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength (nm)":
                                    WAVELENGTHS[:bands],
                                "Mean Intensity":
                                    spectral_profile
                            }
                        )

                        st.line_chart(
                            spectral_df.set_index(
                                "Wavelength (nm)"
                            )
                        )

                    st.subheader(
                        "Spectral Statistics"
                    )

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

    st.header(
        "Tile-Level 3D-CNN Performance"
    )

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

    st.subheader(
        "Classification Report"
    )

    if not report_df.empty:

        st.dataframe(
            report_df,
            use_container_width=True
        )

    else:

        st.info(
            "Classification report was not found."
        )

    st.subheader(
        "Confusion Matrix"
    )

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

    st.subheader(
        "Training History"
    )

    if not history_df.empty:

        numeric_columns = (
            history_df
            .select_dtypes(
                include=np.number
            )
            .columns
            .tolist()
        )

        if numeric_columns:

            st.line_chart(
                history_df[
                    numeric_columns
                ]
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

    st.header(
        "🔬 AI Hyperspectral Prediction"
    )

    st.write(
        "Upload a hyperspectral NPZ tile and run "
        "the trained tile-level 3D-CNN model."
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

        st.subheader(
            "Example Input"
        )

        st.code(
            "data/raw/hyperspectral/0/0001.npz"
        )

    else:

        try:

            data = np.load(
                uploaded_file
            )

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

                    # ------------------------------------------------
                    # INPUT VISUALIZATION
                    # ------------------------------------------------

                    st.subheader(
                        "1. Input Visualization"
                    )

                    selected_band = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=EXPECTED_BANDS - 1,
                        value=0,
                        key="prediction_band"
                    )

                    selected_wavelength = (
                        WAVELENGTHS[
                            selected_band
                        ]
                    )

                    original_band = hsi[
                        :,
                        :,
                        selected_band
                    ]

                    display_band = (
                        normalize_band_for_display(
                            original_band
                        )
                    )

                    # ------------------------------------------------
                    # PREPARE MODEL INPUT
                    # ------------------------------------------------

                    model_tensor = (
                        prepare_hyperspectral_tensor(
                            hsi
                        )
                    )

                    model_band = (
                        model_tensor[
                            0,
                            0,
                            selected_band
                        ]
                        .detach()
                        .cpu()
                        .numpy()
                    )

                    model_band_display = (
                        normalize_band_for_display(
                            model_band
                        )
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
                            "#### Model Input Preview"
                        )

                        st.image(
                            model_band_display,
                            caption=(
                                f"Band {selected_band + 1} "
                                f"• {selected_wavelength} nm "
                                f"• 32 × 32"
                            ),
                            use_container_width=True
                        )

                    # ------------------------------------------------
                    # PREDICTION BUTTON
                    # ------------------------------------------------

                    st.subheader(
                        "2. Run AI Prediction"
                    )

                    if st.button(
                        "🔍 Run Tile-Level AI Prediction",
                        type="primary",
                        use_container_width=True,
                        key="run_prediction"
                    ):

                        try:

                            (
                                predicted_class,
                                confidence,
                                probabilities,
                                _
                            ) = run_prediction(
                                hsi
                            )

                            st.session_state[
                                "prediction_result"
                            ] = {
                                "predicted_class":
                                    predicted_class,

                                "confidence":
                                    confidence,

                                "probabilities":
                                    probabilities
                            }

                        except Exception as exc:

                            st.error(
                                f"Prediction failed: {exc}"
                            )

                    # ------------------------------------------------
                    # RESULT
                    # ------------------------------------------------

                    result = (
                        st.session_state.get(
                            "prediction_result"
                        )
                    )

                    if result is not None:

                        predicted_class = int(
                            result[
                                "predicted_class"
                            ]
                        )

                        confidence = float(
                            result[
                                "confidence"
                            ]
                        )

                        probabilities = np.asarray(
                            result[
                                "probabilities"
                            ]
                        )

                        predicted_name = (
                            CLASS_NAMES.get(
                                predicted_class,
                                f"Class {predicted_class}"
                            )
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

                        # ------------------------------------------------
                        # PROBABILITIES
                        # ------------------------------------------------

                        st.subheader(
                            "4. Class Probability Distribution"
                        )

                        probability_df = pd.DataFrame(
                            {
                                "Class": [
                                    CLASS_NAMES[0],
                                    CLASS_NAMES[1],
                                    CLASS_NAMES[2]
                                ],
                                "Probability (%)": (
                                    probabilities * 100
                                )
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
                            ]
                            .map(
                                lambda value:
                                    f"{value:.2f}%"
                            )
                        )

                        st.dataframe(
                            display_probability_df,
                            hide_index=True,
                            use_container_width=True
                        )

                        # ------------------------------------------------
                        # SPECTRAL SIGNATURE
                        # ------------------------------------------------

                        st.subheader(
                            "5. Mean Spectral Signature"
                        )

                        spectral_profile = hsi.mean(
                            axis=(0, 1)
                        )

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength (nm)":
                                    WAVELENGTHS,
                                "Mean Intensity":
                                    spectral_profile
                            }
                        )

                        st.line_chart(
                            spectral_df.set_index(
                                "Wavelength (nm)"
                            )
                        )

                        # ------------------------------------------------
                        # INTERPRETATION
                        # ------------------------------------------------

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

                        # ------------------------------------------------
                        # CONFIDENCE
                        # ------------------------------------------------

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

                        # ------------------------------------------------
                        # MODEL DETAILS
                        # ------------------------------------------------

                        st.subheader(
                            "8. Inference Details"
                        )

                        col1, col2 = st.columns(2)

                        with col1:

                            st.write(
                                "**Architecture:** 3D-CNN"
                            )

                            st.write(
                                "**Input:** 20 spectral bands"
                            )

                            st.write(
                                "**Model Input:** 20 × 32 × 32"
                            )

                            st.write(
                                "**Output Classes:** 3"
                            )

                        with col2:

                            st.write(
                                "**Parameters:** 18,243"
                            )

                            st.write(
                                "**Inference:** CPU compatible"
                            )

                            st.write(
                                "**Prediction Type:** "
                                "Tile-level classification"
                            )

                            st.write(
                                "**Checkpoint:** "
                                "terraspectra_3dcnn_tilesplit_best.pt"
                            )

                        st.info(
                            "This model performs tile-level classification. "
                            "It does not produce pixel-level disease "
                            "segmentation or a true disease heatmap."
                        )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# PAGE 6 — MONITORING
# ============================================================

elif page == "Monitoring":

    st.header(
        "Agricultural Monitoring"
    )

    st.info(
        "The current monitoring page is a prototype. "
        "GIS-linked geospatial inference will be added "
        "in a later project phase."
    )

    np.random.seed(42)

    monitoring_grid = pd.DataFrame(
        np.random.rand(
            12,
            12
        )
    )

    st.subheader(
        "Field Risk Visualization Prototype"
    )

    st.dataframe(
        monitoring_grid.style.format(
            "{:.2f}"
        ),
        use_container_width=True
    )

    st.info(
        "Future versions will connect model predictions "
        "with geographic coordinates to provide field-level "
        "disease monitoring and risk mapping."
    )