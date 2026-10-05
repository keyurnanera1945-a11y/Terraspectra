from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ============================================================
# OPTIONAL PYTORCH IMPORT
# ============================================================

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    F = None
    TORCH_AVAILABLE = False


# ============================================================
# PROJECT IMPORTS
# ============================================================

import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

try:
    from model_3dcnn import TerraSpectra3DCNN
except ImportError:
    TerraSpectra3DCNN = None


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TerraSpectra | AI Monitoring",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# PATHS
# ============================================================

PATCH_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hyperspectral_patches.csv"
)

# Current prediction model
TILESPLIT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_tilesplit_best.pt"
)

# Tile-level evaluation
TILESPLIT_CONFUSION_MATRIX_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_confusion_matrix.csv"
)

TILESPLIT_CLASSIFICATION_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_classification_report.csv"
)

TILESPLIT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_predictions.csv"
)

TILESPLIT_HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_training_history.csv"
)

# Older balanced-model evaluation files
BALANCED_EVALUATION_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "evaluation_balanced"
)

BALANCED_REPORT_PATH = (
    BALANCED_EVALUATION_DIR
    / "classification_report.txt"
)

BALANCED_CONFUSION_PATH = (
    BALANCED_EVALUATION_DIR
    / "confusion_matrix.png"
)

BALANCED_PREDICTIONS_PATH = (
    BALANCED_EVALUATION_DIR
    / "predictions.csv"
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

EXPECTED_BANDS = 20
PATCH_SIZE = 32

WAVELENGTHS = [
    420,
    440,
    500,
    520,
    540,
    560,
    580,
    600,
    620,
    640,
    660,
    680,
    700,
    720,
    740,
    760,
    770,
    800,
    850,
    900,
]


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background-color: #f7f9fc;
    }

    section[data-testid="stSidebar"] {
        background-color: #111827;
    }

    section[data-testid="stSidebar"] * {
        color: #f3f4f6;
    }

    .main-title {
        font-size: 2.4rem;
        font-weight: 750;
        margin-bottom: 0.15rem;
        color: #111827;
    }

    .subtitle {
        font-size: 1rem;
        color: #6b7280;
        margin-bottom: 1.5rem;
    }

    .section-title {
        font-size: 1.45rem;
        font-weight: 700;
        color: #111827;
        margin-top: 1rem;
        margin-bottom: 0.5rem;
    }

    .info-card {
        background: white;
        border: 1px solid #e5e7eb;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 15px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
    }

    .status-card {
        background: white;
        border-radius: 14px;
        padding: 18px;
        border: 1px solid #e5e7eb;
        min-height: 120px;
    }

    .small-label {
        color: #6b7280;
        font-size: 0.82rem;
        margin-bottom: 5px;
    }

    .big-value {
        color: #111827;
        font-size: 1.7rem;
        font-weight: 700;
    }

    .model-badge {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 20px;
        background: #e8f5e9;
        color: #166534;
        font-size: 0.82rem;
        font-weight: 600;
    }

    .warning-note {
        background: #fff7ed;
        border-left: 4px solid #f97316;
        padding: 12px 15px;
        border-radius: 8px;
        margin: 12px 0;
    }

    .footer {
        margin-top: 40px;
        padding-top: 20px;
        border-top: 1px solid #e5e7eb;
        color: #6b7280;
        text-align: center;
        font-size: 0.85rem;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_column(df, candidates):
    """
    Find the first matching column from a list of possible names.
    """
    if df is None or df.empty:
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = candidate.strip().lower()

        if key in normalized:
            return normalized[key]

    for col in df.columns:
        col_lower = str(col).strip().lower()

        for candidate in candidates:
            if candidate.strip().lower() in col_lower:
                return col

    return None


def normalize_band_for_display(band):
    """
    Convert raw hyperspectral values into a safe 0-1 image.

    This prevents Streamlit/PIL errors such as:
    'Data is outside [0, 255] and clamp is not set.'
    """
    band = np.asarray(band, dtype=np.float32)

    if band.ndim != 2:
        raise ValueError(
            f"Expected a 2D spectral band, received shape {band.shape}."
        )

    if not np.isfinite(band).all():
        raise ValueError(
            "Spectral band contains NaN or infinite values."
        )

    min_value = float(band.min())
    max_value = float(band.max())

    if max_value > min_value:
        band = (
            (band - min_value)
            / (max_value - min_value)
        )
    else:
        band = np.zeros_like(band)

    return np.clip(band, 0.0, 1.0)


@st.cache_data
def load_patch_dataframe():
    """
    Load extracted hyperspectral patch metadata.
    """
    if not PATCH_CSV_PATH.exists():
        return None

    try:
        return pd.read_csv(PATCH_CSV_PATH)
    except Exception:
        return None


@st.cache_data
def load_tilesplit_confusion_matrix():
    """
    Load tile-level confusion matrix.
    """
    if not TILESPLIT_CONFUSION_MATRIX_PATH.exists():
        return None

    try:
        df = pd.read_csv(
            TILESPLIT_CONFUSION_MATRIX_PATH,
            index_col=0
        )

        df = df.apply(
            pd.to_numeric,
            errors="coerce"
        ).fillna(0)

        return df.astype(int)

    except Exception:
        return None


@st.cache_data
def load_tilesplit_report():
    """
    Load tile-level classification report.
    """
    if not TILESPLIT_CLASSIFICATION_REPORT_PATH.exists():
        return None

    try:
        df = pd.read_csv(
            TILESPLIT_CLASSIFICATION_REPORT_PATH,
            index_col=0
        )

        df.index = df.index.astype(str)

        return df

    except Exception:
        return None


@st.cache_data
def load_tilesplit_predictions():
    """
    Load tile-level prediction results.
    """
    if not TILESPLIT_PREDICTIONS_PATH.exists():
        return None

    try:
        return pd.read_csv(
            TILESPLIT_PREDICTIONS_PATH
        )

    except Exception:
        return None


@st.cache_data
def load_tilesplit_history():
    """
    Load tile-level training history.
    """
    if not TILESPLIT_HISTORY_PATH.exists():
        return None

    try:
        return pd.read_csv(
            TILESPLIT_HISTORY_PATH
        )

    except Exception:
        return None


def get_report_value(report, row_name, column_name):
    """
    Safely retrieve a classification-report value.
    """
    if report is None:
        return None

    if row_name not in report.index:
        return None

    if column_name not in report.columns:
        return None

    value = pd.to_numeric(
        report.loc[row_name, column_name],
        errors="coerce"
    )

    if pd.isna(value):
        return None

    return float(value)


def percentage_value(value):
    """
    Convert a 0-1 metric to percentage.
    """
    if value is None:
        return None

    if abs(value) <= 1.5:
        return value * 100

    return value


def prepare_hyperspectral_tensor(hsi):
    """
    Convert HSI array:

        H x W x Bands

    into:

        1 x 1 x Bands x 32 x 32

    required by TerraSpectra3DCNN.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed in the current environment."
        )

    hsi = np.asarray(
        hsi,
        dtype=np.float32
    )

    if hsi.ndim != 3:
        raise ValueError(
            "Expected HSI data with shape "
            "(Height, Width, Bands). "
            f"Received {hsi.shape}."
        )

    if hsi.shape[2] != EXPECTED_BANDS:
        raise ValueError(
            f"Expected {EXPECTED_BANDS} spectral bands, "
            f"but uploaded file contains {hsi.shape[2]} bands."
        )

    if not np.isfinite(hsi).all():
        raise ValueError(
            "Hyperspectral data contains NaN or infinite values."
        )

    # Raw HSI is uint16.
    # Normalize using the same 16-bit range used during training.
    hsi = hsi / 65535.0

    hsi = np.clip(
        hsi,
        0.0,
        1.0
    )

    # HWC -> CHW
    tensor = torch.from_numpy(
        hsi
    ).permute(
        2,
        0,
        1
    )

    # [Bands, H, W]
    # ->
    # [1, 1, Bands, H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Resize:
    # [1, 1, 20, H, W]
    # ->
    # [1, 1, 20, 32, 32]
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


@st.cache_resource
def load_prediction_model():
    """
    Load the TILE-LEVEL 3D-CNN model.

    This is the model trained using the tile-level
    train/validation split.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed."
        )

    if TerraSpectra3DCNN is None:
        raise RuntimeError(
            "Could not import TerraSpectra3DCNN "
            "from src/model_3dcnn.py."
        )

    if not TILESPLIT_MODEL_PATH.exists():
        raise FileNotFoundError(
            "Tile-level model was not found:\n"
            f"{TILESPLIT_MODEL_PATH}"
        )

    model = TerraSpectra3DCNN(
        in_channels=1,
        num_classes=3
    )

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

    # Support different checkpoint formats.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):
        state_dict = checkpoint["model_state_dict"]

    elif (
        isinstance(checkpoint, dict)
        and "state_dict" in checkpoint
    ):
        state_dict = checkpoint["state_dict"]

    else:
        state_dict = checkpoint

    model.load_state_dict(
        state_dict
    )

    model.eval()

    return model


def get_model_prediction(model, tensor):
    """
    Generate class prediction and probabilities.
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is unavailable."
        )

    with torch.no_grad():

        output = model(
            tensor
        )

        probabilities = torch.softmax(
            output,
            dim=1
        )[0]

    predicted_class = int(
        torch.argmax(
            probabilities
        ).item()
    )

    return (
        predicted_class,
        probabilities.cpu().numpy()
    )


def create_tile_classification_table(report):
    """
    Create a clean class-level metrics table.
    """

    if report is None:
        return None

    rows = []

    for class_id in range(3):

        row_name = str(class_id)

        if row_name not in report.index:
            continue

        precision = get_report_value(
            report,
            row_name,
            "precision"
        )

        recall = get_report_value(
            report,
            row_name,
            "recall"
        )

        f1 = get_report_value(
            report,
            row_name,
            "f1-score"
        )

        support = get_report_value(
            report,
            row_name,
            "support"
        )

        rows.append(
            {
                "Class": CLASS_NAMES[class_id],
                "Precision (%)": (
                    round(
                        percentage_value(precision),
                        2
                    )
                    if precision is not None
                    else None
                ),
                "Recall (%)": (
                    round(
                        percentage_value(recall),
                        2
                    )
                    if recall is not None
                    else None
                ),
                "F1 Score (%)": (
                    round(
                        percentage_value(f1),
                        2
                    )
                    if f1 is not None
                    else None
                ),
                "Support": (
                    int(support)
                    if support is not None
                    else None
                ),
            }
        )

    if not rows:
        return None

    return pd.DataFrame(rows)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:1.65rem;
            font-weight:750;
            margin-bottom:2px;
        ">
            🌱 TerraSpectra
        </div>

        <div style="
            font-size:0.85rem;
            color:#9ca3af;
            margin-bottom:25px;
        ">
            Hyperspectral AI Monitoring
        </div>
        """,
        unsafe_allow_html=True
    )

    page = st.radio(
        "Navigation",
        [
            "Overview",
            "Data Explorer",
            "Spectral Analysis",
            "Model Performance",
            "Prediction",
            "Monitoring",
        ],
    )

    st.markdown("---")

    st.markdown(
        """
        <div style="
            font-size:0.78rem;
            color:#9ca3af;
            line-height:1.6;
        ">
        <b>AI Model</b><br>
        Tile-Level 3D-CNN<br><br>

        <b>Input</b><br>
        20-band hyperspectral imagery<br><br>

        <b>Patch Size</b><br>
        32 × 32 pixels
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# LOAD DATA
# ============================================================

patch_df = load_patch_dataframe()

tile_report = load_tilesplit_report()

tile_cm = load_tilesplit_confusion_matrix()

tile_predictions = load_tilesplit_predictions()

tile_history = load_tilesplit_history()


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">TerraSpectra</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            AI-powered hyperspectral analysis for early potato
            disease monitoring.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "HSI Tiles",
            "1,115"
        )

    with col2:
        st.metric(
            "Spectral Bands",
            "20"
        )

    with col3:
        st.metric(
            "Extracted Patches",
            "8,688"
        )

    with col4:
        st.metric(
            "Validation Samples",
            "1,725"
        )

    st.markdown(
        '<div class="section-title">Current AI Model</div>',
        unsafe_allow_html=True
    )

    model_col1, model_col2 = st.columns(
        [2, 1]
    )

    with model_col1:

        st.markdown(
            """
            <div class="info-card">

            <span class="model-badge">
            TILE-LEVEL 3D-CNN
            </span>

            <h3>
            Hyperspectral Disease Classifier
            </h3>

            <p style="color:#6b7280;">
            The current prediction system uses the model trained
            with a tile-level train/validation split. This reduces
            the risk of having patches from the same source tile
            appear in both training and validation sets.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with model_col2:

        tile_accuracy = get_report_value(
            tile_report,
            "accuracy",
            "f1-score"
        )

        if tile_accuracy is not None:
            tile_accuracy_pct = percentage_value(
                tile_accuracy
            )

            st.metric(
                "Validation Accuracy",
                f"{tile_accuracy_pct:.2f}%"
            )

        st.metric(
            "Model Parameters",
            "18,243"
        )

    st.markdown(
        '<div class="section-title">Project Summary</div>',
        unsafe_allow_html=True
    )

    summary_col1, summary_col2 = st.columns(2)

    with summary_col1:

        st.markdown(
            """
            <div class="info-card">

            <h4>Hyperspectral Input</h4>

            <p style="color:#6b7280;">
            Each image contains 20 spectral bands covering
            visible and near-infrared wavelengths. Spectral
            information is converted into fixed-size patches
            for deep learning.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    with summary_col2:

        st.markdown(
            """
            <div class="info-card">

            <h4>AI Classification</h4>

            <p style="color:#6b7280;">
            The 3D-CNN learns spatial and spectral patterns
            simultaneously and predicts three disease-related
            classes.
            </p>

            </div>
            """,
            unsafe_allow_html=True
        )

    if tile_report is not None:

        class_table = create_tile_classification_table(
            tile_report
        )

        if class_table is not None:

            st.markdown(
                '<div class="section-title">Validation Class Performance</div>',
                unsafe_allow_html=True
            )

            st.dataframe(
                class_table,
                width="stretch",
                hide_index=True
            )

            st.info(
                "The current validation results show that "
                "Disease Class 2 requires further model improvement."
            )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.markdown(
        '<div class="main-title">Data Explorer</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Explore extracted hyperspectral patches and class distribution.
        </div>
        """,
        unsafe_allow_html=True
    )

    if patch_df is None:

        st.error(
            "Patch metadata file was not found:\n"
            f"{PATCH_CSV_PATH}"
        )

    else:

        total_patches = len(
            patch_df
        )

        st.metric(
            "Total Patch Samples",
            f"{total_patches:,}"
        )

        st.markdown(
            '<div class="section-title">Dataset Information</div>',
            unsafe_allow_html=True
        )

        st.dataframe(
            patch_df.head(100),
            width="stretch"
        )

        st.markdown(
            '<div class="section-title">Class Distribution</div>',
            unsafe_allow_html=True
        )

        label_column = find_column(
            patch_df,
            [
                "label",
                "class",
                "target",
                "class_id"
            ]
        )

        if label_column is not None:

            class_counts = (
                patch_df[label_column]
                .value_counts()
                .sort_index()
            )

            class_distribution = pd.DataFrame(
                {
                    "Class": [
                        CLASS_NAMES.get(
                            int(index),
                            str(index)
                        )
                        for index in class_counts.index
                    ],
                    "Samples": class_counts.values
                }
            )

            st.bar_chart(
                class_distribution.set_index(
                    "Class"
                )
            )

            st.dataframe(
                class_distribution,
                width="stretch",
                hide_index=True
            )

        else:

            st.warning(
                "Could not automatically identify the class column."
            )


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.markdown(
        '<div class="main-title">Spectral Analysis</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Inspect individual hyperspectral bands and their
            spectral characteristics.
        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload a hyperspectral .npz file",
        type=["npz"],
        key="spectral_npz"
    )

    if uploaded_file is not None:

        try:

            npz_data = np.load(
                uploaded_file
            )

            if "im" not in npz_data:

                st.error(
                    "The uploaded NPZ file does not contain "
                    "the expected 'im' array."
                )

            else:

                hsi = npz_data["im"]

                if hsi.ndim != 3:

                    st.error(
                        f"Expected HSI shape (H, W, Bands), "
                        f"received {hsi.shape}."
                    )

                elif hsi.shape[2] != EXPECTED_BANDS:

                    st.error(
                        f"Expected {EXPECTED_BANDS} bands, "
                        f"received {hsi.shape[2]}."
                    )

                else:

                    st.success(
                        f"Loaded hyperspectral image: "
                        f"{hsi.shape[0]} × "
                        f"{hsi.shape[1]} × "
                        f"{hsi.shape[2]}"
                    )

                    selected_band = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=EXPECTED_BANDS - 1,
                        value=0
                    )

                    display_band = normalize_band_for_display(
                        hsi[:, :, selected_band]
                    )

                    wavelength = WAVELENGTHS[
                        selected_band
                    ]

                    st.image(
                        display_band,
                        caption=(
                            f"Band {selected_band + 1} "
                            f"• approximately {wavelength} nm"
                        ),
                        width="stretch"
                    )

                    band_values = np.asarray(
                        hsi[:, :, selected_band],
                        dtype=np.float32
                    )

                    spectral_col1, spectral_col2, spectral_col3 = (
                        st.columns(3)
                    )

                    with spectral_col1:
                        st.metric(
                            "Minimum",
                            f"{band_values.min():.2f}"
                        )

                    with spectral_col2:
                        st.metric(
                            "Maximum",
                            f"{band_values.max():.2f}"
                        )

                    with spectral_col3:
                        st.metric(
                            "Mean",
                            f"{band_values.mean():.2f}"
                        )

                    st.markdown(
                        '<div class="section-title">Spectral Profile</div>',
                        unsafe_allow_html=True
                    )

                    mean_spectrum = hsi.mean(
                        axis=(0, 1)
                    )

                    spectral_df = pd.DataFrame(
                        {
                            "Wavelength (nm)": WAVELENGTHS,
                            "Mean Intensity": mean_spectrum
                        }
                    )

                    st.line_chart(
                        spectral_df.set_index(
                            "Wavelength (nm)"
                        )
                    )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.markdown(
        '<div class="main-title">Model Performance</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Evaluation results for the tile-level 3D-CNN classifier.
        </div>
        """,
        unsafe_allow_html=True
    )

    # --------------------------------------------------------
    # Model status
    # --------------------------------------------------------

    if TILESPLIT_MODEL_PATH.exists():

        st.success(
            "Tile-level prediction model is available."
        )

    else:

        st.error(
            "Tile-level model file was not found:\n"
            f"{TILESPLIT_MODEL_PATH}"
        )

    # --------------------------------------------------------
    # Main metrics
    # --------------------------------------------------------

    accuracy = get_report_value(
        tile_report,
        "accuracy",
        "f1-score"
    )

    macro_f1 = get_report_value(
        tile_report,
        "macro avg",
        "f1-score"
    )

    weighted_f1 = get_report_value(
        tile_report,
        "weighted avg",
        "f1-score"
    )

    total_validation = None

    if tile_predictions is not None:
        total_validation = len(
            tile_predictions
        )

    metric1, metric2, metric3, metric4 = st.columns(4)

    with metric1:

        if accuracy is not None:
            st.metric(
                "Validation Accuracy",
                f"{percentage_value(accuracy):.2f}%"
            )
        else:
            st.metric(
                "Validation Accuracy",
                "N/A"
            )

    with metric2:

        if macro_f1 is not None:
            st.metric(
                "Macro F1",
                f"{percentage_value(macro_f1):.2f}%"
            )
        else:
            st.metric(
                "Macro F1",
                "N/A"
            )

    with metric3:

        if weighted_f1 is not None:
            st.metric(
                "Weighted F1",
                f"{percentage_value(weighted_f1):.2f}%"
            )
        else:
            st.metric(
                "Weighted F1",
                "N/A"
            )

    with metric4:

        if total_validation is not None:
            st.metric(
                "Validation Samples",
                f"{total_validation:,}"
            )
        else:
            st.metric(
                "Validation Samples",
                "1,725"
            )

    # --------------------------------------------------------
    # Class performance
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Class-Level Performance</div>',
        unsafe_allow_html=True
    )

    class_table = create_tile_classification_table(
        tile_report
    )

    if class_table is not None:

        st.dataframe(
            class_table,
            width="stretch",
            hide_index=True
        )

        chart_data = class_table[
            [
                "Class",
                "Precision (%)",
                "Recall (%)",
                "F1 Score (%)"
            ]
        ].copy()

        chart_data = chart_data.set_index(
            "Class"
        )

        st.bar_chart(
            chart_data
        )

    else:

        st.warning(
            "Tile-level classification report is not available."
        )

    # --------------------------------------------------------
    # Confusion Matrix
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True
    )

    if tile_cm is not None:

        cm_display = tile_cm.copy()

        if cm_display.shape == (3, 3):

            cm_display.index = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2]
            ]

            cm_display.columns = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2]
            ]

        st.dataframe(
            cm_display,
            width="stretch"
        )

        st.caption(
            "Rows represent actual classes and columns represent "
            "predicted classes."
        )

        row_totals = tile_cm.sum(
            axis=1
        ).replace(
            0,
            np.nan
        )

        normalized_cm = (
            tile_cm
            .div(
                row_totals,
                axis=0
            )
            * 100
        )

        if normalized_cm.shape == (3, 3):

            normalized_cm.index = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2]
            ]

            normalized_cm.columns = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2]
            ]

        st.markdown(
            "#### Normalized Confusion Matrix (%)"
        )

        st.dataframe(
            normalized_cm.round(2),
            width="stretch"
        )

    else:

        st.warning(
            "Tile-level confusion matrix was not found."
        )

    # --------------------------------------------------------
    # Training History
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Training History</div>',
        unsafe_allow_html=True
    )

    if tile_history is not None:

        epoch_column = find_column(
            tile_history,
            [
                "epoch"
            ]
        )

        val_accuracy_column = find_column(
            tile_history,
            [
                "val_accuracy",
                "validation_accuracy",
                "val_acc"
            ]
        )

        train_accuracy_column = find_column(
            tile_history,
            [
                "train_accuracy",
                "training_accuracy",
                "train_acc"
            ]
        )

        if epoch_column is not None:

            history_display = tile_history.copy()

            if val_accuracy_column is not None:

                val_acc = pd.to_numeric(
                    history_display[
                        val_accuracy_column
                    ],
                    errors="coerce"
                )

                if val_acc.max() <= 1.5:
                    val_acc = val_acc * 100

                history_display[
                    "Validation Accuracy (%)"
                ] = val_acc

            if train_accuracy_column is not None:

                train_acc = pd.to_numeric(
                    history_display[
                        train_accuracy_column
                    ],
                    errors="coerce"
                )

                if train_acc.max() <= 1.5:
                    train_acc = train_acc * 100

                history_display[
                    "Training Accuracy (%)"
                ] = train_acc

            history_chart_columns = []

            if "Training Accuracy (%)" in history_display:
                history_chart_columns.append(
                    "Training Accuracy (%)"
                )

            if "Validation Accuracy (%)" in history_display:
                history_chart_columns.append(
                    "Validation Accuracy (%)"
                )

            if history_chart_columns:

                chart_history = (
                    history_display[
                        [epoch_column]
                        + history_chart_columns
                    ]
                    .set_index(epoch_column)
                )

                st.line_chart(
                    chart_history
                )

                if (
                    "Validation Accuracy (%)"
                    in history_display
                ):

                    best_idx = (
                        history_display[
                            "Validation Accuracy (%)"
                        ]
                        .idxmax()
                    )

                    best_epoch = history_display.loc[
                        best_idx,
                        epoch_column
                    ]

                    best_val = history_display.loc[
                        best_idx,
                        "Validation Accuracy (%)"
                    ]

                    st.info(
                        f"Best validation accuracy occurred "
                        f"at epoch {best_epoch}: "
                        f"{best_val:.2f}%."
                    )

        with st.expander(
            "View training history table"
        ):

            st.dataframe(
                tile_history,
                width="stretch",
                hide_index=True
            )

    else:

        st.info(
            "Training history is not available."
        )

    # --------------------------------------------------------
    # Important interpretation
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Evaluation Note</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="warning-note">

        The tile-level model achieves strong overall validation
        accuracy, but the class-level results are uneven.
        In particular, Disease Class 2 currently has very low
        or zero recall. Therefore, overall accuracy should not
        be interpreted as equal performance across all disease
        classes.

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.markdown(
        '<div class="main-title">AI Prediction</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Upload a hyperspectral image and classify it using
            the tile-level 3D-CNN model.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="info-card">

        <span class="model-badge">
        ACTIVE MODEL
        </span>

        <h3>Tile-Level 3D-CNN</h3>

        <p style="color:#6b7280;">
        Prediction uses the model trained with the tile-level
        train/validation split.
        </p>

        </div>
        """,
        unsafe_allow_html=True
    )

    uploaded_file = st.file_uploader(
        "Upload hyperspectral .npz file",
        type=["npz"],
        key="prediction_npz"
    )

    if uploaded_file is not None:

        try:

            npz_data = np.load(
                uploaded_file
            )

            if "im" not in npz_data:

                st.error(
                    "The uploaded NPZ file does not contain "
                    "the expected 'im' array."
                )

            else:

                hsi = npz_data["im"]

                if hsi.ndim != 3:

                    st.error(
                        f"Invalid HSI shape: {hsi.shape}. "
                        "Expected (Height, Width, Bands)."
                    )

                elif hsi.shape[2] != EXPECTED_BANDS:

                    st.error(
                        f"This model expects {EXPECTED_BANDS} "
                        f"spectral bands, but the uploaded "
                        f"file contains {hsi.shape[2]}."
                    )

                else:

                    st.success(
                        f"Hyperspectral image loaded successfully: "
                        f"{hsi.shape[0]} × "
                        f"{hsi.shape[1]} × "
                        f"{hsi.shape[2]}"
                    )

                    # ------------------------------------------------
                    # Image preview
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Spectral Preview</div>',
                        unsafe_allow_html=True
                    )

                    selected_band = st.slider(
                        "Select band for visualization",
                        min_value=0,
                        max_value=EXPECTED_BANDS - 1,
                        value=0,
                        key="prediction_band"
                    )

                    display_band = normalize_band_for_display(
                        hsi[:, :, selected_band]
                    )

                    wavelength = WAVELENGTHS[
                        selected_band
                    ]

                    st.image(
                        display_band,
                        caption=(
                            f"Band {selected_band + 1} "
                            f"• {wavelength} nm"
                        ),
                        width="stretch"
                    )

                    # ------------------------------------------------
                    # Input information
                    # ------------------------------------------------

                    info1, info2, info3 = st.columns(3)

                    with info1:
                        st.metric(
                            "Height",
                            hsi.shape[0]
                        )

                    with info2:
                        st.metric(
                            "Width",
                            hsi.shape[1]
                        )

                    with info3:
                        st.metric(
                            "Bands",
                            hsi.shape[2]
                        )

                    st.markdown(
                        '<div class="section-title">AI Classification</div>',
                        unsafe_allow_html=True
                    )

                    # ------------------------------------------------
                    # Prediction
                    # ------------------------------------------------

                    if not TORCH_AVAILABLE:

                        st.error(
                            "PyTorch is not available in this "
                            "Python environment."
                        )

                    elif TerraSpectra3DCNN is None:

                        st.error(
                            "TerraSpectra3DCNN could not be imported "
                            "from src/model_3dcnn.py."
                        )

                    elif not TILESPLIT_MODEL_PATH.exists():

                        st.error(
                            "Tile-level model file was not found:\n"
                            f"{TILESPLIT_MODEL_PATH}"
                        )

                    else:

                        if st.button(
                            "🔍 Run Tile-Level AI Prediction",
                            type="primary",
                            width="stretch"
                        ):

                            try:

                                with st.spinner(
                                    "Preparing hyperspectral tensor..."
                                ):

                                    tensor = (
                                        prepare_hyperspectral_tensor(
                                            hsi
                                        )
                                    )

                                with st.spinner(
                                    "Running tile-level 3D-CNN..."
                                ):

                                    model = (
                                        load_prediction_model()
                                    )

                                    (
                                        predicted_class,
                                        probabilities
                                    ) = get_model_prediction(
                                        model,
                                        tensor
                                    )

                                predicted_name = (
                                    CLASS_NAMES[
                                        predicted_class
                                    ]
                                )

                                confidence = (
                                    probabilities[
                                        predicted_class
                                    ] * 100
                                )

                                # ----------------------------------------
                                # Result
                                # ----------------------------------------

                                st.success(
                                    "Prediction completed successfully."
                                )

                                result_col1, result_col2 = (
                                    st.columns(2)
                                )

                                with result_col1:

                                    st.markdown(
                                        """
                                        <div class="status-card">
                                        <div class="small-label">
                                        Predicted Class
                                        </div>
                                        <div class="big-value">
                                        """
                                        + predicted_name
                                        + """
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                with result_col2:

                                    st.markdown(
                                        """
                                        <div class="status-card">
                                        <div class="small-label">
                                        Model Confidence
                                        </div>
                                        <div class="big-value">
                                        """
                                        + f"{confidence:.2f}%"
                                        + """
                                        </div>
                                        </div>
                                        """,
                                        unsafe_allow_html=True
                                    )

                                # ----------------------------------------
                                # Probability chart
                                # ----------------------------------------

                                st.markdown(
                                    "#### Class Probabilities"
                                )

                                probability_df = pd.DataFrame(
                                    {
                                        "Class": [
                                            CLASS_NAMES[i]
                                            for i in range(
                                                len(
                                                    probabilities
                                                )
                                            )
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

                                st.dataframe(
                                    probability_df.assign(
                                        **{
                                            "Probability (%)":
                                            probability_df[
                                                "Probability (%)"
                                            ].round(2)
                                        }
                                    ),
                                    width="stretch",
                                    hide_index=True
                                )

                                st.caption(
                                    "Model: "
                                    "terraspectra_3dcnn_tilesplit_best.pt"
                                )

                            except Exception as exc:

                                st.error(
                                    "Prediction failed."
                                )

                                st.exception(
                                    exc
                                )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.markdown(
        '<div class="main-title">Monitoring</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <div class="subtitle">
            Spatial-style visualization for hyperspectral disease
            monitoring.
        </div>
        """,
        unsafe_allow_html=True
    )

    st.info(
        "The current monitoring view is a dashboard visualization "
        "prototype. Real GIS/geospatial integration can be added "
        "when GPS-linked field data is available."
    )

    # Reproducible simulated monitoring grid
    rng = np.random.default_rng(42)

    monitoring_grid = rng.integers(
        0,
        3,
        size=(12, 12)
    )

    monitoring_counts = np.bincount(
        monitoring_grid.flatten(),
        minlength=3
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Healthy Regions",
            int(monitoring_counts[0])
        )

    with col2:
        st.metric(
            "Disease Class 1",
            int(monitoring_counts[1])
        )

    with col3:
        st.metric(
            "Disease Class 2",
            int(monitoring_counts[2])
        )

    st.markdown(
        '<div class="section-title">Disease Monitoring Grid</div>',
        unsafe_allow_html=True
    )

    monitoring_df = pd.DataFrame(
        monitoring_grid
    )

    monitoring_df = monitoring_df.replace(
        {
            0: "Healthy",
            1: "Disease 1",
            2: "Disease 2"
        }
    )

    st.dataframe(
        monitoring_df,
        width="stretch"
    )

    st.markdown(
        '<div class="section-title">Class Distribution</div>',
        unsafe_allow_html=True
    )

    monitoring_chart = pd.DataFrame(
        {
            "Class": [
                "Healthy / Normal",
                "Disease Class 1",
                "Disease Class 2"
            ],
            "Regions": monitoring_counts
        }
    )

    st.bar_chart(
        monitoring_chart.set_index(
            "Class"
        )
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TerraSpectra • Hyperspectral AI Monitoring System
        <br>
        Tile-Level 3D-CNN • 20 Spectral Bands • 3-Class Classification
    </div>
    """,
    unsafe_allow_html=True
)