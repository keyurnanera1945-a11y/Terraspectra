from pathlib import Path
import inspect
import sys

import numpy as np
import pandas as pd
import streamlit as st

try:
    import torch
    import torch.nn.functional as F
except ImportError:
    torch = None
    F = None


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_3dcnn import TerraSpectra3DCNN


PATCH_CSV_PATH = PROJECT_ROOT / "outputs" / "hyperspectral_patches.csv"

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

    .main {
        background-color: #f7f9fc;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
        max-width: 1450px;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 18px;
        background: linear-gradient(
            135deg,
            #0f172a 0%,
            #1e293b 55%,
            #334155 100%
        );
        color: white;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.1rem;
    }

    .hero p {
        margin-top: 0.5rem;
        color: #cbd5e1;
        font-size: 1rem;
    }

    .metric-card {
        padding: 1.2rem;
        border-radius: 16px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 2px 8px rgba(15, 23, 42, 0.05);
        min-height: 125px;
    }

    .metric-label {
        color: #64748b;
        font-size: 0.85rem;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 1.65rem;
        font-weight: 750;
        margin-top: 0.3rem;
    }

    .metric-sub {
        color: #64748b;
        font-size: 0.78rem;
        margin-top: 0.2rem;
    }

    .prediction-card {
        padding: 1.5rem;
        border-radius: 18px;
        background: white;
        border: 1px solid #e2e8f0;
        box-shadow: 0 3px 12px rgba(15, 23, 42, 0.07);
        text-align: center;
        min-height: 210px;
    }

    .prediction-title {
        color: #64748b;
        font-size: 0.9rem;
        font-weight: 650;
    }

    .prediction-class {
        color: #0f172a;
        font-size: 1.8rem;
        font-weight: 800;
        margin-top: 0.7rem;
    }

    .confidence {
        color: #2563eb;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.5rem;
    }

    .section-title {
        font-size: 1.3rem;
        font-weight: 750;
        color: #0f172a;
        margin-top: 1.2rem;
        margin-bottom: 0.7rem;
    }

    .info-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        color: #475569;
        line-height: 1.55;
    }

    .warning-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #fff7ed;
        border: 1px solid #fed7aa;
        color: #9a3412;
    }

    .success-box {
        padding: 1rem 1.2rem;
        border-radius: 14px;
        background: #f0fdf4;
        border: 1px solid #bbf7d0;
        color: #166534;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def find_column(df, candidates):
    """Find the first matching column from candidate names."""

    for candidate in candidates:
        if candidate in df.columns:
            return candidate

    lower_map = {
        str(col).lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    return None


def normalize_band_for_display(band):
    """
    Convert raw hyperspectral values into a safe
    0-1 image for Streamlit display.
    """

    band = np.asarray(
        band,
        dtype=np.float32
    )

    if not np.isfinite(band).all():
        raise ValueError(
            "Spectral band contains NaN or infinite values."
        )

    min_val = float(band.min())
    max_val = float(band.max())

    if max_val > min_val:
        band = (
            band - min_val
        ) / (
            max_val - min_val
        )
    else:
        band = np.zeros_like(band)

    return np.clip(
        band,
        0.0,
        1.0
    )


def load_patch_data():
    if not PATCH_CSV_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(PATCH_CSV_PATH)
    except Exception:
        return pd.DataFrame()


def load_predictions():
    if not TILESPLIT_PREDICTIONS_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            TILESPLIT_PREDICTIONS_PATH
        )
    except Exception:
        return pd.DataFrame()


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


def load_history():
    if not TILESPLIT_HISTORY_PATH.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(
            TILESPLIT_HISTORY_PATH
        )
    except Exception:
        return pd.DataFrame()


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


def get_report_value(report, row, column):
    if report.empty:
        return None

    try:
        value = report.loc[row, column]
        return float(value)
    except Exception:
        return None


def percentage_value(value):
    if value is None:
        return "N/A"

    return f"{value * 100:.2f}%"


def prepare_hyperspectral_tensor(hsi):
    """
    Convert HSI from:
        H x W x Bands

    into model input:
        1 x 1 x Bands x 32 x 32
    """

    if torch is None or F is None:
        raise RuntimeError(
            "PyTorch is not installed in the current environment."
        )

    hsi = np.asarray(
        hsi,
        dtype=np.float32
    )

    if hsi.ndim != 3:
        raise ValueError(
            "Expected hyperspectral image with 3 dimensions."
        )

    height, width, bands = hsi.shape

    if bands != EXPECTED_BANDS:
        raise ValueError(
            f"Expected {EXPECTED_BANDS} spectral bands, "
            f"but received {bands}."
        )

    if not np.isfinite(hsi).all():
        raise ValueError(
            "Hyperspectral image contains NaN or infinite values."
        )

    # Normalize uint16-style hyperspectral values.
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
        align_corners=False,
    )

    return tensor


def create_terraspectra_model():
    """
    Create TerraSpectra3DCNN while remaining compatible
    with the current constructor implementation.
    """

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
            f"Model file not found:\n"
            f"{TILESPLIT_MODEL_PATH}"
        )

    model = create_terraspectra_model()

    try:
        checkpoint = torch.load(
            TILESPLIT_MODEL_PATH,
            map_location="cpu",
            weights_only=False,
        )
    except TypeError:
        checkpoint = torch.load(
            TILESPLIT_MODEL_PATH,
            map_location="cpu",
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
        tensor,
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.markdown(
    """
    <div style="
        font-size: 1.45rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    ">
        🌱 TerraSpectra
    </div>

    <div style="
        color: #64748b;
        font-size: 0.85rem;
        margin-bottom: 1rem;
    ">
        Hyperspectral AI Monitoring
    </div>
    """,
    unsafe_allow_html=True,
)

page = st.sidebar.radio(
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

st.sidebar.markdown("---")

st.sidebar.caption(
    "3D-CNN • Hyperspectral Imaging • Potato Disease Analysis"
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
# HERO
# ============================================================

st.markdown(
    """
    <div class="hero">

        <h1>🌱 TerraSpectra AI</h1>

        <p>
            Hyperspectral potato disease monitoring
            powered by a tile-level 3D Convolutional Neural Network.
        </p>

    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.subheader("System Overview")

    total_tiles = 1115
    total_bands = EXPECTED_BANDS
    total_patches = (
        len(patch_df)
        if not patch_df.empty
        else 8688
    )

    validation_samples = 1725

    accuracy = 0.9078

    cols = st.columns(4)

    metrics = [
        (
            "HSI Tiles",
            f"{total_tiles:,}",
            "Hyperspectral image tiles",
        ),
        (
            "Spectral Bands",
            str(total_bands),
            "Bands per HSI tile",
        ),
        (
            "Extracted Patches",
            f"{total_patches:,}",
            "Training-ready patches",
        ),
        (
            "Validation Accuracy",
            f"{accuracy * 100:.2f}%",
            "Tile-level 3D-CNN",
        ),
    ]

    for col, metric in zip(
        cols,
        metrics
    ):

        label, value, sub = metric

        col.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

                <div class="metric-sub">
                    {sub}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Current AI Model</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="info-box">

        <b>Model:</b> Tile-Level 3D-CNN<br>
        <b>Input:</b> 20-band hyperspectral tile<br>
        <b>Input resolution:</b> 32 × 32<br>
        <b>Output classes:</b> 3<br>
        <b>Validation samples:</b> 1,725<br>
        <b>Best validation accuracy:</b> 90.78%<br>
        <b>Parameters:</b> 18,243

        </div>
        """,
        unsafe_allow_html=True,
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.subheader("Hyperspectral Dataset Explorer")

    if patch_df.empty:

        st.warning(
            "Patch dataset CSV was not found."
        )

    else:

        st.write(
            f"Dataset contains "
            f"**{len(patch_df):,} extracted patches**."
        )

        st.dataframe(
            patch_df.head(100),
            use_container_width=True,
        )

        label_column = find_column(
            patch_df,
            [
                "label",
                "class",
                "target",
                "disease",
            ],
        )

        if label_column:

            st.markdown(
                '<div class="section-title">Class Distribution</div>',
                unsafe_allow_html=True,
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
# SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.subheader("Spectral Analysis")

    uploaded_file = st.file_uploader(
        "Upload a hyperspectral .npz file",
        type=["npz"],
        key="spectral_upload",
    )

    if uploaded_file is not None:

        try:

            data = np.load(
                uploaded_file
            )

            if "im" not in data:

                st.error(
                    "The uploaded NPZ file does not contain an 'im' array."
                )

            else:

                hsi = data["im"]

                if hsi.ndim != 3:

                    st.error(
                        "Expected a 3-dimensional hyperspectral image."
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
                    )

                    display_band = normalize_band_for_display(
                        hsi[:, :, band_number]
                    )

                    col1, col2 = st.columns(
                        2
                    )

                    with col1:

                        st.image(
                            display_band,
                            caption=(
                                f"Band {band_number + 1} "
                                f"({WAVELENGTHS[band_number]} nm)"
                                if band_number < len(WAVELENGTHS)
                                else f"Band {band_number + 1}"
                            ),
                            use_container_width=True,
                        )

                    with col2:

                        spectral_profile = hsi.mean(
                            axis=(0, 1)
                        )

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength (nm)": WAVELENGTHS[
                                    :bands
                                ],
                                "Mean Intensity": spectral_profile,
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

    st.subheader("Tile-Level 3D-CNN Performance")

    cols = st.columns(4)

    performance_metrics = [
        (
            "Accuracy",
            "90.78%",
            "Validation",
        ),
        (
            "Class 0 Recall",
            "96.48%",
            "Healthy / Normal",
        ),
        (
            "Class 1 Recall",
            "44.29%",
            "Disease Class 1",
        ),
        (
            "Class 2 Recall",
            "0.00%",
            "Disease Class 2",
        ),
    ]

    for col, metric in zip(
        cols,
        performance_metrics
    ):

        label, value, sub = metric

        col.markdown(
            f"""
            <div class="metric-card">

                <div class="metric-label">
                    {label}
                </div>

                <div class="metric-value">
                    {value}
                </div>

                <div class="metric-sub">
                    {sub}
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown(
        '<div class="section-title">Classification Report</div>',
        unsafe_allow_html=True,
    )

    if not report_df.empty:

        st.dataframe(
            report_df,
            use_container_width=True,
        )

    st.markdown(
        '<div class="section-title">Confusion Matrix</div>',
        unsafe_allow_html=True,
    )

    if not confusion_df.empty:

        st.dataframe(
            confusion_df,
            use_container_width=True,
        )

    st.markdown(
        """
        <div class="warning-box">

        <b>Important:</b>
        The overall accuracy is 90.78%, but the model currently
        has very weak performance on Disease Class 2.
        The 0% recall means the current model should not be
        interpreted as a reliable detector for that class.

        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        '<div class="section-title">Training History</div>',
        unsafe_allow_html=True,
    )

    if not history_df.empty:

        numeric_columns = history_df.select_dtypes(
            include=np.number
        ).columns.tolist()

        if numeric_columns:

            st.line_chart(
                history_df[
                    numeric_columns
                ]
            )


# ============================================================
# STEP 14 — PREDICTION VISUALIZATION
# ============================================================

elif page == "Prediction":

    st.subheader(
        "🔬 AI Hyperspectral Prediction"
    )

    st.markdown(
        """
        Upload an HSI tile to run the trained
        tile-level 3D-CNN model and visualize
        its prediction.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload hyperspectral NPZ tile",
        type=["npz"],
        key="prediction_upload",
    )

    if uploaded_file is None:

        st.markdown(
            """
            <div class="info-box">

            <b>Supported input:</b> `.npz` hyperspectral tile<br>
            <b>Expected array:</b> `im`<br>
            <b>Expected shape:</b> H × W × 20 spectral bands

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:

            data = np.load(
                uploaded_file
            )

            if "im" not in data:

                st.error(
                    "The uploaded NPZ file does not contain an 'im' array."
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
                    # VISUALIZATION CONTROLS
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Input Visualization</div>',
                        unsafe_allow_html=True,
                    )

                    selected_band = st.slider(
                        "Select spectral band for visualization",
                        min_value=0,
                        max_value=EXPECTED_BANDS - 1,
                        value=0,
                    )

                    selected_wavelength = (
                        WAVELENGTHS[selected_band]
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

                    model_tensor = prepare_hyperspectral_tensor(
                        hsi
                    )

                    # Extract model-view band.
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

                    image_col1, image_col2 = st.columns(
                        2
                    )

                    with image_col1:

                        st.markdown(
                            "#### Original Spectral Band"
                        )

                        st.image(
                            display_band,
                            caption=(
                                f"Band {selected_band + 1} "
                                f"• {selected_wavelength} nm "
                                f"• Original {height}×{width}"
                            ),
                            use_container_width=True,
                        )

                    with image_col2:

                        st.markdown(
                            "#### Model Input Preview"
                        )

                        st.image(
                            model_band_display,
                            caption=(
                                f"Band {selected_band + 1} "
                                f"• {selected_wavelength} nm "
                                f"• Resized 32×32"
                            ),
                            use_container_width=True,
                        )

                    # ------------------------------------------------
                    # RUN PREDICTION
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">AI Prediction</div>',
                        unsafe_allow_html=True,
                    )

                    if st.button(
                        "🔍 Run Tile-Level AI Prediction",
                        type="primary",
                        use_container_width=True,
                    ):

                        try:

                            (
                                predicted_class,
                                confidence,
                                probabilities,
                                tensor,
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
                                    probabilities,

                                "hsi":
                                    hsi,

                                "selected_band":
                                    selected_band,
                            }

                        except Exception as exc:

                            st.error(
                                f"Prediction failed: {exc}"
                            )

                    # ------------------------------------------------
                    # DISPLAY STORED RESULT
                    # ------------------------------------------------

                    result = st.session_state.get(
                        "prediction_result"
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

                        predicted_name = CLASS_NAMES.get(
                            predicted_class,
                            f"Class {predicted_class}"
                        )

                        # --------------------------------------------
                        # RESULT CARD
                        # --------------------------------------------

                        st.markdown(
                            '<div class="section-title">Prediction Result</div>',
                            unsafe_allow_html=True,
                        )

                        result_col1, result_col2, result_col3 = st.columns(
                            3
                        )

                        with result_col1:

                            st.markdown(
                                f"""
                                <div class="prediction-card">

                                    <div class="prediction-title">
                                        Predicted Class
                                    </div>

                                    <div class="prediction-class">
                                        {predicted_name}
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with result_col2:

                            st.markdown(
                                f"""
                                <div class="prediction-card">

                                    <div class="prediction-title">
                                        Model Confidence
                                    </div>

                                    <div class="confidence">
                                        {confidence * 100:.2f}%
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with result_col3:

                            st.markdown(
                                f"""
                                <div class="prediction-card">

                                    <div class="prediction-title">
                                        Spectral Bands Used
                                    </div>

                                    <div class="prediction-class">
                                        {bands}
                                    </div>

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # --------------------------------------------
                        # PROBABILITY VISUALIZATION
                        # --------------------------------------------

                        st.markdown(
                            '<div class="section-title">Class Probability Distribution</div>',
                            unsafe_allow_html=True,
                        )

                        probability_df = pd.DataFrame(
                            {
                                "Class": [
                                    CLASS_NAMES[0],
                                    CLASS_NAMES[1],
                                    CLASS_NAMES[2],
                                ],
                                "Probability": (
                                    probabilities * 100
                                ),
                            }
                        )

                        st.bar_chart(
                            probability_df.set_index(
                                "Class"
                            ),
                            y="Probability",
                        )

                        probability_display = (
                            probability_df.copy()
                        )

                        probability_display[
                            "Probability"
                        ] = probability_display[
                            "Probability"
                        ].map(
                            lambda x: f"{x:.2f}%"
                        )

                        st.dataframe(
                            probability_display,
                            hide_index=True,
                            use_container_width=True,
                        )

                        # --------------------------------------------
                        # SPECTRAL SIGNATURE
                        # --------------------------------------------

                        st.markdown(
                            '<div class="section-title">Mean Spectral Signature</div>',
                            unsafe_allow_html=True,
                        )

                        spectral_profile = hsi.mean(
                            axis=(0, 1)
                        )

                        spectral_df = pd.DataFrame(
                            {
                                "Wavelength (nm)": WAVELENGTHS,
                                "Mean Intensity": (
                                    spectral_profile
                                ),
                            }
                        )

                        st.line_chart(
                            spectral_df.set_index(
                                "Wavelength (nm)"
                            )
                        )

                        # --------------------------------------------
                        # INTERPRETATION
                        # --------------------------------------------

                        st.markdown(
                            '<div class="section-title">Prediction Interpretation</div>',
                            unsafe_allow_html=True,
                        )

                        if predicted_class == 0:

                            st.markdown(
                                """
                                <div class="success-box">

                                The tile was classified as
                                <b>Healthy / Normal</b> by the current
                                3D-CNN model.

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        elif predicted_class == 1:

                            st.markdown(
                                """
                                <div class="warning-box">

                                The tile was classified as
                                <b>Disease Class 1</b>.
                                This is an AI model prediction and
                                should be treated as an automated
                                screening result.

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        else:

                            st.markdown(
                                """
                                <div class="warning-box">

                                The tile was classified as
                                <b>Disease Class 2</b>.

                                <br><br>

                                The current validation results show
                                very weak recall for this class,
                                so this prediction requires additional
                                validation.

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        # --------------------------------------------
                        # MODEL DETAILS
                        # --------------------------------------------

                        st.markdown(
                            '<div class="section-title">Inference Details</div>',
                            unsafe_allow_html=True,
                        )

                        detail_col1, detail_col2 = st.columns(
                            2
                        )

                        with detail_col1:

                            st.markdown(
                                """
                                <div class="info-box">

                                <b>Architecture:</b> 3D-CNN<br>
                                <b>Input:</b> 20 spectral bands<br>
                                <b>Model input:</b> 20 × 32 × 32<br>
                                <b>Output classes:</b> 3<br>
                                <b>Parameters:</b> 18,243

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        with detail_col2:

                            st.markdown(
                                """
                                <div class="info-box">

                                <b>Model:</b> Tile-level split model<br>
                                <b>Checkpoint:</b>
                                terraspectra_3dcnn_tilesplit_best.pt<br>
                                <b>Inference:</b> CPU compatible<br>
                                <b>Visualization:</b>
                                Spectral + probability analysis

                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                        st.caption(
                            "Note: This visualization represents a tile-level classification. "
                            "It does not indicate pixel-level disease segmentation."
                        )

        except Exception as exc:

            st.error(
                f"Unable to process uploaded file: {exc}"
            )


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.subheader(
        "Agricultural Monitoring"
    )

    st.info(
        "Monitoring visualization is currently a prototype. "
        "GIS-linked geospatial inference will be added in the next phase."
    )

    np.random.seed(42)

    monitoring_grid = pd.DataFrame(
        np.random.rand(
            12,
            12
        )
    )

    st.subheader(
        "Field Risk Visualization"
    )

    st.dataframe(
        monitoring_grid.style.format(
            "{:.2f}"
        ),
        use_container_width=True,
    )

    st.markdown(
        """
        <div class="info-box">

        Future versions will connect model predictions
        with geospatial coordinates to provide field-level
        disease monitoring and risk mapping.

        </div>
        """,
        unsafe_allow_html=True,
    )