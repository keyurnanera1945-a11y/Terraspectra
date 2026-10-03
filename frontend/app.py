from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st


# ============================================================
# OPTIONAL TORCH IMPORT
# ============================================================

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except Exception:
    torch = None
    F = None
    TORCH_AVAILABLE = False


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ============================================================
# FILE PATHS
# ============================================================

PATCH_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hyperspectral_patches.csv"
)

# ------------------------------------------------------------
# Balanced model
# ------------------------------------------------------------

BALANCED_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_balanced_best.pt"
)

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

# ------------------------------------------------------------
# Tile-level model
# ------------------------------------------------------------

TILESPLIT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_tilesplit_best.pt"
)

TILESPLIT_HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_training_history.csv"
)

TILESPLIT_CONFUSION_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_confusion_matrix.csv"
)

TILESPLIT_REPORT_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_classification_report.csv"
)

TILESPLIT_PREDICTIONS_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_predictions.csv"
)

TILESPLIT_TRAIN_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "train_tilesplit_patches.csv"
)

TILESPLIT_VAL_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "val_tilesplit_patches.csv"
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

CLASS_SHORT_NAMES = {
    0: "Healthy",
    1: "Disease 1",
    2: "Disease 2",
}

EXPECTED_BANDS = 20
PATCH_SIZE = 32

APPROX_WAVELENGTHS = [
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
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TerraSpectra AI",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .main-title {
        font-size: 42px;
        font-weight: 750;
        letter-spacing: -1px;
        margin-bottom: 4px;
    }

    .subtitle {
        font-size: 17px;
        opacity: 0.72;
        margin-bottom: 28px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 28px;
        margin-bottom: 14px;
    }

    .small-label {
        font-size: 13px;
        opacity: 0.65;
        margin-bottom: 4px;
    }

    .status-card {
        padding: 18px;
        border-radius: 14px;
        border: 1px solid rgba(120, 120, 120, 0.22);
        margin-bottom: 12px;
    }

    .prediction-card {
        padding: 24px;
        border-radius: 16px;
        border: 1px solid rgba(0, 180, 100, 0.30);
        margin-top: 16px;
        margin-bottom: 18px;
    }

    .info-card {
        padding: 20px;
        border-radius: 14px;
        border: 1px solid rgba(100, 150, 220, 0.28);
        margin-top: 15px;
        margin-bottom: 15px;
    }

    div[data-testid="stMetric"] {
        padding: 8px 0;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# IMAGE DISPLAY HELPER
# ============================================================

def normalize_for_display(image):
    """
    Convert arbitrary numerical image data into a safe
    uint8 image in the range [0, 255].

    IMPORTANT:
    This function is ONLY used for visualization.

    The original HSI data remains unchanged for prediction.
    """

    image = np.asarray(image)

    if image.size == 0:
        return image

    image = image.astype(np.float32)

    finite_mask = np.isfinite(image)

    if not finite_mask.any():
        return np.zeros(image.shape, dtype=np.uint8)

    valid_values = image[finite_mask]

    min_value = float(valid_values.min())
    max_value = float(valid_values.max())

    if max_value <= min_value:
        return np.zeros(
            image.shape,
            dtype=np.uint8,
        )

    normalized = (
        (image - min_value)
        / (max_value - min_value)
    )

    normalized = np.nan_to_num(
        normalized,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    normalized = np.clip(
        normalized,
        0.0,
        1.0,
    )

    return (
        normalized * 255
    ).astype(np.uint8)


# ============================================================
# DATAFRAME HELPERS
# ============================================================

def find_column(df, candidates):
    """
    Find a dataframe column using case-insensitive matching.
    """

    if df is None:
        return None

    lower_map = {
        str(column).strip().lower(): column
        for column in df.columns
    }

    for candidate in candidates:

        key = str(candidate).strip().lower()

        if key in lower_map:
            return lower_map[key]

    return None


# ============================================================
# LOAD PATCH DATA
# ============================================================

@st.cache_data
def load_patch_dataframe():

    if not PATCH_CSV_PATH.exists():

        return None, (
            f"Patch CSV was not found:\n"
            f"{PATCH_CSV_PATH}"
        )

    try:

        df = pd.read_csv(
            PATCH_CSV_PATH
        )

        return df, None

    except Exception as e:

        return None, (
            f"Unable to read patch CSV: {e}"
        )


# ============================================================
# BALANCED MODEL METRICS
# ============================================================

def parse_balanced_metrics():

    metrics = {
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    }

    if not BALANCED_REPORT_PATH.exists():
        return metrics

    try:

        text = BALANCED_REPORT_PATH.read_text(
            encoding="utf-8"
        )

        for line in text.splitlines():

            line = line.strip()

            if line.startswith(
                "Validation Accuracy:"
            ):

                value = (
                    line.split(
                        ":",
                        1,
                    )[1]
                    .replace(
                        "%",
                        "",
                    )
                    .strip()
                )

                metrics["accuracy"] = float(
                    value
                )

            elif line.startswith(
                "Macro F1:"
            ):

                value = (
                    line.split(
                        ":",
                        1,
                    )[1]
                    .strip()
                )

                metrics["macro_f1"] = (
                    float(value) * 100
                )

            elif line.startswith(
                "Weighted F1:"
            ):

                value = (
                    line.split(
                        ":",
                        1,
                    )[1]
                    .strip()
                )

                metrics["weighted_f1"] = (
                    float(value) * 100
                )

    except Exception:
        pass

    return metrics


# ============================================================
# TILE-LEVEL RESULTS
# ============================================================

@st.cache_data
def load_tilesplit_results():

    results = {
        "accuracy": 90.78,
        "best_epoch": 2,
        "validation_samples": 1725,
        "confusion": None,
        "report": None,
        "predictions": None,
        "history": None,
    }

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    if TILESPLIT_HISTORY_PATH.exists():

        try:

            history = pd.read_csv(
                TILESPLIT_HISTORY_PATH
            )

            results["history"] = history

            if "val_accuracy" in history.columns:

                best_index = history[
                    "val_accuracy"
                ].idxmax()

                results["best_epoch"] = int(
                    history.loc[
                        best_index,
                        "epoch",
                    ]
                )

                results["accuracy"] = float(
                    history.loc[
                        best_index,
                        "val_accuracy",
                    ]
                )

        except Exception:
            pass

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if TILESPLIT_CONFUSION_PATH.exists():

        try:

            confusion = pd.read_csv(
                TILESPLIT_CONFUSION_PATH,
                header=None,
            )

            # Handle possible saved index column.
            if confusion.shape[1] == 4:

                try:

                    first_values = (
                        confusion.iloc[:, 0]
                        .astype(int)
                        .tolist()
                    )

                    if first_values == [0, 1, 2]:

                        confusion = confusion.iloc[
                            :,
                            1:,
                        ]

                except Exception:
                    pass

            confusion = confusion.iloc[
                :3,
                :3,
            ]

            confusion.columns = [
                "Healthy",
                "Disease 1",
                "Disease 2",
            ]

            confusion.index = [
                "Healthy",
                "Disease 1",
                "Disease 2",
            ]

            results["confusion"] = confusion

        except Exception:
            pass

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    if TILESPLIT_REPORT_PATH.exists():

        try:

            report = pd.read_csv(
                TILESPLIT_REPORT_PATH
            )

            results["report"] = report

        except Exception:
            pass

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    if TILESPLIT_PREDICTIONS_PATH.exists():

        try:

            predictions = pd.read_csv(
                TILESPLIT_PREDICTIONS_PATH
            )

            results["predictions"] = predictions

            results["validation_samples"] = len(
                predictions
            )

        except Exception:
            pass

    return results


# ============================================================
# PREPARE MODEL INPUT
# ============================================================

def prepare_hyperspectral_tensor(hsi):

    if not TORCH_AVAILABLE:

        raise RuntimeError(
            "PyTorch is not available in the current environment."
        )

    hsi = np.asarray(hsi)

    if hsi.ndim != 3:

        raise ValueError(
            "Expected H × W × Bands array. "
            f"Received {hsi.shape}."
        )

    height, width, bands = hsi.shape

    if bands != EXPECTED_BANDS:

        raise ValueError(
            f"The model requires exactly "
            f"{EXPECTED_BANDS} spectral bands. "
            f"Received {bands}."
        )

    data = hsi.astype(
        np.float32
    )

    if not np.isfinite(data).all():

        raise ValueError(
            "The uploaded HSI contains "
            "NaN or infinite values."
        )

    # Normalize ONLY for model input.
    #
    # Dataset training uses uint16 / 65535 normalization.
    # Keep this consistent with training.

    if data.max() > 1.0:

        data = data / 65535.0

    data = np.clip(
        data,
        0.0,
        1.0,
    )

    # HWC -> CHW
    data = np.transpose(
        data,
        (2, 0, 1),
    )

    tensor = torch.from_numpy(
        data
    )

    # [Bands, H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # [1, 1, Bands, 32, 32]
    tensor = F.interpolate(
        tensor,
        size=(
            EXPECTED_BANDS,
            PATCH_SIZE,
            PATCH_SIZE,
        ),
        mode="trilinear",
        align_corners=False,
    )

    return tensor.float()


# ============================================================
# LOAD MODEL
# ============================================================

@st.cache_resource
def load_model():

    if not TORCH_AVAILABLE:

        return None, (
            "PyTorch is not available."
        )

    if not BALANCED_MODEL_PATH.exists():

        return None, (
            "Balanced model was not found:\n"
            f"{BALANCED_MODEL_PATH}"
        )

    try:

        from src.model_3dcnn import (
            TerraSpectra3DCNN
        )

        model = TerraSpectra3DCNN(
            num_classes=3
        )

        checkpoint = torch.load(
            BALANCED_MODEL_PATH,
            map_location="cpu",
        )

        if isinstance(
            checkpoint,
            dict,
        ):

            if "model_state_dict" in checkpoint:

                state_dict = (
                    checkpoint[
                        "model_state_dict"
                    ]
                )

            elif "state_dict" in checkpoint:

                state_dict = (
                    checkpoint[
                        "state_dict"
                    ]
                )

            else:

                state_dict = checkpoint

        else:

            state_dict = checkpoint

        cleaned_state_dict = {}

        for key, value in (
            state_dict.items()
        ):

            clean_key = key

            if clean_key.startswith(
                "module."
            ):

                clean_key = (
                    clean_key[7:]
                )

            cleaned_state_dict[
                clean_key
            ] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        model.eval()

        return model, None

    except Exception as e:

        return None, (
            "Unable to load the balanced "
            f"3D CNN:\n{e}"
        )


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_model_prediction(
    model,
    tensor,
):

    with torch.no_grad():

        outputs = model(
            tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1,
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

    return (
        predicted_class,
        confidence,
        probabilities.cpu().numpy(),
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title(
    "🌱 TerraSpectra"
)

st.sidebar.caption(
    "Hyperspectral AI Crop Disease Intelligence"
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

st.sidebar.divider()

st.sidebar.caption(
    "TerraSpectra AI • 3D CNN"
)


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        '<div class="main-title">🌱 TerraSpectra</div>',
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div class="subtitle">
        Hyperspectral AI for crop-condition and disease analysis
        </div>
        """,
        unsafe_allow_html=True,
    )

    # --------------------------------------------------------
    # TOP METRICS
    # --------------------------------------------------------

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Hyperspectral Tiles",
            "1,115",
        )

    with col2:

        st.metric(
            "Spectral Bands",
            "20",
        )

    with col3:

        st.metric(
            "Extracted Patches",
            "8,688",
        )

    with col4:

        st.metric(
            "CNN Parameters",
            "18,243",
        )

    # --------------------------------------------------------
    # INTRODUCTION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">About TerraSpectra</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        TerraSpectra uses hyperspectral imagery and
        spectral-spatial deep learning to analyze crop
        conditions. The system processes 20-band
        hyperspectral data using a 3D Convolutional Neural
        Network.
        """
    )

    # --------------------------------------------------------
    # DATASET
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True,
    )

    dataset_df = pd.DataFrame(
        {
            "Property": [
                "Hyperspectral tiles",
                "Spectral bands",
                "Extracted patches",
                "Patch dimensions",
                "Data format",
                "Target classes",
            ],
            "Value": [
                "1,115",
                "20",
                "8,688",
                "32 × 32",
                "NPZ",
                "3",
            ],
        }
    )

    st.dataframe(
        dataset_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Model Architecture</div>',
        unsafe_allow_html=True,
    )

    model_df = pd.DataFrame(
        {
            "Property": [
                "Architecture",
                "Model",
                "Input",
                "Parameters",
                "Output classes",
            ],
            "Value": [
                "3D CNN",
                "TerraSpectra3DCNN",
                "20 × 32 × 32",
                "18,243",
                "3",
            ],
        }
    )

    st.dataframe(
        model_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # CURRENT TILE MODEL
    # --------------------------------------------------------

    tile_results = load_tilesplit_results()

    st.markdown(
        '<div class="section-title">Current Tile-Level Evaluation</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3 = st.columns(3)

    with c1:

        st.metric(
            "Validation Accuracy",
            f"{tile_results['accuracy']:.2f}%",
        )

    with c2:

        st.metric(
            "Validation Samples",
            f"{tile_results['validation_samples']:,}",
        )

    with c3:

        st.metric(
            "Best Epoch",
            tile_results["best_epoch"],
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title(
        "📊 Data Explorer"
    )

    st.write(
        "Explore the generated hyperspectral patch dataset."
    )

    df, error = load_patch_dataframe()

    if error:

        st.error(error)

    else:

        st.success(
            f"Loaded {len(df):,} hyperspectral patches."
        )

        class_col = find_column(
            df,
            [
                "class_id",
                "class",
                "label",
            ],
        )

        tile_col = find_column(
            df,
            [
                "tile_id",
                "tile",
            ],
        )

        width_col = find_column(
            df,
            [
                "patch_width",
                "width",
            ],
        )

        height_col = find_column(
            df,
            [
                "patch_height",
                "height",
            ],
        )

        col1, col2, col3, col4 = (
            st.columns(4)
        )

        with col1:

            st.metric(
                "Total Patches",
                f"{len(df):,}",
            )

        with col2:

            st.metric(
                "Classes",
                (
                    int(
                        df[class_col].nunique()
                    )
                    if class_col
                    else "N/A"
                ),
            )

        with col3:

            st.metric(
                "Tiles",
                (
                    int(
                        df[tile_col].nunique()
                    )
                    if tile_col
                    else "N/A"
                ),
            )

        with col4:

            if width_col:

                st.metric(
                    "Average Patch Width",
                    f"{df[width_col].mean():.1f}",
                )

            else:

                st.metric(
                    "Average Patch Width",
                    "N/A",
                )

        # ----------------------------------------------------
        # CLASS DISTRIBUTION
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Class Distribution</div>',
            unsafe_allow_html=True,
        )

        if class_col:

            counts = (
                df[class_col]
                .value_counts()
                .sort_index()
            )

            distribution = pd.DataFrame(
                {
                    "Class": [
                        CLASS_NAMES.get(
                            int(index),
                            f"Class {index}",
                        )
                        for index in counts.index
                    ],
                    "Samples": counts.values,
                }
            )

            st.dataframe(
                distribution,
                width="stretch",
                hide_index=True,
            )

            st.bar_chart(
                distribution.set_index(
                    "Class"
                ),
                width="stretch",
            )

        # ----------------------------------------------------
        # PATCH DATA
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Patch Metadata</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(100),
            width="stretch",
            height=450,
        )

        # ----------------------------------------------------
        # MISSING VALUES
        # ----------------------------------------------------

        st.markdown(
            '<div class="section-title">Data Quality</div>',
            unsafe_allow_html=True,
        )

        missing_df = pd.DataFrame(
            {
                "Column": df.columns,
                "Missing Values": [
                    int(
                        df[column].isna().sum()
                    )
                    for column in df.columns
                ],
            }
        )

        st.dataframe(
            missing_df,
            width="stretch",
            hide_index=True,
        )


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.title(
        "📈 Spectral Analysis"
    )

    st.write(
        """
        Principal Component Analysis summarizes the
        high-dimensional spectral information contained in
        the hyperspectral dataset.
        """
    )

    # --------------------------------------------------------
    # PCA METRICS
    # --------------------------------------------------------

    col1, col2, col3 = (
        st.columns(3)
    )

    with col1:

        st.metric(
            "PC1 Variance",
            "71.93%",
        )

    with col2:

        st.metric(
            "PC2 Variance",
            "27.31%",
        )

    with col3:

        st.metric(
            "PC3 Variance",
            "0.33%",
        )

    # --------------------------------------------------------
    # PCA TABLE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Principal Components</div>',
        unsafe_allow_html=True,
    )

    pca_df = pd.DataFrame(
        {
            "Principal Component": [
                "PC1",
                "PC2",
                "PC3",
                "PC4",
                "PC5",
            ],
            "Explained Variance (%)": [
                71.93,
                27.31,
                0.33,
                0.18,
                0.09,
            ],
        }
    )

    st.dataframe(
        pca_df,
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        pca_df.set_index(
            "Principal Component"
        ),
        width="stretch",
    )

    # --------------------------------------------------------
    # SPECTRAL BANDS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Spectral Bands</div>',
        unsafe_allow_html=True,
    )

    band_df = pd.DataFrame(
        {
            "Band": list(range(1, 21)),
            "Approx. Wavelength (nm)": (
                APPROX_WAVELENGTHS
            ),
        }
    )

    st.dataframe(
        band_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # HSI VIEWER
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Hyperspectral Band Viewer</div>',
        unsafe_allow_html=True,
    )

    spectral_file = st.file_uploader(
        "Upload an HSI .npz file",
        type=["npz"],
        key="spectral_npz",
    )

    if spectral_file is not None:

        try:

            npz_data = np.load(
                spectral_file
            )

            if "im" not in npz_data:

                st.error(
                    "The NPZ file does not contain "
                    "the required 'im' array."
                )

            else:

                hsi = npz_data["im"]

                if hsi.ndim != 3:

                    st.error(
                        f"Expected H × W × Bands. "
                        f"Received {hsi.shape}."
                    )

                else:

                    height, width, bands = (
                        hsi.shape
                    )

                    st.success(
                        f"Loaded HSI: "
                        f"{height} × {width} × {bands}"
                    )

                    selected_band = st.slider(
                        "Select spectral band",
                        0,
                        bands - 1,
                        0,
                        key="spectral_band",
                    )

                    # IMPORTANT:
                    # Normalize before displaying.
                    display_band = (
                        normalize_for_display(
                            hsi[
                                :,
                                :,
                                selected_band,
                            ]
                        )
                    )

                    st.image(
                        display_band,
                        caption=(
                            f"Band {selected_band + 1}"
                        ),
                        width="stretch",
                    )

                    values = hsi[
                        :,
                        :,
                        selected_band,
                    ].astype(
                        np.float32
                    )

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    with c1:

                        st.metric(
                            "Minimum",
                            f"{values.min():.2f}",
                        )

                    with c2:

                        st.metric(
                            "Maximum",
                            f"{values.max():.2f}",
                        )

                    with c3:

                        st.metric(
                            "Mean",
                            f"{values.mean():.2f}",
                        )

                    with c4:

                        st.metric(
                            "Std",
                            f"{values.std():.2f}",
                        )

        except Exception as e:

            st.error(
                f"Unable to read the NPZ file: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title(
        "🧠 Model Performance"
    )

    st.write(
        """
        Evaluation results from the TerraSpectra 3D CNN
        experiments, including the tile-level validation model.
        """
    )

    # ========================================================
    # BALANCED METRICS
    # ========================================================

    balanced_metrics = (
        parse_balanced_metrics()
    )

    balanced_accuracy = (
        balanced_metrics["accuracy"]
        if balanced_metrics["accuracy"]
        is not None
        else 92.12
    )

    balanced_macro_f1 = (
        balanced_metrics["macro_f1"]
        if balanced_metrics["macro_f1"]
        is not None
        else 59.80
    )

    balanced_weighted_f1 = (
        balanced_metrics["weighted_f1"]
        if balanced_metrics["weighted_f1"]
        is not None
        else 92.03
    )

    tile_results = (
        load_tilesplit_results()
    )

    tile_accuracy = (
        tile_results["accuracy"]
    )

    # ========================================================
    # EXPERIMENT COMPARISON
    # ========================================================

    st.markdown(
        '<div class="section-title">Experiment Comparison</div>',
        unsafe_allow_html=True,
    )

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Baseline 3D CNN",
                "Augmented 3D CNN",
                "Focal Loss",
                "Balanced Sampling",
                "Tile-Level Split",
            ],
            "Accuracy (%)": [
                92.75,
                84.81,
                7.54,
                balanced_accuracy,
                tile_accuracy,
            ],
        }
    )

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        performance_df.set_index(
            "Model"
        ),
        width="stretch",
    )

    # ========================================================
    # TILE MODEL SUMMARY
    # ========================================================

    st.markdown(
        '<div class="section-title">Tile-Level 3D CNN</div>',
        unsafe_allow_html=True,
    )

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Validation Accuracy",
            f"{tile_accuracy:.2f}%",
        )

    with c2:

        st.metric(
            "Validation Samples",
            f"{tile_results['validation_samples']:,}",
        )

    with c3:

        st.metric(
            "Best Epoch",
            tile_results["best_epoch"],
        )

    with c4:

        st.metric(
            "Parameters",
            "18,243",
        )

    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">Tile-Level Confusion Matrix</div>',
        unsafe_allow_html=True,
    )

    confusion = (
        tile_results["confusion"]
    )

    if confusion is not None:

        st.dataframe(
            confusion,
            width="stretch",
        )

        st.caption(
            "Rows represent actual classes. "
            "Columns represent predicted classes."
        )

    else:

        st.warning(
            "Tile-level confusion matrix file was not found."
        )

    # ========================================================
    # CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">Tile-Level Classification Report</div>',
        unsafe_allow_html=True,
    )

    report = (
        tile_results["report"]
    )

    if report is not None:

        report_display = (
            report.copy()
        )

        # Handle common sklearn column names.
        rename_map = {
            "precision": "Precision",
            "recall": "Recall",
            "f1-score": "F1 Score",
            "f1_score": "F1 Score",
            "support": "Support",
        }

        report_display = (
            report_display.rename(
                columns=rename_map
            )
        )

        for column in [
            "Precision",
            "Recall",
            "F1 Score",
        ]:

            if column in report_display.columns:

                report_display[column] = (
                    pd.to_numeric(
                        report_display[column],
                        errors="coerce",
                    )
                    * 100
                ).round(2)

        st.dataframe(
            report_display,
            width="stretch",
            hide_index=True,
        )

    else:

        st.warning(
            "Tile-level classification report file "
            "was not found."
        )

    # ========================================================
    # PER-CLASS PERFORMANCE
    # ========================================================

    if report is not None:

        report_copy = (
            report.copy()
        )

        lower_columns = {
            str(column)
            .strip()
            .lower(): column
            for column in report_copy.columns
        }

        class_column = (
            lower_columns.get(
                "class"
            )
        )

        precision_column = (
            lower_columns.get(
                "precision"
            )
        )

        recall_column = (
            lower_columns.get(
                "recall"
            )
        )

        f1_column = (
            lower_columns.get(
                "f1-score"
            )
            or lower_columns.get(
                "f1_score"
            )
        )

        if all(
            column is not None
            for column in [
                class_column,
                precision_column,
                recall_column,
                f1_column,
            ]
        ):

            class_rows = []

            for _, row in report_copy.iterrows():

                try:

                    class_id = int(
                        float(
                            row[
                                class_column
                            ]
                        )
                    )

                    if class_id in [
                        0,
                        1,
                        2,
                    ]:

                        class_rows.append(
                            (
                                class_id,
                                float(
                                    row[
                                        precision_column
                                    ]
                                ) * 100,
                                float(
                                    row[
                                        recall_column
                                    ]
                                ) * 100,
                                float(
                                    row[
                                        f1_column
                                    ]
                                ) * 100,
                            )
                        )

                except Exception:
                    continue

            if class_rows:

                st.markdown(
                    '<div class="section-title">Per-Class Performance</div>',
                    unsafe_allow_html=True,
                )

                class_metrics = pd.DataFrame(
                    {
                        "Class": [
                            CLASS_NAMES[
                                item[0]
                            ]
                            for item in class_rows
                        ],
                        "Precision (%)": [
                            item[1]
                            for item in class_rows
                        ],
                        "Recall (%)": [
                            item[2]
                            for item in class_rows
                        ],
                        "F1 Score (%)": [
                            item[3]
                            for item in class_rows
                        ],
                    }
                )

                st.dataframe(
                    class_metrics,
                    width="stretch",
                    hide_index=True,
                )

    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    history = (
        tile_results["history"]
    )

    if history is not None:

        st.markdown(
            '<div class="section-title">Training History</div>',
            unsafe_allow_html=True,
        )

        history_display = (
            history.copy()
        )

        rename_history = {
            "epoch": "Epoch",
            "train_loss": "Train Loss",
            "val_loss": "Validation Loss",
            "train_accuracy": "Train Accuracy",
            "val_accuracy": "Validation Accuracy",
        }

        history_display = (
            history_display.rename(
                columns=rename_history
            )
        )

        st.dataframe(
            history_display,
            width="stretch",
            hide_index=True,
        )

        if (
            "Epoch" in history_display.columns
            and "Validation Accuracy"
            in history_display.columns
        ):

            chart = (
                history_display[
                    [
                        "Epoch",
                        "Validation Accuracy",
                    ]
                ]
                .set_index("Epoch")
            )

            st.markdown(
                "#### Validation Accuracy"
            )

            st.line_chart(
                chart,
                width="stretch",
            )

    # ========================================================
    # MODEL INFORMATION
    # ========================================================

    st.markdown(
        '<div class="section-title">Tile-Level Model Information</div>',
        unsafe_allow_html=True,
    )

    model_info = pd.DataFrame(
        {
            "Property": [
                "Architecture",
                "Model",
                "Input",
                "Parameters",
                "Training Patches",
                "Validation Patches",
                "Training Strategy",
                "Split Strategy",
                "Best Epoch",
                "Validation Accuracy",
            ],
            "Value": [
                "3D CNN",
                "TerraSpectra3DCNN",
                "20 × 32 × 32",
                "18,243",
                "6,963",
                "1,725",
                "Balanced Sampling + Augmentation",
                "Tile-Level Split",
                str(
                    tile_results[
                        "best_epoch"
                    ]
                ),
                f"{tile_accuracy:.2f}%",
            ],
        }
    )

    st.dataframe(
        model_info,
        width="stretch",
        hide_index=True,
    )

    # ========================================================
    # BALANCED MODEL DETAILS
    # ========================================================

    st.markdown(
        '<div class="section-title">Balanced Sampling Model</div>',
        unsafe_allow_html=True,
    )

    b1, b2, b3 = (
        st.columns(3)
    )

    with b1:

        st.metric(
            "Validation Accuracy",
            f"{balanced_accuracy:.2f}%",
        )

    with b2:

        st.metric(
            "Macro F1",
            f"{balanced_macro_f1:.2f}%",
        )

    with b3:

        st.metric(
            "Weighted F1",
            f"{balanced_weighted_f1:.2f}%",
        )

    if BALANCED_CONFUSION_PATH.exists():

        st.image(
            str(
                BALANCED_CONFUSION_PATH
            ),
            caption="Balanced Sampling Confusion Matrix",
            width="stretch",
        )

    if BALANCED_REPORT_PATH.exists():

        try:

            balanced_report_text = (
                BALANCED_REPORT_PATH.read_text(
                    encoding="utf-8"
                )
            )

            with st.expander(
                "View Balanced Classification Report"
            ):

                st.code(
                    balanced_report_text,
                    language="text",
                )

        except Exception:
            pass

    # ========================================================
    # TILE PREDICTIONS
    # ========================================================

    predictions = (
        tile_results["predictions"]
    )

    if predictions is not None:

        st.markdown(
            '<div class="section-title">Tile-Level Validation Predictions</div>',
            unsafe_allow_html=True,
        )

        st.write(
            f"{len(predictions):,} validation predictions"
        )

        st.dataframe(
            predictions.head(100),
            width="stretch",
            hide_index=True,
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.title(
        "🔬 Hyperspectral Disease Prediction"
    )

    st.write(
        """
        Upload a hyperspectral NPZ tile and run the trained
        TerraSpectra 3D CNN.
        """
    )

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    uploaded_file = st.file_uploader(
        "Upload hyperspectral NPZ file",
        type=["npz"],
        key="prediction_npz",
    )

    if uploaded_file is None:

        st.info(
            """
            Upload an `.npz` file containing an `im` array.

            Example:

            data/raw/hyperspectral/0/0001.npz
            """
        )

    else:

        try:

            npz_data = np.load(
                uploaded_file
            )

            if "im" not in npz_data:

                st.error(
                    "Invalid NPZ file. "
                    "Required array: 'im'."
                )

            else:

                # IMPORTANT:
                # Keep original HSI values.
                # Do NOT normalize this array for prediction.
                hsi = np.asarray(
                    npz_data["im"]
                )

                if hsi.ndim != 3:

                    st.error(
                        "Invalid HSI shape. "
                        f"Expected H × W × Bands, "
                        f"received {hsi.shape}."
                    )

                else:

                    height, width, bands = (
                        hsi.shape
                    )

                    st.success(
                        "Hyperspectral image loaded successfully."
                    )

                    # ------------------------------------------------
                    # IMAGE INFORMATION
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Image Information</div>',
                        unsafe_allow_html=True,
                    )

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    with c1:

                        st.metric(
                            "Height",
                            height,
                        )

                    with c2:

                        st.metric(
                            "Width",
                            width,
                        )

                    with c3:

                        st.metric(
                            "Spectral Bands",
                            bands,
                        )

                    with c4:

                        st.metric(
                            "Data Type",
                            str(hsi.dtype),
                        )

                    # ------------------------------------------------
                    # BAND VIEWER
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Spectral Band Viewer</div>',
                        unsafe_allow_html=True,
                    )

                    selected_band = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=bands - 1,
                        value=0,
                        key="prediction_band",
                    )

                    # FIX:
                    # Normalize raw uint16 / hyperspectral values
                    # before passing to st.image().
                    display_band = (
                        normalize_for_display(
                            hsi[
                                :,
                                :,
                                selected_band,
                            ]
                        )
                    )

                    st.image(
                        display_band,
                        caption=(
                            f"Spectral Band "
                            f"{selected_band + 1}"
                        ),
                        width="stretch",
                    )

                    # ------------------------------------------------
                    # IMAGE STATISTICS
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Image Statistics</div>',
                        unsafe_allow_html=True,
                    )

                    stats = hsi.astype(
                        np.float32
                    )

                    c1, c2, c3, c4 = (
                        st.columns(4)
                    )

                    with c1:

                        st.metric(
                            "Minimum",
                            f"{stats.min():.2f}",
                        )

                    with c2:

                        st.metric(
                            "Maximum",
                            f"{stats.max():.2f}",
                        )

                    with c3:

                        st.metric(
                            "Mean",
                            f"{stats.mean():.2f}",
                        )

                    with c4:

                        st.metric(
                            "Std",
                            f"{stats.std():.2f}",
                        )

                    # ------------------------------------------------
                    # MODEL VALIDATION
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">AI Prediction</div>',
                        unsafe_allow_html=True,
                    )

                    if bands != EXPECTED_BANDS:

                        st.error(
                            f"Model input mismatch: "
                            f"this model requires "
                            f"{EXPECTED_BANDS} bands, "
                            f"but the uploaded image contains "
                            f"{bands} bands."
                        )

                    elif not TORCH_AVAILABLE:

                        st.error(
                            """
                            PyTorch is not available in the
                            current Python environment.
                            """
                        )

                    else:

                        model, model_error = (
                            load_model()
                        )

                        if model is None:

                            st.error(
                                "The prediction model "
                                "could not be loaded."
                            )

                            if model_error:

                                st.code(
                                    model_error
                                )

                        else:

                            st.success(
                                "TerraSpectra 3D CNN is ready."
                            )

                            predict_button = (
                                st.button(
                                    "🚀 Run Disease Prediction",
                                    type="primary",
                                    width="stretch",
                                )
                            )

                            if predict_button:

                                with st.spinner(
                                    "Running hyperspectral inference..."
                                ):

                                    try:

                                        tensor = (
                                            prepare_hyperspectral_tensor(
                                                hsi
                                            )
                                        )

                                        (
                                            predicted_class,
                                            confidence,
                                            probabilities,
                                        ) = (
                                            get_model_prediction(
                                                model,
                                                tensor,
                                            )
                                        )

                                    except Exception as e:

                                        st.error(
                                            "Prediction failed."
                                        )

                                        st.exception(
                                            e
                                        )

                                    else:

                                        class_name = (
                                            CLASS_NAMES.get(
                                                predicted_class,
                                                f"Class {predicted_class}",
                                            )
                                        )

                                        # ------------------------------------------------
                                        # RESULT
                                        # ------------------------------------------------

                                        st.markdown(
                                            '<div class="section-title">Prediction Result</div>',
                                            unsafe_allow_html=True,
                                        )

                                        r1, r2 = (
                                            st.columns(2)
                                        )

                                        with r1:

                                            st.metric(
                                                "Predicted Class",
                                                class_name,
                                            )

                                        with r2:

                                            st.metric(
                                                "Confidence",
                                                f"{confidence * 100:.2f}%",
                                            )

                                        st.markdown(
                                            f"""
                                            <div class="prediction-card">
                                                <b>AI Classification</b>
                                                <br><br>
                                                <span style="font-size:28px;">
                                                    {class_name}
                                                </span>
                                                <br><br>
                                                Model confidence:
                                                <b>{confidence * 100:.2f}%</b>
                                            </div>
                                            """,
                                            unsafe_allow_html=True,
                                        )

                                        # ------------------------------------------------
                                        # PROBABILITIES
                                        # ------------------------------------------------

                                        st.markdown(
                                            '<div class="section-title">Class Probabilities</div>',
                                            unsafe_allow_html=True,
                                        )

                                        probability_df = (
                                            pd.DataFrame(
                                                {
                                                    "Class": [
                                                        CLASS_NAMES.get(
                                                            i,
                                                            f"Class {i}",
                                                        )
                                                        for i in range(
                                                            len(
                                                                probabilities
                                                            )
                                                        )
                                                    ],
                                                    "Probability (%)": [
                                                        round(
                                                            float(
                                                                p
                                                            )
                                                            * 100,
                                                            2,
                                                        )
                                                        for p in probabilities
                                                    ],
                                                }
                                            )
                                        )

                                        st.dataframe(
                                            probability_df,
                                            width="stretch",
                                            hide_index=True,
                                        )

                                        st.bar_chart(
                                            probability_df.set_index(
                                                "Class"
                                            ),
                                            width="stretch",
                                        )


        except Exception as e:

            st.error(
                "Unable to process the uploaded file."
            )

            st.exception(e)


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.title(
        "🛰 Field Monitoring"
    )

    st.write(
        """
        Prototype visualization of crop-condition predictions
        across an agricultural field grid.
        """
    )

    st.info(
        """
        The current grid is simulated for dashboard
        demonstration. It can later be connected to actual
        UAV and GIS prediction outputs.
        """
    )

    rng = np.random.default_rng(
        42
    )

    field_data = rng.integers(
        0,
        3,
        size=(12, 12),
    )

    field_df = pd.DataFrame(
        field_data
    )

    st.markdown(
        '<div class="section-title">Field Disease Grid</div>',
        unsafe_allow_html=True,
    )

    st.dataframe(
        field_df,
        width="stretch",
        height=450,
    )

    unique, counts = np.unique(
        field_data,
        return_counts=True,
    )

    summary = {
        int(label): int(count)
        for label, count in zip(
            unique,
            counts,
        )
    }

    c1, c2, c3, c4 = (
        st.columns(4)
    )

    with c1:

        st.metric(
            "Grid Cells",
            int(
                field_data.size
            ),
        )

    with c2:

        st.metric(
            "Healthy",
            summary.get(
                0,
                0,
            ),
        )

    with c3:

        st.metric(
            "Disease 1",
            summary.get(
                1,
                0,
            ),
        )

    with c4:

        st.metric(
            "Disease 2",
            summary.get(
                2,
                0,
            ),
        )

    st.markdown(
        '<div class="section-title">Monitoring Summary</div>',
        unsafe_allow_html=True,
    )

    monitoring_df = pd.DataFrame(
        {
            "Condition": [
                "Healthy / Normal",
                "Disease Class 1",
                "Disease Class 2",
            ],
            "Grid Cells": [
                summary.get(0, 0),
                summary.get(1, 0),
                summary.get(2, 0),
            ],
        }
    )

    st.dataframe(
        monitoring_df,
        width="stretch",
        hide_index=True,
    )

    st.bar_chart(
        monitoring_df.set_index(
            "Condition"
        ),
        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "TerraSpectra AI • Hyperspectral Crop Analysis"
)