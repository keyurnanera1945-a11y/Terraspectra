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
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent


# ------------------------------------------------------------
# PATCH DATA
# ------------------------------------------------------------

PATCH_CSV_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "hyperspectral_patches.csv"
)


# ------------------------------------------------------------
# BALANCED MODEL
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

CLASSIFICATION_REPORT_PATH = (
    BALANCED_EVALUATION_DIR
    / "classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    BALANCED_EVALUATION_DIR
    / "confusion_matrix.png"
)

PREDICTIONS_PATH = (
    BALANCED_EVALUATION_DIR
    / "predictions.csv"
)


# ------------------------------------------------------------
# TILE-LEVEL MODEL
# ------------------------------------------------------------

TILESPLIT_MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_tilesplit_best.pt"
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

TILESPLIT_HISTORY_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "tilesplit_training_history.csv"
)

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


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

CLASS_DESCRIPTIONS = {
    0: (
        "The model classified this hyperspectral sample "
        "as Healthy / Normal."
    ),
    1: (
        "The model detected spectral-spatial characteristics "
        "associated with Disease Class 1."
    ),
    2: (
        "The model detected spectral-spatial characteristics "
        "associated with Disease Class 2."
    ),
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
    page_title="TerraSpectra",
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

    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 18px;
        opacity: 0.8;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 24px;
        font-weight: 650;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .success-box {
        padding: 20px;
        border-radius: 12px;
        border: 1px solid rgba(0, 180, 100, 0.4);
        margin-top: 15px;
        margin-bottom: 15px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(100, 150, 220, 0.35);
        margin-top: 15px;
        margin-bottom: 15px;
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
    Find a dataframe column using case-insensitive matching.
    """

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
# LOAD PATCH DATAFRAME
# ============================================================

@st.cache_data
def load_patch_dataframe():
    """
    Load hyperspectral patch metadata.
    """

    if not PATCH_CSV_PATH.exists():
        return None, (
            f"Patch CSV was not found:\n{PATCH_CSV_PATH}"
        )

    try:
        df = pd.read_csv(PATCH_CSV_PATH)

        return df, None

    except Exception as e:
        return None, f"Unable to read patch CSV: {e}"


# ============================================================
# BALANCED MODEL METRICS
# ============================================================

def parse_balanced_metrics():
    """
    Read real balanced-model metrics from
    classification_report.txt.
    """

    metrics = {
        "accuracy": None,
        "macro_f1": None,
        "weighted_f1": None,
    }

    if not CLASSIFICATION_REPORT_PATH.exists():
        return metrics

    try:
        text = CLASSIFICATION_REPORT_PATH.read_text(
            encoding="utf-8"
        )

        for line in text.splitlines():

            line = line.strip()

            if line.startswith("Validation Accuracy:"):

                value = (
                    line.split(":", 1)[1]
                    .replace("%", "")
                    .strip()
                )

                metrics["accuracy"] = float(value)

            elif line.startswith("Macro F1:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                )

                metrics["macro_f1"] = float(value) * 100

            elif line.startswith("Weighted F1:"):

                value = (
                    line.split(":", 1)[1]
                    .strip()
                )

                metrics["weighted_f1"] = float(value) * 100

    except Exception:
        pass

    return metrics


# ============================================================
# TILE-LEVEL METRICS
# ============================================================

@st.cache_data
def load_tilesplit_metrics():
    """
    Load tile-level evaluation results.
    """

    result = {
        "accuracy": 90.78,
        "best_epoch": 2,
        "validation_samples": 1725,
        "confusion_matrix": None,
        "classification_report": None,
        "predictions": None,
        "history": None,
        "error": None,
    }

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    if TILESPLIT_CONFUSION_MATRIX_PATH.exists():

        try:

            confusion_df = pd.read_csv(
                TILESPLIT_CONFUSION_MATRIX_PATH,
                header=None,
            )

            # Remove accidental index column if present.
            if confusion_df.shape[1] == 4:

                first_column = confusion_df.iloc[:, 0]

                try:

                    if list(
                        first_column.astype(int)
                    ) == [0, 1, 2]:

                        confusion_df = confusion_df.iloc[
                            :,
                            1:
                        ]

                except Exception:
                    pass

            confusion_df = confusion_df.iloc[
                :3,
                :3
            ]

            confusion_df.columns = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2],
            ]

            confusion_df.index = [
                CLASS_NAMES[0],
                CLASS_NAMES[1],
                CLASS_NAMES[2],
            ]

            result["confusion_matrix"] = confusion_df

        except Exception as e:

            result["error"] = (
                f"Unable to read tile-level confusion matrix: {e}"
            )

    # --------------------------------------------------------
    # CLASSIFICATION REPORT
    # --------------------------------------------------------

    if TILESPLIT_CLASSIFICATION_REPORT_PATH.exists():

        try:

            report_df = pd.read_csv(
                TILESPLIT_CLASSIFICATION_REPORT_PATH
            )

            result["classification_report"] = report_df

        except Exception as e:

            if result["error"] is None:
                result["error"] = (
                    "Unable to read tile-level "
                    f"classification report: {e}"
                )

    # --------------------------------------------------------
    # PREDICTIONS
    # --------------------------------------------------------

    if TILESPLIT_PREDICTIONS_PATH.exists():

        try:

            predictions_df = pd.read_csv(
                TILESPLIT_PREDICTIONS_PATH
            )

            result["predictions"] = predictions_df

            result["validation_samples"] = len(
                predictions_df
            )

        except Exception as e:

            if result["error"] is None:
                result["error"] = (
                    "Unable to read tile-level "
                    f"predictions: {e}"
                )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    if TILESPLIT_HISTORY_PATH.exists():

        try:

            history_df = pd.read_csv(
                TILESPLIT_HISTORY_PATH
            )

            result["history"] = history_df

            if "val_accuracy" in history_df.columns:

                best_index = history_df[
                    "val_accuracy"
                ].idxmax()

                result["best_epoch"] = int(
                    history_df.loc[
                        best_index,
                        "epoch",
                    ]
                )

                result["accuracy"] = float(
                    history_df.loc[
                        best_index,
                        "val_accuracy",
                    ]
                )

        except Exception:
            pass

    return result


# ============================================================
# PREPARE HYPERSPECTRAL TENSOR
# ============================================================

def prepare_hyperspectral_tensor(hsi):
    """
    Convert HSI from H x W x Bands into
    model input shape:

        [1, 1, 20, 32, 32]
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not available in the current environment."
        )

    hsi = np.asarray(hsi)

    if hsi.ndim != 3:
        raise ValueError(
            f"Expected H x W x Bands array, received {hsi.shape}"
        )

    height, width, bands = hsi.shape

    if bands != EXPECTED_BANDS:
        raise ValueError(
            f"Model requires exactly {EXPECTED_BANDS} bands. "
            f"Received {bands}."
        )

    data = hsi.astype(np.float32)

    if not np.isfinite(data).all():
        raise ValueError(
            "HSI contains NaN or infinite values."
        )

    if data.max() > 1.0:
        data = data / 65535.0

    data = np.clip(data, 0.0, 1.0)

    # HWC -> CHW
    data = np.transpose(data, (2, 0, 1))

    tensor = torch.from_numpy(data)

    # [Bands, H, W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # [1, 1, Bands, H, W]
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
# LOAD BALANCED MODEL
# ============================================================

@st.cache_resource
def load_model():
    """
    Load the balanced TerraSpectra 3D CNN.
    """

    if not TORCH_AVAILABLE:
        return None, (
            "PyTorch is not available. "
            "Install PyTorch in the project virtual environment."
        )

    if not BALANCED_MODEL_PATH.exists():
        return None, (
            "Balanced model was not found:\n"
            f"{BALANCED_MODEL_PATH}"
        )

    try:

        from src.model_3dcnn import TerraSpectra3DCNN

        model = TerraSpectra3DCNN(
            num_classes=3
        )

        checkpoint = torch.load(
            BALANCED_MODEL_PATH,
            map_location="cpu",
        )

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:
                state_dict = checkpoint["model_state_dict"]

            elif "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]

            else:
                state_dict = checkpoint

        else:
            state_dict = checkpoint

        cleaned_state_dict = {}

        for key, value in state_dict.items():

            new_key = key

            if new_key.startswith("module."):
                new_key = new_key[7:]

            cleaned_state_dict[new_key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=True,
        )

        model.eval()

        return model, None

    except Exception as e:

        return None, (
            f"Unable to load balanced 3D CNN:\n{e}"
        )


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_model_prediction(model, tensor):
    """
    Generate prediction and class probabilities.
    """

    with torch.no_grad():

        outputs = model(tensor)

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
            probabilities[predicted_class].item()
        )

    return (
        predicted_class,
        confidence,
        probabilities.cpu().numpy(),
    )


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 TerraSpectra")

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

st.sidebar.markdown(
    """
    **Project Pipeline**

    UAV Hyperspectral Data  
    ↓  
    Spectral Analysis  
    ↓  
    Patch Extraction  
    ↓  
    3D CNN  
    ↓  
    Disease Prediction  
    ↓  
    Field Monitoring
    """
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
        AI-powered hyperspectral crop disease intelligence
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.write(
        """
        TerraSpectra analyzes hyperspectral imagery using
        spectral-spatial deep learning to identify crop
        conditions and support agricultural field monitoring.
        """
    )

    st.markdown(
        '<div class="section-title">Project Overview</div>',
        unsafe_allow_html=True,
    )

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "HSI Tiles",
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
            "3D CNN Parameters",
            "18,243",
        )

    st.markdown(
        '<div class="section-title">Dataset</div>',
        unsafe_allow_html=True,
    )

    dataset_df = pd.DataFrame(
        {
            "Property": [
                "Hyperspectral Tiles",
                "Spectral Bands",
                "Patch Samples",
                "Training Patches",
                "Validation Patches",
                "Input Patch Size",
                "Data Format",
            ],
            "Value": [
                "1,115",
                "20",
                "8,688",
                "6,950",
                "1,738",
                "32 × 32",
                "NPZ",
            ],
        }
    )

    st.dataframe(
        dataset_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Model</div>',
        unsafe_allow_html=True,
    )

    model_df = pd.DataFrame(
        {
            "Property": [
                "Architecture",
                "Model",
                "Input",
                "Classes",
                "Training Strategy",
            ],
            "Value": [
                "3D CNN",
                "TerraSpectra3DCNN",
                "20 × 32 × 32",
                "3",
                "Balanced Sampling",
            ],
        }
    )

    st.dataframe(
        model_df,
        width="stretch",
        hide_index=True,
    )

    st.info(
        """
        The current prediction model is the balanced-sampling
        TerraSpectra 3D CNN.
        """
    )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("📊 Data Explorer")

    st.write(
        """
        Explore the hyperspectral patch metadata generated
        during dataset preparation.
        """
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

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Total Patches",
                f"{len(df):,}",
            )

        with col2:

            if class_col:
                st.metric(
                    "Classes",
                    int(df[class_col].nunique()),
                )
            else:
                st.metric(
                    "Classes",
                    "N/A",
                )

        with col3:

            if tile_col:
                st.metric(
                    "Tiles",
                    int(df[tile_col].nunique()),
                )
            else:
                st.metric(
                    "Tiles",
                    "N/A",
                )

        with col4:

            if width_col and height_col:

                mean_size = (
                    df[width_col].mean()
                )

                st.metric(
                    "Avg Patch Width",
                    f"{mean_size:.1f}",
                )

            else:

                st.metric(
                    "Avg Patch Width",
                    "N/A",
                )

        st.markdown(
            '<div class="section-title">Class Distribution</div>',
            unsafe_allow_html=True,
        )

        if class_col:

            class_counts = (
                df[class_col]
                .value_counts()
                .sort_index()
            )

            class_distribution = pd.DataFrame(
                {
                    "Class": [
                        CLASS_NAMES.get(
                            int(index),
                            f"Class {index}",
                        )
                        for index in class_counts.index
                    ],
                    "Samples": class_counts.values,
                }
            )

            st.dataframe(
                class_distribution,
                width="stretch",
                hide_index=True,
            )

            chart_df = class_distribution.set_index(
                "Class"
            )

            st.bar_chart(
                chart_df,
                width="stretch",
            )

        st.markdown(
            '<div class="section-title">Patch Metadata</div>',
            unsafe_allow_html=True,
        )

        st.dataframe(
            df.head(100),
            width="stretch",
            height=450,
        )

        st.markdown(
            '<div class="section-title">CSV Information</div>',
            unsafe_allow_html=True,
        )

        info_col1, info_col2 = st.columns(2)

        with info_col1:

            st.write("**Column names:**")

            st.code(
                "\n".join(
                    str(column)
                    for column in df.columns
                )
            )

        with info_col2:

            st.write("**Missing values:**")

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

    st.title("📈 Spectral Analysis")

    st.write(
        """
        Principal Component Analysis was used to analyze
        high-dimensional hyperspectral information.
        """
    )

    col1, col2, col3 = st.columns(3)

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

    st.markdown(
        '<div class="section-title">Principal Component Analysis</div>',
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

    st.markdown(
        '<div class="section-title">Spectral Bands</div>',
        unsafe_allow_html=True,
    )

    band_df = pd.DataFrame(
        {
            "Band": list(range(1, 21)),
            "Approx. Wavelength (nm)": APPROX_WAVELENGTHS,
        }
    )

    st.dataframe(
        band_df,
        width="stretch",
        hide_index=True,
    )

    st.info(
        """
        The TerraSpectra 3D CNN uses 20 spectral bands
        as the spectral dimension of its input.
        """
    )

    st.markdown(
        '<div class="section-title">Upload HSI for Spectral Inspection</div>',
        unsafe_allow_html=True,
    )

    spectral_file = st.file_uploader(
        "Upload a hyperspectral .npz file",
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
                    "The uploaded NPZ file does not contain "
                    "the required 'im' array."
                )

            else:

                hsi = npz_data["im"]

                if hsi.ndim != 3:

                    st.error(
                        f"Expected H × W × Bands, "
                        f"received {hsi.shape}"
                    )

                else:

                    height, width, bands_count = hsi.shape

                    st.success(
                        f"Loaded HSI: "
                        f"{height} × {width} × {bands_count}"
                    )

                    selected_band = st.slider(
                        "Select spectral band",
                        min_value=0,
                        max_value=bands_count - 1,
                        value=0,
                        key="spectral_band",
                    )

                    st.image(
                        hsi[:, :, selected_band],
                        caption=(
                            f"Spectral Band "
                            f"{selected_band + 1}"
                        ),
                        width="stretch",
                    )

                    spectral_values = hsi[
                        :,
                        :,
                        selected_band,
                    ].astype(float)

                    stat_col1, stat_col2 = st.columns(2)

                    with stat_col1:

                        st.metric(
                            "Minimum",
                            f"{spectral_values.min():.2f}",
                        )

                    with stat_col2:

                        st.metric(
                            "Maximum",
                            f"{spectral_values.max():.2f}",
                        )

                    stat_col3, stat_col4 = st.columns(2)

                    with stat_col3:

                        st.metric(
                            "Mean",
                            f"{spectral_values.mean():.2f}",
                        )

                    with stat_col4:

                        st.metric(
                            "Std",
                            f"{spectral_values.std():.2f}",
                        )

        except Exception as e:

            st.error(
                f"Unable to read the NPZ file: {e}"
            )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🧠 Model Performance")

    st.write(
        """
        Comparison of the TerraSpectra 3D CNN training
        approaches evaluated during model development.
        """
    )

    # ========================================================
    # MODEL COMPARISON
    # ========================================================

    balanced_metrics = parse_balanced_metrics()

    balanced_accuracy = (
        balanced_metrics["accuracy"]
        if balanced_metrics["accuracy"] is not None
        else 92.12
    )

    balanced_macro_f1 = (
        balanced_metrics["macro_f1"]
        if balanced_metrics["macro_f1"] is not None
        else 59.80
    )

    balanced_weighted_f1 = (
        balanced_metrics["weighted_f1"]
        if balanced_metrics["weighted_f1"] is not None
        else 92.03
    )

    tilesplit_metrics = load_tilesplit_metrics()

    tilesplit_accuracy = tilesplit_metrics[
        "accuracy"
    ]

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
                tilesplit_accuracy,
            ],
            "Macro F1 (%)": [
                57.21,
                42.60,
                11.22,
                balanced_macro_f1,
                45.00,
            ],
            "Weighted F1 (%)": [
                92.27,
                86.59,
                3.39,
                balanced_weighted_f1,
                89.00,
            ],
        }
    )

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True,
    )

    st.markdown(
        '<div class="section-title">Accuracy Comparison</div>',
        unsafe_allow_html=True,
    )

    accuracy_chart = (
        performance_df
        .set_index("Model")[["Accuracy (%)"]]
    )

    st.bar_chart(
        accuracy_chart,
        width="stretch",
    )

    # ========================================================
    # BALANCED SAMPLING MODEL
    # ========================================================

    st.markdown(
        '<div class="section-title">Balanced Sampling 3D CNN</div>',
        unsafe_allow_html=True,
    )

    m1, m2, m3 = st.columns(3)

    with m1:

        st.metric(
            "Validation Accuracy",
            f"{balanced_accuracy:.4f}%",
        )

    with m2:

        st.metric(
            "Macro F1",
            f"{balanced_macro_f1:.2f}%",
        )

    with m3:

        st.metric(
            "Weighted F1",
            f"{balanced_weighted_f1:.2f}%",
        )

    st.divider()

    # ========================================================
    # BALANCED CONFUSION MATRIX
    # ========================================================

    st.markdown(
        '<div class="section-title">Balanced Model Confusion Matrix</div>',
        unsafe_allow_html=True,
    )

    if CONFUSION_MATRIX_PATH.exists():

        st.image(
            str(CONFUSION_MATRIX_PATH),
            caption=(
                "Balanced Sampling 3D CNN "
                "Confusion Matrix"
            ),
            width="stretch",
        )

    else:

        st.info(
            "Balanced confusion matrix was not found at:\n"
            f"{CONFUSION_MATRIX_PATH}"
        )

    # ========================================================
    # BALANCED CLASSIFICATION REPORT
    # ========================================================

    st.markdown(
        '<div class="section-title">Balanced Classification Report</div>',
        unsafe_allow_html=True,
    )

    if CLASSIFICATION_REPORT_PATH.exists():

        try:

            report_text = (
                CLASSIFICATION_REPORT_PATH.read_text(
                    encoding="utf-8"
                )
            )

            st.code(
                report_text,
                language="text",
            )

        except Exception as e:

            st.warning(
                f"Unable to read classification report: {e}"
            )

    else:

        st.info(
            "Balanced classification report was not found."
        )

    # ========================================================
    # BALANCED PREDICTIONS
    # ========================================================

    if PREDICTIONS_PATH.exists():

        try:

            predictions_df = pd.read_csv(
                PREDICTIONS_PATH
            )

            st.markdown(
                '<div class="section-title">Balanced Validation Predictions</div>',
                unsafe_allow_html=True,
            )

            st.write(
                f"Validation predictions: "
                f"{len(predictions_df):,} samples"
            )

            st.dataframe(
                predictions_df.head(100),
                width="stretch",
                hide_index=True,
            )

        except Exception as e:

            st.warning(
                f"Unable to read predictions CSV: {e}"
            )

    # ========================================================
    # TILE-LEVEL MODEL
    # ========================================================

    st.divider()

    st.markdown(
        '<div class="section-title">🧩 Tile-Level Split 3D CNN</div>',
        unsafe_allow_html=True,
    )

    st.write(
        """
        The tile-level split prevents patches from the same
        hyperspectral tile from being distributed between
        training and validation sets. This provides a more
        realistic evaluation of model generalization to
        unseen tiles.
        """
    )

    # --------------------------------------------------------
    # TILE-LEVEL SUMMARY METRICS
    # --------------------------------------------------------

    tile_col1, tile_col2, tile_col3, tile_col4 = (
        st.columns(4)
    )

    with tile_col1:

        st.metric(
            "Validation Accuracy",
            f"{tilesplit_accuracy:.2f}%",
        )

    with tile_col2:

        st.metric(
            "Best Epoch",
            tilesplit_metrics["best_epoch"],
        )

    with tile_col3:

        st.metric(
            "Validation Samples",
            f"{tilesplit_metrics['validation_samples']:,}",
        )

    with tile_col4:

        st.metric(
            "Model Parameters",
            "18,243",
        )

    # --------------------------------------------------------
    # TILE-LEVEL CONFUSION MATRIX
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Tile-Level Confusion Matrix</div>',
        unsafe_allow_html=True,
    )

    confusion_df = tilesplit_metrics[
        "confusion_matrix"
    ]

    if confusion_df is not None:

        st.dataframe(
            confusion_df,
            width="stretch",
        )

        st.caption(
            "Rows represent actual classes and columns "
            "represent predicted classes."
        )

    else:

        st.info(
            "Tile-level confusion matrix was not found."
        )

    # --------------------------------------------------------
    # TILE-LEVEL CLASSIFICATION REPORT
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Tile-Level Classification Report</div>',
        unsafe_allow_html=True,
    )

    tile_report_df = tilesplit_metrics[
        "classification_report"
    ]

    if tile_report_df is not None:

        report_display = tile_report_df.copy()

        # Rename common sklearn columns for dashboard display.
        report_display = report_display.rename(
            columns={
                "precision": "Precision",
                "recall": "Recall",
                "f1-score": "F1 Score",
                "f1_score": "F1 Score",
                "support": "Support",
            }
        )

        # Convert numeric metrics to readable percentages.
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
                    ) * 100
                ).round(2)

        st.dataframe(
            report_display,
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Tile-level classification report was not found."
        )

    # --------------------------------------------------------
    # TILE-LEVEL CLASS METRICS
    # --------------------------------------------------------

    if tile_report_df is not None:

        report_copy = tile_report_df.copy()

        report_copy.columns = [
            str(column).strip().lower()
            for column in report_copy.columns
        ]

        class_rows = report_copy[
            report_copy["class"].astype(str).isin(
                [
                    "0",
                    "1",
                    "2",
                    "0.0",
                    "1.0",
                    "2.0",
                ]
            )
        ].copy()

        if len(class_rows) > 0:

            st.markdown(
                '<div class="section-title">Per-Class Performance</div>',
                unsafe_allow_html=True,
            )

            metric_columns = st.columns(3)

            for position, (_, row) in enumerate(
                class_rows.iterrows()
            ):

                class_id = int(
                    float(row["class"])
                )

                precision = float(
                    row["precision"]
                ) * 100

                recall = float(
                    row["recall"]
                ) * 100

                f1 = float(
                    row["f1-score"]
                ) * 100

                with metric_columns[position % 3]:

                    st.markdown(
                        f"**{CLASS_NAMES.get(class_id, f'Class {class_id}')}**"
                    )

                    st.metric(
                        "Precision",
                        f"{precision:.2f}%",
                    )

                    st.metric(
                        "Recall",
                        f"{recall:.2f}%",
                    )

                    st.metric(
                        "F1 Score",
                        f"{f1:.2f}%",
                    )

    # --------------------------------------------------------
    # TILE-LEVEL TRAINING HISTORY
    # --------------------------------------------------------

    tile_history_df = tilesplit_metrics[
        "history"
    ]

    if tile_history_df is not None:

        st.markdown(
            '<div class="section-title">Tile-Level Training History</div>',
            unsafe_allow_html=True,
        )

        history_display = tile_history_df.copy()

        if "epoch" in history_display.columns:

            history_display = history_display.rename(
                columns={
                    "epoch": "Epoch",
                    "train_loss": "Train Loss",
                    "val_loss": "Validation Loss",
                    "train_accuracy": "Train Accuracy",
                    "val_accuracy": "Validation Accuracy",
                }
            )

            st.dataframe(
                history_display,
                width="stretch",
                hide_index=True,
            )

        # Validation accuracy chart
        if (
            "Epoch" in history_display.columns
            and "Validation Accuracy"
            in history_display.columns
        ):

            accuracy_history = (
                history_display[
                    [
                        "Epoch",
                        "Validation Accuracy",
                    ]
                ]
                .set_index("Epoch")
            )

            st.markdown(
                "#### Validation Accuracy by Epoch"
            )

            st.line_chart(
                accuracy_history,
                width="stretch",
            )

    # --------------------------------------------------------
    # TILE-LEVEL PREDICTIONS
    # --------------------------------------------------------

    tile_predictions_df = tilesplit_metrics[
        "predictions"
    ]

    st.markdown(
        '<div class="section-title">Tile-Level Validation Predictions</div>',
        unsafe_allow_html=True,
    )

    if tile_predictions_df is not None:

        st.write(
            f"Tile-level validation predictions: "
            f"{len(tile_predictions_df):,} samples"
        )

        st.dataframe(
            tile_predictions_df.head(100),
            width="stretch",
            hide_index=True,
        )

    else:

        st.info(
            "Tile-level prediction file was not found."
        )

    # --------------------------------------------------------
    # TILE-LEVEL MODEL INFORMATION
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Tile-Level Model Information</div>',
        unsafe_allow_html=True,
    )

    tile_model_info = pd.DataFrame(
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
                "Best Validation Accuracy",
                "Model File",
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
                    tilesplit_metrics["best_epoch"]
                ),
                f"{tilesplit_accuracy:.2f}%",
                "terraspectra_3dcnn_tilesplit_best.pt",
            ],
        }
    )

    st.dataframe(
        tile_model_info,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # IMPORTANT OBSERVATION
    # --------------------------------------------------------

    st.info(
        """
        Tile-level evaluation provides a more realistic
        validation setup because complete hyperspectral tiles
        are separated between training and validation.

        The current tile-level result is 90.78% validation
        accuracy. The class-level report should also be
        considered because overall accuracy is influenced by
        the class distribution.
        """
    )

    # --------------------------------------------------------
    # FILE STATUS
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-title">Tile-Level Output Files</div>',
        unsafe_allow_html=True,
    )

    tile_files_df = pd.DataFrame(
        {
            "Output": [
                "Best Model",
                "Training History",
                "Confusion Matrix",
                "Classification Report",
                "Predictions",
            ],
            "Path": [
                str(TILESPLIT_MODEL_PATH.relative_to(PROJECT_ROOT)),
                str(TILESPLIT_HISTORY_PATH.relative_to(PROJECT_ROOT)),
                str(TILESPLIT_CONFUSION_MATRIX_PATH.relative_to(PROJECT_ROOT)),
                str(TILESPLIT_CLASSIFICATION_REPORT_PATH.relative_to(PROJECT_ROOT)),
                str(TILESPLIT_PREDICTIONS_PATH.relative_to(PROJECT_ROOT)),
            ],
            "Status": [
                "Available"
                if TILESPLIT_MODEL_PATH.exists()
                else "Missing",
                "Available"
                if TILESPLIT_HISTORY_PATH.exists()
                else "Missing",
                "Available"
                if TILESPLIT_CONFUSION_MATRIX_PATH.exists()
                else "Missing",
                "Available"
                if TILESPLIT_CLASSIFICATION_REPORT_PATH.exists()
                else "Missing",
                "Available"
                if TILESPLIT_PREDICTIONS_PATH.exists()
                else "Missing",
            ],
        }
    )

    st.dataframe(
        tile_files_df,
        width="stretch",
        hide_index=True,
    )

    if tilesplit_metrics["error"]:

        st.warning(
            tilesplit_metrics["error"]
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.title("🔬 Hyperspectral Disease Prediction")

    st.write(
        """
        Upload a hyperspectral `.npz` tile to run the
        trained balanced-sampling 3D CNN.
        """
    )

    uploaded_file = st.file_uploader(
        "Upload hyperspectral NPZ file",
        type=["npz"],
        key="prediction_npz",
    )

    if uploaded_file is None:

        st.info(
            """
            Upload a `.npz` file containing an `im` array.

            Example dataset file:

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
                    "Required array: 'im'"
                )

            else:

                hsi = npz_data["im"]

                if hsi.ndim != 3:

                    st.error(
                        f"Expected 3D HSI array. "
                        f"Received shape: {hsi.shape}"
                    )

                else:

                    height, width, bands = hsi.shape

                    st.success(
                        "Hyperspectral image loaded successfully."
                    )

                    # ------------------------------------------------
                    # IMAGE INFORMATION
                    # ------------------------------------------------

                    col1, col2, col3, col4 = st.columns(4)

                    with col1:

                        st.metric(
                            "Height",
                            height,
                        )

                    with col2:

                        st.metric(
                            "Width",
                            width,
                        )

                    with col3:

                        st.metric(
                            "Bands",
                            bands,
                        )

                    with col4:

                        st.metric(
                            "Data Type",
                            str(hsi.dtype),
                        )

                    # ------------------------------------------------
                    # SPECTRAL VIEWER
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">Spectral Band Viewer</div>',
                        unsafe_allow_html=True,
                    )

                    selected_band = st.slider(
                        "Select band",
                        min_value=0,
                        max_value=bands - 1,
                        value=0,
                        key="prediction_band",
                    )

                    st.image(
                        hsi[:, :, selected_band],
                        caption=(
                            f"Band {selected_band + 1}"
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

                    stat_col1, stat_col2 = st.columns(2)

                    with stat_col1:

                        st.metric(
                            "Minimum",
                            f"{stats.min():.2f}",
                        )

                    with stat_col2:

                        st.metric(
                            "Maximum",
                            f"{stats.max():.2f}",
                        )

                    stat_col3, stat_col4 = st.columns(2)

                    with stat_col3:

                        st.metric(
                            "Mean",
                            f"{stats.mean():.2f}",
                        )

                    with stat_col4:

                        st.metric(
                            "Std",
                            f"{stats.std():.2f}",
                        )

                    # ------------------------------------------------
                    # AI PREDICTION
                    # ------------------------------------------------

                    st.markdown(
                        '<div class="section-title">AI Prediction</div>',
                        unsafe_allow_html=True,
                    )

                    if bands != EXPECTED_BANDS:

                        st.error(
                            f"This model requires exactly "
                            f"{EXPECTED_BANDS} bands. "
                            f"The uploaded image contains "
                            f"{bands} bands."
                        )

                    elif not TORCH_AVAILABLE:

                        st.error(
                            "PyTorch is not available."
                        )

                    else:

                        model, model_error = load_model()

                        if model is None:

                            st.error(
                                "Model could not be loaded."
                            )

                            st.code(
                                model_error
                                if model_error
                                else "Unknown model error"
                            )

                        else:

                            st.success(
                                "Balanced 3D CNN loaded successfully."
                            )

                            predict_button = st.button(
                                "🚀 Run Disease Prediction",
                                type="primary",
                                width="stretch",
                            )

                            if predict_button:

                                with st.spinner(
                                    "Processing hyperspectral image..."
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
                                            f"Prediction failed: {e}"
                                        )

                                    else:

                                        class_name = (
                                            CLASS_NAMES.get(
                                                predicted_class,
                                                f"Class {predicted_class}",
                                            )
                                        )

                                        st.success(
                                            f"Prediction: {class_name}"
                                        )

                                        result_col1, result_col2 = (
                                            st.columns(2)
                                        )

                                        with result_col1:

                                            st.metric(
                                                "Predicted Class",
                                                class_name,
                                            )

                                        with result_col2:

                                            st.metric(
                                                "Confidence",
                                                f"{confidence * 100:.2f}%",
                                            )

                                        st.markdown(
                                            f"""
                                            <div class="success-box">
                                            <b>Result:</b>
                                            {class_name}
                                            <br><br>
                                            {CLASS_DESCRIPTIONS.get(
                                                predicted_class,
                                                "Model prediction completed.",
                                            )}
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

                                        probability_df = pd.DataFrame(
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
                                                    float(
                                                        p * 100
                                                    )
                                                    for p in probabilities
                                                ],
                                            }
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
                f"Unable to process uploaded file: {e}"
            )


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.title("🛰 Field Monitoring")

    st.write(
        """
        Prototype interface for visualizing crop-condition
        predictions across an agricultural field grid.
        """
    )

    st.info(
        """
        The current field grid is simulated for dashboard
        demonstration. It can later be connected to real
        UAV/GIS prediction outputs.
        """
    )

    rng = np.random.default_rng(42)

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

    col1, col2, col3, col4 = st.columns(4)

    with col1:

        st.metric(
            "Total Grid Cells",
            int(field_data.size),
        )

    with col2:

        st.metric(
            "Healthy",
            summary.get(0, 0),
        )

    with col3:

        st.metric(
            "Disease Class 1",
            summary.get(1, 0),
        )

    with col4:

        st.metric(
            "Disease Class 2",
            summary.get(2, 0),
        )

    st.markdown(
        '<div class="section-title">Monitoring Workflow</div>',
        unsafe_allow_html=True,
    )

    workflow_col1, workflow_col2, workflow_col3 = (
        st.columns(3)
    )

    with workflow_col1:

        st.markdown(
            """
            ### 1️⃣ UAV Capture

            Capture hyperspectral imagery
            over the agricultural field.
            """
        )

    with workflow_col2:

        st.markdown(
            """
            ### 2️⃣ AI Analysis

            Process spectral-spatial information
            using the trained 3D CNN.
            """
        )

    with workflow_col3:

        st.markdown(
            """
            ### 3️⃣ Field Decision

            Identify areas requiring
            further inspection.
            """
        )


# ============================================================
# FOOTER
# ============================================================

st.sidebar.divider()

st.sidebar.caption(
    "TerraSpectra • AI-powered hyperspectral crop analysis"
)