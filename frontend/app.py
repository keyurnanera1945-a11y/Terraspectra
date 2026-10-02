import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# Optional plotting
import matplotlib.pyplot as plt

# Optional torch
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except Exception:
    TORCH_AVAILABLE = False

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

PATCHES_CSV = OUTPUTS_DIR / "hyperspectral_patches.csv"
TRAIN_CSV = OUTPUTS_DIR / "train_patches.csv"
VAL_CSV = OUTPUTS_DIR / "val_patches.csv"

BALANCED_MODEL = (
    OUTPUTS_DIR
    / "models"
    / "terraspectra_3dcnn_balanced_best.pt"
)

BASELINE_MODEL = (
    OUTPUTS_DIR
    / "models"
    / "terraspectra_3dcnn_best.pt"
)

AUGMENTED_MODEL = (
    OUTPUTS_DIR
    / "models"
    / "terraspectra_3dcnn_augmented_best.pt"
)

EVALUATION_DIR = OUTPUTS_DIR / "evaluation_balanced"
PREDICTIONS_CSV = EVALUATION_DIR / "predictions.csv"


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
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>
    .main {
        background-color: #f7f9fb;
    }

    .block-container {
        padding-top: 1.5rem;
        padding-bottom: 2rem;
    }

    .hero {
        padding: 1.5rem;
        border-radius: 16px;
        background: linear-gradient(
            135deg,
            #e8f5e9 0%,
            #f1f8e9 50%,
            #e3f2fd 100%
        );
        border: 1px solid #d5e8d4;
        margin-bottom: 1.5rem;
    }

    .hero h1 {
        margin-bottom: 0.3rem;
    }

    .hero p {
        color: #4f5b62;
        font-size: 1.05rem;
    }

    .info-card {
        padding: 1rem;
        border-radius: 12px;
        background: white;
        border: 1px solid #e5e7eb;
        margin-bottom: 1rem;
    }

    .small-text {
        color: #667085;
        font-size: 0.9rem;
    }

    .status-ok {
        color: #16803c;
        font-weight: 600;
    }

    .status-warning {
        color: #b54708;
        font-weight: 600;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# CONSTANTS
# ============================================================

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

CLASS_COLORS = {
    0: "Healthy",
    1: "Disease 1",
    2: "Disease 2",
}

SPECTRAL_BANDS = [
    420,
    440,
    460,
    480,
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
    780,
    800,
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def safe_read_csv(path):
    """Read a CSV safely."""
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as exc:
        st.error(f"Unable to read {path.name}: {exc}")
    return None


def find_column(df, candidates):
    """Find a column using case-insensitive matching."""
    if df is None:
        return None

    normalized = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    for candidate in candidates:
        key = str(candidate).strip().lower()

        if key in normalized:
            return normalized[key]

    return None


def load_patch_dataframe():
    """Load the main hyperspectral patch metadata CSV."""
    return safe_read_csv(PATCHES_CSV)


@st.cache_data
def cached_patch_dataframe():
    """Cached patch dataframe."""
    if not PATCHES_CSV.exists():
        return None

    try:
        return pd.read_csv(PATCHES_CSV)
    except Exception:
        return None


def get_class_counts(df):
    """Return class counts."""
    class_col = find_column(
        df,
        ["class_id", "class", "label", "target"]
    )

    if class_col is None:
        return pd.Series(dtype=int)

    return df[class_col].value_counts().sort_index()


def class_name(class_id):
    try:
        return CLASS_NAMES.get(int(class_id), f"Class {class_id}")
    except Exception:
        return str(class_id)


def normalize_hsi(hsi):
    """
    Normalize HSI data to float32 [0,1].
    Expected original shape: H x W x Bands.
    """

    arr = np.asarray(hsi)

    if arr.ndim != 3:
        raise ValueError(
            f"Expected 3D HSI array, received shape {arr.shape}"
        )

    arr = arr.astype(np.float32)

    if np.nanmax(arr) > 1.0:
        arr = arr / 65535.0

    arr = np.nan_to_num(
        arr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0,
    )

    arr = np.clip(arr, 0.0, 1.0)

    return arr


def prepare_hsi_tensor(hsi):
    """
    Convert HWC hyperspectral image to model input:

    HWC
      ↓
    CHW
      ↓
    [1,1,C,H,W]
      ↓
    spatial resize to 32x32
    """

    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch is not installed in the current environment."
        )

    arr = normalize_hsi(hsi)

    height, width, bands = arr.shape

    if bands != 20:
        raise ValueError(
            f"Model expects 20 spectral bands, but received {bands}."
        )

    # HWC -> CHW
    arr = np.transpose(arr, (2, 0, 1))

    tensor = torch.from_numpy(arr).float()

    # [C,H,W] -> [1,1,C,H,W]
    tensor = tensor.unsqueeze(0).unsqueeze(0)

    # Resize only spatial dimensions.
    # [1,1,20,H,W] -> [1,1,20,32,32]
    tensor = F.interpolate(
        tensor,
        size=(20, 32, 32),
        mode="trilinear",
        align_corners=False,
    )

    return tensor


def load_model_class():
    """Load TerraSpectra model class."""
    try:
        from src.model_3dcnn import TerraSpectra3DCNN
        return TerraSpectra3DCNN
    except Exception as exc:
        return None


def extract_state_dict(checkpoint):
    """Extract a state dictionary from common checkpoint formats."""

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:
            return checkpoint["model_state_dict"]

        if "state_dict" in checkpoint:
            return checkpoint["state_dict"]

        # Raw state dict
        if all(
            isinstance(k, str)
            for k in checkpoint.keys()
        ):
            return checkpoint

    raise ValueError(
        "Unsupported model checkpoint format."
    )


@st.cache_resource
def load_model(model_path):
    """Load TerraSpectra 3D CNN."""

    if not TORCH_AVAILABLE:
        return None, "PyTorch is unavailable."

    if not model_path.exists():
        return None, (
            f"Model file not found:\n{model_path}"
        )

    model_class = load_model_class()

    if model_class is None:
        return None, (
            "Could not import TerraSpectra3DCNN "
            "from src.model_3dcnn."
        )

    try:
        model = model_class()

        checkpoint = torch.load(
            model_path,
            map_location="cpu",
        )

        state_dict = extract_state_dict(checkpoint)

        # Handle DataParallel checkpoints.
        cleaned_state_dict = {}

        for key, value in state_dict.items():
            if key.startswith("module."):
                key = key[7:]

            cleaned_state_dict[key] = value

        model.load_state_dict(
            cleaned_state_dict,
            strict=False,
        )

        model.eval()

        return model, "Model loaded successfully."

    except Exception as exc:
        return None, f"Model loading failed: {exc}"


def predict_hsi(model, hsi):
    """Run model prediction."""

    tensor = prepare_hsi_tensor(hsi)

    with torch.no_grad():
        output = model(tensor)

        if isinstance(output, (tuple, list)):
            output = output[0]

        probabilities = torch.softmax(
            output,
            dim=1,
        )

        predicted_class = int(
            torch.argmax(
                probabilities,
                dim=1,
            ).item()
        )

        confidence = float(
            probabilities[0, predicted_class].item()
        )

    return (
        predicted_class,
        confidence,
        probabilities[0].cpu().numpy(),
    )


def load_prediction_results():
    return safe_read_csv(PREDICTIONS_CSV)


# ============================================================
# SIDEBAR
# ============================================================

st.sidebar.title("🌱 TerraSpectra AI")

st.sidebar.markdown(
    """
    **Hyperspectral Crop Disease Intelligence**

    Analyze UAV hyperspectral imagery using
    PCA and a 3D CNN deep-learning pipeline.
    """
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

st.sidebar.markdown("### System Status")

if PATCHES_CSV.exists():
    st.sidebar.success("Patch dataset available")
else:
    st.sidebar.warning("Patch dataset not found")

if TORCH_AVAILABLE:
    st.sidebar.success("PyTorch available")
else:
    st.sidebar.warning("PyTorch unavailable")

if BALANCED_MODEL.exists():
    st.sidebar.success("Balanced model available")
else:
    st.sidebar.warning("Balanced model not found")


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    st.markdown(
        """
        <div class="hero">
            <h1>🌱 TerraSpectra AI</h1>
            <p>
                Hyperspectral crop-disease intelligence using
                PCA-based spectral analysis and a 3D CNN.
            </p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    df = cached_patch_dataframe()

    if df is not None:

        class_col = find_column(
            df,
            ["class_id", "class", "label"]
        )

        tile_col = find_column(
            df,
            ["tile_id", "tile"]
        )

        width_col = find_column(
            df,
            ["patch_width", "width"]
        )

        height_col = find_column(
            df,
            ["patch_height", "height"]
        )

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(
                "Hyperspectral Patches",
                f"{len(df):,}",
            )

        with col2:
            if tile_col:
                st.metric(
                    "Tiles",
                    f"{df[tile_col].nunique():,}",
                )
            else:
                st.metric("Tiles", "1,115")

        with col3:
            if class_col:
                st.metric(
                    "Classes",
                    f"{df[class_col].nunique()}",
                )
            else:
                st.metric("Classes", "3")

        with col4:
            if width_col and height_col:
                avg_size = (
                    df[width_col].mean()
                    * df[height_col].mean()
                )
                st.metric(
                    "Avg Patch Area",
                    f"{avg_size:.0f} px²",
                )
            else:
                st.metric("Spectral Bands", "20")

        st.divider()

        st.subheader("Project Pipeline")

        pipeline_cols = st.columns(5)

        pipeline = [
            ("01", "UAV HSI", "20-band hyperspectral imagery"),
            ("02", "PCA", "Spectral dimensionality analysis"),
            ("03", "Patch Extraction", "Disease-region patches"),
            ("04", "3D CNN", "Deep-learning classification"),
            ("05", "Prediction", "Disease monitoring"),
        ]

        for col, item in zip(pipeline_cols, pipeline):
            with col:
                st.markdown(
                    f"""
                    <div class="info-card">
                        <h3>{item[0]}</h3>
                        <b>{item[1]}</b>
                        <p class="small-text">
                            {item[2]}
                        </p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

        st.subheader("Dataset Distribution")

        if class_col:

            counts = (
                df[class_col]
                .value_counts()
                .sort_index()
            )

            chart_df = pd.DataFrame(
                {
                    "Class": [
                        class_name(i)
                        for i in counts.index
                    ],
                    "Samples": counts.values,
                }
            )

            st.bar_chart(
                chart_df.set_index("Class"),
                width="stretch",
            )

        st.subheader("Dataset Information")

        info_col1, info_col2 = st.columns(2)

        with info_col1:
            st.markdown(
                """
                **Hyperspectral Data**

                - 1,115 HSI tiles
                - 20 spectral bands
                - UAV-based imagery
                - NPZ storage format
                - Spectral range used by the project:
                  approximately 420–800 nm
                """
            )

        with info_col2:
            st.markdown(
                """
                **Machine Learning**

                - PCA preprocessing
                - 3D CNN architecture
                - 32 × 32 spatial patches
                - Three classification classes
                - Balanced training strategy
                """
            )

    else:

        st.warning(
            "The hyperspectral patch dataset could not be found."
        )

        st.info(
            f"Expected file:\n{PATCHES_CSV}"
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    st.title("🔎 Data Explorer")

    df = cached_patch_dataframe()

    if df is None:

        st.error(
            "hyperspectral_patches.csv was not found."
        )

        st.code(str(PATCHES_CSV))

    else:

        st.success(
            f"Loaded {len(df):,} hyperspectral patch records."
        )

        class_col = find_column(
            df,
            ["class_id", "class", "label"]
        )

        tile_col = find_column(
            df,
            ["tile_id", "tile"]
        )

        width_col = find_column(
            df,
            ["patch_width", "width"]
        )

        height_col = find_column(
            df,
            ["patch_height", "height"]
        )

        # ----------------------------------------------------
        # Metrics
        # ----------------------------------------------------

        c1, c2, c3, c4 = st.columns(4)

        with c1:
            st.metric(
                "Total Patches",
                f"{len(df):,}",
            )

        with c2:
            if tile_col:
                st.metric(
                    "Unique Tiles",
                    f"{df[tile_col].nunique():,}",
                )
            else:
                st.metric(
                    "Unique Tiles",
                    "N/A",
                )

        with c3:
            if width_col:
                st.metric(
                    "Min Patch Width",
                    f"{df[width_col].min():.0f}",
                )
            else:
                st.metric(
                    "Min Patch Width",
                    "N/A",
                )

        with c4:
            if height_col:
                st.metric(
                    "Max Patch Height",
                    f"{df[height_col].max():.0f}",
                )
            else:
                st.metric(
                    "Max Patch Height",
                    "N/A",
                )

        st.divider()

        # ----------------------------------------------------
        # Filters
        # ----------------------------------------------------

        st.subheader("Filters")

        filter_col1, filter_col2 = st.columns(2)

        filtered_df = df.copy()

        with filter_col1:

            if class_col:

                class_values = sorted(
                    df[class_col]
                    .dropna()
                    .unique()
                    .tolist()
                )

                selected_classes = st.multiselect(
                    "Select Classes",
                    options=class_values,
                    default=class_values,
                    format_func=class_name,
                )

                filtered_df = filtered_df[
                    filtered_df[class_col].isin(
                        selected_classes
                    )
                ]

        with filter_col2:

            search_text = st.text_input(
                "Search HSI file / tile",
                placeholder="Example: 0001.npz",
            )

            if search_text:

                mask = pd.Series(
                    False,
                    index=filtered_df.index,
                )

                if "hsi_file" in filtered_df.columns:
                    mask = (
                        filtered_df["hsi_file"]
                        .astype(str)
                        .str.contains(
                            search_text,
                            case=False,
                            na=False,
                        )
                    )

                if tile_col:
                    mask = (
                        mask
                        |
                        filtered_df[tile_col]
                        .astype(str)
                        .str.contains(
                            search_text,
                            case=False,
                            na=False,
                        )
                    )

                filtered_df = filtered_df[mask]

        st.write(
            f"Showing **{len(filtered_df):,}** records."
        )

        # ----------------------------------------------------
        # Class distribution
        # ----------------------------------------------------

        if class_col:

            st.subheader("Class Distribution")

            counts = (
                filtered_df[class_col]
                .value_counts()
                .sort_index()
            )

            chart_df = pd.DataFrame(
                {
                    "Class": [
                        class_name(i)
                        for i in counts.index
                    ],
                    "Samples": counts.values,
                }
            )

            st.bar_chart(
                chart_df.set_index("Class"),
                width="stretch",
            )

        # ----------------------------------------------------
        # Patch dimensions
        # ----------------------------------------------------

        if width_col and height_col:

            st.subheader("Patch Dimensions")

            dimension_df = filtered_df[
                [
                    width_col,
                    height_col,
                ]
            ].copy()

            dimension_df.columns = [
                "Patch Width",
                "Patch Height",
            ]

            st.dataframe(
                dimension_df.describe(),
                width="stretch",
            )

        # ----------------------------------------------------
        # Dataset table
        # ----------------------------------------------------

        st.subheader("Patch Metadata")

        display_df = filtered_df.copy()

        if class_col:

            display_df["class_name"] = (
                display_df[class_col]
                .apply(class_name)
            )

        st.dataframe(
            display_df,
            width="stretch",
            height=450,
        )


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    st.title("📈 Spectral Analysis")

    st.markdown(
        """
        Hyperspectral imagery contains information across multiple
        wavelength bands. TerraSpectra uses spectral information
        together with spatial features for crop-disease detection.
        """
    )

    st.subheader("Spectral Bands")

    band_df = pd.DataFrame(
        {
            "Band": range(1, 21),
            "Wavelength (nm)": SPECTRAL_BANDS,
        }
    )

    st.dataframe(
        band_df,
        width="stretch",
    )

    st.subheader("PCA Explained Variance")

    pca_data = pd.DataFrame(
        {
            "Principal Component": [
                "PC1",
                "PC2",
                "PC3",
            ],
            "Explained Variance (%)": [
                71.93,
                27.31,
                0.33,
            ],
        }
    )

    col1, col2 = st.columns(2)

    with col1:

        st.bar_chart(
            pca_data.set_index(
                "Principal Component"
            ),
            width="stretch",
        )

    with col2:

        st.metric(
            "PC1",
            "71.93%",
        )

        st.metric(
            "PC2",
            "27.31%",
        )

        st.metric(
            "PC3",
            "0.33%",
        )

        st.info(
            "The first two principal components explain "
            "approximately 99.24% of the variance."
        )

    st.subheader("Example Spectral Profiles")

    # Generate representative illustrative profiles.
    # This section is visualization-only and does not replace
    # actual per-pixel spectral measurements.

    rng = np.random.default_rng(42)

    fig, ax = plt.subplots(
        figsize=(10, 4)
    )

    wavelengths = np.array(SPECTRAL_BANDS)

    for label, base, noise in [
        ("Healthy / Normal", 0.55, 0.025),
        ("Disease Class 1", 0.48, 0.03),
        ("Disease Class 2", 0.42, 0.035),
    ]:

        profile = (
            base
            + 0.12
            * np.sin(
                wavelengths / 75
            )
            + rng.normal(
                0,
                noise,
                len(wavelengths),
            )
        )

        profile = np.clip(
            profile,
            0,
            1,
        )

        ax.plot(
            wavelengths,
            profile,
            marker="o",
            label=label,
        )

    ax.set_xlabel(
        "Wavelength (nm)"
    )

    ax.set_ylabel(
        "Normalized Reflectance"
    )

    ax.set_title(
        "Illustrative Spectral Profiles"
    )

    ax.legend()

    ax.grid(alpha=0.25)

    st.pyplot(
        fig,
        width="stretch",
    )

    st.caption(
        "The example curves above are illustrative. "
        "Use actual HSI samples for scientific spectral comparison."
    )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    st.title("🧠 Model Performance")

    st.markdown(
        """
        TerraSpectra uses a 3D CNN to learn spatial and spectral
        features from 20-band hyperspectral patches.
        """
    )

    st.subheader("Model Comparison")

    performance_df = pd.DataFrame(
        {
            "Model": [
                "Baseline 3D CNN",
                "Augmented 3D CNN",
                "Focal Loss",
                "Balanced Sampling",
            ],
            "Accuracy (%)": [
                92.75,
                84.81,
                7.54,
                92.12,
            ],
            "Macro F1 (%)": [
                57.21,
                42.60,
                11.22,
                59.80,
            ],
            "Weighted F1 (%)": [
                92.27,
                86.59,
                3.39,
                92.03,
            ],
        }
    )

    st.dataframe(
        performance_df,
        width="stretch",
        hide_index=True,
    )

    st.subheader("Accuracy Comparison")

    accuracy_chart = (
        performance_df
        .set_index("Model")[["Accuracy (%)"]]
    )

    st.bar_chart(
        accuracy_chart,
        width="stretch",
    )

    st.subheader("Balanced 3D CNN")

    m1, m2, m3 = st.columns(3)

    with m1:
        st.metric(
            "Validation Accuracy",
            "92.12%",
        )

    with m2:
        st.metric(
            "Macro F1",
            "59.80%",
        )

    with m3:
        st.metric(
            "Weighted F1",
            "92.03%",
        )

    st.divider()

    st.subheader("Training Configuration")

    config_df = pd.DataFrame(
        {
            "Parameter": [
                "Architecture",
                "Input",
                "Spectral Bands",
                "Patch Size",
                "Optimizer",
                "Learning Rate",
                "Training Strategy",
            ],
            "Value": [
                "3D CNN",
                "Hyperspectral patch",
                "20",
                "32 × 32",
                "Adam",
                "0.001",
                "Balanced Sampling",
            ],
        }
    )

    st.dataframe(
        config_df,
        width="stretch",
        hide_index=True,
    )

    # --------------------------------------------------------
    # Evaluation predictions
    # --------------------------------------------------------

    st.subheader("Evaluation Predictions")

    prediction_df = load_prediction_results()

    if prediction_df is not None:

        st.success(
            f"Loaded {len(prediction_df):,} evaluation predictions."
        )

        st.dataframe(
            prediction_df.head(100),
            width="stretch",
            height=400,
        )

    else:

        st.info(
            "Balanced evaluation predictions were not found."
        )

        st.caption(
            f"Expected: {PREDICTIONS_CSV}"
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    st.title("🔬 Hyperspectral Prediction")

    st.markdown(
        """
        Upload a hyperspectral `.npz` file containing an
        `im` array with shape:

        **Height × Width × 20 spectral bands**
        """
    )

    uploaded_file = st.file_uploader(
        "Upload HSI NPZ file",
        type=["npz"],
    )

    st.divider()

    model_path = BALANCED_MODEL

    if not model_path.exists():

        st.warning(
            "Balanced model was not found."
        )

        st.info(
            f"Expected model:\n{model_path}"
        )

        if BASELINE_MODEL.exists():
            st.info(
                "A baseline model is available. "
                "You can switch the model path in the code if required."
            )

    if uploaded_file is not None:

        try:

            loaded = np.load(
                uploaded_file,
                allow_pickle=False,
            )

            keys = loaded.files

            st.write(
                "**NPZ keys:**",
                ", ".join(keys),
            )

            if "im" not in keys:

                st.error(
                    "The uploaded NPZ does not contain an 'im' array."
                )

            else:

                hsi = loaded["im"]

                st.success(
                    f"HSI loaded successfully: {hsi.shape}"
                )

                if hsi.ndim != 3:

                    st.error(
                        "Expected a 3-dimensional HSI array."
                    )

                else:

                    height, width, bands = hsi.shape

                    c1, c2, c3 = st.columns(3)

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
                            "Bands",
                            bands,
                        )

                    if bands != 20:

                        st.error(
                            f"The TerraSpectra model expects "
                            f"20 bands, but this file has {bands}."
                        )

                    else:

                        # ------------------------------------------------
                        # HSI preview
                        # ------------------------------------------------

                        st.subheader(
                            "RGB-style Spectral Preview"
                        )

                        normalized = normalize_hsi(
                            hsi
                        )

                        # Select approximate visible bands.
                        # For a 20-band cube, use three representative bands.
                        rgb_indices = [
                            13,
                            9,
                            5,
                        ]

                        rgb = normalized[
                            :,
                            :,
                            rgb_indices,
                        ]

                        rgb = np.clip(
                            rgb * 2.5,
                            0,
                            1,
                        )

                        st.image(
                            rgb,
                            caption="Hyperspectral composite preview",
                            width="stretch",
                        )

                        st.subheader(
                            "Model Prediction"
                        )

                        if not TORCH_AVAILABLE:

                            st.error(
                                "PyTorch is not available."
                            )

                        elif not model_path.exists():

                            st.error(
                                f"Model not found:\n{model_path}"
                            )

                        else:

                            model, message = load_model(
                                model_path
                            )

                            if model is None:

                                st.error(message)

                            else:

                                try:

                                    predicted_class, confidence, probabilities = (
                                        predict_hsi(
                                            model,
                                            hsi,
                                        )
                                    )

                                    result_col1, result_col2 = st.columns(
                                        2
                                    )

                                    with result_col1:

                                        st.metric(
                                            "Predicted Class",
                                            class_name(
                                                predicted_class
                                            ),
                                        )

                                    with result_col2:

                                        st.metric(
                                            "Confidence",
                                            f"{confidence * 100:.2f}%",
                                        )

                                    st.success(
                                        f"Prediction: "
                                        f"{class_name(predicted_class)}"
                                    )

                                    probability_df = pd.DataFrame(
                                        {
                                            "Class": [
                                                class_name(i)
                                                for i in range(
                                                    len(
                                                        probabilities
                                                    )
                                                )
                                            ],
                                            "Probability": [
                                                float(p) * 100
                                                for p in probabilities
                                            ],
                                        }
                                    )

                                    st.subheader(
                                        "Class Probabilities"
                                    )

                                    st.bar_chart(
                                        probability_df.set_index(
                                            "Class"
                                        ),
                                        width="stretch",
                                    )

                                    st.dataframe(
                                        probability_df,
                                        width="stretch",
                                        hide_index=True,
                                    )

                                except Exception as exc:

                                    st.error(
                                        f"Prediction failed: {exc}"
                                    )

        except Exception as exc:

            st.error(
                f"Unable to process the uploaded file: {exc}"
            )


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.title("🛰️ Field Monitoring")

    st.markdown(
        """
        Prototype monitoring dashboard for visualizing
        disease predictions across a field grid.
        """
    )

    st.info(
        "The current monitoring map is a prototype visualization. "
        "Connect georeferenced UAV predictions for real field deployment."
    )

    rng = np.random.default_rng(42)

    grid_size = 12

    grid = rng.choice(
        [0, 1, 2],
        size=(grid_size, grid_size),
        p=[0.72, 0.18, 0.10],
    )

    fig, ax = plt.subplots(
        figsize=(8, 6)
    )

    image = ax.imshow(
        grid,
        interpolation="nearest",
    )

    ax.set_title(
        "Prototype Disease Monitoring Grid"
    )

    ax.set_xlabel(
        "Field X"
    )

    ax.set_ylabel(
        "Field Y"
    )

    ax.set_xticks(
        range(grid_size)
    )

    ax.set_yticks(
        range(grid_size)
    )

    plt.colorbar(
        image,
        ax=ax,
        ticks=[0, 1, 2],
        label="Class ID",
    )

    st.pyplot(
        fig,
        width="stretch",
    )

    st.subheader(
        "Monitoring Summary"
    )

    total_cells = grid.size

    healthy_cells = int(
        np.sum(grid == 0)
    )

    disease1_cells = int(
        np.sum(grid == 1)
    )

    disease2_cells = int(
        np.sum(grid == 2)
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.metric(
            "Total Cells",
            total_cells,
        )

    with c2:
        st.metric(
            "Healthy",
            healthy_cells,
        )

    with c3:
        st.metric(
            "Disease 1",
            disease1_cells,
        )

    with c4:
        st.metric(
            "Disease 2",
            disease2_cells,
        )

    monitoring_df = pd.DataFrame(
        {
            "Class": [
                "Healthy / Normal",
                "Disease Class 1",
                "Disease Class 2",
            ],
            "Cells": [
                healthy_cells,
                disease1_cells,
                disease2_cells,
            ],
        }
    )

    st.subheader(
        "Class Distribution"
    )

    st.bar_chart(
        monitoring_df.set_index("Class"),
        width="stretch",
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TerraSpectra AI • Hyperspectral Crop Disease Intelligence • "
    "Python + Streamlit + PyTorch + 3D CNN"
)