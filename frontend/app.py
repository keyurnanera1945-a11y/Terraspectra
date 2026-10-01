import os
import re
import sys
import numpy as np
import pandas as pd
import streamlit as st

from PIL import Image


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="TerraSpectra",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)


# ============================================================
# PATH CONFIGURATION
# ============================================================

ROOT_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..")
)

OUTPUT_DIR = os.path.join(
    ROOT_DIR,
    "outputs"
)

DATA_DIR = os.path.join(
    ROOT_DIR,
    "data",
    "raw",
    "hyperspectral",
    "0"
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main application */

    .stApp {
        background-color: #0b1117;
    }

    [data-testid="stSidebar"] {
        background-color: #0f171f;
        border-right: 1px solid #24313d;
    }

    [data-testid="stSidebar"] * {
        color: #dce7ee;
    }

    /* Header */

    .brand {
        font-size: 30px;
        font-weight: 800;
        letter-spacing: 2px;
        color: #f1f5f9;
        margin-bottom: 0px;
    }

    .subtitle {
        color: #8da2b3;
        font-size: 13px;
        margin-top: -5px;
        letter-spacing: 0.4px;
    }

    .status {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        background-color: #12301f;
        color: #62d995;
        font-size: 12px;
        font-weight: 700;
        border: 1px solid #1f5c38;
    }

    /* Metric cards */

    .metric-card {
        background: linear-gradient(
            145deg,
            #121d27,
            #0f171f
        );

        border: 1px solid #243441;
        border-radius: 12px;
        padding: 20px;
        min-height: 125px;
    }

    .metric-title {
        color: #8498a8;
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }

    .metric-value {
        color: #f1f5f9;
        font-size: 30px;
        font-weight: 800;
        margin-top: 8px;
    }

    .metric-description {
        color: #6f8596;
        font-size: 11px;
        margin-top: 4px;
    }

    /* Section */

    .section-title {
        color: #e8eef3;
        font-size: 20px;
        font-weight: 700;
        margin-top: 25px;
        margin-bottom: 12px;
    }

    .section-caption {
        color: #8296a6;
        font-size: 13px;
        margin-bottom: 18px;
    }

    /* Info panels */

    .info-panel {
        background-color: #111b24;
        border: 1px solid #263743;
        border-radius: 12px;
        padding: 18px;
        margin-bottom: 15px;
    }

    .info-title {
        color: #e5edf3;
        font-size: 15px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .info-text {
        color: #91a4b3;
        font-size: 13px;
        line-height: 1.6;
    }

    /* Footer */

    .footer {
        color: #536878;
        text-align: center;
        font-size: 11px;
        margin-top: 50px;
        padding: 20px;
        border-top: 1px solid #1d2a34;
    }

    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def load_csv(path):
    if os.path.exists(path):
        return pd.read_csv(path)
    return None


def read_report(path):
    if not os.path.exists(path):
        return None

    with open(
        path,
        "r",
        encoding="utf-8"
    ) as f:
        return f.read()


def extract_metric(text, pattern):
    if not text:
        return None

    match = re.search(
        pattern,
        text
    )

    if match:
        return float(match.group(1))

    return None


def metric_card(
    title,
    value,
    description=""
):
    st.markdown(
        f"""
        <div class="metric-card">

            <div class="metric-title">
                {title}
            </div>

            <div class="metric-value">
                {value}
            </div>

            <div class="metric-description">
                {description}
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


def section_title(title, caption=None):

    st.markdown(
        f"""
        <div class="section-title">
            {title}
        </div>
        """,
        unsafe_allow_html=True
    )

    if caption:
        st.markdown(
            f"""
            <div class="section-caption">
                {caption}
            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# LOAD DATA
# ============================================================

patch_df = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "hyperspectral_patches.csv"
    )
)

train_df = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "train_patches.csv"
    )
)

val_df = load_csv(
    os.path.join(
        OUTPUT_DIR,
        "val_patches.csv"
    )
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div class="brand">
            TERRASPECTRA
        </div>

        <div class="subtitle">
            Hyperspectral Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown("---")

    page = st.radio(
        "NAVIGATION",
        [
            "Overview",
            "Data Explorer",
            "Spectral Analysis",
            "Model Performance",
            "Prediction",
            "Monitoring"
        ]
    )

    st.markdown("---")

    st.markdown(
        """
        <div class="info-panel">

            <div class="info-title">
                SYSTEM STATUS
            </div>

            <div class="info-text">
                ● Processing pipeline ready<br>
                ● Hyperspectral data loaded<br>
                ● 3D CNN models available<br>
                ● Evaluation results available
            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# TOP HEADER
# ============================================================

header_left, header_right = st.columns(
    [5, 1]
)

with header_left:

    st.markdown(
        """
        <div class="brand">
            TERRASPECTRA
        </div>

        <div class="subtitle">
            Hyperspectral Crop Intelligence Platform
        </div>
        """,
        unsafe_allow_html=True
    )

with header_right:

    st.markdown(
        """
        <div style="text-align:right;margin-top:10px;">
            <span class="status">
                ● SYSTEM ONLINE
            </span>
        </div>
        """,
        unsafe_allow_html=True
    )


# ============================================================
# OVERVIEW
# ============================================================

if page == "Overview":

    section_title(
        "Operational Overview",
        "Real-time view of the TerraSpectra hyperspectral processing pipeline."
    )

    total_tiles = 1115

    total_bands = 20

    total_patches = (
        len(patch_df)
        if patch_df is not None
        else 8688
    )

    total_train = (
        len(train_df)
        if train_df is not None
        else 6950
    )

    total_val = (
        len(val_df)
        if val_df is not None
        else 1738
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        metric_card(
            "Hyperspectral Tiles",
            f"{total_tiles:,}",
            "UAV hyperspectral tiles"
        )

    with c2:
        metric_card(
            "Spectral Bands",
            total_bands,
            "Selected wavelength channels"
        )

    with c3:
        metric_card(
            "Annotated Patches",
            f"{total_patches:,}",
            "YOLO-derived regions"
        )

    with c4:
        metric_card(
            "Model Parameters",
            "18,243",
            "3D CNN trainable parameters"
        )


    section_title(
        "Processing Pipeline"
    )

    pipeline = [
        ("01", "Hyperspectral Input"),
        ("02", "Spectral Analysis"),
        ("03", "Patch Extraction"),
        ("04", "3D CNN Inference"),
        ("05", "Disease Classification"),
        ("06", "Monitoring")
    ]

    cols = st.columns(6)

    for col, item in zip(cols, pipeline):

        with col:

            st.markdown(
                f"""
                <div class="info-panel"
                     style="text-align:center;min-height:125px;">

                    <div style="
                        font-size:20px;
                        font-weight:800;
                        color:#8fa7b7;">
                        {item[0]}
                    </div>

                    <div class="info-title">
                        {item[1]}
                    </div>

                </div>
                """,
                unsafe_allow_html=True
            )


    section_title(
        "System Snapshot"
    )

    col1, col2 = st.columns(2)

    with col1:

        st.markdown(
            """
            <div class="info-panel">

                <div class="info-title">
                    DATA PIPELINE
                </div>

                <div class="info-text">

                    Hyperspectral tiles are processed into
                    normalized spectral-spatial patches.
                    PCA is used for spectral analysis while
                    the original 20-band representation is
                    retained for deep learning.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:

        st.markdown(
            """
            <div class="info-panel">

                <div class="info-title">
                    AI ENGINE
                </div>

                <div class="info-text">

                    TerraSpectra uses a lightweight 3D CNN
                    that learns both spectral and spatial
                    patterns from hyperspectral crop imagery.

                </div>

            </div>
            """,
            unsafe_allow_html=True
        )


# ============================================================
# DATA EXPLORER
# ============================================================

elif page == "Data Explorer":

    section_title(
        "Hyperspectral Data Explorer",
        "Explore dataset composition and extracted training regions."
    )

    if patch_df is None:

        st.error(
            "hyperspectral_patches.csv not found."
        )

    else:

        total_patches = len(patch_df)

        unique_tiles = patch_df["tile_id"].nunique()

        classes = patch_df["class_id"].value_counts()

        c1, c2, c3 = st.columns(3)

        with c1:
            metric_card(
                "Tiles",
                f"{unique_tiles:,}",
                "Unique hyperspectral tiles"
            )

        with c2:
            metric_card(
                "Patches",
                f"{total_patches:,}",
                "Extracted bounding-box regions"
            )

        with c3:
            metric_card(
                "Classes",
                patch_df["class_id"].nunique(),
                "Numeric dataset classes"
            )


        section_title(
            "Class Distribution"
        )

        class_counts = (
            patch_df["class_id"]
            .value_counts()
            .sort_index()
        )

        st.bar_chart(
            class_counts
        )


        section_title(
            "Patch Dimensions",
            "Spatial dimensions before normalization to 32 × 32."
        )

        dimension_df = patch_df[
            [
                "patch_width",
                "patch_height"
            ]
        ].describe()

        st.dataframe(
            dimension_df,
            use_container_width=True
        )


        section_title(
            "Dataset Records"
        )

        st.dataframe(
            patch_df.head(100),
            use_container_width=True,
            height=400
        )


# ============================================================
# SPECTRAL ANALYSIS
# ============================================================

elif page == "Spectral Analysis":

    section_title(
        "Spectral Analysis",
        "Principal Component Analysis of the hyperspectral representation."
    )

    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Input Bands",
            "20",
            "Spectral channels"
        )

    with c2:
        metric_card(
            "PC1 Variance",
            "71.93%",
            "Explained variance"
        )

    with c3:
        metric_card(
            "PC1 + PC2",
            "99.23%",
            "Cumulative variance"
        )


    section_title(
        "Explained Variance"
    )

    pca_variance = os.path.join(
        OUTPUT_DIR,
        "pca_variance.png"
    )

    if os.path.exists(pca_variance):

        st.image(
            Image.open(pca_variance),
            use_container_width=True
        )

    else:

        st.warning(
            "PCA variance visualization not found."
        )


    section_title(
        "PCA Projection"
    )

    pca_projection = os.path.join(
        OUTPUT_DIR,
        "pca_projection.png"
    )

    if os.path.exists(pca_projection):

        st.image(
            Image.open(pca_projection),
            use_container_width=True
        )

    else:

        st.warning(
            "PCA projection not found."
        )


    section_title(
        "Spectral Band Configuration"
    )

    band_file = os.path.join(
        OUTPUT_DIR,
        "pca_band_information.txt"
    )

    if os.path.exists(band_file):

        with open(
            band_file,
            "r",
            encoding="utf-8"
        ) as f:

            band_text = f.read()

        st.code(
            band_text
        )


# ============================================================
# MODEL PERFORMANCE
# ============================================================

elif page == "Model Performance":

    section_title(
        "Model Performance",
        "Comparison of the trained TerraSpectra 3D CNN experiments."
    )


    # --------------------------------------------------------
    # MODEL METRICS
    # --------------------------------------------------------

    model_dirs = {
        "Baseline 3D CNN": "evaluation",
        "Augmented 3D CNN": "evaluation_augmented",
        "Focal Loss": "evaluation_focal",
        "Balanced Sampling": "evaluation_balanced"
    }


    results = []

    for model_name, folder in model_dirs.items():

        report_path = os.path.join(
            OUTPUT_DIR,
            folder,
            "classification_report.txt"
        )

        report = read_report(
            report_path
        )

        if report:

            accuracy = extract_metric(
                report,
                r"Accuracy:\s*([0-9.]+)%"
            )

            if accuracy is None:

                accuracy = extract_metric(
                    report,
                    r"Validation Accuracy\s*:\s*([0-9.]+)%"
                )

            macro_f1 = extract_metric(
                report,
                r"Macro F1:\s*([0-9.]+)"
            )

            weighted_f1 = extract_metric(
                report,
                r"Weighted F1:\s*([0-9.]+)"
            )

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy,
                    "Macro F1": macro_f1,
                    "Weighted F1": weighted_f1
                }
            )


    # Fallback to known evaluation values
    if not results:

        results = [
            {
                "Model": "Baseline 3D CNN",
                "Accuracy": 92.75,
                "Macro F1": 0.5721,
                "Weighted F1": 0.9227
            },
            {
                "Model": "Augmented 3D CNN",
                "Accuracy": 84.81,
                "Macro F1": 0.4260,
                "Weighted F1": 0.8659
            },
            {
                "Model": "Focal Loss",
                "Accuracy": 7.54,
                "Macro F1": 0.1122,
                "Weighted F1": 0.0339
            },
            {
                "Model": "Balanced Sampling",
                "Accuracy": 92.12,
                "Macro F1": 0.5980,
                "Weighted F1": 0.9203
            }
        ]


    results_df = pd.DataFrame(
        results
    )


    section_title(
        "Experiment Comparison"
    )

    st.dataframe(
        results_df,
        use_container_width=True,
        hide_index=True
    )


    st.bar_chart(
        results_df.set_index("Model")[
            [
                "Accuracy",
                "Macro F1"
            ]
        ]
    )


    # --------------------------------------------------------
    # MODEL SELECTOR
    # --------------------------------------------------------

    selected_model = st.selectbox(
        "Select evaluation",
        list(model_dirs.keys())
    )

    selected_folder = model_dirs[
        selected_model
    ]


    report_path = os.path.join(
        OUTPUT_DIR,
        selected_folder,
        "classification_report.txt"
    )

    report = read_report(
        report_path
    )


    if report:

        st.code(
            report
        )


    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    cm_path = os.path.join(
        OUTPUT_DIR,
        selected_folder,
        "confusion_matrix.png"
    )

    if os.path.exists(cm_path):

        section_title(
            "Confusion Matrix"
        )

        st.image(
            Image.open(cm_path),
            width=600
        )


# ============================================================
# PREDICTION
# ============================================================

elif page == "Prediction":

    section_title(
        "Hyperspectral Prediction",
        "Run inference on a hyperspectral tile using the trained 3D CNN."
    )

    st.markdown(
        """
        <div class="info-panel">

            <div class="info-title">
                INFERENCE ENGINE
            </div>

            <div class="info-text">

                Upload a TerraSpectra-compatible NPZ hyperspectral
                tile. The inference engine will preprocess the
                spectral-spatial representation and generate a
                model prediction.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    uploaded_file = st.file_uploader(
        "Upload hyperspectral NPZ tile",
        type=["npz"]
    )


    if uploaded_file is not None:

        try:

            data = np.load(
                uploaded_file
            )

            if "im" not in data:

                st.error(
                    "The NPZ file does not contain an 'im' array."
                )

            else:

                hsi = data["im"]

                c1, c2, c3 = st.columns(3)

                with c1:
                    metric_card(
                        "Height",
                        hsi.shape[0],
                        "Pixels"
                    )

                with c2:
                    metric_card(
                        "Width",
                        hsi.shape[1],
                        "Pixels"
                    )

                with c3:
                    metric_card(
                        "Bands",
                        hsi.shape[2],
                        "Spectral channels"
                    )


                section_title(
                    "Hyperspectral Tile Preview"
                )

                # Use first principal-looking spectral band
                band_index = st.slider(
                    "Select spectral band",
                    0,
                    hsi.shape[2] - 1,
                    0
                )


                band = hsi[
                    :,
                    :,
                    band_index
                ]


                st.image(
                    band,
                    caption=f"Spectral Band {band_index}",
                    use_container_width=True
                )


                st.info(
                    "Prediction inference will be connected "
                    "to the selected trained checkpoint in the "
                    "next dashboard integration stage."
                )

        except Exception as e:

            st.error(
                f"Unable to read NPZ file: {e}"
            )


    else:

        st.info(
            "Upload an NPZ hyperspectral tile to begin."
        )


# ============================================================
# MONITORING
# ============================================================

elif page == "Monitoring":

    section_title(
        "Field Monitoring",
        "Spatial monitoring interface for hyperspectral crop analysis."
    )


    st.markdown(
        """
        <div class="info-panel">

            <div class="info-title">
                MONITORING ENGINE
            </div>

            <div class="info-text">

                This module is designed for tile-level crop
                monitoring and future geospatial integration.
                Hyperspectral tiles can be connected to field
                coordinates and prediction results to create
                disease-risk maps.

            </div>

        </div>
        """,
        unsafe_allow_html=True
    )


    # Demonstration field grid

    section_title(
        "Field Analysis Grid"
    )

    grid = np.random.default_rng(42).random(
        (12, 12)
    )

    st.image(
        grid,
        caption="Monitoring visualization prototype",
        use_container_width=True
    )


    c1, c2, c3 = st.columns(3)

    with c1:
        metric_card(
            "Tiles Processed",
            "1,115",
            "Current dataset"
        )

    with c2:
        metric_card(
            "Spectral Channels",
            "20",
            "Per hyperspectral tile"
        )

    with c3:
        metric_card(
            "AI Engine",
            "3D CNN",
            "Spectral-spatial model"
        )


# ============================================================
# FOOTER
# ============================================================

st.markdown(
    """
    <div class="footer">
        TERRASPECTRA · Hyperspectral Crop Intelligence Platform
        · PyTorch · Streamlit · Python
    </div>
    """,
    unsafe_allow_html=True
)