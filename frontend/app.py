from pathlib import Path
from io import BytesIO

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import pydeck as pdk


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
# DARK MODE
# ============================================================
# This applies the dashboard theme directly from app.py.
# No config.toml or external CSS file is required.

st.markdown(
    """
    <style>
        /* Main application */
        .stApp {
            background-color: #0b1120;
            color: #f8fafc;
        }

        /* Main content */
        .main {
            background-color: #0b1120;
        }

        /* Sidebar */
        section[data-testid="stSidebar"] {
            background-color: #080d19;
            border-right: 1px solid #1e293b;
        }

        section[data-testid="stSidebar"] * {
            color: #f8fafc;
        }

        /* Headings */
        h1, h2, h3, h4 {
            color: #f8fafc !important;
        }

        /* Normal text */
        p, label, span {
            color: #cbd5e1;
        }

        /* Metric cards */
        div[data-testid="stMetric"] {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 12px;
            padding: 15px;
        }

        div[data-testid="stMetricLabel"] {
            color: #94a3b8 !important;
        }

        div[data-testid="stMetricValue"] {
            color: #f8fafc !important;
        }

        /* Buttons */
        .stButton > button {
            background-color: #16a34a;
            color: white;
            border: 1px solid #22c55e;
            border-radius: 8px;
            font-weight: 600;
        }

        .stButton > button:hover {
            background-color: #15803d;
            color: white;
            border-color: #4ade80;
        }

        /* Select boxes */
        div[data-baseweb="select"] > div {
            background-color: #111827;
            border-color: #334155;
            color: #f8fafc;
        }

        /* Text inputs */
        input {
            background-color: #111827 !important;
            color: #f8fafc !important;
        }

        /* Number inputs */
        div[data-baseweb="input"] {
            background-color: #111827;
        }

        /* File uploader */
        section[data-testid="stFileUploaderDropzone"] {
            background-color: #111827;
            border: 1px dashed #475569;
            border-radius: 12px;
        }

        /* Tabs */
        button[data-baseweb="tab"] {
            color: #94a3b8;
        }

        button[data-baseweb="tab"][aria-selected="true"] {
            color: #22c55e;
        }

        /* Expanders */
        details {
            background-color: #111827;
            border: 1px solid #1e293b;
            border-radius: 10px;
        }

        /* Dataframes */
        div[data-testid="stDataFrame"] {
            border: 1px solid #1e293b;
            border-radius: 10px;
        }

        /* Alerts */
        div[data-testid="stAlert"] {
            border-radius: 10px;
        }

        /* Progress bar */
        div[data-testid="stProgressBar"] > div > div {
            background-color: #22c55e;
        }

        /* Horizontal separators */
        hr {
            border-color: #1e293b;
        }

        /* Radio buttons */
        div[role="radiogroup"] label {
            color: #cbd5e1;
        }

        /* Multiselect */
        div[data-baseweb="tag"] {
            background-color: #166534;
        }

        /* Captions */
        .stCaption {
            color: #64748b !important;
        }

        /* Scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }

        ::-webkit-scrollbar-track {
            background: #0b1120;
        }

        ::-webkit-scrollbar-thumb {
            background: #334155;
            border-radius: 10px;
        }

        ::-webkit-scrollbar-thumb:hover {
            background: #475569;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"

DATA_DIR = (
    ROOT
    / "data"
    / "raw"
    / "hyperspectral"
    / "0"
)

LABEL_DIR = (
    ROOT
    / "data"
    / "raw"
    / "hyperspectral"
    / "labels"
    / "0"
)

MODELS_DIR = OUTPUTS / "models"

PATCH_CSV = OUTPUTS / "hyperspectral_patches.csv"

GEO_CSV = OUTPUTS / "geospatial_predictions.csv"

CONFUSION_CSV = (
    OUTPUTS / "tilesplit_confusion_matrix.csv"
)

REPORT_CSV = (
    OUTPUTS / "tilesplit_classification_report.csv"
)

HISTORY_CSV = (
    OUTPUTS / "tilesplit_training_history.csv"
)

EXPECTED_BANDS = 20


# ============================================================
# FASTAPI
# ============================================================

API_URLS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]


# ============================================================
# SPECTRAL BANDS
# ============================================================

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
# CLASS DEFINITIONS
# ============================================================

CLASS_NAMES = {
    0: "Healthy",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

CLASS_COLORS = {
    "Healthy": "#22c55e",
    "Disease Class 1": "#f59e0b",
    "Disease Class 2": "#ef4444",
    "Unknown": "#94a3b8",
}


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Spectral Analysis"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def fmt_number(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return str(value)


def normalize_image(image):
    image = np.asarray(image)

    if image.size == 0:
        return image

    image = image.astype(np.float32)

    minimum = np.nanmin(image)
    maximum = np.nanmax(image)

    if maximum - minimum == 0:
        return np.zeros_like(image)

    return (image - minimum) / (
        maximum - minimum
    )


def normalize_cube_shape(cube):
    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected 3D cube, received {cube.shape}"
        )

    if cube.shape[-1] == EXPECTED_BANDS:
        return cube

    if cube.shape[0] == EXPECTED_BANDS:
        return np.transpose(
            cube,
            (1, 2, 0),
        )

    if cube.shape[1] == EXPECTED_BANDS:
        return np.transpose(
            cube,
            (0, 2, 1),
        )

    raise ValueError(
        f"Could not identify {EXPECTED_BANDS} bands "
        f"in cube shape {cube.shape}"
    )


def load_npz_bytes(uploaded_file):
    data = np.load(
        BytesIO(uploaded_file.getvalue())
    )

    if "im" in data.files:
        cube = data["im"]
    else:
        cube = data[data.files[0]]

    return normalize_cube_shape(cube)


def load_dataset_cube(file_path):
    data = np.load(file_path)

    if "im" in data.files:
        cube = data["im"]
    else:
        cube = data[data.files[0]]

    return normalize_cube_shape(cube)


def load_dataframe(path):
    if not path.exists():
        return pd.DataFrame()

    try:
        return pd.read_csv(path)
    except Exception:
        return pd.DataFrame()


def find_column(df, candidates):
    if df.empty:
        return None

    normalized = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:

        key = candidate.lower().strip()

        if key in normalized:
            return normalized[key]

    for col in df.columns:

        column_name = str(col).lower()

        for candidate in candidates:

            if candidate.lower() in column_name:
                return col

    return None


def class_from_value(value):

    try:

        value_int = int(float(value))

        if value_int in CLASS_NAMES:
            return CLASS_NAMES[value_int]

    except Exception:
        pass

    text = str(value).strip().lower()

    if "healthy" in text:
        return "Healthy"

    if "class 1" in text:
        return "Disease Class 1"

    if "disease 1" in text:
        return "Disease Class 1"

    if "class 2" in text:
        return "Disease Class 2"

    if "disease 2" in text:
        return "Disease Class 2"

    return "Unknown"


def get_dataset_stats():

    patch_df = load_dataframe(
        PATCH_CSV
    )

    tile_count = 0

    if DATA_DIR.exists():

        tile_count = len(
            list(
                DATA_DIR.glob("*.npz")
            )
        )

    patch_count = len(patch_df)

    return tile_count, patch_count


def get_model_parameter_count():

    candidates = [
        MODELS_DIR
        / "terraspectra_3dcnn_balanced_best.pt",

        MODELS_DIR
        / "terraspectra_3dcnn_best.pt",

        MODELS_DIR
        / "tilesplit_best.pt",
    ]

    for model_path in candidates:

        if not model_path.exists():
            continue

        try:

            import torch

            checkpoint = torch.load(
                model_path,
                map_location="cpu",
            )

            if isinstance(
                checkpoint,
                dict,
            ):

                state_dict = checkpoint.get(
                    "model_state_dict",
                    checkpoint.get(
                        "state_dict",
                        checkpoint,
                    ),
                )

                total = 0

                for value in state_dict.values():

                    if hasattr(
                        value,
                        "numel",
                    ):
                        total += value.numel()

                if total > 0:
                    return total

        except Exception:
            pass

    return 18243


def load_confusion_matrix():

    df = load_dataframe(
        CONFUSION_CSV
    )

    if df.empty:
        return None

    numeric = df.select_dtypes(
        include=np.number
    )

    if (
        numeric.shape[0] >= 3
        and numeric.shape[1] >= 3
    ):

        return numeric.iloc[
            :3,
            :3,
        ].values

    return None


def calculate_metrics():

    matrix = load_confusion_matrix()

    if matrix is None:
        return None

    matrix = np.asarray(matrix)

    total = matrix.sum()

    if total > 0:
        accuracy = (
            np.trace(matrix)
            / total
        )
    else:
        accuracy = 0

    precisions = []
    recalls = []
    f1_scores = []

    for i in range(
        len(matrix)
    ):

        tp = matrix[i, i]

        fp = (
            matrix[:, i].sum()
            - tp
        )

        fn = (
            matrix[i, :].sum()
            - tp
        )

        if tp + fp > 0:
            precision = (
                tp / (tp + fp)
            )
        else:
            precision = 0

        if tp + fn > 0:
            recall = (
                tp / (tp + fn)
            )
        else:
            recall = 0

        if precision + recall > 0:

            f1 = (
                2
                * precision
                * recall
                / (
                    precision
                    + recall
                )
            )

        else:
            f1 = 0

        precisions.append(
            precision
        )

        recalls.append(
            recall
        )

        f1_scores.append(
            f1
        )

    return {
        "accuracy": accuracy,
        "macro_precision": np.mean(
            precisions
        ),
        "macro_recall": np.mean(
            recalls
        ),
        "macro_f1": np.mean(
            f1_scores
        ),
        "precision": precisions,
        "recall": recalls,
        "f1": f1_scores,
    }


# ============================================================
# FASTAPI FUNCTIONS
# ============================================================

def check_api():

    for base_url in API_URLS:

        try:

            response = requests.get(
                f"{base_url}/health",
                timeout=2,
            )

            if response.status_code == 200:
                return base_url

        except Exception:
            continue

    return None


def parse_prediction_response(data):

    result = {
        "class_id": None,
        "class_name": "Unknown",
        "confidence": 0.0,
        "probabilities": {},
    }

    if not isinstance(
        data,
        dict,
    ):
        return result

    possible_class = (
        data.get("predicted_class")
        or data.get("class_id")
        or data.get("prediction")
        or data.get("predicted_label")
    )

    if possible_class is not None:

        try:

            class_id = int(
                possible_class
            )

            result["class_id"] = (
                class_id
            )

            result["class_name"] = (
                CLASS_NAMES.get(
                    class_id,
                    f"Class {class_id}",
                )
            )

        except Exception:

            result["class_name"] = str(
                possible_class
            )

    possible_confidence = (
        data.get("confidence")
        or data.get("probability")
        or data.get("score")
    )

    if possible_confidence is not None:

        try:

            confidence = float(
                possible_confidence
            )

            if confidence > 1:
                confidence /= 100

            result["confidence"] = (
                confidence
            )

        except Exception:
            pass

    probabilities = (
        data.get("probabilities")
        or data.get(
            "class_probabilities"
        )
        or data.get("probs")
    )

    if isinstance(
        probabilities,
        dict,
    ):

        parsed = {}

        for key, value in (
            probabilities.items()
        ):

            try:

                parsed[
                    class_from_value(key)
                ] = float(value)

            except Exception:
                continue

        result["probabilities"] = (
            parsed
        )

    elif isinstance(
        probabilities,
        list,
    ):

        parsed = {}

        for i, value in enumerate(
            probabilities
        ):

            try:

                parsed[
                    CLASS_NAMES.get(
                        i,
                        f"Class {i}",
                    )
                ] = float(value)

            except Exception:
                continue

        result["probabilities"] = (
            parsed
        )

    return result


def call_prediction_api(
    base_url,
    uploaded_file,
):

    files = {
        "file": (
            uploaded_file.name,
            uploaded_file.getvalue(),
            "application/octet-stream",
        )
    }

    endpoints = [
        "/predict",
        "/prediction",
        "/api/predict",
    ]

    last_error = None

    for endpoint in endpoints:

        try:

            response = requests.post(
                f"{base_url}{endpoint}",
                files=files,
                timeout=120,
            )

            if response.status_code == 200:
                return response.json()

            last_error = (
                f"{endpoint}: "
                f"HTTP {response.status_code}"
            )

        except Exception as exc:

            last_error = str(exc)

    raise RuntimeError(
        last_error
        or "Prediction API request failed."
    )


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("🌱 TerraSpectra AI")

    st.caption(
        "Hyperspectral Potato Disease Intelligence"
    )

    st.divider()

    st.subheader("Navigation")

    page = st.radio(
        "Select module",
        [
            "Spectral Analysis",
            "Disease Prediction",
            "Monitoring",
        ],
        index=[
            "Spectral Analysis",
            "Disease Prediction",
            "Monitoring",
        ].index(
            st.session_state.page
        ),
    )

    st.session_state.page = page

    st.divider()

    st.subheader("System Status")

    api_status = check_api()

    if api_status:
        st.success(
            "● FastAPI Online"
        )
    else:
        st.error(
            "● FastAPI Offline"
        )

    tile_count, patch_count = (
        get_dataset_stats()
    )

    st.metric(
        "Hyperspectral Tiles",
        fmt_number(tile_count),
    )

    st.metric(
        "Labeled Patches",
        fmt_number(patch_count),
    )

    st.caption(
        "20 bands • 3 disease classes"
    )


# ============================================================
# MAIN HEADER
# ============================================================

st.title(
    "🌱 TerraSpectra AI"
)

st.caption(
    "AI-powered hyperspectral agriculture intelligence"
)

st.divider()


# ============================================================
# PAGE 1
# SPECTRAL ANALYSIS
# ============================================================

if page == "Spectral Analysis":

    st.header(
        "🔬 Spectral Analysis"
    )

    st.write(
        "Explore hyperspectral imagery, spectral bands, "
        "pixel signatures and PCA information."
    )

    upload_tab, dataset_tab, pca_tab = st.tabs(
        [
            "📤 Upload Cube",
            "📁 Dataset Explorer",
            "📊 PCA Analysis",
        ]
    )

    # ========================================================
    # UPLOAD TAB
    # ========================================================

    with upload_tab:

        uploaded = st.file_uploader(
            "Upload hyperspectral NPZ",
            type=["npz"],
            key="spectral_upload",
        )

        cube = None

        if uploaded is not None:

            try:

                cube = load_npz_bytes(
                    uploaded
                )

                st.success(
                    f"Hyperspectral cube loaded: "
                    f"{cube.shape}"
                )

            except Exception as exc:

                st.error(
                    f"Unable to read file: {exc}"
                )

        elif DATA_DIR.exists():

            files = sorted(
                DATA_DIR.glob("*.npz")
            )

            if files:

                selected_file = (
                    st.selectbox(
                        "Or select dataset tile",
                        files,
                        format_func=lambda x: x.name,
                    )
                )

                try:

                    cube = load_dataset_cube(
                        selected_file
                    )

                except Exception as exc:

                    st.error(
                        f"Unable to load tile: {exc}"
                    )

        if cube is not None:

            st.divider()

            height, width, bands = (
                cube.shape
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Image Height",
                f"{height}px",
            )

            c2.metric(
                "Image Width",
                f"{width}px",
            )

            c3.metric(
                "Spectral Bands",
                bands,
            )

            st.subheader(
                "Spectral Band Viewer"
            )

            band_number = st.slider(
                "Select band",
                min_value=0,
                max_value=bands - 1,
                value=0,
            )

            band_image = normalize_image(
                cube[
                    :,
                    :,
                    band_number,
                ]
            )

            wavelength = (
                WAVELENGTHS[
                    band_number
                ]
                if band_number
                < len(WAVELENGTHS)
                else "Unknown"
            )

            st.image(
                band_image,
                caption=(
                    f"Band {band_number + 1} "
                    f"• {wavelength} nm"
                ),
                width="stretch",
            )

            st.subheader(
                "Band Statistics"
            )

            selected_band = cube[
                :,
                :,
                band_number,
            ]

            stats_df = pd.DataFrame(
                {
                    "Metric": [
                        "Minimum",
                        "Maximum",
                        "Mean",
                        "Median",
                        "Standard Deviation",
                    ],
                    "Value": [
                        float(
                            np.min(
                                selected_band
                            )
                        ),
                        float(
                            np.max(
                                selected_band
                            )
                        ),
                        float(
                            np.mean(
                                selected_band
                            )
                        ),
                        float(
                            np.median(
                                selected_band
                            )
                        ),
                        float(
                            np.std(
                                selected_band
                            )
                        ),
                    ],
                }
            )

            st.dataframe(
                stats_df,
                width="stretch",
                hide_index=True,
            )

            st.subheader(
                "Pixel Spectral Signature"
            )

            col_x, col_y = st.columns(2)

            with col_x:

                pixel_x = st.number_input(
                    "X coordinate",
                    min_value=0,
                    max_value=width - 1,
                    value=width // 2,
                )

            with col_y:

                pixel_y = st.number_input(
                    "Y coordinate",
                    min_value=0,
                    max_value=height - 1,
                    value=height // 2,
                )

            spectrum = cube[
                int(pixel_y),
                int(pixel_x),
                :,
            ]

            wavelengths = (
                WAVELENGTHS[:bands]
                if bands
                <= len(WAVELENGTHS)
                else list(range(bands))
            )

            spectrum_df = pd.DataFrame(
                {
                    "Wavelength": wavelengths,
                    "Spectral Value": spectrum,
                }
            )

            fig = px.line(
                spectrum_df,
                x="Wavelength",
                y="Spectral Value",
                markers=True,
                title="Pixel Spectral Signature",
                template="plotly_dark",
            )

            fig.update_layout(
                height=450,
                paper_bgcolor="#0b1120",
                plot_bgcolor="#111827",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )


    # ========================================================
    # DATASET TAB
    # ========================================================

    with dataset_tab:

        st.subheader(
            "Dataset Overview"
        )

        tile_count, patch_count = (
            get_dataset_stats()
        )

        c1, c2, c3, c4 = st.columns(4)

        c1.metric(
            "Tiles",
            fmt_number(tile_count),
        )

        c2.metric(
            "Labeled Patches",
            fmt_number(patch_count),
        )

        c3.metric(
            "Spectral Bands",
            EXPECTED_BANDS,
        )

        c4.metric(
            "CNN Parameters",
            fmt_number(
                get_model_parameter_count()
            ),
        )

        patch_df = load_dataframe(
            PATCH_CSV
        )

        if not patch_df.empty:

            class_col = find_column(
                patch_df,
                [
                    "class",
                    "label",
                    "target",
                    "disease_class",
                ],
            )

            if class_col:

                display_df = (
                    patch_df.copy()
                )

                display_df[
                    "Class Name"
                ] = display_df[
                    class_col
                ].apply(
                    class_from_value
                )

                distribution = (
                    display_df[
                        "Class Name"
                    ]
                    .value_counts()
                    .reset_index()
                )

                distribution.columns = [
                    "Class",
                    "Samples",
                ]

                fig = px.bar(
                    distribution,
                    x="Class",
                    y="Samples",
                    color="Class",
                    color_discrete_map=CLASS_COLORS,
                    title="Patch Distribution",
                    template="plotly_dark",
                )

                fig.update_layout(
                    showlegend=False,
                    height=420,
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                with st.expander(
                    "View patch dataset"
                ):

                    st.dataframe(
                        display_df.head(
                            200
                        ),
                        width="stretch",
                        hide_index=True,
                    )

            else:

                st.dataframe(
                    patch_df.head(200),
                    width="stretch",
                    hide_index=True,
                )

        else:

            st.info(
                "hyperspectral_patches.csv was not found."
            )


    # ========================================================
    # PCA TAB
    # ========================================================

    with pca_tab:

        st.subheader(
            "Principal Component Analysis"
        )

        st.write(
            "PCA reduces hyperspectral dimensionality "
            "while preserving the majority of spectral variance."
        )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "PC1",
            "71.93%",
        )

        c2.metric(
            "PC2",
            "27.31%",
        )

        c3.metric(
            "PC1 + PC2",
            "99.24%",
        )

        pca_df = pd.DataFrame(
            {
                "Component": [
                    "PC1",
                    "PC2",
                    "PC3",
                ],
                "Variance": [
                    71.93,
                    27.31,
                    0.99,
                ],
            }
        )

        fig = px.bar(
            pca_df,
            x="Component",
            y="Variance",
            title="Explained Spectral Variance",
            template="plotly_dark",
        )

        fig.update_layout(
            yaxis_title="Variance (%)",
            height=430,
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.success(
            "PC1 and PC2 together capture approximately "
            "99% of the spectral variance."
        )


# ============================================================
# PAGE 2
# DISEASE PREDICTION
# ============================================================

elif page == "Disease Prediction":

    st.header(
        "🧠 Disease Prediction"
    )

    st.write(
        "Upload a hyperspectral NPZ cube and run "
        "AI-based disease classification."
    )

    api_url = check_api()

    if api_url:

        st.success(
            f"FastAPI connected: {api_url}"
        )

    else:

        st.error(
            "FastAPI backend is offline."
        )

        st.code(
            "python -m uvicorn api.main:app --reload"
        )

    uploaded = st.file_uploader(
        "Upload hyperspectral NPZ",
        type=["npz"],
        key="prediction_upload",
    )

    if uploaded is not None:

        try:

            cube = load_npz_bytes(
                uploaded
            )

            st.success(
                f"Valid hyperspectral cube: "
                f"{cube.shape}"
            )

            st.subheader(
                "Input Preview"
            )

            preview_band = st.slider(
                "Preview spectral band",
                0,
                cube.shape[-1] - 1,
                0,
                key="prediction_band",
            )

            preview = normalize_image(
                cube[
                    :,
                    :,
                    preview_band,
                ]
            )

            st.image(
                preview,
                caption=(
                    f"Band {preview_band + 1}"
                ),
                width="stretch",
            )

            st.subheader(
                "Input Validation"
            )

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Height",
                cube.shape[0],
            )

            c2.metric(
                "Width",
                cube.shape[1],
            )

            c3.metric(
                "Bands",
                cube.shape[2],
            )

            if (
                cube.shape[-1]
                != EXPECTED_BANDS
            ):

                st.error(
                    f"Expected {EXPECTED_BANDS} bands, "
                    f"received {cube.shape[-1]}."
                )

            else:

                if st.button(
                    "🚀 Run Disease Prediction",
                    type="primary",
                    width="stretch",
                ):

                    if not api_url:

                        st.error(
                            "Start the FastAPI backend first."
                        )

                    else:

                        with st.spinner(
                            "Running AI inference..."
                        ):

                            try:

                                response = (
                                    call_prediction_api(
                                        api_url,
                                        uploaded,
                                    )
                                )

                                result = (
                                    parse_prediction_response(
                                        response
                                    )
                                )

                                st.session_state[
                                    "prediction_result"
                                ] = result

                                st.success(
                                    "Prediction completed."
                                )

                            except Exception as exc:

                                st.error(
                                    f"Prediction failed: {exc}"
                                )

        except Exception as exc:

            st.error(
                f"Invalid NPZ file: {exc}"
            )


    # ========================================================
    # PREDICTION RESULT
    # ========================================================

    result = (
        st.session_state
        .prediction_result
    )

    if result:

        st.divider()

        st.subheader(
            "Prediction Result"
        )

        class_name = (
            result["class_name"]
        )

        confidence = (
            result["confidence"]
        )

        if class_name == "Healthy":

            st.success(
                f"🌱 Prediction: {class_name}"
            )

        elif (
            class_name
            == "Disease Class 1"
        ):

            st.warning(
                f"⚠️ Prediction: {class_name}"
            )

        elif (
            class_name
            == "Disease Class 2"
        ):

            st.error(
                f"🚨 Prediction: {class_name}"
            )

        else:

            st.info(
                f"Prediction: {class_name}"
            )

        c1, c2, c3 = st.columns(3)

        c1.metric(
            "Predicted Class",
            class_name,
        )

        c2.metric(
            "Confidence",
            f"{confidence * 100:.2f}%",
        )

        c3.metric(
            "Class ID",
            (
                result["class_id"]
                if result["class_id"]
                is not None
                else "N/A"
            ),
        )

        st.progress(
            min(
                max(
                    confidence,
                    0.0,
                ),
                1.0,
            ),
            text=(
                f"Confidence: "
                f"{confidence * 100:.2f}%"
            ),
        )

        probabilities = (
            result.get(
                "probabilities",
                {},
            )
        )

        if probabilities:

            st.subheader(
                "Class Probabilities"
            )

            probability_df = (
                pd.DataFrame(
                    {
                        "Class": list(
                            probabilities.keys()
                        ),
                        "Probability": [
                            float(value)
                            for value in probabilities.values()
                        ],
                    }
                )
            )

            probability_df[
                "Probability (%)"
            ] = (
                probability_df[
                    "Probability"
                ]
                * 100
            )

            fig = px.bar(
                probability_df,
                x="Class",
                y="Probability (%)",
                color="Class",
                color_discrete_map=CLASS_COLORS,
                title="Prediction Probability",
                template="plotly_dark",
            )

            fig.update_layout(
                showlegend=False,
                height=430,
                paper_bgcolor="#0b1120",
                plot_bgcolor="#111827",
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

            st.dataframe(
                probability_df,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# PAGE 3
# MONITORING
# ============================================================

elif page == "Monitoring":

    st.header(
        "🗺️ Field Monitoring"
    )

    st.write(
        "Monitor disease distribution, geospatial predictions "
        "and machine-learning performance."
    )

    geo_df = load_dataframe(
        GEO_CSV
    )

    report_df = load_dataframe(
        REPORT_CSV
    )

    history_df = load_dataframe(
        HISTORY_CSV
    )

    metrics = calculate_metrics()

    # ========================================================
    # KPI SECTION
    # ========================================================

    tile_count, patch_count = (
        get_dataset_stats()
    )

    total_locations = len(
        geo_df
    )

    disease_locations = 0

    if not geo_df.empty:

        disease_col = find_column(
            geo_df,
            [
                "Disease Class",
                "Disease",
                "Class",
                "Prediction",
                "Predicted Class",
                "label",
            ],
        )

        if disease_col:

            geo_df[
                "_ClassName"
            ] = geo_df[
                disease_col
            ].apply(
                class_from_value
            )

            disease_locations = int(
                (
                    geo_df[
                        "_ClassName"
                    ]
                    != "Healthy"
                ).sum()
            )

    c1, c2, c3, c4, c5 = (
        st.columns(5)
    )

    c1.metric(
        "Hyperspectral Tiles",
        fmt_number(tile_count),
    )

    c2.metric(
        "Labeled Patches",
        fmt_number(patch_count),
    )

    c3.metric(
        "Map Locations",
        fmt_number(total_locations),
    )

    c4.metric(
        "Disease Locations",
        fmt_number(
            disease_locations
        ),
    )

    if metrics:

        c5.metric(
            "Model Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%",
        )

    else:

        c5.metric(
            "Model Accuracy",
            "N/A",
        )

    st.divider()

    # ========================================================
    # REAL OPENSTREETMAP
    # ========================================================

    st.subheader(
        "🌍 Disease Monitoring Map"
    )

    if geo_df.empty:

        st.warning(
            "Geospatial prediction data not found."
        )

        st.info(
            "Expected: outputs/geospatial_predictions.csv"
        )

    else:

        lat_col = find_column(
            geo_df,
            [
                "Latitude",
                "latitude",
                "lat",
            ],
        )

        lon_col = find_column(
            geo_df,
            [
                "Longitude",
                "longitude",
                "lon",
                "lng",
            ],
        )

        disease_col = find_column(
            geo_df,
            [
                "Disease Class",
                "Disease",
                "Class",
                "Prediction",
                "Predicted Class",
                "label",
            ],
        )

        if (
            not lat_col
            or not lon_col
        ):

            st.error(
                "Latitude or Longitude columns "
                "were not found."
            )

        else:

            map_df = geo_df.copy()

            map_df["Latitude"] = (
                pd.to_numeric(
                    map_df[
                        lat_col
                    ],
                    errors="coerce",
                )
            )

            map_df["Longitude"] = (
                pd.to_numeric(
                    map_df[
                        lon_col
                    ],
                    errors="coerce",
                )
            )

            map_df = map_df.dropna(
                subset=[
                    "Latitude",
                    "Longitude",
                ]
            )

            if disease_col:

                map_df[
                    "Disease Class"
                ] = map_df[
                    disease_col
                ].apply(
                    class_from_value
                )

            else:

                map_df[
                    "Disease Class"
                ] = "Unknown"

            # ------------------------------------------------
            # FILTER
            # ------------------------------------------------

            selected_classes = (
                st.multiselect(
                    "Filter disease classes",
                    [
                        "Healthy",
                        "Disease Class 1",
                        "Disease Class 2",
                    ],
                    default=[
                        "Healthy",
                        "Disease Class 1",
                        "Disease Class 2",
                    ],
                )
            )

            filtered_map = map_df[
                map_df[
                    "Disease Class"
                ].isin(
                    selected_classes
                )
            ].copy()

            # ------------------------------------------------
            # MAP COLORS
            # ------------------------------------------------

            def map_color(
                class_name
            ):

                if (
                    class_name
                    == "Healthy"
                ):
                    return [
                        34,
                        197,
                        94,
                        230,
                    ]

                if (
                    class_name
                    == "Disease Class 1"
                ):
                    return [
                        245,
                        158,
                        11,
                        235,
                    ]

                if (
                    class_name
                    == "Disease Class 2"
                ):
                    return [
                        239,
                        68,
                        68,
                        240,
                    ]

                return [
                    148,
                    163,
                    184,
                    220,
                ]

            filtered_map[
                "Color"
            ] = filtered_map[
                "Disease Class"
            ].apply(
                map_color
            )

            # ------------------------------------------------
            # MAP CENTER
            # ------------------------------------------------

            if not filtered_map.empty:

                center_lat = float(
                    filtered_map[
                        "Latitude"
                    ].mean()
                )

                center_lon = float(
                    filtered_map[
                        "Longitude"
                    ].mean()
                )

                lat_range = (
                    filtered_map[
                        "Latitude"
                    ].max()
                    - filtered_map[
                        "Latitude"
                    ].min()
                )

                lon_range = (
                    filtered_map[
                        "Longitude"
                    ].max()
                    - filtered_map[
                        "Longitude"
                    ].min()
                )

                max_range = max(
                    float(
                        lat_range
                    ),
                    float(
                        lon_range
                    ),
                )

                if max_range < 0.01:

                    zoom = 14

                elif max_range < 0.05:

                    zoom = 12

                elif max_range < 0.2:

                    zoom = 10

                elif max_range < 1:

                    zoom = 7

                else:

                    zoom = 4

            else:

                center_lat = 22.3072
                center_lon = 72.1362
                zoom = 5

            # ------------------------------------------------
            # OPENSTREETMAP TILE LAYER
            # ------------------------------------------------

            osm_layer = pdk.Layer(
                "TileLayer",
                data=(
                    "https://tile.openstreetmap.org/"
                    "{z}/{x}/{y}.png"
                ),
                min_zoom=0,
                max_zoom=19,
                tile_size=256,
            )

            # ------------------------------------------------
            # DISEASE POINT LAYER
            # ------------------------------------------------

            point_layer = pdk.Layer(
                "ScatterplotLayer",
                data=filtered_map,
                get_position=[
                    "Longitude",
                    "Latitude",
                ],
                get_fill_color="Color",
                get_radius=50,
                radius_min_pixels=5,
                radius_max_pixels=25,
                pickable=True,
                stroked=True,
                get_line_color=[
                    255,
                    255,
                    255,
                    180,
                ],
                line_width_min_pixels=1,
            )

            # ------------------------------------------------
            # MAP VIEW
            # ------------------------------------------------

            view_state = pdk.ViewState(
                latitude=center_lat,
                longitude=center_lon,
                zoom=zoom,
                pitch=0,
                bearing=0,
            )

            deck = pdk.Deck(
                layers=[
                    osm_layer,
                    point_layer,
                ],
                initial_view_state=view_state,
                tooltip={
                    "text": (
                        "Disease: {Disease Class}\n"
                        "Latitude: {Latitude}\n"
                        "Longitude: {Longitude}"
                    )
                },
            )

            st.pydeck_chart(
                deck,
                width="stretch",
                height=650,
            )

            st.caption(
                "Street map © OpenStreetMap contributors. "
                "Markers represent the disease predictions "
                "contained in the geospatial dataset."
            )

            # ------------------------------------------------
            # LEGEND
            # ------------------------------------------------

            st.subheader(
                "Map Legend"
            )

            l1, l2, l3 = (
                st.columns(3)
            )

            l1.success(
                "🟢 Healthy"
            )

            l2.warning(
                "🟠 Disease Class 1"
            )

            l3.error(
                "🔴 Disease Class 2"
            )

            with st.expander(
                "View mapped predictions"
            ):

                st.dataframe(
                    filtered_map,
                    width="stretch",
                    hide_index=True,
                )

    st.divider()

    # ========================================================
    # DISEASE DISTRIBUTION
    # ========================================================

    st.subheader(
        "📊 Disease Distribution"
    )

    if not geo_df.empty:

        disease_col = find_column(
            geo_df,
            [
                "Disease Class",
                "Disease",
                "Class",
                "Prediction",
                "Predicted Class",
                "label",
            ],
        )

        if disease_col:

            distribution = (
                geo_df[
                    disease_col
                ]
                .apply(
                    class_from_value
                )
                .value_counts()
                .reindex(
                    [
                        "Healthy",
                        "Disease Class 1",
                        "Disease Class 2",
                    ],
                    fill_value=0,
                )
                .reset_index()
            )

            distribution.columns = [
                "Disease Class",
                "Locations",
            ]

            c1, c2 = (
                st.columns(2)
            )

            with c1:

                fig = px.pie(
                    distribution,
                    names="Disease Class",
                    values="Locations",
                    color="Disease Class",
                    color_discrete_map=CLASS_COLORS,
                    hole=0.55,
                    title="Disease Distribution",
                    template="plotly_dark",
                )

                fig.update_layout(
                    height=430,
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            with c2:

                fig = px.bar(
                    distribution,
                    x="Disease Class",
                    y="Locations",
                    color="Disease Class",
                    color_discrete_map=CLASS_COLORS,
                    title="Disease Class Counts",
                    template="plotly_dark",
                )

                fig.update_layout(
                    showlegend=False,
                    height=430,
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )


    # ========================================================
    # MODEL DIAGNOSTICS
    # ========================================================

    st.divider()

    st.subheader(
        "🎯 Model Diagnostics"
    )

    if metrics:

        c1, c2, c3, c4 = (
            st.columns(4)
        )

        c1.metric(
            "Accuracy",
            f"{metrics['accuracy'] * 100:.2f}%",
        )

        c2.metric(
            "Macro Precision",
            f"{metrics['macro_precision'] * 100:.2f}%",
        )

        c3.metric(
            "Macro Recall",
            f"{metrics['macro_recall'] * 100:.2f}%",
        )

        c4.metric(
            "Macro F1",
            f"{metrics['macro_f1'] * 100:.2f}%",
        )

        performance_df = (
            pd.DataFrame(
                {
                    "Class": [
                        "Healthy",
                        "Disease Class 1",
                        "Disease Class 2",
                    ],
                    "Precision": [
                        metrics[
                            "precision"
                        ][0],
                        metrics[
                            "precision"
                        ][1],
                        metrics[
                            "precision"
                        ][2],
                    ],
                    "Recall": [
                        metrics[
                            "recall"
                        ][0],
                        metrics[
                            "recall"
                        ][1],
                        metrics[
                            "recall"
                        ][2],
                    ],
                    "F1 Score": [
                        metrics[
                            "f1"
                        ][0],
                        metrics[
                            "f1"
                        ][1],
                        metrics[
                            "f1"
                        ][2],
                    ],
                }
            )
        )

        performance_long = (
            performance_df.melt(
                id_vars="Class",
                var_name="Metric",
                value_name="Score",
            )
        )

        performance_long[
            "Score"
        ] *= 100

        fig = px.bar(
            performance_long,
            x="Class",
            y="Score",
            color="Metric",
            barmode="group",
            title="Class-wise Model Performance",
            template="plotly_dark",
        )

        fig.update_layout(
            yaxis_title="Score (%)",
            height=450,
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        st.dataframe(
            performance_df.style.format(
                {
                    "Precision": "{:.2%}",
                    "Recall": "{:.2%}",
                    "F1 Score": "{:.2%}",
                }
            ),
            width="stretch",
            hide_index=True,
        )


    # ========================================================
    # CONFUSION MATRIX
    # ========================================================

    matrix = (
        load_confusion_matrix()
    )

    if matrix is not None:

        st.subheader(
            "Confusion Matrix"
        )

        class_labels = [
            "Healthy",
            "Disease Class 1",
            "Disease Class 2",
        ]

        fig = go.Figure(
            data=go.Heatmap(
                z=matrix,
                x=class_labels,
                y=class_labels,
                text=matrix,
                texttemplate="%{text}",
                colorscale="Viridis",
                hovertemplate=(
                    "Actual: %{y}<br>"
                    "Predicted: %{x}<br>"
                    "Samples: %{z}"
                    "<extra></extra>"
                ),
            )
        )

        fig.update_layout(
            title="Model Confusion Matrix",
            xaxis_title="Predicted Class",
            yaxis_title="Actual Class",
            template="plotly_dark",
            height=500,
            paper_bgcolor="#0b1120",
            plot_bgcolor="#111827",
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        matrix_df = pd.DataFrame(
            matrix,
            index=class_labels,
            columns=class_labels,
        )

        st.dataframe(
            matrix_df,
            width="stretch",
        )


    # ========================================================
    # TRAINING HISTORY
    # ========================================================

    if not history_df.empty:

        st.subheader(
            "📈 Training History"
        )

        epoch_col = find_column(
            history_df,
            [
                "epoch",
                "Epoch",
            ],
        )

        accuracy_col = find_column(
            history_df,
            [
                "val_accuracy",
                "validation_accuracy",
                "accuracy",
                "val_acc",
            ],
        )

        loss_col = find_column(
            history_df,
            [
                "val_loss",
                "validation_loss",
                "loss",
            ],
        )

        if epoch_col:

            chart_data = {
                "Epoch": history_df[
                    epoch_col
                ]
            }

            if accuracy_col:

                chart_data[
                    "Validation Accuracy"
                ] = history_df[
                    accuracy_col
                ]

            if loss_col:

                chart_data[
                    "Validation Loss"
                ] = history_df[
                    loss_col
                ]

            chart_df = pd.DataFrame(
                chart_data
            )

            value_columns = [
                column
                for column
                in chart_df.columns
                if column != "Epoch"
            ]

            if value_columns:

                fig = px.line(
                    chart_df,
                    x="Epoch",
                    y=value_columns,
                    markers=True,
                    title="Training Progress",
                    template="plotly_dark",
                )

                fig.update_layout(
                    height=450,
                    paper_bgcolor="#0b1120",
                    plot_bgcolor="#111827",
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

        with st.expander(
            "View training history"
        ):

            st.dataframe(
                history_df,
                width="stretch",
                hide_index=True,
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "TerraSpectra AI • Hyperspectral Agriculture Intelligence • "
    "PCA • 3D CNN • FastAPI • Streamlit"
)