from pathlib import Path
import io
import json

import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go


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
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

OUTPUTS = ROOT / "outputs"
DATA_DIR = ROOT / "data" / "raw" / "hyperspectral" / "0"
LABEL_DIR = ROOT / "data" / "raw" / "labels" / "0"
MODELS_DIR = OUTPUTS / "models"

PATCH_CSV = OUTPUTS / "hyperspectral_patches.csv"
GEO_CSV = OUTPUTS / "geospatial_predictions.csv"

CONFUSION_CSV = OUTPUTS / "tilesplit_confusion_matrix.csv"
REPORT_CSV = OUTPUTS / "tilesplit_classification_report.csv"
HISTORY_CSV = OUTPUTS / "tilesplit_training_history.csv"

API_URLS = [
    "http://127.0.0.1:8000",
    "http://localhost:8000",
]

EXPECTED_BANDS = 20


# ============================================================
# SPECTRAL INFORMATION
# ============================================================

WAVELENGTHS = [
    420, 440, 500, 520, 540,
    560, 580, 600, 620, 640,
    660, 680, 700, 720, 740,
    760, 770, 800, 850, 900
]

CLASS_NAMES = {
    0: "Healthy",
    1: "Disease Class 1",
    2: "Disease Class 2",
}

CLASS_COLORS = {
    "Healthy": "#22c55e",
    "Disease Class 1": "#f59e0b",
    "Disease Class 2": "#ef4444",
}

ACCENT_CYAN = "#22d3ee"
ACCENT_PURPLE = "#a855f7"
ACCENT_BLUE = "#3b82f6"
ACCENT_GREEN = "#22c55e"
ACCENT_ORANGE = "#f59e0b"
ACCENT_RED = "#ef4444"


# ============================================================
# SESSION STATE
# ============================================================

if "page" not in st.session_state:
    st.session_state.page = "Spectral Analysis"

if "prediction_result" not in st.session_state:
    st.session_state.prediction_result = None


def navigate(page):
    st.session_state.page = page
    st.rerun()


# ============================================================
# DARK PROFESSIONAL THEME
# ============================================================

st.markdown(
    """
    <style>

    .stApp {
        background:
            radial-gradient(circle at 10% 10%, rgba(34,211,238,0.07), transparent 28%),
            radial-gradient(circle at 90% 20%, rgba(168,85,247,0.07), transparent 28%),
            radial-gradient(circle at 50% 100%, rgba(34,197,94,0.04), transparent 30%),
            #070b14;
        color: #f8fafc;
    }

    [data-testid="stSidebar"] {
        background: #090e19;
        border-right: 1px solid rgba(148,163,184,0.12);
    }

    [data-testid="stSidebar"] * {
        color: #e2e8f0;
    }

    h1, h2, h3 {
        color: #f8fafc !important;
        letter-spacing: -0.02em;
    }

    p, label, span {
        color: #cbd5e1;
    }

    .hero {
        padding: 26px 30px;
        border-radius: 20px;
        margin-bottom: 24px;
        background:
            linear-gradient(
                135deg,
                rgba(34,211,238,0.12),
                rgba(168,85,247,0.10),
                rgba(34,197,94,0.06)
            );
        border: 1px solid rgba(34,211,238,0.18);
        box-shadow: 0 12px 40px rgba(0,0,0,0.28);
    }

    .hero-title {
        font-size: 38px;
        font-weight: 800;
        margin-bottom: 5px;
        color: #f8fafc;
    }

    .hero-subtitle {
        color: #94a3b8;
        font-size: 15px;
    }

    .section-label {
        color: #22d3ee;
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.16em;
        text-transform: uppercase;
        margin-bottom: 5px;
    }

    .metric-card {
        background: linear-gradient(
            145deg,
            rgba(15,23,42,0.95),
            rgba(15,23,42,0.72)
        );
        border: 1px solid rgba(148,163,184,0.12);
        border-radius: 17px;
        padding: 20px;
        min-height: 125px;
        box-shadow: 0 8px 30px rgba(0,0,0,0.22);
    }

    .metric-label {
        color: #94a3b8;
        font-size: 12px;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }

    .metric-value {
        color: #f8fafc;
        font-size: 29px;
        font-weight: 800;
        margin-top: 7px;
    }

    .metric-note {
        color: #64748b;
        font-size: 12px;
        margin-top: 5px;
    }

    .status-online {
        color: #22c55e;
        font-weight: 700;
    }

    .status-offline {
        color: #ef4444;
        font-weight: 700;
    }

    .feature-card {
        background: rgba(15,23,42,0.72);
        border: 1px solid rgba(148,163,184,0.10);
        border-radius: 16px;
        padding: 20px;
        min-height: 150px;
    }

    .feature-title {
        font-weight: 800;
        color: #f8fafc;
        margin-bottom: 7px;
    }

    .feature-text {
        color: #94a3b8;
        font-size: 13px;
        line-height: 1.6;
    }

    .prediction-box {
        border-radius: 20px;
        padding: 28px;
        margin-top: 15px;
        border: 1px solid rgba(34,211,238,0.20);
        background:
            linear-gradient(
                145deg,
                rgba(34,211,238,0.08),
                rgba(168,85,247,0.07)
            );
    }

    .prediction-class {
        font-size: 34px;
        font-weight: 850;
        color: #f8fafc;
    }

    .prediction-confidence {
        font-size: 18px;
        color: #22d3ee;
        font-weight: 750;
    }

    .small-muted {
        color: #64748b;
        font-size: 12px;
    }

    div[data-testid="stFileUploader"] {
        background: rgba(15,23,42,0.55);
        border-radius: 15px;
    }

    .stButton > button {
        border-radius: 11px;
        border: 1px solid rgba(148,163,184,0.15);
        background: rgba(15,23,42,0.85);
        color: #e2e8f0;
        font-weight: 700;
        transition: all 0.2s ease;
    }

    .stButton > button:hover {
        border-color: rgba(34,211,238,0.55);
        color: #22d3ee;
        transform: translateY(-1px);
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPERS
# ============================================================

def fmt_number(value):
    try:
        return f"{int(value):,}"
    except Exception:
        return "—"


def normalize_image(image):
    image = np.asarray(image, dtype=np.float32)

    if image.size == 0:
        return np.zeros_like(image)

    image = np.nan_to_num(
        image,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )

    minimum = float(np.min(image))
    maximum = float(np.max(image))

    if maximum <= minimum:
        return np.zeros_like(image, dtype=np.float32)

    return (image - minimum) / (maximum - minimum)


def normalize_cube_shape(cube):
    """
    Convert supported cube formats to H x W x Bands.
    """

    cube = np.asarray(cube)

    if cube.ndim != 3:
        raise ValueError(
            f"Expected 3D hyperspectral cube, received shape {cube.shape}"
        )

    shape = cube.shape

    # H x W x Bands
    if shape[-1] == EXPECTED_BANDS:
        return cube

    # Bands x H x W
    if shape[0] == EXPECTED_BANDS:
        return np.transpose(cube, (1, 2, 0))

    # H x Bands x W
    if shape[1] == EXPECTED_BANDS:
        return np.transpose(cube, (0, 2, 1))

    raise ValueError(
        f"Could not find {EXPECTED_BANDS} spectral bands in cube shape {shape}"
    )


def load_npz_bytes(raw_bytes):
    with np.load(
        io.BytesIO(raw_bytes),
        allow_pickle=False
    ) as data:

        if "im" not in data:
            raise ValueError(
                "The uploaded NPZ does not contain the required 'im' array."
            )

        cube = normalize_cube_shape(data["im"])

    return cube.astype(np.float32)


def load_dataset_cube(filename):
    path = DATA_DIR / filename

    if not path.exists():
        return None

    try:
        with np.load(path, allow_pickle=False) as data:
            if "im" not in data:
                return None

            return normalize_cube_shape(data["im"]).astype(np.float32)

    except Exception:
        return None


def load_dataframe(path):
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception:
        pass

    return pd.DataFrame()


def find_column(df, candidates):
    if df.empty:
        return None

    lower_map = {
        str(col).lower().strip(): col
        for col in df.columns
    }

    for candidate in candidates:
        if candidate.lower() in lower_map:
            return lower_map[candidate.lower()]

    for col in df.columns:
        col_lower = str(col).lower()

        for candidate in candidates:
            if candidate.lower() in col_lower:
                return col

    return None


def class_from_value(value):
    if pd.isna(value):
        return "Unknown"

    text = str(value).strip().lower()

    if text in {"0", "healthy", "normal", "class 0", "healthy / normal"}:
        return "Healthy"

    if text in {"1", "disease class 1", "class 1", "disease1", "disease 1"}:
        return "Disease Class 1"

    if text in {"2", "disease class 2", "class 2", "disease2", "disease 2"}:
        return "Disease Class 2"

    try:
        number = int(float(value))

        if number in CLASS_NAMES:
            return CLASS_NAMES[number]

    except Exception:
        pass

    return str(value)


def load_patch_data():
    return load_dataframe(PATCH_CSV)


def get_dataset_stats():
    patch_df = load_patch_data()

    total_tiles = 0
    total_patches = 0

    if DATA_DIR.exists():
        total_tiles = len(list(DATA_DIR.glob("*.npz")))

    if not patch_df.empty:
        total_patches = len(patch_df)

    class_counts = {
        "Healthy": 8060,
        "Disease Class 1": 360,
        "Disease Class 2": 268,
    }

    if not patch_df.empty:
        possible_class_col = find_column(
            patch_df,
            [
                "class_id",
                "class",
                "label",
                "target",
                "disease_class",
            ],
        )

        if possible_class_col:
            values = patch_df[possible_class_col].apply(class_from_value)
            counts = values.value_counts()

            for class_name in class_counts:
                if class_name in counts:
                    class_counts[class_name] = int(
                        counts[class_name]
                    )

    if total_patches == 0:
        total_patches = sum(class_counts.values())

    return total_tiles, total_patches, class_counts


def model_parameter_count():
    default_params = 18243

    model_files = [
        MODELS_DIR / "terraspectra_3dcnn_balanced_best.pt",
        MODELS_DIR / "terraspectra_3dcnn_best.pt",
    ]

    for path in model_files:
        if not path.exists():
            continue

        try:
            import torch

            checkpoint = torch.load(
                path,
                map_location="cpu"
            )

            if isinstance(checkpoint, dict):
                state_dict = checkpoint.get(
                    "state_dict",
                    checkpoint.get("model_state_dict")
                )

                if state_dict:
                    return sum(
                        value.numel()
                        for value in state_dict.values()
                        if hasattr(value, "numel")
                    )

        except Exception:
            continue

    return default_params


# ============================================================
# MODEL METRICS
# ============================================================

def load_confusion_matrix():
    if CONFUSION_CSV.exists():
        try:
            matrix = pd.read_csv(
                CONFUSION_CSV,
                index_col=0
            )

            matrix = matrix.apply(
                pd.to_numeric,
                errors="coerce"
            ).fillna(0)

            if matrix.shape == (3, 3):
                return matrix.values.astype(int)

        except Exception:
            pass

    return np.array(
        [
            [1535, 55, 1],
            [39, 31, 0],
            [62, 2, 0],
        ]
    )


def calculate_metrics(matrix):
    matrix = np.asarray(matrix)

    total = matrix.sum()

    if total == 0:
        return 0.0, 0.0, []

    accuracy = np.trace(matrix) / total

    class_metrics = []

    for i in range(matrix.shape[0]):
        tp = matrix[i, i]

        fp = matrix[:, i].sum() - tp
        fn = matrix[i, :].sum() - tp

        precision = (
            tp / (tp + fp)
            if (tp + fp) > 0
            else 0
        )

        recall = (
            tp / (tp + fn)
            if (tp + fn) > 0
            else 0
        )

        f1 = (
            2 * precision * recall /
            (precision + recall)
            if (precision + recall) > 0
            else 0
        )

        class_metrics.append(
            {
                "Class": CLASS_NAMES.get(i, f"Class {i}"),
                "Precision": precision,
                "Recall": recall,
                "F1 Score": f1,
                "Support": matrix[i].sum(),
            }
        )

    macro_f1 = np.mean(
        [item["F1 Score"] for item in class_metrics]
    )

    return accuracy, macro_f1, class_metrics


# ============================================================
# API
# ============================================================

def api_health_check():
    for base_url in API_URLS:
        try:
            response = requests.get(
                f"{base_url}/health",
                timeout=5
            )

            if response.status_code < 500:
                return True, base_url

        except Exception:
            continue

    return False, None


def parse_prediction_response(payload):
    if not isinstance(payload, dict):
        return None

    data = payload

    # Support nested API responses
    for key in ["data", "result", "prediction"]:
        if isinstance(data.get(key), dict):
            data = data[key]
            break

    prediction = None
    confidence = None
    probabilities = None

    prediction_keys = [
        "predicted_class",
        "predicted_class_id",
        "class_id",
        "prediction",
        "predicted_label",
        "label",
        "class",
    ]

    for key in prediction_keys:
        if key in data:
            prediction = data[key]
            break

    confidence_keys = [
        "confidence",
        "prediction_confidence",
        "probability",
        "score",
    ]

    for key in confidence_keys:
        if key in data:
            confidence = data[key]
            break

    for key in [
        "probabilities",
        "class_probabilities",
        "probs",
        "probability_distribution",
    ]:
        if key in data:
            probabilities = data[key]
            break

    if prediction is None:
        return None

    # Convert class ID to class name
    try:
        if isinstance(prediction, (int, float)):
            prediction_id = int(prediction)

            if prediction_id in CLASS_NAMES:
                prediction = CLASS_NAMES[prediction_id]

        elif isinstance(prediction, str):
            prediction = class_from_value(prediction)

    except Exception:
        prediction = str(prediction)

    # Confidence
    try:
        confidence = float(confidence)

        if confidence <= 1:
            confidence *= 100

    except Exception:
        confidence = None

    return {
        "class": str(prediction),
        "confidence": confidence,
        "probabilities": probabilities,
        "raw": payload,
    }


def call_prediction_api(file_bytes, filename):
    """
    Tries common FastAPI endpoints.

    No external API key is required.
    """

    errors = []

    endpoints = [
        "/predict",
        "/api/predict",
        "/prediction",
        "/api/prediction",
    ]

    for base_url in API_URLS:

        for endpoint in endpoints:

            try:
                response = requests.post(
                    base_url + endpoint,
                    files={
                        "file": (
                            filename,
                            file_bytes,
                            "application/octet-stream",
                        )
                    },
                    timeout=120,
                )

                if response.status_code == 404:
                    continue

                if response.status_code >= 400:
                    errors.append(
                        f"{endpoint}: HTTP {response.status_code} - "
                        f"{response.text[:250]}"
                    )
                    continue

                try:
                    payload = response.json()
                except Exception:
                    errors.append(
                        f"{endpoint}: API returned non-JSON response."
                    )
                    continue

                result = parse_prediction_response(payload)

                if result is not None:
                    return result, None

                errors.append(
                    f"{endpoint}: Response format not recognized."
                )

            except requests.exceptions.ConnectionError:
                errors.append(
                    f"{base_url}: backend connection failed."
                )

            except requests.exceptions.Timeout:
                errors.append(
                    f"{endpoint}: prediction request timed out."
                )

            except Exception as exc:
                errors.append(
                    f"{endpoint}: {str(exc)}"
                )

    return None, "\n".join(errors[-8:])


# ============================================================
# COMMON HEADER
# ============================================================

def render_header(title, subtitle):
    st.markdown(
        f"""
        <div class="hero">
            <div class="section-label">TerraSpectra AI</div>
            <div class="hero-title">{title}</div>
            <div class="hero-subtitle">{subtitle}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_page_navigation():
    cols = st.columns(3)

    pages = [
        ("🔬", "Spectral Analysis"),
        ("🧠", "Disease Prediction"),
        ("🗺️", "Monitoring"),
    ]

    for col, (icon, page) in zip(cols, pages):
        with col:
            if st.button(
                f"{icon}  {page}",
                width="stretch",
                key=f"nav_{page}",
            ):
                navigate(page)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.markdown(
        """
        <div style="
            font-size:24px;
            font-weight:800;
            color:#f8fafc;
            margin-bottom:3px;
        ">
            🌱 TerraSpectra
        </div>
        <div style="
            color:#22d3ee;
            font-size:12px;
            font-weight:700;
            letter-spacing:0.12em;
        ">
            HYPERSPECTRAL AI
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.divider()

    selected_page = st.radio(
        "NAVIGATION",
        [
            "Spectral Analysis",
            "Disease Prediction",
            "Monitoring",
        ],
        index=[
            "Spectral Analysis",
            "Disease Prediction",
            "Monitoring",
        ].index(st.session_state.page),
    )

    if selected_page != st.session_state.page:
        st.session_state.page = selected_page
        st.rerun()

    st.divider()

    total_tiles, total_patches, class_counts = get_dataset_stats()

    st.markdown(
        "### Dataset"
    )

    st.caption(
        f"🛰️ {fmt_number(total_tiles)} hyperspectral tiles"
    )

    st.caption(
        f"🧩 {fmt_number(total_patches)} labeled patches"
    )

    st.caption(
        "🌈 20 spectral bands"
    )

    st.divider()

    st.markdown(
        "### Model"
    )

    st.caption(
        "3D CNN • Tile Split"
    )

    st.caption(
        "90.78% validation accuracy"
    )

    st.caption(
        f"{fmt_number(model_parameter_count())} parameters"
    )

    st.divider()

    st.caption(
        "TerraSpectra AI • Research Dashboard"
    )


# ============================================================
# PAGE 1 — SPECTRAL ANALYSIS
# ============================================================

if st.session_state.page == "Spectral Analysis":

    render_header(
        "Spectral Intelligence",
        "Explore hyperspectral signatures, band responses and dimensionality reduction."
    )

    render_page_navigation()

    st.write("")

    total_tiles, total_patches, class_counts = get_dataset_stats()

    # --------------------------------------------------------
    # KPI ROW
    # --------------------------------------------------------

    cols = st.columns(4)

    metrics = [
        (
            "HYPERSPECTRAL TILES",
            fmt_number(total_tiles),
            "UAV captured samples",
        ),
        (
            "SPECTRAL BANDS",
            "20",
            "420–900 nm",
        ),
        (
            "LABELED PATCHES",
            fmt_number(total_patches),
            "Disease detection samples",
        ),
        (
            "PCA INFORMATION",
            "99.23%",
            "PC1 + PC2",
        ),
    ]

    for col, (label, value, note) in zip(cols, metrics):

        with col:
            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">{label}</div>
                    <div class="metric-value">{value}</div>
                    <div class="metric-note">{note}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.write("")

    # --------------------------------------------------------
    # DATA SOURCE
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">DATA SOURCE</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload hyperspectral NPZ file",
        type=["npz"],
        key="spectral_upload",
    )

    cube = None
    source_name = None

    if uploaded is not None:

        try:
            cube = load_npz_bytes(
                uploaded.getvalue()
            )

            source_name = uploaded.name

            st.success(
                f"Loaded {uploaded.name} • Shape: {cube.shape}"
            )

        except Exception as exc:

            st.error(
                f"Unable to read hyperspectral file: {exc}"
            )

    else:

        available_tiles = []

        if DATA_DIR.exists():
            available_tiles = sorted(
                DATA_DIR.glob("*.npz")
            )

        if available_tiles:

            selected_tile = st.selectbox(
                "Or select a dataset tile",
                available_tiles,
                format_func=lambda x: x.name,
            )

            cube = load_dataset_cube(
                selected_tile.name
            )

            source_name = selected_tile.name

    # --------------------------------------------------------
    # SPECTRAL VISUALIZATION
    # --------------------------------------------------------

    if cube is not None:

        st.divider()

        left, right = st.columns([1.25, 1])

        with left:

            st.markdown(
                '<div class="section-label">SPECTRAL BAND MAP</div>',
                unsafe_allow_html=True,
            )

            band_index = st.slider(
                "Select spectral band",
                min_value=0,
                max_value=EXPECTED_BANDS - 1,
                value=8,
            )

            wavelength = WAVELENGTHS[band_index]

            band_image = normalize_image(
                cube[:, :, band_index]
            )

            fig = px.imshow(
                band_image,
                color_continuous_scale="Viridis",
                aspect="auto",
                labels={
                    "x": "Pixel X",
                    "y": "Pixel Y",
                    "color": "Normalized Response",
                },
            )

            fig.update_layout(
                title=f"Band {band_index + 1} • {wavelength} nm",
                height=470,
                template="plotly_dark",
                margin=dict(l=10, r=10, t=50, b=10),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        with right:

            st.markdown(
                '<div class="section-label">CUBE INFORMATION</div>',
                unsafe_allow_html=True,
            )

            st.markdown(
                f"""
                <div class="feature-card">
                    <div class="feature-title">
                        {source_name}
                    </div>
                    <div class="feature-text">
                        Height: {cube.shape[0]} pixels<br>
                        Width: {cube.shape[1]} pixels<br>
                        Spectral bands: {cube.shape[2]}<br>
                        Data type: float32<br>
                        Selected wavelength: {wavelength} nm
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.write("")

            st.markdown(
                '<div class="section-label">BAND STATISTICS</div>',
                unsafe_allow_html=True,
            )

            band = cube[:, :, band_index]

            stats = pd.DataFrame(
                {
                    "Statistic": [
                        "Minimum",
                        "Maximum",
                        "Mean",
                        "Std. Deviation",
                    ],
                    "Value": [
                        float(np.min(band)),
                        float(np.max(band)),
                        float(np.mean(band)),
                        float(np.std(band)),
                    ],
                }
            )

            st.dataframe(
                stats,
                hide_index=True,
                width="stretch",
            )

        # ----------------------------------------------------
        # SPECTRAL SIGNATURE
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-label">SPECTRAL SIGNATURE</div>',
            unsafe_allow_html=True,
        )

        h, w, bands = cube.shape

        x_pixel = st.slider(
            "Pixel X",
            0,
            w - 1,
            w // 2,
        )

        y_pixel = st.slider(
            "Pixel Y",
            0,
            h - 1,
            h // 2,
        )

        spectrum = cube[
            y_pixel,
            x_pixel,
            :
        ]

        spectrum_norm = normalize_image(
            spectrum
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=WAVELENGTHS,
                y=spectrum_norm,
                mode="lines+markers",
                line=dict(
                    width=3
                ),
                marker=dict(
                    size=7
                ),
                name="Spectral Response",
            )
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            xaxis_title="Wavelength (nm)",
            yaxis_title="Normalized Spectral Response",
            hovermode="x unified",
            margin=dict(l=10, r=10, t=20, b=10),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

        # ----------------------------------------------------
        # PCA
        # ----------------------------------------------------

        st.divider()

        st.markdown(
            '<div class="section-label">DIMENSIONALITY REDUCTION</div>',
            unsafe_allow_html=True,
        )

        try:

            from sklearn.decomposition import PCA

            pixels = cube.reshape(
                -1,
                cube.shape[-1]
            )

            max_samples = min(
                10000,
                len(pixels)
            )

            rng = np.random.default_rng(42)

            if len(pixels) > max_samples:
                indexes = rng.choice(
                    len(pixels),
                    max_samples,
                    replace=False,
                )

                sample = pixels[indexes]

            else:
                sample = pixels

            sample = (
                sample - sample.mean(axis=0)
            ) / (
                sample.std(axis=0) + 1e-8
            )

            pca = PCA(
                n_components=3
            )

            transformed = pca.fit_transform(
                sample
            )

            explained = (
                pca.explained_variance_ratio_
                * 100
            )

            pc_cols = st.columns(3)

            for i, col in enumerate(pc_cols):

                with col:

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                PRINCIPAL COMPONENT {i + 1}
                            </div>
                            <div class="metric-value">
                                {explained[i]:.2f}%
                            </div>
                            <div class="metric-note">
                                Explained variance
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

            pca_df = pd.DataFrame(
                {
                    "PC1": transformed[:, 0],
                    "PC2": transformed[:, 1],
                    "PC3": transformed[:, 2],
                }
            )

            fig = px.scatter_3d(
                pca_df,
                x="PC1",
                y="PC2",
                z="PC3",
                opacity=0.65,
            )

            fig.update_layout(
                template="plotly_dark",
                height=600,
                margin=dict(
                    l=0,
                    r=0,
                    t=20,
                    b=0,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

        except Exception as exc:

            st.warning(
                f"PCA visualization unavailable: {exc}"
            )


# ============================================================
# PAGE 2 — DISEASE PREDICTION
# ============================================================

elif st.session_state.page == "Disease Prediction":

    render_header(
        "AI Disease Prediction",
        "Upload a hyperspectral tile and obtain an AI-driven disease classification."
    )

    render_page_navigation()

    st.write("")

    # --------------------------------------------------------
    # API STATUS
    # --------------------------------------------------------

    api_online, api_base = api_health_check()

    cols = st.columns(4)

    with cols[0]:
        status_text = (
            "ONLINE"
            if api_online
            else "OFFLINE"
        )

        status_color = (
            "#22c55e"
            if api_online
            else "#ef4444"
        )

        st.markdown(
            f"""
            <div class="metric-card">
                <div class="metric-label">AI API</div>
                <div class="metric-value"
                     style="color:{status_color};">
                    {status_text}
                </div>
                <div class="metric-note">
                    {api_base if api_base else "Start FastAPI backend"}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[1]:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    MODEL
                </div>
                <div class="metric-value">
                    3D CNN
                </div>
                <div class="metric-note">
                    Hyperspectral classifier
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[2]:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    VALIDATION
                </div>
                <div class="metric-value">
                    90.78%
                </div>
                <div class="metric-note">
                    Tile-split accuracy
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with cols[3]:

        st.markdown(
            """
            <div class="metric-card">
                <div class="metric-label">
                    CLASSES
                </div>
                <div class="metric-value">
                    3
                </div>
                <div class="metric-note">
                    Healthy + 2 disease classes
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.write("")

    # --------------------------------------------------------
    # UPLOAD
    # --------------------------------------------------------

    st.markdown(
        '<div class="section-label">INPUT DATA</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload hyperspectral NPZ",
        type=["npz"],
        key="prediction_upload",
        help="NPZ must contain an array named 'im'.",
    )

    if uploaded is None:

        st.info(
            "Upload an NPZ hyperspectral tile to begin disease prediction."
        )

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    Expected Input
                </div>

                <div class="feature-text">
                    • File format: NPZ<br>
                    • Required array: im<br>
                    • Spectral bands: 20<br>
                    • Supported layouts: H×W×20, 20×H×W, H×20×W
                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    else:

        try:

            file_bytes = uploaded.getvalue()

            cube = load_npz_bytes(
                file_bytes
            )

            st.success(
                f"Input loaded successfully • {uploaded.name} • {cube.shape}"
            )

            # ------------------------------------------------
            # PREVIEW
            # ------------------------------------------------

            st.divider()

            left, right = st.columns([1.2, 1])

            with left:

                st.markdown(
                    '<div class="section-label">INPUT PREVIEW</div>',
                    unsafe_allow_html=True,
                )

                preview_band = st.slider(
                    "Preview band",
                    0,
                    EXPECTED_BANDS - 1,
                    8,
                    key="prediction_band",
                )

                preview = normalize_image(
                    cube[:, :, preview_band]
                )

                fig = px.imshow(
                    preview,
                    color_continuous_scale="Viridis",
                    aspect="auto",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=430,
                    title=(
                        f"Band {preview_band + 1} • "
                        f"{WAVELENGTHS[preview_band]} nm"
                    ),
                    margin=dict(
                        l=10,
                        r=10,
                        t=50,
                        b=10,
                    ),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

            with right:

                st.markdown(
                    '<div class="section-label">INPUT VALIDATION</div>',
                    unsafe_allow_html=True,
                )

                validation_data = pd.DataFrame(
                    {
                        "Property": [
                            "Filename",
                            "Shape",
                            "Height",
                            "Width",
                            "Bands",
                            "Data Type",
                            "Minimum",
                            "Maximum",
                        ],
                        "Value": [
                            uploaded.name,
                            str(cube.shape),
                            cube.shape[0],
                            cube.shape[1],
                            cube.shape[2],
                            str(cube.dtype),
                            f"{np.min(cube):.3f}",
                            f"{np.max(cube):.3f}",
                        ],
                    }
                )

                st.dataframe(
                    validation_data,
                    hide_index=True,
                    width="stretch",
                )

            # ------------------------------------------------
            # PREDICTION
            # ------------------------------------------------

            st.divider()

            st.markdown(
                '<div class="section-label">AI INFERENCE</div>',
                unsafe_allow_html=True,
            )

            predict_button = st.button(
                "🚀  RUN DISEASE PREDICTION",
                width="stretch",
                type="primary",
            )

            if predict_button:

                with st.spinner(
                    "Running hyperspectral AI inference..."
                ):

                    result, error = call_prediction_api(
                        file_bytes,
                        uploaded.name,
                    )

                if result is not None:

                    st.session_state.prediction_result = result

                else:

                    st.session_state.prediction_result = None

                    st.error(
                        "Prediction could not be completed."
                    )

                    with st.expander(
                        "Show prediction diagnostics"
                    ):

                        st.code(
                            error
                            or
                            "No diagnostic information available."
                        )

                        st.info(
                            "Start the FastAPI backend with:\n\n"
                            "python -m uvicorn api.main:app --reload"
                        )

            # ------------------------------------------------
            # RESULT
            # ------------------------------------------------

            result = st.session_state.prediction_result

            if result is not None:

                predicted_class = result["class"]
                confidence = result["confidence"]

                if predicted_class == "Healthy":
                    result_color = ACCENT_GREEN

                elif predicted_class == "Disease Class 1":
                    result_color = ACCENT_ORANGE

                elif predicted_class == "Disease Class 2":
                    result_color = ACCENT_RED

                else:
                    result_color = ACCENT_CYAN

                confidence_text = (
                    f"{confidence:.2f}%"
                    if confidence is not None
                    else "Available"
                )

                st.markdown(
                    f"""
                    <div class="prediction-box">

                        <div class="section-label">
                            PREDICTION RESULT
                        </div>

                        <div class="prediction-class"
                             style="color:{result_color};">
                            {predicted_class}
                        </div>

                        <div class="prediction-confidence">
                            Confidence: {confidence_text}
                        </div>

                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                probabilities = result.get(
                    "probabilities"
                )

                if probabilities is not None:

                    try:

                        if isinstance(
                            probabilities,
                            dict
                        ):

                            probability_df = pd.DataFrame(
                                {
                                    "Class": [
                                        class_from_value(k)
                                        for k in probabilities.keys()
                                    ],
                                    "Probability": [
                                        float(v) * 100
                                        if float(v) <= 1
                                        else float(v)
                                        for v in probabilities.values()
                                    ],
                                }
                            )

                        elif isinstance(
                            probabilities,
                            (list, tuple)
                        ):

                            probability_df = pd.DataFrame(
                                {
                                    "Class": [
                                        CLASS_NAMES.get(
                                            i,
                                            f"Class {i}"
                                        )
                                        for i in range(
                                            len(probabilities)
                                        )
                                    ],
                                    "Probability": [
                                        float(v) * 100
                                        if float(v) <= 1
                                        else float(v)
                                        for v in probabilities
                                    ],
                                }
                            )

                        else:
                            probability_df = None

                        if (
                            probability_df is not None
                            and not probability_df.empty
                        ):

                            st.write("")

                            fig = px.bar(
                                probability_df,
                                x="Probability",
                                y="Class",
                                orientation="h",
                                text="Probability",
                            )

                            fig.update_traces(
                                texttemplate="%{text:.2f}%",
                                textposition="outside",
                            )

                            fig.update_layout(
                                template="plotly_dark",
                                height=300,
                                xaxis_title="Probability (%)",
                                yaxis_title="",
                                margin=dict(
                                    l=10,
                                    r=40,
                                    t=20,
                                    b=10,
                                ),
                            )

                            st.plotly_chart(
                                fig,
                                width="stretch",
                            )

                    except Exception:
                        pass

        except Exception as exc:

            st.error(
                f"Invalid hyperspectral file: {exc}"
            )


# ============================================================
# PAGE 3 — MONITORING
# ============================================================

elif st.session_state.page == "Monitoring":

    render_header(
        "Field Monitoring",
        "Geospatial disease intelligence, model performance and field-level risk overview."
    )

    render_page_navigation()

    st.write("")

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    geo_df = load_dataframe(
        GEO_CSV
    )

    matrix = load_confusion_matrix()

    accuracy, macro_f1, class_metrics = calculate_metrics(
        matrix
    )

    total_tiles, total_patches, class_counts = get_dataset_stats()

    # --------------------------------------------------------
    # KPI CARDS
    # --------------------------------------------------------

    cols = st.columns(5)

    monitoring_metrics = [
        (
            "MONITORED TILES",
            fmt_number(total_tiles),
            "Hyperspectral samples",
        ),
        (
            "LABELED PATCHES",
            fmt_number(total_patches),
            "Training / validation",
        ),
        (
            "MODEL ACCURACY",
            f"{accuracy * 100:.2f}%",
            "Tile-split validation",
        ),
        (
            "MACRO F1",
            f"{macro_f1 * 100:.2f}%",
            "Across 3 classes",
        ),
        (
            "MODEL PARAMETERS",
            fmt_number(model_parameter_count()),
            "3D CNN",
        ),
    ]

    for col, (label, value, note) in zip(
        cols,
        monitoring_metrics
    ):

        with col:

            st.markdown(
                f"""
                <div class="metric-card">
                    <div class="metric-label">
                        {label}
                    </div>
                    <div class="metric-value">
                        {value}
                    </div>
                    <div class="metric-note">
                        {note}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------
    # DISEASE DISTRIBUTION
    # --------------------------------------------------------

    st.divider()

    left, right = st.columns([1, 1.2])

    with left:

        st.markdown(
            '<div class="section-label">DISEASE DISTRIBUTION</div>',
            unsafe_allow_html=True,
        )

        distribution_df = pd.DataFrame(
            {
                "Class": list(
                    class_counts.keys()
                ),
                "Samples": list(
                    class_counts.values()
                ),
            }
        )

        fig = px.pie(
            distribution_df,
            names="Class",
            values="Samples",
            hole=0.58,
            color="Class",
            color_discrete_map=CLASS_COLORS,
        )

        fig.update_layout(
            template="plotly_dark",
            height=400,
            margin=dict(
                l=10,
                r=10,
                t=20,
                b=10,
            ),
            legend=dict(
                orientation="h",
                y=-0.05,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:

        st.markdown(
            '<div class="section-label">CLASS PERFORMANCE</div>',
            unsafe_allow_html=True,
        )

        if class_metrics:

            performance_df = pd.DataFrame(
                class_metrics
            )

            display_df = performance_df.copy()

            for column in [
                "Precision",
                "Recall",
                "F1 Score",
            ]:

                display_df[column] = (
                    display_df[column] * 100
                ).round(2)

            st.dataframe(
                display_df,
                hide_index=True,
                width="stretch",
            )

            fig = go.Figure()

            for metric in [
                "Precision",
                "Recall",
                "F1 Score",
            ]:

                fig.add_trace(
                    go.Bar(
                        name=metric,
                        x=display_df["Class"],
                        y=display_df[metric],
                    )
                )

            fig.update_layout(
                template="plotly_dark",
                barmode="group",
                height=330,
                yaxis_title="Score (%)",
                margin=dict(
                    l=10,
                    r=10,
                    t=20,
                    b=10,
                ),
            )

            st.plotly_chart(
                fig,
                width="stretch",
            )

    # --------------------------------------------------------
    # GEOSPATIAL MONITORING
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">GEOSPATIAL DISEASE INTELLIGENCE</div>',
        unsafe_allow_html=True,
    )

    if geo_df.empty:

        st.info(
            "No geospatial prediction file was found."
        )

        st.caption(
            f"Expected file: {GEO_CSV}"
        )

    else:

        lat_col = find_column(
            geo_df,
            [
                "latitude",
                "lat",
                "y",
            ],
        )

        lon_col = find_column(
            geo_df,
            [
                "longitude",
                "lon",
                "lng",
                "x",
            ],
        )

        class_col = find_column(
            geo_df,
            [
                "class",
                "class_id",
                "prediction",
                "predicted_class",
                "label",
                "disease_class",
            ],
        )

        if (
            lat_col is None
            or lon_col is None
        ):

            st.warning(
                "Latitude/longitude columns were not detected in geospatial_predictions.csv."
            )

            st.dataframe(
                geo_df.head(100),
                width="stretch",
            )

        else:

            map_df = geo_df.copy()

            map_df["Latitude"] = pd.to_numeric(
                map_df[lat_col],
                errors="coerce",
            )

            map_df["Longitude"] = pd.to_numeric(
                map_df[lon_col],
                errors="coerce",
            )

            if class_col:

                map_df["Disease Class"] = (
                    map_df[class_col]
                    .apply(class_from_value)
                )

            else:

                map_df["Disease Class"] = (
                    "Unknown"
                )

            map_df = map_df.dropna(
                subset=[
                    "Latitude",
                    "Longitude",
                ]
            )

            # ----------------------------------------------
            # CLASS FILTER
            # ----------------------------------------------

            available_classes = [
                value
                for value in [
                    "Healthy",
                    "Disease Class 1",
                    "Disease Class 2",
                ]
                if value in set(
                    map_df["Disease Class"]
                )
            ]

            selected_classes = st.multiselect(
                "Display classes",
                available_classes,
                default=available_classes,
            )

            filtered_map = map_df[
                map_df["Disease Class"].isin(
                    selected_classes
                )
            ]

            # ----------------------------------------------
            # PLOTLY GEO MAP
            # ----------------------------------------------

            if filtered_map.empty:

                st.warning(
                    "No points match the selected classes."
                )

            else:

                fig = px.scatter_geo(
                    filtered_map,
                    lat="Latitude",
                    lon="Longitude",
                    color="Disease Class",
                    color_discrete_map=CLASS_COLORS,
                    hover_name="Disease Class",
                    hover_data={
                        "Latitude": ":.6f",
                        "Longitude": ":.6f",
                    },
                    projection="natural earth",
                )

                fig.update_traces(
                    marker=dict(
                        size=9,
                        opacity=0.85,
                        line=dict(
                            width=1
                        ),
                    )
                )

                fig.update_geos(
                    showland=True,
                    landcolor="#111827",
                    showocean=True,
                    oceancolor="#050b16",
                    showcountries=True,
                    countrycolor="#475569",
                    coastlinecolor="#64748b",
                    showlakes=True,
                    lakecolor="#0b1220",
                    bgcolor="rgba(0,0,0,0)",
                )

                fig.update_layout(
                    template="plotly_dark",
                    height=600,
                    margin=dict(
                        l=0,
                        r=0,
                        t=10,
                        b=0,
                    ),
                    legend=dict(
                        title="Disease Status",
                        orientation="h",
                        y=0.02,
                        x=0.02,
                    ),
                )

                st.plotly_chart(
                    fig,
                    width="stretch",
                )

                st.caption(
                    "Monitoring coordinates are displayed from the available geospatial prediction dataset. "
                    "If the dataset uses simulated/demo coordinates, treat the map as a visualization rather than live GPS."
                )

            # ----------------------------------------------
            # GEOSPATIAL SUMMARY
            # ----------------------------------------------

            st.write("")

            summary_cols = st.columns(3)

            for col, class_name in zip(
                summary_cols,
                [
                    "Healthy",
                    "Disease Class 1",
                    "Disease Class 2",
                ],
            ):

                count = int(
                    (
                        map_df["Disease Class"]
                        == class_name
                    ).sum()
                )

                with col:

                    color = CLASS_COLORS[
                        class_name
                    ]

                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <div class="metric-label">
                                {class_name}
                            </div>
                            <div class="metric-value"
                                 style="color:{color};">
                                {count:,}
                            </div>
                            <div class="metric-note">
                                Geospatial predictions
                            </div>
                        </div>
                        """,
                        unsafe_allow_html=True,
                    )

    # --------------------------------------------------------
    # CONFUSION MATRIX
    # --------------------------------------------------------

    st.divider()

    st.markdown(
        '<div class="section-label">MODEL DIAGNOSTICS</div>',
        unsafe_allow_html=True,
    )

    left, right = st.columns([1, 1])

    with left:

        matrix_df = pd.DataFrame(
            matrix,
            index=[
                "Healthy",
                "Disease Class 1",
                "Disease Class 2",
            ],
            columns=[
                "Healthy",
                "Disease Class 1",
                "Disease Class 2",
            ],
        )

        fig = px.imshow(
            matrix_df,
            text_auto=True,
            color_continuous_scale="Viridis",
            labels={
                "x": "Predicted",
                "y": "Actual",
                "color": "Samples",
            },
        )

        fig.update_layout(
            template="plotly_dark",
            height=430,
            margin=dict(
                l=10,
                r=10,
                t=30,
                b=10,
            ),
        )

        st.plotly_chart(
            fig,
            width="stretch",
        )

    with right:

        st.markdown(
            """
            <div class="feature-card">

                <div class="feature-title">
                    Model Intelligence
                </div>

                <div class="feature-text">

                    <b>Architecture</b><br>
                    3D Convolutional Neural Network<br><br>

                    <b>Input</b><br>
                    20-band hyperspectral patches<br><br>

                    <b>Validation Strategy</b><br>
                    Tile-level train/validation split<br><br>

                    <b>Validation Accuracy</b><br>
                    90.78%<br><br>

                    <b>Primary Objective</b><br>
                    Detect healthy potato regions and
                    distinguish two disease classes.

                </div>

            </div>
            """,
            unsafe_allow_html=True,
        )

    # --------------------------------------------------------
    # TRAINING HISTORY
    # --------------------------------------------------------

    if HISTORY_CSV.exists():

        st.divider()

        st.markdown(
            '<div class="section-label">TRAINING INTELLIGENCE</div>',
            unsafe_allow_html=True,
        )

        history_df = load_dataframe(
            HISTORY_CSV
        )

        if not history_df.empty:

            numeric_cols = history_df.select_dtypes(
                include=np.number
            ).columns.tolist()

            if numeric_cols:

                epoch_col = find_column(
                    history_df,
                    [
                        "epoch",
                        "epochs",
                    ],
                )

                if epoch_col:

                    metric_options = [
                        col
                        for col in numeric_cols
                        if col != epoch_col
                    ]

                    if metric_options:

                        selected_metric = st.selectbox(
                            "Training metric",
                            metric_options,
                        )

                        fig = px.line(
                            history_df,
                            x=epoch_col,
                            y=selected_metric,
                            markers=True,
                        )

                        fig.update_layout(
                            template="plotly_dark",
                            height=400,
                            xaxis_title="Epoch",
                            yaxis_title=selected_metric,
                            margin=dict(
                                l=10,
                                r=10,
                                t=20,
                                b=10,
                            ),
                        )

                        st.plotly_chart(
                            fig,
                            width="stretch",
                        )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.markdown(
    """
    <div style="
        text-align:center;
        padding:15px;
        color:#475569;
        font-size:12px;
    ">
        TerraSpectra AI • Hyperspectral Potato Disease Detection
        • 3D CNN • Spectral Intelligence • Geospatial Monitoring
    </div>
    """,
    unsafe_allow_html=True,
)