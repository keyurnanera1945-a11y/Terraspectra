from pathlib import Path
import inspect
import sys

import numpy as np
import torch
import torch.nn.functional as F

from fastapi import FastAPI, File, HTTPException, UploadFile


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_3dcnn import TerraSpectra3DCNN


MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_tilesplit_best.pt"
)


# ============================================================
# CONSTANTS
# ============================================================

EXPECTED_BANDS = 20
PATCH_SIZE = 32

CLASS_NAMES = {
    0: "Healthy / Normal",
    1: "Disease Class 1",
    2: "Disease Class 2",
}


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title="TerraSpectra AI API",
    description=(
        "FastAPI inference service for hyperspectral "
        "potato disease classification using a "
        "tile-level 3D-CNN."
    ),
    version="1.0.0",
)


# ============================================================
# MODEL CREATION
# ============================================================

def create_model():

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

        model = TerraSpectra3DCNN(
            **kwargs
        )

    except TypeError:

        model = TerraSpectra3DCNN()

    return model


# ============================================================
# LOAD TRAINED MODEL
# ============================================================

def load_model():

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    model = create_model()

    try:

        checkpoint = torch.load(
            MODEL_PATH,
            map_location="cpu",
            weights_only=False
        )

    except TypeError:

        checkpoint = torch.load(
            MODEL_PATH,
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


MODEL = load_model()


# ============================================================
# PREPARE HYPERSPECTRAL DATA
# ============================================================

def prepare_tensor(hsi):

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
            "Hyperspectral data contains "
            "NaN or infinite values."
        )

    # Raw uint16 → 0-1
    hsi = hsi / 65535.0

    hsi = np.clip(
        hsi,
        0.0,
        1.0
    )

    # HWC → CHW
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


# ============================================================
# PREDICTION
# ============================================================

def predict(hsi):

    tensor = prepare_tensor(
        hsi
    )

    with torch.no_grad():

        logits = MODEL(
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
        probability_values
    )


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.get("/")
def root():

    return {
        "project": "TerraSpectra AI",
        "status": "API is running",
        "model": "Tile-Level 3D-CNN",
        "input_bands": EXPECTED_BANDS,
        "input_size": "20 x 32 x 32",
        "classes": CLASS_NAMES,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():

    return {
        "status": "healthy",
        "model_loaded": MODEL is not None,
        "model_file": MODEL_PATH.name,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.post("/predict")
async def predict_hyperspectral(
    file: UploadFile = File(...)
):

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No file name provided."
        )

    if not file.filename.lower().endswith(
        ".npz"
    ):

        raise HTTPException(
            status_code=400,
            detail="Only .npz hyperspectral files are supported."
        )

    try:

        file_bytes = await file.read()

        if not file_bytes:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty."
            )

        from io import BytesIO

        data = np.load(
            BytesIO(file_bytes)
        )

        if "im" not in data:

            raise HTTPException(
                status_code=400,
                detail=(
                    "The NPZ file must contain "
                    "an 'im' array."
                )
            )

        hsi = data["im"]

        predicted_class, confidence, probabilities = (
            predict(hsi)
        )

        probability_dict = {
            CLASS_NAMES[index]:
                round(
                    float(
                        probabilities[index]
                    ),
                    6
                )
            for index in range(3)
        }

        return {
            "success": True,
            "filename": file.filename,
            "input_shape": list(hsi.shape),
            "model_input": [
                20,
                32,
                32
            ],
            "predicted_class": predicted_class,
            "class_name": CLASS_NAMES[
                predicted_class
            ],
            "confidence": round(
                confidence,
                6
            ),
            "probabilities": probability_dict,
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Inference failed: {exc}"
        )