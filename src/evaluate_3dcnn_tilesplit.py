import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

VAL_CSV = "outputs/val_tilesplit_patches.csv"

HSI_ROOT = "data/raw/hyperspectral/0"

MODEL_PATH = (
    "outputs/models/"
    "terraspectra_3dcnn_tilesplit_best.pt"
)

OUTPUT_DIR = "outputs"

CONFUSION_MATRIX_PATH = (
    f"{OUTPUT_DIR}/tilesplit_confusion_matrix.csv"
)

CLASSIFICATION_REPORT_PATH = (
    f"{OUTPUT_DIR}/tilesplit_classification_report.csv"
)

PREDICTIONS_PATH = (
    f"{OUTPUT_DIR}/tilesplit_predictions.csv"
)

BATCH_SIZE = 16

NUM_CLASSES = 3

NUM_WORKERS = 0


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


print("=" * 65)
print("TERRASPECTRA TILE-LEVEL 3D-CNN EVALUATION")
print("=" * 65)

print()
print("Device:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(VAL_CSV):

    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )


if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model checkpoint not found:\n{MODEL_PATH}"
    )


if not os.path.exists(HSI_ROOT):

    raise FileNotFoundError(
        f"Hyperspectral directory not found:\n{HSI_ROOT}"
    )


# ============================================================
# LOAD VALIDATION DATASET
# ============================================================

print()
print("-" * 65)
print("Loading validation dataset...")
print("-" * 65)

val_dataset = TerraSpectraDataset(
    csv_file=VAL_CSV,
    hsi_root=HSI_ROOT,
    augment=False
)

print()
print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# VALIDATION DATALOADER
# ============================================================

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)


# ============================================================
# CREATE MODEL
# ============================================================

print()
print("-" * 65)
print("Loading 3D-CNN model...")
print("-" * 65)

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(device)

model.eval()


print()
print("Model loaded successfully.")

if "epoch" in checkpoint:

    print(
        "Best epoch:",
        checkpoint["epoch"]
    )

if "val_accuracy" in checkpoint:

    print(
        "Saved validation accuracy:",
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )


# ============================================================
# EVALUATION
# ============================================================

all_predictions = []
all_labels = []

print()
print("-" * 65)
print("Running evaluation...")
print("-" * 65)

with torch.no_grad():

    for patches, labels in val_loader:

        patches = patches.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Dataset shape:
        # [B, 20, 32, 32]
        #
        # 3D-CNN shape:
        # [B, 1, 20, 32, 32]

        patches = patches.unsqueeze(1)

        outputs = model(patches)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_labels.extend(
            labels.cpu().numpy()
        )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_predictions = np.array(
    all_predictions
)

all_labels = np.array(
    all_labels
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


print()
print("=" * 65)
print("EVALUATION RESULTS")
print("=" * 65)

print()
print(
    f"Overall Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES))
)


print()
print("Confusion Matrix:")
print()

print(cm)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=list(range(NUM_CLASSES)),
    target_names=[
        "Class 0",
        "Class 1",
        "Class 2"
    ],
    output_dict=True,
    zero_division=0
)


report_df = pd.DataFrame(
    report
).transpose()


print()
print("Classification Report:")
print()

print(
    classification_report(
        all_labels,
        all_predictions,
        labels=list(range(NUM_CLASSES)),
        target_names=[
            "Class 0",
            "Class 1",
            "Class 2"
        ],
        zero_division=0
    )
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

cm_df = pd.DataFrame(
    cm,
    index=[
        "Actual_Class_0",
        "Actual_Class_1",
        "Actual_Class_2"
    ],
    columns=[
        "Predicted_Class_0",
        "Predicted_Class_1",
        "Predicted_Class_2"
    ]
)

cm_df.to_csv(
    CONFUSION_MATRIX_PATH
)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_df.to_csv(
    CLASSIFICATION_REPORT_PATH
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame(
    {
        "actual_class": all_labels,
        "predicted_class": all_predictions
    }
)

predictions_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)


# ============================================================
# FINAL SUMMARY
# ============================================================

print("=" * 65)
print("EVALUATION COMPLETED SUCCESSFULLY")
print("=" * 65)

print()
print(
    f"Overall Accuracy : {accuracy * 100:.2f}%"
)

print()
print("Saved files:")

print(
    f"1. {CONFUSION_MATRIX_PATH}"
)

print(
    f"2. {CLASSIFICATION_REPORT_PATH}"
)

print(
    f"3. {PREDICTIONS_PATH}"
)

print()
print("=" * 65)