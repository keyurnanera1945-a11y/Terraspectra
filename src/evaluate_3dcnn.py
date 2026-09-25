from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay,
    precision_recall_fscore_support
)

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

VAL_CSV = PROJECT_ROOT / "outputs" / "val_patches.csv"

MODEL_PATH = (
    PROJECT_ROOT
    / "outputs"
    / "models"
    / "terraspectra_3dcnn_best.pt"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs" / "evaluation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

CONFUSION_MATRIX_PATH = (
    OUTPUT_DIR / "confusion_matrix.png"
)

REPORT_PATH = (
    OUTPUT_DIR / "classification_report.txt"
)

PREDICTIONS_PATH = (
    OUTPUT_DIR / "predictions.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
NUM_CLASSES = 3

CLASS_NAMES = [
    "Class 0",
    "Class 1",
    "Class 2"
]


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TERRASPECTRA 3D CNN MODEL EVALUATION")
print("=" * 70)

print(f"\nDevice     : {DEVICE}")
print(f"Model      : {MODEL_PATH}")
print(f"Validation : {VAL_CSV}")


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading validation dataset...")

val_dataset = TerraSpectraDataset(
    csv_file=VAL_CSV
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(f"Validation samples: {len(val_dataset)}")
print(f"Validation batches : {len(val_loader)}")


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading best model...")

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

model = model.to(DEVICE)
model.eval()

print("Best model loaded successfully.")

if "epoch" in checkpoint:
    print(
        f"Best checkpoint epoch : "
        f"{checkpoint['epoch']}"
    )

if "best_val_accuracy" in checkpoint:
    print(
        f"Saved validation accuracy : "
        f"{checkpoint['best_val_accuracy']:.2f}%"
    )


# ============================================================
# PREDICTION
# ============================================================

print("\nRunning predictions...")

all_labels = []
all_predictions = []
all_probabilities = []

with torch.no_grad():

    for batch_index, (inputs, labels) in enumerate(
        val_loader
    ):

        # Dataset output:
        # [B, 20, 32, 32]
        #
        # 3D CNN input:
        # [B, 1, 20, 32, 32]

        inputs = inputs.unsqueeze(1)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(inputs)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy()
        )

        if (
            batch_index + 1
        ) % 25 == 0:

            print(
                f"Processed "
                f"{batch_index + 1}/"
                f"{len(val_loader)} batches"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

all_labels = np.array(
    all_labels
)

all_predictions = np.array(
    all_predictions
)

all_probabilities = np.array(
    all_probabilities
)


# ============================================================
# OVERALL ACCURACY
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

print("\n" + "=" * 70)
print("OVERALL RESULTS")
print("=" * 70)

print(
    f"\nValidation Accuracy: "
    f"{accuracy * 100:.2f}%"
)


# ============================================================
# CLASSIFICATION REPORT
# ============================================================

report = classification_report(
    all_labels,
    all_predictions,
    labels=[0, 1, 2],
    target_names=CLASS_NAMES,
    digits=4,
    zero_division=0
)

print("\nClassification Report:")
print("-" * 70)
print(report)


# ============================================================
# PRECISION / RECALL / F1
# ============================================================

precision, recall, f1, support = (
    precision_recall_fscore_support(
        all_labels,
        all_predictions,
        labels=[0, 1, 2],
        zero_division=0
    )
)

print("Per-class metrics:")
print("-" * 70)

for i in range(NUM_CLASSES):

    print(
        f"{CLASS_NAMES[i]:<10} | "
        f"Precision: {precision[i]:.4f} | "
        f"Recall: {recall[i]:.4f} | "
        f"F1: {f1[i]:.4f} | "
        f"Support: {support[i]}"
    )


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions,
    labels=[0, 1, 2]
)

print("\nConfusion Matrix:")
print("-" * 70)
print(cm)


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

fig, ax = plt.subplots(
    figsize=(7, 6)
)

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=CLASS_NAMES
)

display.plot(
    ax=ax,
    values_format="d",
    cmap="Blues"
)

ax.set_title(
    "TerraSpectra 3D CNN Confusion Matrix"
)

plt.tight_layout()

plt.savefig(
    CONFUSION_MATRIX_PATH,
    dpi=200
)

plt.close()

print(
    f"\nConfusion matrix saved:"
)

print(CONFUSION_MATRIX_PATH)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

with open(
    REPORT_PATH,
    "w",
    encoding="utf-8"
) as file:

    file.write(
        "TERRASPECTRA 3D CNN MODEL EVALUATION\n"
    )

    file.write(
        "=" * 70 + "\n\n"
    )

    file.write(
        f"Validation samples: "
        f"{len(all_labels)}\n"
    )

    file.write(
        f"Accuracy: "
        f"{accuracy * 100:.2f}%\n\n"
    )

    file.write(
        "Classification Report\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(report)

    file.write(
        "\n\nConfusion Matrix\n"
    )

    file.write(
        "-" * 70 + "\n"
    )

    file.write(
        str(cm)
    )


print(
    f"\nClassification report saved:"
)

print(REPORT_PATH)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = pd.DataFrame({
    "actual_class": all_labels,
    "predicted_class": all_predictions,
    "class_0_probability": all_probabilities[:, 0],
    "class_1_probability": all_probabilities[:, 1],
    "class_2_probability": all_probabilities[:, 2]
})

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print(
    f"\nPredictions saved:"
)

print(PREDICTIONS_PATH)


# ============================================================
# COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("MODEL EVALUATION COMPLETED")
print("=" * 70)