import os
import sys
import torch
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

sys.path.append(
    os.path.dirname(os.path.abspath(__file__))
)

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

VAL_CSV = "outputs/val_patches.csv"

MODEL_PATH = (
    "outputs/models/"
    "terraspectra_3dcnn_balanced_best.pt"
)

OUTPUT_DIR = "outputs/evaluation_balanced"

BATCH_SIZE = 16

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


print("=" * 70)
print("TerraSpectra - Balanced Sampling 3D CNN Evaluation")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

val_dataset = TerraSpectraDataset(
    VAL_CSV,
    augment=False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print(
    "Validation samples:",
    len(val_dataset)
)


# ============================================================
# MODEL
# ============================================================

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE,
    weights_only=False
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print(
    "Loaded checkpoint from epoch:",
    checkpoint["epoch"]
)

print(
    "Checkpoint validation loss:",
    f'{checkpoint["val_loss"]:.4f}'
)

print(
    "Checkpoint validation accuracy:",
    f'{checkpoint["val_accuracy"] * 100:.2f}%'
)


# ============================================================
# EVALUATION
# ============================================================

model.eval()

all_labels = []
all_predictions = []

with torch.no_grad():

    for inputs, labels in val_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Add channel dimension
        inputs = inputs.unsqueeze(1)

        outputs = model(inputs)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

macro_f1 = f1_score(
    all_labels,
    all_predictions,
    average="macro",
    zero_division=0
)

weighted_f1 = f1_score(
    all_labels,
    all_predictions,
    average="weighted",
    zero_division=0
)

report = classification_report(
    all_labels,
    all_predictions,
    digits=4,
    zero_division=0
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("EVALUATION RESULTS")
print("=" * 70)

print()
print(
    f"Validation Accuracy : {accuracy * 100:.2f}%"
)

print(
    f"Macro F1            : {macro_f1:.4f}"
)

print(
    f"Weighted F1         : {weighted_f1:.4f}"
)

print()
print("Classification Report")
print("-" * 70)
print(report)

print("Confusion Matrix")
print("-" * 70)
print(cm)


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(
        "TerraSpectra - Balanced Sampling 3D CNN\n"
    )

    f.write(
        f"Validation Accuracy: "
        f"{accuracy * 100:.4f}%\n"
    )

    f.write(
        f"Macro F1: {macro_f1:.4f}\n"
    )

    f.write(
        f"Weighted F1: {weighted_f1:.4f}\n\n"
    )

    f.write(
        "Classification Report\n"
    )

    f.write(report)

    f.write(
        "\nConfusion Matrix\n"
    )

    f.write(
        str(cm)
    )


# ============================================================
# SAVE CONFUSION MATRIX
# ============================================================

plt.figure(
    figsize=(7, 7)
)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=[
        "Class 0",
        "Class 1",
        "Class 2"
    ]
)

disp.plot(
    values_format="d"
)

plt.title(
    "TerraSpectra - Balanced Sampling 3D CNN"
)

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


# ============================================================
# SAVE PREDICTIONS
# ============================================================

predictions_df = pd.DataFrame({
    "true_label": all_labels,
    "predicted_label": all_predictions
})

predictions_path = os.path.join(
    OUTPUT_DIR,
    "predictions.csv"
)

predictions_df.to_csv(
    predictions_path,
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()
print("Saved files:")
print(report_path)
print(cm_path)
print(predictions_path)

print()
print("=" * 70)
print("Evaluation completed successfully!")
print("=" * 70)