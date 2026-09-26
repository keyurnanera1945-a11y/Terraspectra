import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score
)

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

VAL_CSV = "outputs/val_patches.csv"

MODEL_PATH = (
    "outputs/models/"
    "terraspectra_3dcnn_focal_best.pt"
)

OUTPUT_DIR = "outputs/evaluation_focal"

REPORT_PATH = (
    f"{OUTPUT_DIR}/classification_report.txt"
)

CONFUSION_MATRIX_PATH = (
    f"{OUTPUT_DIR}/confusion_matrix.png"
)

PREDICTIONS_PATH = (
    f"{OUTPUT_DIR}/predictions.csv"
)

BATCH_SIZE = 16
NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TerraSpectra - Focal Loss 3D CNN Evaluation")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Output directory
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Validation dataset
    # --------------------------------------------------------

    val_dataset = TerraSpectraDataset(
        csv_file=VAL_CSV,
        hsi_root="data/raw/hyperspectral/0",
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

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TerraSpectra3DCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    # --------------------------------------------------------
    # Load Focal Loss checkpoint
    #
    # weights_only=False is required for checkpoints
    # created with the current PyTorch version.
    # --------------------------------------------------------

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=False
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model.eval()

    print(
        "Loaded checkpoint from epoch:",
        checkpoint["epoch"]
    )

    print(
        "Checkpoint validation loss:",
        f"{checkpoint['val_loss']:.4f}"
    )

    print(
        "Checkpoint validation accuracy:",
        f"{checkpoint['val_accuracy'] * 100:.2f}%"
    )

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for batch_x, batch_y in val_loader:

            # [B, 20, 32, 32]
            # ->
            # [B, 1, 20, 32, 32]

            batch_x = batch_x.unsqueeze(1)

            batch_x = batch_x.to(DEVICE)

            outputs = model(batch_x)

            probabilities = torch.softmax(
                outputs,
                dim=1
            )

            predictions = torch.argmax(
                probabilities,
                dim=1
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch_y.numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    # --------------------------------------------------------
    # Convert to NumPy
    # --------------------------------------------------------

    y_true = np.array(
        all_labels
    )

    y_pred = np.array(
        all_predictions
    )

    probabilities = np.array(
        all_probabilities
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    macro_f1 = f1_score(
        y_true,
        y_pred,
        average="macro",
        zero_division=0
    )

    weighted_f1 = f1_score(
        y_true,
        y_pred,
        average="weighted",
        zero_division=0
    )

    report = classification_report(
        y_true,
        y_pred,
        digits=4,
        zero_division=0
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EVALUATION RESULTS")
    print("=" * 70)

    print()
    print(
        f"Validation Accuracy : "
        f"{accuracy * 100:.2f}%"
    )

    print(
        f"Macro F1            : "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1         : "
        f"{weighted_f1:.4f}"
    )

    print()
    print("Classification Report")
    print("-" * 70)

    print(report)

    print("Confusion Matrix")
    print("-" * 70)

    print(cm)

    # --------------------------------------------------------
    # Save classification report
    # --------------------------------------------------------

    with open(
        REPORT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        f.write(
            "TerraSpectra - Focal Loss 3D CNN Evaluation\n"
        )

        f.write(
            "=" * 60 + "\n\n"
        )

        f.write(
            f"Validation Accuracy: "
            f"{accuracy * 100:.2f}%\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n\n"
        )

        f.write(
            "Classification Report\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(report)

        f.write(
            "\n\nConfusion Matrix\n"
        )

        f.write(
            "-" * 60 + "\n"
        )

        f.write(
            np.array2string(cm)
        )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    predictions_df = pd.DataFrame(
        {
            "true_label": y_true,
            "predicted_label": y_pred,
            "probability_class_0": probabilities[:, 0],
            "probability_class_1": probabilities[:, 1],
            "probability_class_2": probabilities[:, 2]
        }
    )

    predictions_df.to_csv(
        PREDICTIONS_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Confusion matrix image
    # --------------------------------------------------------

    import matplotlib.pyplot as plt

    plt.figure(
        figsize=(6, 5)
    )

    plt.imshow(
        cm,
        interpolation="nearest"
    )

    plt.title(
        "Focal Loss 3D CNN Confusion Matrix"
    )

    plt.colorbar()

    plt.xlabel(
        "Predicted Label"
    )

    plt.ylabel(
        "True Label"
    )

    plt.xticks(
        range(NUM_CLASSES)
    )

    plt.yticks(
        range(NUM_CLASSES)
    )

    for i in range(NUM_CLASSES):

        for j in range(NUM_CLASSES):

            plt.text(
                j,
                i,
                str(cm[i, j]),
                ha="center",
                va="center"
            )

    plt.tight_layout()

    plt.savefig(
        CONFUSION_MATRIX_PATH,
        dpi=200
    )

    plt.close()

    # --------------------------------------------------------
    # Final output
    # --------------------------------------------------------

    print()
    print("Saved files:")
    print(REPORT_PATH)
    print(CONFUSION_MATRIX_PATH)
    print(PREDICTIONS_PATH)

    print()
    print("=" * 70)
    print("Evaluation completed successfully!")
    print("=" * 70)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()