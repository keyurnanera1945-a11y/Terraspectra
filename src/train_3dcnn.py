from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

TRAIN_CSV = PROJECT_ROOT / "outputs" / "train_patches.csv"
VAL_CSV = PROJECT_ROOT / "outputs" / "val_patches.csv"

MODEL_DIR = PROJECT_ROOT / "outputs" / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

BEST_MODEL_PATH = MODEL_DIR / "terraspectra_3dcnn_best.pt"


# ============================================================
# CONFIGURATION
# ============================================================

BATCH_SIZE = 16
NUM_EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 3

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

torch.manual_seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("TERRASPECTRA 3D CNN TRAINING")
print("=" * 70)

print(f"\nDevice       : {DEVICE}")
print(f"Batch size   : {BATCH_SIZE}")
print(f"Epochs       : {NUM_EPOCHS}")
print(f"Learning rate: {LEARNING_RATE}")


# ============================================================
# DATASETS
# ============================================================

print("\nLoading training dataset...")

train_dataset = TerraSpectraDataset(
    csv_file=TRAIN_CSV
)

print("\nLoading validation dataset...")

val_dataset = TerraSpectraDataset(
    csv_file=VAL_CSV
)


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


print("\nDataLoaders created.")

print(f"Training batches   : {len(train_loader)}")
print(f"Validation batches : {len(val_loader)}")


# ============================================================
# CLASS WEIGHTS
# ============================================================

train_labels = train_dataset.df["class_id"].values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print("\nTraining class counts:")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id}: {count}")


# ------------------------------------------------------------
# Balanced class weights
# ------------------------------------------------------------

total_samples = len(train_labels)

class_weights = total_samples / (
    NUM_CLASSES * class_counts
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
).to(DEVICE)


print("\nClass weights:")

for class_id, weight in enumerate(class_weights):
    print(
        f"Class {class_id}: {weight.item():.4f}"
    )


# ============================================================
# MODEL
# ============================================================

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


print("\nModel loaded.")

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

print(f"Trainable parameters: {total_parameters:,}")


# ============================================================
# LOSS FUNCTION
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING VARIABLES
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# TRAINING LOOP
# ============================================================

for epoch in range(NUM_EPOCHS):

    print("\n" + "=" * 70)
    print(
        f"Epoch {epoch + 1}/{NUM_EPOCHS}"
    )
    print("=" * 70)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_index, (inputs, labels) in enumerate(train_loader):

        # Add channel dimension
        # [B, 20, 32, 32]
        # →
        # [B, 1, 20, 32, 32]

        inputs = inputs.unsqueeze(1)

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)

        # Calculate loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update weights
        optimizer.step()

        # Statistics
        running_loss += (
            loss.item() * inputs.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

    train_loss = running_loss / total

    train_accuracy = (
        correct / total
    ) * 100


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.unsqueeze(1)

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() * inputs.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = (
        val_loss_total / val_total
    )

    val_accuracy = (
        val_correct / val_total
    ) * 100


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"\nTrain Loss     : {train_loss:.4f}"
    )

    print(
        f"Train Accuracy : {train_accuracy:.2f}%"
    )

    print(
        f"Val Loss       : {val_loss:.4f}"
    )

    print(
        f"Val Accuracy   : {val_accuracy:.2f}%"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "num_classes": NUM_CLASSES,
                "best_val_accuracy": best_val_accuracy,
                "epoch": epoch + 1
            },
            BEST_MODEL_PATH
        )

        print(
            f"\n✓ Best model saved!"
        )

        print(
            f"Path: {BEST_MODEL_PATH}"
        )


# ============================================================
# TRAINING COMPLETE
# ============================================================

print("\n" + "=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print(
    f"\nBest validation accuracy: "
    f"{best_val_accuracy:.2f}%"
)

print(
    f"\nBest model:"
)

print(BEST_MODEL_PATH)

print("\n" + "=" * 70)