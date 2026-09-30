import os
import sys
import torch
import numpy as np
import pandas as pd

from torch.utils.data import DataLoader, WeightedRandomSampler

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = "outputs/train_patches.csv"
VAL_CSV = "outputs/val_patches.csv"

MODEL_DIR = "outputs/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "terraspectra_3dcnn_balanced_best.pt"
)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(MODEL_DIR, exist_ok=True)


print("=" * 70)
print("TerraSpectra - Balanced Sampling 3D CNN Training")
print("=" * 70)

print()
print("Device:", DEVICE)


# ============================================================
# LOAD DATASETS
# ============================================================

train_dataset = TerraSpectraDataset(
    TRAIN_CSV,
    augment=False
)

val_dataset = TerraSpectraDataset(
    VAL_CSV,
    augment=False
)

print("Training samples  :", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# GET TRAINING LABELS
# ============================================================

_df = pd.read_csv(TRAIN_CSV, encoding="utf-8-sig")
_df.columns = _df.columns.str.strip().str.lower()
print("Train CSV columns:", _df.columns.tolist())
train_labels = _df["class_id"].astype(int).values

class_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

print()
print("Training class distribution:")

for class_id, count in enumerate(class_counts):
    print(f"Class {class_id}: {count}")


# ============================================================
# CALCULATE SAMPLE WEIGHTS
# ============================================================

class_weights = 1.0 / class_counts

sample_weights = np.array([
    class_weights[label]
    for label in train_labels
])

sample_weights = torch.DoubleTensor(sample_weights)


# ============================================================
# BALANCED SAMPLER
# ============================================================

sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True
)


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=sampler,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)

print()
print("Model parameters:", sum(
    p.numel() for p in model.parameters()
))


# ============================================================
# LOSS
# ============================================================

criterion = torch.nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="min",
    factor=0.5,
    patience=2
)


# ============================================================
# TRAINING
# ============================================================

best_val_loss = float("inf")

print()
print("=" * 70)
print("STARTING TRAINING")
print("=" * 70)

for epoch in range(EPOCHS):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in train_loader:

        inputs = inputs.to(DEVICE)
        labels = labels.to(DEVICE)

        # Add channel dimension
        inputs = inputs.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(inputs)

        loss = criterion(outputs, labels)

        loss.backward()

        optimizer.step()

        running_loss += loss.item() * labels.size(0)

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    train_loss = running_loss / total
    train_acc = correct / total


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for inputs, labels in val_loader:

            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            inputs = inputs.unsqueeze(1)

            outputs = model(inputs)

            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_correct += (
                predictions == labels
            ).sum().item()

            val_total += labels.size(0)

    val_loss = val_running_loss / val_total
    val_acc = val_correct / val_total


    # --------------------------------------------------------
    # SCHEDULER
    # --------------------------------------------------------

    scheduler.step(val_loss)

    current_lr = optimizer.param_groups[0]["lr"]


    # --------------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------------

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc * 100:.2f}% | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc * 100:.2f}% | "
        f"LR: {current_lr:.6f}"
    )


    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        torch.save(
            {
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_loss": val_loss,
                "val_accuracy": val_acc
            },
            MODEL_PATH
        )

        print("  -> Best model saved")


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 70)
print("TRAINING COMPLETED")
print("=" * 70)

print("Best validation loss:", best_val_loss)
print("Model saved to:", MODEL_PATH)