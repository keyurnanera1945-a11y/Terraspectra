import os
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, WeightedRandomSampler

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_CSV = "outputs/train_tilesplit_patches.csv"
VAL_CSV = "outputs/val_tilesplit_patches.csv"

HSI_ROOT = "data/raw/hyperspectral/0"

MODEL_OUTPUT = "outputs/models/terraspectra_3dcnn_tilesplit_best.pt"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 3

NUM_WORKERS = 0

USE_BALANCED_SAMPLING = True
USE_AUGMENTATION = True

RANDOM_SEED = 42


# ============================================================
# REPRODUCIBILITY
# ============================================================

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(RANDOM_SEED)


# ============================================================
# DEVICE
# ============================================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("=" * 65)
print("TERRASPECTRA TILE-LEVEL 3D-CNN TRAINING")
print("=" * 65)

print()
print("Device:", device)


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAIN_CSV):
    raise FileNotFoundError(
        f"Training CSV not found:\n{TRAIN_CSV}"
    )

if not os.path.exists(VAL_CSV):
    raise FileNotFoundError(
        f"Validation CSV not found:\n{VAL_CSV}"
    )

if not os.path.exists(HSI_ROOT):
    raise FileNotFoundError(
        f"Hyperspectral directory not found:\n{HSI_ROOT}"
    )


# ============================================================
# LOAD DATASETS
# ============================================================

print()
print("-" * 65)
print("Loading datasets...")
print("-" * 65)

train_dataset = TerraSpectraDataset(
    csv_file=TRAIN_CSV,
    hsi_root=HSI_ROOT,
    augment=USE_AUGMENTATION
)

val_dataset = TerraSpectraDataset(
    csv_file=VAL_CSV,
    hsi_root=HSI_ROOT,
    augment=False
)

print()
print("Training samples  :", len(train_dataset))
print("Validation samples:", len(val_dataset))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_labels = train_dataset.df["class_id"].astype(int).values
val_labels = val_dataset.df["class_id"].astype(int).values

train_counts = np.bincount(
    train_labels,
    minlength=NUM_CLASSES
)

val_counts = np.bincount(
    val_labels,
    minlength=NUM_CLASSES
)

print()
print("Training class distribution:")
for class_id in range(NUM_CLASSES):
    print(
        f"Class {class_id}: {train_counts[class_id]}"
    )

print()
print("Validation class distribution:")
for class_id in range(NUM_CLASSES):
    print(
        f"Class {class_id}: {val_counts[class_id]}"
    )


# ============================================================
# BALANCED SAMPLING
# ============================================================

train_sampler = None
shuffle_train = True

if USE_BALANCED_SAMPLING:

    print()
    print("-" * 65)
    print("Using balanced sampling")
    print("-" * 65)

    class_weights = np.zeros(NUM_CLASSES)

    for class_id in range(NUM_CLASSES):

        if train_counts[class_id] > 0:

            class_weights[class_id] = (
                1.0 / train_counts[class_id]
            )

    sample_weights = np.array(
        [
            class_weights[label]
            for label in train_labels
        ],
        dtype=np.float64
    )

    train_sampler = WeightedRandomSampler(
        weights=torch.as_tensor(
            sample_weights,
            dtype=torch.double
        ),
        num_samples=len(sample_weights),
        replacement=True
    )

    shuffle_train = False

    print()
    print("Class sampling weights:")

    for class_id in range(NUM_CLASSES):
        print(
            f"Class {class_id}: "
            f"{class_weights[class_id]:.8f}"
        )


# ============================================================
# DATA LOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=shuffle_train,
    num_workers=NUM_WORKERS,
    pin_memory=torch.cuda.is_available()
)

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
print("Creating 3D-CNN model...")
print("-" * 65)

model = TerraSpectra3DCNN(
    num_classes=NUM_CLASSES
)

model = model.to(device)


# ============================================================
# MODEL PARAMETERS
# ============================================================

total_parameters = sum(
    p.numel()
    for p in model.parameters()
)

trainable_parameters = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

print()
print("Total parameters     :", f"{total_parameters:,}")
print("Trainable parameters :", f"{trainable_parameters:,}")


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING FUNCTION
# ============================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device
):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for patches, labels in loader:

        patches = patches.to(
            device,
            non_blocking=True
        )

        labels = labels.to(
            device,
            non_blocking=True
        )

        # Dataset output:
        # [B, 20, 32, 32]
        #
        # 3D CNN input:
        # [B, 1, 20, 32, 32]

        patches = patches.unsqueeze(1)

        optimizer.zero_grad()

        outputs = model(patches)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() * labels.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        correct += (
            predictions == labels
        ).sum().item()

        total += labels.size(0)

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return epoch_loss, epoch_accuracy


# ============================================================
# VALIDATION FUNCTION
# ============================================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    all_predictions = []
    all_labels = []

    with torch.no_grad():

        for patches, labels in loader:

            patches = patches.to(
                device,
                non_blocking=True
            )

            labels = labels.to(
                device,
                non_blocking=True
            )

            patches = patches.unsqueeze(1)

            outputs = model(patches)

            loss = criterion(
                outputs,
                labels
            )

            running_loss += (
                loss.item() * labels.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            correct += (
                predictions == labels
            ).sum().item()

            total += labels.size(0)

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

    epoch_loss = running_loss / total
    epoch_accuracy = correct / total

    return (
        epoch_loss,
        epoch_accuracy,
        np.array(all_predictions),
        np.array(all_labels)
    )


# ============================================================
# TRAINING LOOP
# ============================================================

best_val_accuracy = 0.0

history = []


print()
print("=" * 65)
print("STARTING TILE-LEVEL TRAINING")
print("=" * 65)

for epoch in range(1, EPOCHS + 1):

    train_loss, train_accuracy = train_one_epoch(
        model,
        train_loader,
        criterion,
        optimizer,
        device
    )

    (
        val_loss,
        val_accuracy,
        predictions,
        labels
    ) = validate(
        model,
        val_loader,
        criterion,
        device
    )

    history.append(
        {
            "epoch": epoch,
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "val_loss": val_loss,
            "val_accuracy": val_accuracy
        }
    )

    print()
    print(
        f"Epoch [{epoch:02d}/{EPOCHS}]"
    )

    print(
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_accuracy * 100:.2f}%"
    )

    print(
        f"Val Loss:   {val_loss:.4f} | "
        f"Val Acc:   {val_accuracy * 100:.2f}%"
    )

    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        os.makedirs(
            os.path.dirname(MODEL_OUTPUT),
            exist_ok=True
        )

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_accuracy": val_accuracy,
                "num_classes": NUM_CLASSES
            },
            MODEL_OUTPUT
        )

        print()
        print(
            "✓ Best model saved!"
        )

        print(
            f"  Validation accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"  Path: {MODEL_OUTPUT}"
        )


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_df = pd.DataFrame(history)

history_path = (
    "outputs/"
    "tilesplit_training_history.csv"
)

history_df.to_csv(
    history_path,
    index=False
)


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 65)
print("TILE-LEVEL TRAINING COMPLETED")
print("=" * 65)

print()
print(
    f"Best validation accuracy: "
    f"{best_val_accuracy * 100:.2f}%"
)

print()
print(
    "Best model:"
)

print(
    MODEL_OUTPUT
)

print()
print(
    "Training history:"
)

print(
    history_path
)

print()
print("=" * 65)