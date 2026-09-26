import os
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score

from dataset import TerraSpectraDataset
from model_3dcnn import TerraSpectra3DCNN


# ============================================================
# CONFIGURATION
# ============================================================

PATCH_CSV = "outputs/hyperspectral_patches.csv"

TRAIN_CSV = "outputs/train_patches.csv"
VAL_CSV = "outputs/val_patches.csv"

MODEL_PATH = "outputs/models/terraspectra_3dcnn_augmented_best.pt"

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TerraSpectra - Augmented 3D CNN Training")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load training dataset
    # --------------------------------------------------------

    train_dataset = TerraSpectraDataset(
        csv_file=TRAIN_CSV,
        hsi_root="data/raw/hyperspectral/0",
        augment=True
    )

    # --------------------------------------------------------
    # Load validation dataset
    #
    # IMPORTANT:
    # No augmentation on validation data.
    # --------------------------------------------------------

    val_dataset = TerraSpectraDataset(
        csv_file=VAL_CSV,
        hsi_root="data/raw/hyperspectral/0",
        augment=False
    )

    print("Training samples  :", len(train_dataset))
    print("Validation samples:", len(val_dataset))

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

    train_df = pd.read_csv(TRAIN_CSV)

    class_counts = (
        train_df["class_id"]
        .value_counts()
        .sort_index()
    )

    print()
    print("Training class distribution:")
    print(class_counts)

    total_samples = len(train_df)

    class_weights = []

    for class_id in range(NUM_CLASSES):

        count = class_counts.get(
            class_id,
            1
        )

        weight = total_samples / (
            NUM_CLASSES * count
        )

        class_weights.append(weight)

    class_weights = torch.tensor(
        class_weights,
        dtype=torch.float32
    ).to(DEVICE)

    print()
    print("Class weights:")

    for i, weight in enumerate(class_weights):

        print(
            f"Class {i}: {weight.item():.4f}"
        )

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TerraSpectra3DCNN(
        num_classes=NUM_CLASSES
    )

    model = model.to(DEVICE)

    print()
    print(
        "Trainable parameters:",
        sum(
            p.numel()
            for p in model.parameters()
            if p.requires_grad
        )
    )

    # --------------------------------------------------------
    # Loss function
    # --------------------------------------------------------

    criterion = torch.nn.CrossEntropyLoss(
        weight=class_weights
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # Learning-rate scheduler
    # --------------------------------------------------------

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=0.5,
        patience=2
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    best_val_loss = float("inf")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    for epoch in range(EPOCHS):

        # ====================================================
        # TRAIN
        # ====================================================

        model.train()

        train_losses = []
        train_predictions = []
        train_labels = []

        for batch_x, batch_y in train_loader:

            # ------------------------------------------------
            # Add channel dimension for 3D CNN
            #
            # [B, 20, 32, 32]
            # ->
            # [B, 1, 20, 32, 32]
            # ------------------------------------------------

            batch_x = batch_x.unsqueeze(1)

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(batch_x)

            # Loss
            loss = criterion(
                outputs,
                batch_y
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            train_losses.append(
                loss.item()
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            train_predictions.extend(
                predictions.detach().cpu().numpy()
            )

            train_labels.extend(
                batch_y.detach().cpu().numpy()
            )

        # ----------------------------------------------------
        # Training metrics
        # ----------------------------------------------------

        train_loss = np.mean(train_losses)

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions
        )

        # ====================================================
        # VALIDATION
        # ====================================================

        model.eval()

        val_losses = []
        val_predictions = []
        val_labels = []

        with torch.no_grad():

            for batch_x, batch_y in val_loader:

                batch_x = batch_x.unsqueeze(1)

                batch_x = batch_x.to(DEVICE)
                batch_y = batch_y.to(DEVICE)

                outputs = model(batch_x)

                loss = criterion(
                    outputs,
                    batch_y
                )

                val_losses.append(
                    loss.item()
                )

                predictions = torch.argmax(
                    outputs,
                    dim=1
                )

                val_predictions.extend(
                    predictions.cpu().numpy()
                )

                val_labels.extend(
                    batch_y.cpu().numpy()
                )

        # ----------------------------------------------------
        # Validation metrics
        # ----------------------------------------------------

        val_loss = np.mean(val_losses)

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        # ----------------------------------------------------
        # Update scheduler
        # ----------------------------------------------------

        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Print results
        # ----------------------------------------------------

        print(
            f"Epoch {epoch + 1:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_accuracy * 100:.2f}% | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_accuracy * 100:.2f}% | "
            f"LR: {current_lr:.6f}"
        )

        # ----------------------------------------------------
        # Save best model
        # ----------------------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                    "val_accuracy": val_accuracy
                },
                MODEL_PATH
            )

            print(
                f"  -> Best model saved: {MODEL_PATH}"
            )

    print()
    print("=" * 70)
    print("Training completed!")
    print("=" * 70)

    print()
    print("Best validation loss:", best_val_loss)

    print()
    print("Best model:")
    print(MODEL_PATH)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()