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

TRAIN_CSV = "outputs/train_patches.csv"
VAL_CSV = "outputs/val_patches.csv"

MODEL_PATH = (
    "outputs/models/"
    "terraspectra_3dcnn_focal_best.pt"
)

BATCH_SIZE = 16
EPOCHS = 10
LEARNING_RATE = 0.001

NUM_CLASSES = 3

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# FOCAL LOSS
# ============================================================

class FocalLoss(torch.nn.Module):

    def __init__(
        self,
        alpha=None,
        gamma=2.0
    ):
        super().__init__()

        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):

        # Standard cross entropy per sample
        ce_loss = torch.nn.functional.cross_entropy(
            inputs,
            targets,
            weight=self.alpha,
            reduction="none"
        )

        # Probability of the correct class
        pt = torch.exp(-ce_loss)

        # Focal loss
        focal_loss = (
            (1 - pt) ** self.gamma
        ) * ce_loss

        return focal_loss.mean()


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("TerraSpectra - 3D CNN with Focal Loss")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    train_dataset = TerraSpectraDataset(
        csv_file=TRAIN_CSV,
        hsi_root="data/raw/hyperspectral/0",
        augment=False
    )

    val_dataset = TerraSpectraDataset(
        csv_file=VAL_CSV,
        hsi_root="data/raw/hyperspectral/0",
        augment=False
    )

    print(
        "Training samples  :",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(val_dataset)
    )

    # --------------------------------------------------------
    # DataLoader
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
    # Class distribution
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

    # --------------------------------------------------------
    # Calculate class weights
    # --------------------------------------------------------

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
    print("Focal Loss class weights:")

    for i, weight in enumerate(class_weights):

        print(
            f"Class {i}: {weight.item():.4f}"
        )

    # --------------------------------------------------------
    # Model
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
    # Focal Loss
    # --------------------------------------------------------

    criterion = FocalLoss(
        alpha=class_weights,
        gamma=2.0
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
    # Best model tracking
    # --------------------------------------------------------

    best_val_loss = float("inf")

    os.makedirs(
        os.path.dirname(MODEL_PATH),
        exist_ok=True
    )

    # ========================================================
    # TRAINING LOOP
    # ========================================================

    for epoch in range(EPOCHS):

        model.train()

        train_losses = []
        train_predictions = []
        train_labels = []

        # ----------------------------------------------------
        # Training
        # ----------------------------------------------------

        for batch_x, batch_y in train_loader:

            batch_x = batch_x.unsqueeze(1)

            batch_x = batch_x.to(DEVICE)
            batch_y = batch_y.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_x)

            loss = criterion(
                outputs,
                batch_y
            )

            loss.backward()

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

        train_loss = np.mean(
            train_losses
        )

        train_accuracy = accuracy_score(
            train_labels,
            train_predictions
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

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

        val_loss = np.mean(
            val_losses
        )

        val_accuracy = accuracy_score(
            val_labels,
            val_predictions
        )

        scheduler.step(
            val_loss
        )

        current_lr = optimizer.param_groups[0]["lr"]

        # ----------------------------------------------------
        # Print
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
        # Save best checkpoint
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
                "  -> Best model saved:",
                MODEL_PATH
            )

    print()
    print("=" * 70)
    print("Focal Loss training completed!")
    print("=" * 70)

    print()
    print(
        "Best validation loss:",
        best_val_loss
    )

    print()
    print(
        "Model:",
        MODEL_PATH
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()