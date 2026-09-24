from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

CSV_FILE = (
    PROJECT_ROOT
    / "outputs"
    / "hyperspectral_patches.csv"
)


# ============================================================
# DATASET
# ============================================================

class TerraSpectraDataset(Dataset):

    def __init__(
        self,
        csv_file=CSV_FILE,
        target_size=32,
    ):
        self.project_root = PROJECT_ROOT

        self.df = pd.read_csv(csv_file)

        self.target_size = target_size

        print(
            f"Dataset samples: {len(self.df)}"
        )

    # --------------------------------------------------------
    # Number of samples
    # --------------------------------------------------------

    def __len__(self):
        return len(self.df)

    # --------------------------------------------------------
    # Load one hyperspectral sample
    # --------------------------------------------------------

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # ----------------------------------------------------
        # Load NPZ file
        # ----------------------------------------------------

        hsi_path = (
            self.project_root
            / row["hsi_file"]
        )

        data = np.load(hsi_path)

        cube = data["im"].astype(
            np.float32
        )

        # Original shape:
        #
        # H × W × Bands
        #
        # 120 × 120 × 20

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        # ----------------------------------------------------
        # Crop hyperspectral patch
        # ----------------------------------------------------

        patch = cube[
            y1:y2,
            x1:x2,
            :
        ]

        # ----------------------------------------------------
        # Safety check
        # ----------------------------------------------------

        if (
            patch.size == 0
            or patch.shape[0] == 0
            or patch.shape[1] == 0
        ):

            raise RuntimeError(
                f"Empty patch at index {index}"
            )

        # ----------------------------------------------------
        # Normalize spectral values
        # ----------------------------------------------------

        # Dataset values are uint16.
        #
        # Convert approximately to [0, 1]
        # using the observed 16-bit range.

        patch = patch / 65535.0

        patch = np.clip(
            patch,
            0.0,
            1.0
        )

        # ----------------------------------------------------
        # Convert:
        #
        # H × W × Bands
        #
        # to:
        #
        # Bands × H × W
        # ----------------------------------------------------

        patch = np.transpose(
            patch,
            (2, 0, 1)
        )

        # ----------------------------------------------------
        # Convert to PyTorch tensor
        # ----------------------------------------------------

        patch = torch.tensor(
            patch,
            dtype=torch.float32
        )

        # ----------------------------------------------------
        # Resize spatial dimensions
        #
        # 20 × variable_H × variable_W
        #
        # →
        #
        # 20 × 32 × 32
        # ----------------------------------------------------

        patch = patch.unsqueeze(0)

        patch = torch.nn.functional.interpolate(
            patch,
            size=(
                self.target_size,
                self.target_size
            ),
            mode="bilinear",
            align_corners=False
        )

        patch = patch.squeeze(0)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = torch.tensor(
            int(row["class_id"]),
            dtype=torch.long
        )

        return patch, label


# ============================================================
# TEST DATASET
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TERRASPECTRA PYTORCH DATASET TEST")
    print("=" * 65)

    dataset = TerraSpectraDataset()

    print(
        f"\nTotal samples: {len(dataset)}"
    )

    # Load first sample

    sample, label = dataset[0]

    print("\nFirst sample")
    print("-" * 65)

    print(
        f"Tensor shape : {sample.shape}"
    )

    print(
        f"Tensor dtype : {sample.dtype}"
    )

    print(
        f"Tensor min   : {sample.min().item():.6f}"
    )

    print(
        f"Tensor max   : {sample.max().item():.6f}"
    )

    print(
        f"Tensor mean  : {sample.mean().item():.6f}"
    )

    print(
        f"Label        : {label.item()}"
    )

    print("\nExpected tensor format:")
    print("20 spectral bands × 32 × 32")

    print("\n" + "=" * 65)
    print("DATASET TEST COMPLETED")
    print("=" * 65)