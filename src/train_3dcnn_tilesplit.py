import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F


class TerraSpectraDataset(Dataset):

    def __init__(
        self,
        csv_file,
        hsi_root="data/raw/hyperspectral/0",
        augment=False
    ):
        self.df = pd.read_csv(csv_file)
        self.hsi_root = hsi_root
        self.augment = augment

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ==================================================
        # LOAD HYPERSPECTRAL FILE
        # ==================================================

        hsi_file = str(row["hsi_file"])

        # CSV may contain:
        # data\raw\hyperspectral\0\0001.npz
        #
        # or:
        # 0001.npz

        if hsi_file.startswith("data"):
            hsi_path = hsi_file
        else:
            hsi_path = os.path.join(
                self.hsi_root,
                hsi_file
            )

        # Normalize Windows path
        hsi_path = os.path.normpath(hsi_path)

        # Check file exists
        if not os.path.exists(hsi_path):
            raise FileNotFoundError(
                f"Hyperspectral file not found:\n{hsi_path}"
            )

        # ==================================================
        # LOAD NPZ
        # ==================================================

        data = np.load(hsi_path)

        if "im" not in data:
            raise KeyError(
                f"'im' key not found in hyperspectral file:\n{hsi_path}"
            )

        image = data["im"]

        # ==================================================
        # GET BOUNDING BOX
        # ==================================================

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

        # ==================================================
        # VALIDATE BOUNDING BOX
        # ==================================================

        height, width, bands = image.shape

        x1 = max(0, min(x1, width))
        x2 = max(0, min(x2, width))

        y1 = max(0, min(y1, height))
        y2 = max(0, min(y2, height))

        if x2 <= x1 or y2 <= y1:
            raise ValueError(
                f"Invalid bounding box at index {idx}: "
                f"({x1}, {y1}, {x2}, {y2})"
            )

        # ==================================================
        # CROP HYPERSPECTRAL PATCH
        # ==================================================

        patch = image[y1:y2, x1:x2, :]

        # ==================================================
        # CONVERT TO FLOAT32
        # ==================================================

        patch = patch.astype(np.float32)

        # ==================================================
        # NORMALIZATION
        #
        # Original HSI values are uint16.
        # ==================================================

        patch = patch / 65535.0

        patch = np.clip(
            patch,
            0.0,
            1.0
        )

        # ==================================================
        # HWC -> CHW
        #
        # Original:
        # Height x Width x Bands
        #
        # Required:
        # Bands x Height x Width
        # ==================================================

        patch = torch.from_numpy(patch)

        patch = patch.permute(
            2,
            0,
            1
        )

        # ==================================================
        # RESIZE SPATIAL DIMENSIONS
        #
        # Expected output:
        # Bands x 32 x 32
        # ==================================================

        patch = patch.unsqueeze(0)

        patch = F.interpolate(
            patch,
            size=(32, 32),
            mode="bilinear",
            align_corners=False
        )

        patch = patch.squeeze(0)

        # ==================================================
        # DATA AUGMENTATION
        # ==================================================

        if self.augment:

            # ------------------------------------------------
            # Random horizontal flip
            # ------------------------------------------------

            if torch.rand(1).item() < 0.5:

                patch = torch.flip(
                    patch,
                    dims=[2]
                )

            # ------------------------------------------------
            # Random vertical flip
            # ------------------------------------------------

            if torch.rand(1).item() < 0.5:

                patch = torch.flip(
                    patch,
                    dims=[1]
                )

            # ------------------------------------------------
            # Random 90-degree rotation
            # ------------------------------------------------

            k = torch.randint(
                0,
                4,
                (1,)
            ).item()

            if k > 0:

                patch = torch.rot90(
                    patch,
                    k=k,
                    dims=[1, 2]
                )

        # ==================================================
        # LABEL
        # ==================================================

        label = int(row["class_id"])

        return patch.float(), label


# ==========================================================
# TEST DATASET
# ==========================================================

if __name__ == "__main__":

    print("=" * 60)
    print("TerraSpectra Dataset Test")
    print("=" * 60)

    dataset = TerraSpectraDataset(
        csv_file="outputs/hyperspectral_patches.csv",
        hsi_root="data/raw/hyperspectral/0",
        augment=True
    )

    print()
    print("Dataset samples:", len(dataset))

    # ======================================================
    # LOAD FIRST SAMPLE
    # ======================================================

    patch, label = dataset[0]

    print()
    print("First sample information")
    print("-" * 40)

    print("Tensor shape :", patch.shape)
    print("Tensor dtype :", patch.dtype)
    print("Tensor min   :", patch.min().item())
    print("Tensor max   :", patch.max().item())
    print("Tensor mean  :", patch.mean().item())
    print("Label        :", label)

    # ======================================================
    # EXPECTED SHAPE CHECK
    # ======================================================

    assert patch.shape == torch.Size([20, 32, 32]), (
        f"Unexpected tensor shape: {patch.shape}"
    )

    assert patch.dtype == torch.float32, (
        f"Unexpected tensor dtype: {patch.dtype}"
    )

    assert 0.0 <= patch.min().item() <= 1.0, (
        "Tensor contains values below 0"
    )

    assert 0.0 <= patch.max().item() <= 1.0, (
        "Tensor contains values above 1"
    )

    print()
    print("Shape check  : PASSED")
    print("Dtype check  : PASSED")
    print("Range check  : PASSED")

    print()
    print("=" * 60)
    print("Dataset test completed successfully!")
    print("=" * 60)