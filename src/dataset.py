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
        # or only:
        # 0001.npz

        if hsi_file.startswith("data"):
            hsi_path = hsi_file
        else:
            hsi_path = os.path.join(
                self.hsi_root,
                hsi_file
            )

        # Convert Windows path to normalized path
        hsi_path = os.path.normpath(hsi_path)

        # Check file exists
        if not os.path.exists(hsi_path):
            raise FileNotFoundError(
                f"Hyperspectral file not found:\n{hsi_path}"
            )

        # Load NPZ
        data = np.load(hsi_path)

        image = data["im"]

        # ==================================================
        # GET BOUNDING BOX
        # ==================================================

        x1 = int(row["x1"])
        y1 = int(row["y1"])
        x2 = int(row["x2"])
        y2 = int(row["y2"])

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
        # 20 spectral bands
        # 32 x 32 spatial size
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

            # ----------------------------------------------
            # Random horizontal flip
            # ----------------------------------------------

            if torch.rand(1).item() < 0.5:

                patch = torch.flip(
                    patch,
                    dims=[2]
                )

            # ----------------------------------------------
            # Random vertical flip
            # ----------------------------------------------

            if torch.rand(1).item() < 0.5:

                patch = torch.flip(
                    patch,
                    dims=[1]
                )

            # ----------------------------------------------
            # Random 90-degree rotation
            # ----------------------------------------------

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

    # Load first sample

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

    print()
    print("=" * 60)
    print("Dataset test completed successfully!")
    print("=" * 60)