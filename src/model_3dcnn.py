import torch
import torch.nn as nn


class TerraSpectra3DCNN(nn.Module):

    def __init__(self, num_classes=3):

        super().__init__()

        # ====================================================
        # BLOCK 1
        # ====================================================

        self.block1 = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(8),
            nn.ReLU(),

            # Preserve spectral dimension
            # Reduce spatial dimensions
            nn.MaxPool3d(
                kernel_size=(1, 2, 2)
            )
        )

        # ====================================================
        # BLOCK 2
        # ====================================================

        self.block2 = nn.Sequential(
            nn.Conv3d(
                in_channels=8,
                out_channels=16,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(16),
            nn.ReLU(),

            nn.MaxPool3d(
                kernel_size=(2, 2, 2)
            )
        )

        # ====================================================
        # BLOCK 3
        # ====================================================

        self.block3 = nn.Sequential(
            nn.Conv3d(
                in_channels=16,
                out_channels=32,
                kernel_size=3,
                padding=1
            ),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )

        # ====================================================
        # GLOBAL AVERAGE POOLING
        # ====================================================

        self.global_pool = nn.AdaptiveAvgPool3d(
            output_size=1
        )

        # ====================================================
        # CLASSIFIER
        # ====================================================

        self.classifier = nn.Sequential(
            nn.Flatten(),

            nn.Linear(32, 16),
            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(16, num_classes)
        )

    def forward(self, x):

        x = self.block1(x)

        x = self.block2(x)

        x = self.block3(x)

        x = self.global_pool(x)

        x = self.classifier(x)

        return x


# ============================================================
# MODEL TEST
# ============================================================

if __name__ == "__main__":

    print("=" * 65)
    print("TERRASPECTRA 3D CNN MODEL TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Create model
    # --------------------------------------------------------

    model = TerraSpectra3DCNN(num_classes=3)

    print("\nModel created successfully.")

    # --------------------------------------------------------
    # Create dummy hyperspectral input
    # --------------------------------------------------------

    # Batch = 2
    # Channel = 1
    # Spectral bands = 20
    # Height = 32
    # Width = 32

    x = torch.randn(
        2,
        1,
        20,
        32,
        32
    )

    print(f"\nInput shape  : {x.shape}")

    # --------------------------------------------------------
    # Forward pass
    # --------------------------------------------------------

    output = model(x)

    print(f"Output shape : {output.shape}")

    print("\nExpected output:")
    print("2 samples × 3 classes")

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(f"\nTotal parameters     : {total_parameters:,}")
    print(f"Trainable parameters : {trainable_parameters:,}")

    print("\n" + "=" * 65)
    print("3D CNN MODEL TEST COMPLETED")
    print("=" * 65)