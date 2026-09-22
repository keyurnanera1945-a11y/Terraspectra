from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HSI_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "hyperspectral"
    / "0"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET INFORMATION
# ============================================================

SPECTRAL_BANDS = [
    420, 440, 500, 510, 531,
    550, 570, 670, 672, 680,
    690, 695, 700, 708, 710,
    770, 800, 850, 900, 970
]


# ============================================================
# FIND FILES
# ============================================================

hsi_files = sorted(
    HSI_DIR.glob("*.npz")
)


print("=" * 65)
print("TERRASPECTRA PCA ANALYSIS")
print("=" * 65)

print(
    f"\nHyperspectral files : {len(hsi_files)}"
)

print(
    f"Spectral bands     : {len(SPECTRAL_BANDS)}"
)


# ============================================================
# LOAD SAMPLE TILES
# ============================================================

# Use 100 tiles for the initial PCA analysis.
# This keeps the first experiment reasonably fast.

sample_files = hsi_files[:100]

all_pixels = []


print(
    f"\nLoading {len(sample_files)} tiles..."
)


for file in sample_files:

    data = np.load(file)

    cube = data["im"].astype(
        np.float32
    )

    # Expected shape:
    # 120 x 120 x 20

    if cube.shape != (120, 120, 20):

        print(
            f"WARNING: Unexpected shape "
            f"{cube.shape} in {file.name}"
        )

        continue

    # Convert:
    #
    # 120 x 120 x 20
    #
    # into:
    #
    # 14400 x 20

    pixels = cube.reshape(
        -1,
        cube.shape[-1]
    )

    all_pixels.append(pixels)


# ============================================================
# COMBINE PIXELS
# ============================================================

X = np.vstack(all_pixels)


print(
    f"\nPixel matrix shape : {X.shape}"
)


# ============================================================
# SAMPLE PIXELS
# ============================================================

MAX_PIXELS = 100000


if len(X) > MAX_PIXELS:

    print(
        f"Sampling {MAX_PIXELS} pixels..."
    )

    rng = np.random.default_rng(
        42
    )

    indices = rng.choice(
        len(X),
        size=MAX_PIXELS,
        replace=False
    )

    X = X[indices]


print(
    f"Pixels used for PCA : {len(X)}"
)


# ============================================================
# STANDARDIZATION
# ============================================================

print(
    "\nStandardizing spectral bands..."
)


scaler = StandardScaler()

X_scaled = scaler.fit_transform(
    X
)


# ============================================================
# PCA
# ============================================================

print(
    "Running PCA..."
)


pca = PCA(
    n_components=20
)


X_pca = pca.fit_transform(
    X_scaled
)


# ============================================================
# EXPLAINED VARIANCE
# ============================================================

explained_variance = (
    pca.explained_variance_ratio_
)

cumulative_variance = np.cumsum(
    explained_variance
)


print("\nPCA RESULTS")
print("-" * 65)


for i, variance in enumerate(
    explained_variance
):

    print(
        f"PC{i + 1:02d}: "
        f"{variance * 100:.2f}% "
        f"| Cumulative: "
        f"{cumulative_variance[i] * 100:.2f}%"
    )


# ============================================================
# COMPONENTS REQUIRED FOR 95%
# ============================================================

components_95 = (
    np.argmax(
        cumulative_variance >= 0.95
    )
    + 1
)


print(
    "\nComponents required for 95% variance:",
    components_95
)


# ============================================================
# SAVE PCA RESULTS
# ============================================================

np.save(
    OUTPUT_DIR / "pca_components.npy",
    pca.components_
)

np.save(
    OUTPUT_DIR / "pca_explained_variance.npy",
    explained_variance
)

np.save(
    OUTPUT_DIR / "pca_cumulative_variance.npy",
    cumulative_variance
)

np.save(
    OUTPUT_DIR / "pca_mean.npy",
    pca.mean_
)


# ============================================================
# SAVE BAND INFORMATION
# ============================================================

with open(
    OUTPUT_DIR / "pca_band_information.txt",
    "w"
) as file:

    file.write(
        "TerraSpectra PCA Band Information\n"
    )

    file.write(
        "=================================\n\n"
    )

    file.write(
        "Original spectral bands (nm):\n"
    )

    file.write(
        str(SPECTRAL_BANDS)
    )

    file.write("\n\n")

    file.write(
        f"Components for 95% variance: "
        f"{components_95}\n"
    )


# ============================================================
# CUMULATIVE VARIANCE PLOT
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.plot(
    range(1, 21),
    cumulative_variance * 100,
    marker="o"
)


plt.xlabel(
    "Number of Principal Components"
)

plt.ylabel(
    "Cumulative Explained Variance (%)"
)

plt.title(
    "TerraSpectra - PCA Explained Variance"
)

plt.grid(True)

plt.tight_layout()


variance_path = (
    OUTPUT_DIR
    / "pca_variance.png"
)


plt.savefig(
    variance_path,
    dpi=200
)

plt.close()


# ============================================================
# PCA 2D VISUALIZATION
# ============================================================

plt.figure(
    figsize=(10, 6)
)


plt.scatter(
    X_pca[:, 0],
    X_pca[:, 1],
    s=2,
    alpha=0.4
)


plt.xlabel(
    "Principal Component 1"
)

plt.ylabel(
    "Principal Component 2"
)

plt.title(
    "TerraSpectra - PCA Projection"
)

plt.grid(True)

plt.tight_layout()


projection_path = (
    OUTPUT_DIR
    / "pca_projection.png"
)


plt.savefig(
    projection_path,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETED
# ============================================================

print("\nSaved files:")
print(
    f"- {OUTPUT_DIR / 'pca_components.npy'}"
)

print(
    f"- {OUTPUT_DIR / 'pca_explained_variance.npy'}"
)

print(
    f"- {OUTPUT_DIR / 'pca_cumulative_variance.npy'}"
)

print(
    f"- {OUTPUT_DIR / 'pca_mean.npy'}"
)

print(
    f"- {OUTPUT_DIR / 'pca_band_information.txt'}"
)

print(
    f"- {variance_path}"
)

print(
    f"- {projection_path}"
)


print("\n" + "=" * 65)
print("PCA ANALYSIS COMPLETED")
print("=" * 65)