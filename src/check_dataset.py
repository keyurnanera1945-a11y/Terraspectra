import numpy as np
from pathlib import Path

# Project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Hyperspectral folder
HSI_DIR = PROJECT_ROOT / "data" / "raw" / "hyperspectral" / "0"

# Get NPZ files
files = sorted(HSI_DIR.glob("*.npz"))

print("=" * 60)
print("TERRASPECTRA DATASET CHECK")
print("=" * 60)

print(f"Dataset folder: {HSI_DIR}")
print(f"Number of NPZ files: {len(files)}")

if len(files) == 0:
    print("ERROR: No NPZ files found.")
    exit()

# First file
sample_file = files[0]

print("\nFirst file:")
print(sample_file.name)

# Load NPZ
data = np.load(sample_file)

print("\nKeys inside NPZ:")
print(data.files)

for key in data.files:
    arr = data[key]

    print("\n" + "-" * 40)
    print(f"Key       : {key}")
    print(f"Shape     : {arr.shape}")
    print(f"Data type : {arr.dtype}")
    print(f"Min       : {arr.min()}")
    print(f"Max       : {arr.max()}")
    print(f"Mean      : {arr.mean()}")

print("\nDataset check completed.")