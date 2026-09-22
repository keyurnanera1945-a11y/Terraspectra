from pathlib import Path
import numpy as np
from collections import Counter


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

HSI_DIR = PROJECT_ROOT / "data" / "raw" / "hyperspectral" / "0"
LABEL_DIR = PROJECT_ROOT / "data" / "raw" / "labels" / "0"


# ============================================================
# DATASET FILES
# ============================================================

hsi_files = sorted(HSI_DIR.glob("*.npz"))
label_files = sorted(LABEL_DIR.glob("*.txt"))

print("=" * 65)
print("TERRASPECTRA DATASET ANALYSIS")
print("=" * 65)

print(f"\nHyperspectral files : {len(hsi_files)}")
print(f"Label files         : {len(label_files)}")


# ============================================================
# CHECK FILE MATCHING
# ============================================================

hsi_ids = {f.stem for f in hsi_files}
label_ids = {f.stem for f in label_files}

missing_labels = hsi_ids - label_ids
missing_hsi = label_ids - hsi_ids

print("\nFile matching")
print("-" * 65)
print(f"Missing labels : {len(missing_labels)}")
print(f"Missing HSI    : {len(missing_hsi)}")


# ============================================================
# INSPECT FIRST HYPERSPECTRAL FILE
# ============================================================

sample_file = hsi_files[0]

data = np.load(sample_file)
cube = data["im"]

print("\nHyperspectral sample")
print("-" * 65)
print(f"File       : {sample_file.name}")
print(f"Shape      : {cube.shape}")
print(f"Data type  : {cube.dtype}")
print(f"Min        : {cube.min()}")
print(f"Max        : {cube.max()}")
print(f"Mean       : {cube.mean():.4f}")


# ============================================================
# COUNT YOLO CLASSES
# ============================================================

class_counter = Counter()
box_counter = 0
empty_label_files = 0

for label_file in label_files:

    lines = label_file.read_text().strip().splitlines()

    if not lines:
        empty_label_files += 1
        continue

    for line in lines:

        parts = line.split()

        if len(parts) != 5:
            print(f"WARNING: Invalid label format: {label_file.name}")
            continue

        class_id = int(parts[0])

        class_counter[class_id] += 1
        box_counter += 1


# ============================================================
# LABEL SUMMARY
# ============================================================

print("\nYOLO label summary")
print("-" * 65)

print(f"Total bounding boxes : {box_counter}")
print(f"Empty label files   : {empty_label_files}")

for class_id, count in sorted(class_counter.items()):
    print(f"Class {class_id} boxes     : {count}")


# ============================================================
# BAND INFORMATION
# ============================================================

bands = [
    420, 440, 500, 510, 531,
    550, 570, 670, 672, 680,
    690, 695, 700, 708, 710,
    770, 800, 850, 900, 970
]

print("\nSpectral bands")
print("-" * 65)

print(f"Number of bands : {len(bands)}")
print(f"Bands (nm)      : {bands}")


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 65)
print("DATASET CHECK COMPLETED")
print("=" * 65)