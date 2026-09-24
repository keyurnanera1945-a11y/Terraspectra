from pathlib import Path

import numpy as np
import pandas as pd


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

LABEL_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "labels"
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

IMAGE_WIDTH = 120
IMAGE_HEIGHT = 120
NUM_BANDS = 20


# ============================================================
# CONVERT YOLO BOX TO HSI PIXEL COORDINATES
# ============================================================

def yolo_to_pixels(
    class_id,
    center_x,
    center_y,
    width,
    height
):
    """
    Convert normalized YOLO coordinates
    into HSI pixel coordinates.

    YOLO format:

    class_id center_x center_y width height

    Coordinates are normalized between 0 and 1.
    """

    center_x_pixel = center_x * IMAGE_WIDTH
    center_y_pixel = center_y * IMAGE_HEIGHT

    width_pixel = width * IMAGE_WIDTH
    height_pixel = height * IMAGE_HEIGHT

    x1 = int(
        center_x_pixel - width_pixel / 2
    )

    y1 = int(
        center_y_pixel - height_pixel / 2
    )

    x2 = int(
        center_x_pixel + width_pixel / 2
    )

    y2 = int(
        center_y_pixel + height_pixel / 2
    )

    # Keep coordinates inside the HSI tile

    x1 = max(0, min(x1, IMAGE_WIDTH - 1))
    y1 = max(0, min(y1, IMAGE_HEIGHT - 1))

    x2 = max(1, min(x2, IMAGE_WIDTH))
    y2 = max(1, min(y2, IMAGE_HEIGHT))

    return x1, y1, x2, y2


# ============================================================
# START
# ============================================================

print("=" * 70)
print("TERRASPECTRA HYPERSPECTRAL PATCH EXTRACTION")
print("=" * 70)


hsi_files = sorted(
    HSI_DIR.glob("*.npz")
)

print(
    f"\nHyperspectral tiles : {len(hsi_files)}"
)


records = []

total_boxes = 0
valid_boxes = 0
invalid_boxes = 0


# ============================================================
# PROCESS EACH TILE
# ============================================================

for index, hsi_file in enumerate(hsi_files):

    tile_id = hsi_file.stem

    label_file = LABEL_DIR / f"{tile_id}.txt"

    if not label_file.exists():

        print(
            f"WARNING: Missing label for {tile_id}"
        )

        continue


    # --------------------------------------------------------
    # Verify HSI shape
    # --------------------------------------------------------

    data = np.load(hsi_file)

    cube = data["im"]

    if cube.shape != (
        IMAGE_HEIGHT,
        IMAGE_WIDTH,
        NUM_BANDS
    ):

        print(
            f"WARNING: Unexpected shape "
            f"{cube.shape} in {tile_id}"
        )

        continue


    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    lines = (
        label_file
        .read_text()
        .strip()
        .splitlines()
    )


    for box_index, line in enumerate(lines):

        parts = line.split()

        if len(parts) != 5:

            invalid_boxes += 1

            continue


        try:

            class_id = int(parts[0])

            center_x = float(parts[1])
            center_y = float(parts[2])

            width = float(parts[3])
            height = float(parts[4])

        except ValueError:

            invalid_boxes += 1

            continue


        total_boxes += 1


        # ----------------------------------------------------
        # Convert YOLO → HSI coordinates
        # ----------------------------------------------------

        x1, y1, x2, y2 = yolo_to_pixels(
            class_id,
            center_x,
            center_y,
            width,
            height
        )


        # ----------------------------------------------------
        # Validate bounding box
        # ----------------------------------------------------

        patch_width = x2 - x1
        patch_height = y2 - y1


        if (
            patch_width <= 0
            or patch_height <= 0
        ):

            invalid_boxes += 1

            continue


        valid_boxes += 1


        # ----------------------------------------------------
        # Save metadata
        # ----------------------------------------------------

        records.append(
            {
                "tile_id": tile_id,

                "hsi_file":
                    str(
                        hsi_file.relative_to(
                            PROJECT_ROOT
                        )
                    ),

                "box_id": box_index,

                "class_id": class_id,

                "x1": x1,
                "y1": y1,
                "x2": x2,
                "y2": y2,

                "patch_width":
                    patch_width,

                "patch_height":
                    patch_height,

                "center_x":
                    center_x,

                "center_y":
                    center_y,

                "bbox_width":
                    width,

                "bbox_height":
                    height
            }
        )


    # --------------------------------------------------------
    # Progress
    # --------------------------------------------------------

    if (
        index + 1
    ) % 100 == 0:

        print(
            f"Processed "
            f"{index + 1}/{len(hsi_files)} tiles..."
        )


# ============================================================
# CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)


# ============================================================
# SAVE METADATA
# ============================================================

output_file = (
    OUTPUT_DIR
    / "hyperspectral_patches.csv"
)


df.to_csv(
    output_file,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("PATCH EXTRACTION SUMMARY")
print("=" * 70)

print(
    f"\nTotal bounding boxes : {total_boxes}"
)

print(
    f"Valid patches        : {valid_boxes}"
)

print(
    f"Invalid boxes        : {invalid_boxes}"
)


print(
    f"\nMetadata file:"
)

print(
    output_file
)


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

print("\nClass distribution")
print("-" * 70)

if not df.empty:

    class_counts = (
        df["class_id"]
        .value_counts()
        .sort_index()
    )

    for class_id, count in class_counts.items():

        print(
            f"Class {class_id}: {count}"
        )


# ============================================================
# PATCH SIZE STATISTICS
# ============================================================

if not df.empty:

    print("\nPatch size statistics")
    print("-" * 70)

    print(
        f"Minimum width  : "
        f"{df['patch_width'].min()}"
    )

    print(
        f"Maximum width  : "
        f"{df['patch_width'].max()}"
    )

    print(
        f"Average width  : "
        f"{df['patch_width'].mean():.2f}"
    )

    print(
        f"Minimum height : "
        f"{df['patch_height'].min()}"
    )

    print(
        f"Maximum height : "
        f"{df['patch_height'].max()}"
    )

    print(
        f"Average height : "
        f"{df['patch_height'].mean():.2f}"
    )


print("\n" + "=" * 70)
print("PATCH METADATA EXTRACTION COMPLETED")
print("=" * 70)