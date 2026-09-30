import os
import pandas as pd

from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_CSV = "outputs/hyperspectral_patches.csv"

TRAIN_CSV = "outputs/train_tilesplit_patches.csv"
VAL_CSV = "outputs/val_tilesplit_patches.csv"

TEST_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# LOAD DATA
# ============================================================

df = pd.read_csv(INPUT_CSV)

print("=" * 70)
print("TerraSpectra - Tile-Level Train/Validation Split")
print("=" * 70)

print()
print("Total patches:", len(df))
print("Total unique tiles:", df["tile_id"].nunique())


# ============================================================
# GET UNIQUE TILES
# ============================================================

tiles = df["tile_id"].unique()

print()
print("Unique tiles:", len(tiles))


# ============================================================
# SPLIT TILES
# ============================================================

train_tiles, val_tiles = train_test_split(
    tiles,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)


# ============================================================
# CREATE PATCH DATASETS
# ============================================================

train_df = df[
    df["tile_id"].isin(train_tiles)
].copy()

val_df = df[
    df["tile_id"].isin(val_tiles)
].copy()


# ============================================================
# VERIFY NO TILE OVERLAP
# ============================================================

overlap = set(train_df["tile_id"]) & set(
    val_df["tile_id"]
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(
    TRAIN_CSV,
    index=False
)

val_df.to_csv(
    VAL_CSV,
    index=False
)


# ============================================================
# RESULTS
# ============================================================

print()
print("TRAINING")
print("-" * 70)
print("Tiles  :", train_df["tile_id"].nunique())
print("Patches:", len(train_df))

print()
print("VALIDATION")
print("-" * 70)
print("Tiles  :", val_df["tile_id"].nunique())
print("Patches:", len(val_df))

print()
print("Overlapping tiles:", len(overlap))

print()
print("Training class distribution:")
print(train_df["class_id"].value_counts().sort_index())

print()
print("Validation class distribution:")
print(val_df["class_id"].value_counts().sort_index())

print()
print("Saved files:")
print(TRAIN_CSV)
print(VAL_CSV)

print()
print("=" * 70)
print("Tile-level split completed!")
print("=" * 70)