from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_CSV = PROJECT_ROOT / "outputs" / "hyperspectral_patches.csv"

TRAIN_CSV = PROJECT_ROOT / "outputs" / "train_patches.csv"
VAL_CSV = PROJECT_ROOT / "outputs" / "val_patches.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("TERRASPECTRA TRAIN / VALIDATION SPLIT")
print("=" * 65)

df = pd.read_csv(INPUT_CSV)

print(f"\nTotal samples: {len(df)}")

print("\nOriginal class distribution:")
print(df["class_id"].value_counts().sort_index())


# ============================================================
# STRATIFIED SPLIT
# ============================================================

train_df, val_df = train_test_split(
    df,
    test_size=0.20,
    random_state=42,
    stratify=df["class_id"]
)


# ============================================================
# SAVE
# ============================================================

train_df.to_csv(TRAIN_CSV, index=False)
val_df.to_csv(VAL_CSV, index=False)


# ============================================================
# RESULTS
# ============================================================

print("\nSplit completed.")

print(f"Training samples   : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")

print("\nTraining class distribution:")
print(train_df["class_id"].value_counts().sort_index())

print("\nValidation class distribution:")
print(val_df["class_id"].value_counts().sort_index())

print("\nFiles created:")
print(TRAIN_CSV)
print(VAL_CSV)

print("\n" + "=" * 65)
print("TRAIN / VALIDATION SPLIT COMPLETED")
print("=" * 65)