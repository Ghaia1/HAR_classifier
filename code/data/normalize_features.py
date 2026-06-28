"""Apply z-score normalization fitted on training set."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Normalize features using z-score (fitted on train).")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed/features"),
        help="Features directory with class folders",
    )
    parser.add_argument(
        "--scalers-dir",
        type=Path,
        default=Path("data/processed/features/scalers"),
        help="Where to save fitted scalers",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.scalers_dir.mkdir(parents=True, exist_ok=True)

    print("Z-SCORE NORMALIZATION (ONE scaler for all classes)")
    print("=" * 70)
    print()

    # Find all class folders
    class_folders = sorted([d for d in args.features_dir.iterdir() if d.is_dir() and d.name != "scalers"])
    if not class_folders:
        raise RuntimeError(f"No class folders found in: {args.features_dir}")

    # Step 1: Collect all training data across all classes
    print("STEP 1: Collecting training data from all classes...")
    print("-" * 70)

    all_train_data = []
    metadata_cols = ["window_id", "activity_id", "activity_name", "recording_folder"]
    feature_cols = None

    class_info = {}

    for class_folder in class_folders:
        class_name = class_folder.name
        activity_name = class_name.split("_")[1]

        train_csv = class_folder / f"{activity_name}_train.csv"
        if not train_csv.exists():
            print(f"⚠ {activity_name}: train CSV not found, skipping")
            continue

        df_train = pd.read_csv(train_csv)
        
        # Determine feature columns (same for all classes)
        if feature_cols is None:
            feature_cols = [c for c in df_train.columns if c not in metadata_cols]

        all_train_data.append(df_train)
        class_info[activity_name] = {
            "class_folder": class_folder,
            "train_csv": train_csv,
            "val_csv": class_folder / f"{activity_name}_val.csv",
            "test_csv": class_folder / f"{activity_name}_test.csv",
        }

        print(f"  {activity_name}: {len(df_train)} train samples")

    # Combine all training data
    df_all_train = pd.concat(all_train_data, ignore_index=True)
    print(f"\nTotal train samples: {len(df_all_train)}")

    # Step 2: Fit ONE scaler on all training data
    print()
    print("STEP 2: Fitting scaler on all training data...")
    print("-" * 70)

    X_all_train = df_all_train[feature_cols].values
    scaler = StandardScaler()
    scaler.fit(X_all_train)

    print(f"Features: {len(feature_cols)}")
    print(f"Scaler fitted on {len(X_all_train)} samples")

    # Step 3: Transform all train/val/test sets
    print()
    print("STEP 3: Applying scaler to all datasets...")
    print("-" * 70)

    for activity_name, info in class_info.items():
        print(f"\n{activity_name.upper()}")

        # Train
        df_train = pd.read_csv(info["train_csv"])
        X_train = df_train[feature_cols].values
        X_train_scaled = scaler.transform(X_train)
        df_train_normalized = pd.DataFrame(X_train_scaled, columns=feature_cols)
        for col in metadata_cols:
            if col in df_train.columns:
                df_train_normalized[col] = df_train[col].values
        
        train_normalized_csv = info["class_folder"] / f"{activity_name}_train_normalized.csv"
        df_train_normalized.to_csv(train_normalized_csv, index=False)
        print(f"  ✓ {train_normalized_csv.name}")

        # Val
        if info["val_csv"].exists():
            df_val = pd.read_csv(info["val_csv"])
            X_val = df_val[feature_cols].values
            X_val_scaled = scaler.transform(X_val)
            df_val_normalized = pd.DataFrame(X_val_scaled, columns=feature_cols)
            for col in metadata_cols:
                if col in df_val.columns:
                    df_val_normalized[col] = df_val[col].values
            
            val_normalized_csv = info["class_folder"] / f"{activity_name}_val_normalized.csv"
            df_val_normalized.to_csv(val_normalized_csv, index=False)
            print(f"  ✓ {val_normalized_csv.name}")

        # Test
        if info["test_csv"].exists():
            df_test = pd.read_csv(info["test_csv"])
            X_test = df_test[feature_cols].values
            X_test_scaled = scaler.transform(X_test)
            df_test_normalized = pd.DataFrame(X_test_scaled, columns=feature_cols)
            for col in metadata_cols:
                if col in df_test.columns:
                    df_test_normalized[col] = df_test[col].values
            
            test_normalized_csv = info["class_folder"] / f"{activity_name}_test_normalized.csv"
            df_test_normalized.to_csv(test_normalized_csv, index=False)
            print(f"  ✓ {test_normalized_csv.name}")

    # Save the ONE scaler
    scaler_path = args.scalers_dir / "scaler_global.pkl"
    joblib.dump(scaler, scaler_path)
    print()
    print(f"✓ Global scaler saved to: {scaler_path}")

    # Print scaler statistics
    print()
    print("=" * 70)
    print("GLOBAL SCALER STATISTICS (fitted on all training data)")
    print("=" * 70)
    print(f"\nTotal features: {len(feature_cols)}")
    print(f"\nMean (before scaling) - First 10 features:")
    for i in range(min(10, len(scaler.mean_))):
        print(f"  Feature {i}: {scaler.mean_[i]:10.6f}")
    if len(scaler.mean_) > 10:
        print(f"  ... ({len(scaler.mean_) - 10} more)")

    print(f"\nStd Dev (before scaling) - First 10 features:")
    for i in range(min(10, len(scaler.scale_))):
        print(f"  Feature {i}: {scaler.scale_[i]:10.6f}")
    if len(scaler.scale_) > 10:
        print(f"  ... ({len(scaler.scale_) - 10} more)")

    print()
    print("=" * 70)
    print(f"✓ Normalization complete (one scaler for all classes)")
    print()
    print("NEXT: Use *_normalized.csv files for training/validation/testing")


if __name__ == "__main__":
    main()
