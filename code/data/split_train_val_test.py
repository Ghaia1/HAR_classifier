"""Split features into train/validation/test sets (stratified by class)."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split features into train/val/test sets.")
    parser.add_argument(
        "--features-dir",
        type=Path,
        default=Path("data/processed/features"),
        help="Features directory with class folders",
    )
    parser.add_argument(
        "--train-ratio",
        type=float,
        default=0.60,
        help="Ratio of data for training (0-1)",
    )
    parser.add_argument(
        "--val-ratio",
        type=float,
        default=0.20,
        help="Ratio of data for validation (0-1)",
    )
    parser.add_argument(
        "--test-ratio",
        type=float,
        default=0.20,
        help="Ratio of data for testing (0-1)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Validate ratios
    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 0.01:
        raise ValueError(f"Ratios must sum to 1.0, got {total_ratio}")

    print("TRAIN/VALIDATION/TEST SPLIT STRATEGY")
    print("=" * 60)
    print(f"Train: {args.train_ratio:.0%} | Val: {args.val_ratio:.0%} | Test: {args.test_ratio:.0%}")
    print(f"Seed: {args.seed}")
    print()

    # Find all class folders (exclude 'scalers')
    class_folders = sorted([d for d in args.features_dir.iterdir() if d.is_dir() and d.name != "scalers"])
    if not class_folders:
        raise RuntimeError(f"No class folders found in: {args.features_dir}")

    print("JUSTIFICATION FOR SMALL DATASET:")
    print("-" * 60)
    print("• Stratified split: maintains class distribution in each set")
    print("• Shuffled per-class: ensures variety in each split")
    print("• 60/20/20 split: conservative for small dataset")
    print("  - Train (60%): enough to learn without heavy overfitting")
    print("  - Val (20%): tune hyperparameters and early stopping")
    print("  - Test (20%): held-out final evaluation")
    print()

    print("SPLITTING BY CLASS:")
    print("-" * 60)

    total_train = 0
    total_val = 0
    total_test = 0

    for class_folder in class_folders:
        class_name = class_folder.name  # e.g., "0_walking"
        activity_name = class_name.split("_")[1]

        # Find the features CSV
        csv_files = list(class_folder.glob(f"{activity_name}.csv"))
        if not csv_files:
            print(f"⚠ {class_name}: No CSV found, skipping")
            continue

        df = pd.read_csv(csv_files[0])
        n_samples = len(df)

        # Shuffle with seed
        df_shuffled = df.sample(frac=1.0, random_state=args.seed).reset_index(drop=True)

        # Calculate split indices
        n_train = int(n_samples * args.train_ratio)
        n_val = int(n_samples * args.val_ratio)
        n_test = n_samples - n_train - n_val

        # Split
        df_train = df_shuffled[:n_train]
        df_val = df_shuffled[n_train : n_train + n_val]
        df_test = df_shuffled[n_train + n_val :]

        # Save
        train_csv = class_folder / f"{activity_name}_train.csv"
        val_csv = class_folder / f"{activity_name}_val.csv"
        test_csv = class_folder / f"{activity_name}_test.csv"

        df_train.to_csv(train_csv, index=False)
        df_val.to_csv(val_csv, index=False)
        df_test.to_csv(test_csv, index=False)

        total_train += n_train
        total_val += n_val
        total_test += n_test

        print(
            f"{activity_name:15} | Total: {n_samples:3d} | "
            f"Train: {n_train:2d} | Val: {n_val:2d} | Test: {n_test:2d}"
        )

    print()
    print("TOTAL ACROSS ALL CLASSES:")
    print("-" * 60)
    print(f"Train: {total_train} samples ({total_train / (total_train + total_val + total_test):.1%})")
    print(f"Val:   {total_val} samples ({total_val / (total_train + total_val + total_test):.1%})")
    print(f"Test:  {total_test} samples ({total_test / (total_train + total_val + total_test):.1%})")
    print()
    print("✓ Split complete. Files saved in class folders:")
    print(f"  - {activity_name}_train.csv")
    print(f"  - {activity_name}_val.csv")
    print(f"  - {activity_name}_test.csv")


if __name__ == "__main__":
    main()
