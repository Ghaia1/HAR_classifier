"""Extract features from windowed data."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from reev_har.feature_extraction import extract_features


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract features from windowed data.")
    parser.add_argument(
        "--windows-dir",
        type=Path,
        default=Path("data/processed/windowed"),
        help="Windowed data directory",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/features"),
        help="Output features directory (one CSV per class)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Find all class folders
    class_folders = sorted([d for d in args.windows_dir.iterdir() if d.is_dir()])
    if not class_folders:
        raise RuntimeError(f"No class folders found in: {args.windows_dir}")

    print(f"Extracting features from {len(class_folders)} class(es)...\n")

    for class_folder in class_folders:
        class_name = class_folder.name  # e.g., "0_walking"

        # Find the windows CSV file
        window_files = list(class_folder.glob("*.csv"))
        if not window_files:
            print(f"⚠ No CSV files in {class_folder}, skipping")
            continue

        windows_csv = window_files[0]
        print(f"Reading {windows_csv.name}...", end=" ")

        # Load windows
        df_windows = pd.read_csv(windows_csv)
        print(f"{len(df_windows)} windows", end=" | ")

        # Extract features for each window
        features_list = []
        skipped = 0
        for idx, row in df_windows.iterrows():
            try:
                # Parse JSON arrays (handles NaN better than literal_eval)
                window_dict = {
                    "acc_x": json.loads(row["acc_x"]) if isinstance(row["acc_x"], str) else row["acc_x"],
                    "acc_y": json.loads(row["acc_y"]) if isinstance(row["acc_y"], str) else row["acc_y"],
                    "acc_z": json.loads(row["acc_z"]) if isinstance(row["acc_z"], str) else row["acc_z"],
                    "gyr_x": json.loads(row["gyr_x"]) if isinstance(row["gyr_x"], str) else row["gyr_x"],
                    "gyr_y": json.loads(row["gyr_y"]) if isinstance(row["gyr_y"], str) else row["gyr_y"],
                    "gyr_z": json.loads(row["gyr_z"]) if isinstance(row["gyr_z"], str) else row["gyr_z"],
                }
            except (ValueError, json.JSONDecodeError, TypeError):
                # Skip windows with unparseable data
                skipped += 1
                continue

            # Convert to numpy and handle NaN
            import numpy as np
            window_dict_np = {k: np.array(v, dtype=float) for k, v in window_dict.items()}
            
            # Replace NaN with 0
            for k in window_dict_np:
                window_dict_np[k] = np.nan_to_num(window_dict_np[k], nan=0.0)

            # Extract features
            feat = extract_features(window_dict_np)

            # Add metadata
            feat["window_id"] = row["window_id"]
            feat["activity_id"] = row["activity_id"]
            feat["activity_name"] = row["activity_name"]
            feat["recording_folder"] = row["recording_folder"]

            features_list.append(feat)

        df_features = pd.DataFrame(features_list)

        # Save to class folder
        class_output_folder = args.output_dir / class_name
        class_output_folder.mkdir(parents=True, exist_ok=True)
        
        output_csv = class_output_folder / f"{class_name.split('_')[1]}.csv"
        df_features.to_csv(output_csv, index=False)

        if skipped > 0:
            print(f"(skipped {skipped}) | ", end="")
        print(f"✓ {output_csv}")

    print(f"\n✓ All features saved to {args.output_dir}")


if __name__ == "__main__":
    main()
