"""Apply crop boundaries from crop_config.csv to raw sensor data."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from reev_har.data_loading import discover_recordings


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop sensor CSVs using boundaries from crop_config.csv")
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Root data directory")
    parser.add_argument(
        "--crop-config",
        type=Path,
        default=Path("metrics/crop_config.csv"),
        help="CSV with crop boundaries",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/cropped"),
        help="Where to save cropped CSVs (mirrors data/ structure)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Load crop config
    if not args.crop_config.exists():
        raise FileNotFoundError(f"Crop config not found: {args.crop_config}")

    df_crop = pd.read_csv(args.crop_config)
    crop_map = {
        row["recording_folder"]: (row["crop_start_s"], row["crop_end_s"])
        for _, row in df_crop.iterrows()
    }

    recordings = discover_recordings(args.data_dir)
    if not recordings:
        raise RuntimeError(f"No recordings found in: {args.data_dir}")

    print(f"Cropping {len(recordings)} recording(s)...\n")

    for rec in recordings:
        if rec.folder.name not in crop_map:
            print(f"⚠ {rec.folder.name}: not in crop config, skipping")
            continue

        crop_start, crop_end = crop_map[rec.folder.name]

        # Create output folder
        output_folder = args.output_dir / rec.folder.name
        output_folder.mkdir(parents=True, exist_ok=True)

        # Crop Accelerometer
        acc_path = rec.folder / "Accelerometer.csv"
        if acc_path.exists():
            df_acc = pd.read_csv(acc_path).rename(columns={"Time (s)": "time_s"})
            df_acc_cropped = df_acc[(df_acc["time_s"] >= crop_start) & (df_acc["time_s"] <= crop_end)]
            df_acc_cropped = df_acc_cropped.rename(columns={"time_s": "Time (s)"})
            
            output_acc = output_folder / "Accelerometer.csv"
            df_acc_cropped.to_csv(output_acc, index=False)
            print(f"✓ {rec.folder.name}/Accelerometer.csv: {len(df_acc)} → {len(df_acc_cropped)} samples")
        
        # Crop Gyroscope
        gyr_path = rec.folder / "Gyroscope.csv"
        if gyr_path.exists():
            df_gyr = pd.read_csv(gyr_path).rename(columns={"Time (s)": "time_s"})
            df_gyr_cropped = df_gyr[(df_gyr["time_s"] >= crop_start) & (df_gyr["time_s"] <= crop_end)]
            df_gyr_cropped = df_gyr_cropped.rename(columns={"time_s": "Time (s)"})
            
            output_gyr = output_folder / "Gyroscope.csv"
            df_gyr_cropped.to_csv(output_gyr, index=False)
            print(f"✓ {rec.folder.name}/Gyroscope.csv: {len(df_gyr)} → {len(df_gyr_cropped)} samples")

    print(f"\n✓ All cropped data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
