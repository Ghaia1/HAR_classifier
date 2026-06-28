"""Load all recordings and create interactive HTML plots for each."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from reev_har.data_loading import discover_recordings, load_recording
from reev_har.plotting import plot_recording_interactive


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create interactive HTML plots for all recordings.")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data/raw"),
        help="Root data directory to visualize (default: data/raw, use data/processed/cropped for cropped version)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where HTML plots are saved (default: metrics/figures_<data-dir-name>)",
    )
    parser.add_argument(
        "--crop-config",
        type=Path,
        default=Path("metrics/crop_config.csv"),
        help="CSV with crop_start_s and crop_end_s columns (optional)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    
    # Infer output folder name from data directory if not explicitly provided
    if args.output_dir is None:
        data_dir_name = args.data_dir.name
        args.output_dir = Path("metrics") / f"figures_{data_dir_name}"
    
    args.output_dir.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(args.data_dir)
    if not recordings:
        raise RuntimeError(f"No recordings found in: {args.data_dir}")

    # Load crop config if it exists
    crop_config = {}
    if args.crop_config.exists():
        df_crop = pd.read_csv(args.crop_config)
        crop_config = {
            row["recording_folder"]: (row["crop_start_s"], row["crop_end_s"])
            for _, row in df_crop.iterrows()
        }
        print(f"Loaded crop config from {args.crop_config}\n")
    else:
        print(f"No crop config found at {args.crop_config} (proceeding without crop markers)\n")

    print(f"Found {len(recordings)} recording(s).\n")

    for rec in recordings:
        print(f"Loading {rec.folder.name}...", end=" ")
        df = load_recording(rec.folder)
        
        # Get crop times if available
        crop_start, crop_end = None, None
        if rec.folder.name in crop_config:
            crop_start, crop_end = crop_config[rec.folder.name]
        
        output_path = args.output_dir / f"{rec.folder.name}.html"
        plot_recording_interactive(df, output_path, crop_start_s=crop_start, crop_end_s=crop_end)
        print(f"✓ {output_path}")


if __name__ == "__main__":
    main()
