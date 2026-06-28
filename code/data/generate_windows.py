"""Generate sliding windows organized by activity class."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from reev_har.data_loading import discover_recordings, load_recording
from reev_har.windowing import WindowConfig, generate_windows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate sliding windows organized by class.")
    parser.add_argument("--data-dir", type=Path, default=Path("data/processed/cropped"), help="Cropped data directory")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/windowed"),
        help="Root output directory for windowed data",
    )
    parser.add_argument("--window-size-s", type=float, default=2.5, help="Window size in seconds")
    parser.add_argument("--overlap-ratio", type=float, default=0.5, help="Overlap ratio (0-1)")
    parser.add_argument("--sampling-rate", type=int, default=100, help="Sampling rate in Hz")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Create window config
    config = WindowConfig(
        sampling_rate_hz=args.sampling_rate,
        window_size_s=args.window_size_s,
        overlap_ratio=args.overlap_ratio,
    )

    print(f"Window config: {config.window_samples} samples, {config.step_samples} sample step (50% overlap)\n")

    recordings = discover_recordings(args.data_dir)
    if not recordings:
        raise RuntimeError(f"No recordings found in: {args.data_dir}")

    # Group windows by activity
    windows_by_class = {}

    print(f"Generating windows from {len(recordings)} recording(s)...\n")

    for rec in recordings:
        df = load_recording(rec.folder)
        windows = generate_windows(df, config)

        activity_name = rec.activity_name
        activity_id = rec.activity_id

        if activity_name not in windows_by_class:
            windows_by_class[activity_name] = []

        windows_by_class[activity_name].extend(windows)
        print(f"{rec.folder.name:30} | {len(windows):4d} windows")

    # Save each class to its own folder
    print(f"\nSaving windows by class...\n")

    for activity_name, windows in windows_by_class.items():
        # Get activity_id from first window
        activity_id = windows[0]["activity_id"]

        # Create class folder: {activity_id}_{activity_name}
        class_folder = args.output_dir / f"{activity_id}_{activity_name}"
        class_folder.mkdir(parents=True, exist_ok=True)

        # Convert to DataFrame
        df = pd.DataFrame(windows)

        # Save as CSV with activity name
        output_csv = class_folder / f"{activity_name}.csv"
        df.to_csv(output_csv, index=False)

        print(f"{activity_name:15} | {len(windows):4d} windows → {output_csv}")

    print(f"\n✓ All windowed data saved to {args.output_dir}")


if __name__ == "__main__":
    main()
