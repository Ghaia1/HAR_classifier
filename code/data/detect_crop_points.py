"""Auto-detect crop points for all recordings and save to CSV."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

# Add lib to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent.parent / 'lib'))

from reev_har.data_loading import discover_recordings, load_recording
from reev_har.signal_detection import detect_crop_points


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Auto-detect activity start/end points and save crop config."
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data/raw"), help="Root data directory")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("metrics/crop_config.csv"),
        help="Where to save crop configuration CSV",
    )
    parser.add_argument(
        "--class-params",
        type=Path,
        default=Path("metrics/crop_params_per_class.json"),
        help="JSON file with per-class threshold parameters (optional)",
    )
    parser.add_argument(
        "--noise-percentile",
        type=float,
        default=10,
        help="Default percentile for baseline noise level (overridden by JSON if provided)",
    )
    parser.add_argument(
        "--threshold-multiplier",
        type=float,
        default=1.1,
        help="Default activity threshold multiplier (overridden by JSON if provided)",
    )
    parser.add_argument(
        "--min-duration",
        type=float,
        default=1.0,
        help="Minimum activity duration in seconds to count as valid",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    recordings = discover_recordings(args.data_dir)
    if not recordings:
        raise RuntimeError(f"No recordings found in: {args.data_dir}")

    # Load per-class parameters if available
    class_params = {}
    if args.class_params.exists():
        with open(args.class_params) as f:
            class_params = json.load(f)
        print(f"Loaded per-class parameters from {args.class_params}\n")
    else:
        print(f"Using global parameters (class_params file not found)\n")

    print(f"Detecting crop points for {len(recordings)} recording(s)...\n")

    results = []

    for rec in recordings:
        df = load_recording(rec.folder)
        
        # Get parameters for this class (or fall back to global defaults)
        if rec.activity_name in class_params:
            params = class_params[rec.activity_name]
            noise_pct = params.get("noise_percentile", args.noise_percentile)
            thresh_mult = params.get("threshold_multiplier", args.threshold_multiplier)
        else:
            noise_pct = args.noise_percentile
            thresh_mult = args.threshold_multiplier
        
        crop_start, crop_end = detect_crop_points(
            df,
            noise_percentile=noise_pct,
            activity_threshold_multiplier=thresh_mult,
            min_activity_duration_s=args.min_duration,
        )
        duration_original = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
        duration_cropped = crop_end - crop_start

        results.append({
            "recording_folder": rec.folder.name,
            "activity_id": rec.activity_id,
            "activity_name": rec.activity_name,
            "crop_start_s": round(crop_start, 3),
            "crop_end_s": round(crop_end, 3),
            "duration_original_s": round(duration_original, 1),
            "duration_cropped_s": round(duration_cropped, 1),
        })

        print(
            f"{rec.folder.name:30} | "
            f"params(noise%={noise_pct:.0f}, mult={thresh_mult:.2f}) | "
            f"crop [{crop_start:.2f}–{crop_end:.2f}]s | "
            f"duration {duration_cropped:.1f}s"
        )

    df_results = pd.DataFrame(results)
    df_results.to_csv(args.output, index=False)
    print(f"\n✓ Crop config saved to {args.output}\n")

    # Summary stats by activity
    print("Summary by activity class:")
    print(df_results.groupby("activity_name")["duration_cropped_s"].agg(["count", "min", "max", "mean"]))


if __name__ == "__main__":
    main()
