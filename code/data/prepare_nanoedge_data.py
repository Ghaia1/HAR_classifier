#!/usr/bin/env python3
"""
Prepare normalized features for NanoEdge training.
Strips header and metadata columns, keeping only the 64 feature values per line.
"""

import pandas as pd
from pathlib import Path


def main():
    features_dir = Path("data/processed/features")
    
    # Map folder names to activity names
    activity_map = {
        "0_walking": "walking",
        "1_sitting": "sitting",
        "2_running": "running",
        "3_falling": "falling"
    }
    
    # Metadata columns to drop
    metadata_cols = {"window_id", "activity_id", "activity_name", "recording_folder"}
    
    for folder_name, activity_name in activity_map.items():
        activity_dir = features_dir / folder_name
        print(f"\nProcessing {activity_name}...")
        
        for split in ["train", "val", "test"]:
            normalized_file = activity_dir / f"{activity_name}_{split}_normalized.csv"
            
            if not normalized_file.exists():
                print(f"  ⚠️  {normalized_file.name} not found, skipping")
                continue
            
            # Read normalized CSV
            df = pd.read_csv(normalized_file)
            
            # Keep only feature columns (drop metadata)
            feature_cols = [col for col in df.columns if col not in metadata_cols]
            df_features = df[feature_cols]
            
            # Save without header, no index
            output_file = activity_dir / f"{activity_name}_{split}_nanoedge.csv"
            df_features.to_csv(output_file, header=False, index=False)
            
            print(f"  ✓ {output_file.name}: {len(df_features)} samples, {len(feature_cols)} features")


if __name__ == "__main__":
    main()
