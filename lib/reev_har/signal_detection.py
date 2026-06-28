"""Automatic detection of signal activity boundaries."""

from __future__ import annotations

import pandas as pd
import numpy as np


def detect_crop_points(
    df: pd.DataFrame,
    noise_percentile: float = 20,
    activity_threshold_multiplier: float = 1.5,
    min_activity_duration_s: float = 2.0,
) -> tuple[float, float]:
    """Detect start and end times of meaningful activity in a recording.
    
    Uses a threshold-crossing approach on combined acceleration + gyroscope norm.
    
    Args:
        df: DataFrame with columns 'time_s', 'acc_norm', 'gyr_norm'
        noise_percentile: Percentile to define baseline noise (lower = quieter)
        activity_threshold_multiplier: How many times baseline to trigger activity
        min_activity_duration_s: Minimum contiguous active duration to count as valid activity
    
    Returns:
        (crop_start_s, crop_end_s) tuple
    """
    # Normalize signals to same scale (gyro typically smaller than acc)
    acc_norm_scaled = df["acc_norm"] / df["acc_norm"].max() if df["acc_norm"].max() > 0 else df["acc_norm"]
    gyr_norm_scaled = df["gyr_norm"] / df["gyr_norm"].max() if df["gyr_norm"].max() > 0 else df["gyr_norm"]
    
    # Combined norm: average of both scaled signals
    combined = (acc_norm_scaled + gyr_norm_scaled) / 2.0
    
    # Smooth with rolling window
    window = max(1, len(df) // 100)  # ~1% of signal
    smoothed = combined.rolling(window=window, center=True, min_periods=1).mean()
    
    # Baseline: quietest part of signal
    baseline = np.percentile(smoothed, noise_percentile)
    
    # Activity threshold
    threshold = baseline * activity_threshold_multiplier
    
    # Find activity periods
    active = smoothed > threshold
    
    # Find contiguous active regions
    active_diff = np.diff(active.astype(int))
    rising_idx = np.where(active_diff == 1)[0]
    falling_idx = np.where(active_diff == -1)[0]
    
    # Filter by minimum duration
    valid_regions = []
    for rise, fall in zip(rising_idx, falling_idx):
        duration = df["time_s"].iloc[fall] - df["time_s"].iloc[rise]
        if duration >= min_activity_duration_s:
            valid_regions.append((rise, fall))
    
    if not valid_regions:
        # Fallback: use first and last 10% as noise buffer
        duration = df["time_s"].iloc[-1] - df["time_s"].iloc[0]
        crop_start = df["time_s"].iloc[0] + 0.1 * duration
        crop_end = df["time_s"].iloc[-1] - 0.1 * duration
    else:
        # Use first rising edge and last falling edge
        crop_start = df["time_s"].iloc[valid_regions[0][0]]
        crop_end = df["time_s"].iloc[valid_regions[-1][1]]
    
    return float(crop_start), float(crop_end)
