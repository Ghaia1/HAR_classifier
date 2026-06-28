"""Sliding window segmentation of sensor data."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WindowConfig:
    """Configuration for sliding window generation."""

    sampling_rate_hz: int = 100
    window_size_s: float = 2.5
    overlap_ratio: float = 0.5

    @property
    def window_samples(self) -> int:
        """Number of samples per window."""
        return int(self.window_size_s * self.sampling_rate_hz)

    @property
    def step_samples(self) -> int:
        """Step size between windows (samples)."""
        return int(self.window_samples * (1 - self.overlap_ratio))


def generate_windows(
    df: pd.DataFrame,
    config: WindowConfig,
) -> list[dict]:
    """Generate sliding windows from a recording.

    Args:
        df: DataFrame with columns 'time_s', 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z'
        config: WindowConfig with window and overlap parameters

    Returns:
        List of dictionaries, one per window, with metadata and sensor arrays.
    """
    windows = []
    window_samples = config.window_samples
    step_samples = config.step_samples

    activity_id = df["activity_id"].iloc[0]
    activity_name = df["activity_name"].iloc[0]
    recording = df["recording"].iloc[0]

    for start_idx in range(0, len(df) - window_samples + 1, step_samples):
        end_idx = start_idx + window_samples
        window_data = df.iloc[start_idx:end_idx]

        window_record = {
            "window_id": len(windows),
            "activity_id": activity_id,
            "activity_name": activity_name,
            "recording_folder": recording,
            "window_start_idx": start_idx,
            "window_start_time_s": float(window_data["time_s"].iloc[0]),
            "window_end_time_s": float(window_data["time_s"].iloc[-1]),
            "acc_x": window_data["acc_x"].values.tolist(),
            "acc_y": window_data["acc_y"].values.tolist(),
            "acc_z": window_data["acc_z"].values.tolist(),
            "gyr_x": window_data["gyr_x"].values.tolist(),
            "gyr_y": window_data["gyr_y"].values.tolist(),
            "gyr_z": window_data["gyr_z"].values.tolist(),
        }
        windows.append(window_record)

    return windows
