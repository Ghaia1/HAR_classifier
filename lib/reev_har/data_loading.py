from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import pandas as pd


ACC_FILE = "Accelerometer.csv"
GYR_FILE = "Gyroscope.csv"


@dataclass(frozen=True)
class Recording:
    """Single recording folder and parsed activity metadata."""

    activity_id: int
    activity_name: str
    folder: Path


_ACTIVITY_PATTERN = re.compile(r"^(?P<label>\d+)_(?P<name>.+)$")


def discover_recordings(data_dir: Path) -> list[Recording]:
    """Discover all activity recordings from top-level folders in data_dir."""
    recordings: list[Recording] = []
    for folder in sorted(data_dir.iterdir()):
        if not folder.is_dir():
            continue
        if not (folder / ACC_FILE).exists() or not (folder / GYR_FILE).exists():
            continue

        match = _ACTIVITY_PATTERN.match(folder.name)
        if not match:
            continue

        activity_id = int(match.group("label"))
        activity_name = match.group("name")
        recordings.append(Recording(activity_id=activity_id, activity_name=activity_name, folder=folder))

    return recordings


def load_recording(recording_folder: Path, merge_tolerance_s: float = 0.01) -> pd.DataFrame:
    """Load and align accelerometer and gyroscope signals into one DataFrame.

    The function performs a nearest-neighbor time alignment of gyroscope samples on
    accelerometer timestamps because the two streams can be slightly time-shifted.
    """
    acc_path = recording_folder / ACC_FILE
    gyr_path = recording_folder / GYR_FILE

    if not acc_path.exists() or not gyr_path.exists():
        missing = [str(p.name) for p in [acc_path, gyr_path] if not p.exists()]
        raise FileNotFoundError(f"Missing required file(s) in {recording_folder}: {missing}")

    acc = pd.read_csv(acc_path).rename(
        columns={
            "Time (s)": "time_s",
            "Acceleration x (m/s^2)": "acc_x",
            "Acceleration y (m/s^2)": "acc_y",
            "Acceleration z (m/s^2)": "acc_z",
        }
    )
    gyr = pd.read_csv(gyr_path).rename(
        columns={
            "Time (s)": "time_s",
            "Gyroscope x (rad/s)": "gyr_x",
            "Gyroscope y (rad/s)": "gyr_y",
            "Gyroscope z (rad/s)": "gyr_z",
        }
    )

    acc = acc.sort_values("time_s").dropna(subset=["time_s"])
    gyr = gyr.sort_values("time_s").dropna(subset=["time_s"])

    merged = pd.merge_asof(
        left=acc,
        right=gyr,
        on="time_s",
        direction="nearest",
        tolerance=merge_tolerance_s,
    )

    merged["acc_norm"] = (merged["acc_x"] ** 2 + merged["acc_y"] ** 2 + merged["acc_z"] ** 2) ** 0.5
    merged["gyr_norm"] = (merged["gyr_x"] ** 2 + merged["gyr_y"] ** 2 + merged["gyr_z"] ** 2) ** 0.5

    activity_id, activity_name = _parse_activity_from_folder(recording_folder)
    merged["activity_id"] = activity_id
    merged["activity_name"] = activity_name
    merged["recording"] = recording_folder.name

    return merged


def _parse_activity_from_folder(recording_folder: Path) -> tuple[int, str]:
    match = _ACTIVITY_PATTERN.match(recording_folder.name)
    if not match:
        return -1, recording_folder.name
    return int(match.group("label")), match.group("name")
