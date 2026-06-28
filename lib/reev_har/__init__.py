"""Core package for Reev HAR project."""

from .data_loading import discover_recordings, load_recording
from .feature_extraction import extract_features
from .plotting import plot_recording_interactive
from .signal_detection import detect_crop_points
from .windowing import WindowConfig, generate_windows

__all__ = [
    "discover_recordings",
    "load_recording",
    "plot_recording_interactive",
    "detect_crop_points",
    "WindowConfig",
    "generate_windows",
    "extract_features",
]
