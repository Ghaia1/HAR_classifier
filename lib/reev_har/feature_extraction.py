"""Feature extraction for HAR classification."""

from __future__ import annotations

import numpy as np
from scipy import signal as scipy_signal
from scipy.stats import kurtosis, skew


def extract_features(window_dict: dict) -> dict:
    """Extract all features from a single window.

    Args:
        window_dict: Dictionary with 'acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z' (lists)

    Returns:
        Dictionary with extracted features (scalars and simple types)
    """
    features = {}

    # Convert to numpy arrays
    acc_x = np.array(window_dict["acc_x"])
    acc_y = np.array(window_dict["acc_y"])
    acc_z = np.array(window_dict["acc_z"])
    gyr_x = np.array(window_dict["gyr_x"])
    gyr_y = np.array(window_dict["gyr_y"])
    gyr_z = np.array(window_dict["gyr_z"])

    # ========== TIME DOMAIN ==========
    # Acceleration features
    features["acc_x_mean"] = float(np.mean(acc_x))
    features["acc_y_mean"] = float(np.mean(acc_y))
    features["acc_z_mean"] = float(np.mean(acc_z))

    features["acc_x_std"] = float(np.std(acc_x))
    features["acc_y_std"] = float(np.std(acc_y))
    features["acc_z_std"] = float(np.std(acc_z))

    features["acc_x_min"] = float(np.min(acc_x))
    features["acc_y_min"] = float(np.min(acc_y))
    features["acc_z_min"] = float(np.min(acc_z))

    features["acc_x_max"] = float(np.max(acc_x))
    features["acc_y_max"] = float(np.max(acc_y))
    features["acc_z_max"] = float(np.max(acc_z))

    features["acc_x_range"] = float(np.max(acc_x) - np.min(acc_x))
    features["acc_y_range"] = float(np.max(acc_y) - np.min(acc_y))
    features["acc_z_range"] = float(np.max(acc_z) - np.min(acc_z))

    # RMS (root mean square)
    features["acc_x_rms"] = float(np.sqrt(np.mean(acc_x**2)))
    features["acc_y_rms"] = float(np.sqrt(np.mean(acc_y**2)))
    features["acc_z_rms"] = float(np.sqrt(np.mean(acc_z**2)))

    # Zero crossing rate
    features["acc_x_zcr"] = float(_zero_crossing_rate(acc_x))
    features["acc_y_zcr"] = float(_zero_crossing_rate(acc_y))
    features["acc_z_zcr"] = float(_zero_crossing_rate(acc_z))

    # Signal magnitude area (SMA)
    features["acc_sma"] = float(np.mean(np.abs(acc_x) + np.abs(acc_y) + np.abs(acc_z)))

    # Magnitude (norm)
    acc_magnitude = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)
    features["acc_magnitude_mean"] = float(np.mean(acc_magnitude))
    features["acc_magnitude_std"] = float(np.std(acc_magnitude))
    features["acc_magnitude_max"] = float(np.max(acc_magnitude))
    features["acc_magnitude_skewness"] = float(skew(acc_magnitude))
    features["acc_magnitude_kurtosis"] = float(kurtosis(acc_magnitude))

    # Gyroscope features (same structure)
    features["gyr_x_mean"] = float(np.mean(gyr_x))
    features["gyr_y_mean"] = float(np.mean(gyr_y))
    features["gyr_z_mean"] = float(np.mean(gyr_z))

    features["gyr_x_std"] = float(np.std(gyr_x))
    features["gyr_y_std"] = float(np.std(gyr_y))
    features["gyr_z_std"] = float(np.std(gyr_z))

    features["gyr_x_min"] = float(np.min(gyr_x))
    features["gyr_y_min"] = float(np.min(gyr_y))
    features["gyr_z_min"] = float(np.min(gyr_z))

    features["gyr_x_max"] = float(np.max(gyr_x))
    features["gyr_y_max"] = float(np.max(gyr_y))
    features["gyr_z_max"] = float(np.max(gyr_z))

    features["gyr_x_range"] = float(np.max(gyr_x) - np.min(gyr_x))
    features["gyr_y_range"] = float(np.max(gyr_y) - np.min(gyr_y))
    features["gyr_z_range"] = float(np.max(gyr_z) - np.min(gyr_z))

    features["gyr_x_rms"] = float(np.sqrt(np.mean(gyr_x**2)))
    features["gyr_y_rms"] = float(np.sqrt(np.mean(gyr_y**2)))
    features["gyr_z_rms"] = float(np.sqrt(np.mean(gyr_z**2)))

    gyr_magnitude = np.sqrt(gyr_x**2 + gyr_y**2 + gyr_z**2)
    features["gyr_magnitude_mean"] = float(np.mean(gyr_magnitude))
    features["gyr_magnitude_std"] = float(np.std(gyr_magnitude))

    # ========== FREQUENCY DOMAIN ==========
    # FFT on acceleration magnitude
    fft_acc = np.abs(np.fft.fft(acc_magnitude - np.mean(acc_magnitude)))
    freqs = np.fft.fftfreq(len(acc_magnitude), d=0.01)  # 0.01s = 100Hz sampling
    freqs = freqs[: len(freqs) // 2]
    fft_acc = fft_acc[: len(fft_acc) // 2]

    # Dominant frequency
    if len(fft_acc) > 0 and np.max(fft_acc) > 0:
        dominant_freq_idx = np.argmax(fft_acc)
        features["acc_dominant_freq"] = float(freqs[dominant_freq_idx]) if dominant_freq_idx > 0 else 0.0
    else:
        features["acc_dominant_freq"] = 0.0

    # Energy in frequency bands
    band_low = (freqs >= 0.0) & (freqs < 1.0)
    band_mid = (freqs >= 1.0) & (freqs < 3.0)
    band_high = (freqs >= 3.0) & (freqs < 10.0)

    features["acc_energy_low"] = float(np.sum(fft_acc[band_low] ** 2))
    features["acc_energy_mid"] = float(np.sum(fft_acc[band_mid] ** 2))
    features["acc_energy_high"] = float(np.sum(fft_acc[band_high] ** 2))

    # Spectral entropy
    features["acc_spectral_entropy"] = float(_spectral_entropy(fft_acc))

    # Same for gyroscope magnitude
    fft_gyr = np.abs(np.fft.fft(gyr_magnitude - np.mean(gyr_magnitude)))
    fft_gyr = fft_gyr[: len(fft_gyr) // 2]

    if len(fft_gyr) > 0 and np.max(fft_gyr) > 0:
        dominant_freq_idx_gyr = np.argmax(fft_gyr)
        features["gyr_dominant_freq"] = float(freqs[dominant_freq_idx_gyr]) if dominant_freq_idx_gyr > 0 else 0.0
    else:
        features["gyr_dominant_freq"] = 0.0

    features["gyr_energy_low"] = float(np.sum(fft_gyr[band_low] ** 2))
    features["gyr_energy_mid"] = float(np.sum(fft_gyr[band_mid] ** 2))
    features["gyr_energy_high"] = float(np.sum(fft_gyr[band_high] ** 2))

    features["gyr_spectral_entropy"] = float(_spectral_entropy(fft_gyr))

    # ========== CROSS-AXIS FEATURES ==========
    # Correlation between axes
    features["acc_x_y_corr"] = float(np.corrcoef(acc_x, acc_y)[0, 1])
    features["acc_x_z_corr"] = float(np.corrcoef(acc_x, acc_z)[0, 1])
    features["acc_y_z_corr"] = float(np.corrcoef(acc_y, acc_z)[0, 1])

    features["gyr_x_y_corr"] = float(np.corrcoef(gyr_x, gyr_y)[0, 1])
    features["gyr_x_z_corr"] = float(np.corrcoef(gyr_x, gyr_z)[0, 1])
    features["gyr_y_z_corr"] = float(np.corrcoef(gyr_y, gyr_z)[0, 1])

    # Inter-sensor feature: acc/gyr magnitude ratio
    gyr_mag_mean = np.mean(gyr_magnitude)
    acc_mag_mean = np.mean(acc_magnitude)
    features["acc_gyr_magnitude_ratio"] = float(acc_mag_mean / (gyr_mag_mean + 1e-6))  # avoid division by zero

    return features


def _zero_crossing_rate(signal_arr: np.ndarray) -> float:
    """Count zero crossings in signal."""
    zero_crossings = np.where(np.diff(np.sign(signal_arr - np.mean(signal_arr))))[0]
    return len(zero_crossings) / len(signal_arr)


def _spectral_entropy(fft_magnitude: np.ndarray) -> float:
    """Compute spectral entropy (0=periodic, 1=random)."""
    # Normalize to probability distribution
    power = fft_magnitude**2
    power_norm = power / np.sum(power)
    # Entropy: -sum(p * log(p))
    entropy = -np.sum(power_norm[power_norm > 0] * np.log2(power_norm[power_norm > 0]))
    # Normalize by max possible entropy
    max_entropy = np.log2(len(power_norm))
    return entropy / max_entropy if max_entropy > 0 else 0.0
