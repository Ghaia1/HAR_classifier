#!/usr/bin/env python3
"""
Feature importance analyzer - maps feature indices to feature names.
"""

import pandas as pd
import joblib
from pathlib import Path


def main():
    # Load model and get feature importances
    model = joblib.load("model_weights/random_forest_model.pkl")
    importances = model.feature_importances_
    
    # Feature names (in order they appear in normalized CSV)
    feature_names = [
        # Accelerometer time-domain (21)
        'acc_x_mean', 'acc_y_mean', 'acc_z_mean',
        'acc_x_std', 'acc_y_std', 'acc_z_std',
        'acc_x_min', 'acc_y_min', 'acc_z_min',
        'acc_x_max', 'acc_y_max', 'acc_z_max',
        'acc_x_range', 'acc_y_range', 'acc_z_range',
        'acc_x_rms', 'acc_y_rms', 'acc_z_rms',
        'acc_x_zcr', 'acc_y_zcr', 'acc_z_zcr',
        # Accelerometer magnitude (7)
        'acc_sma', 'acc_magnitude_mean', 'acc_magnitude_std', 'acc_magnitude_max',
        'acc_magnitude_skewness', 'acc_magnitude_kurtosis',
        # Gyroscope time-domain (18)
        'gyr_x_mean', 'gyr_y_mean', 'gyr_z_mean',
        'gyr_x_std', 'gyr_y_std', 'gyr_z_std',
        'gyr_x_min', 'gyr_y_min', 'gyr_z_min',
        'gyr_x_max', 'gyr_y_max', 'gyr_z_max',
        'gyr_x_range', 'gyr_y_range', 'gyr_z_range',
        'gyr_x_rms', 'gyr_y_rms', 'gyr_z_rms',
        # Gyroscope magnitude (2)
        'gyr_magnitude_mean', 'gyr_magnitude_std',
        # Frequency domain (14)
        'acc_dominant_freq', 'acc_energy_low', 'acc_energy_mid', 'acc_energy_high', 'acc_spectral_entropy',
        'gyr_dominant_freq', 'gyr_energy_low', 'gyr_energy_mid', 'gyr_energy_high', 'gyr_spectral_entropy',
        # Cross-axis (7)
        'acc_x_y_corr', 'acc_x_z_corr', 'acc_y_z_corr',
        'gyr_x_y_corr', 'gyr_x_z_corr', 'gyr_y_z_corr',
        'acc_gyr_magnitude_ratio'
    ]
    
    # Create DataFrame
    df = pd.DataFrame({
        'feature_name': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False).reset_index(drop=True)
    
    df['cumsum_importance'] = df['importance'].cumsum()
    
    # Print top features
    print("=== Top 20 Most Important Features ===")
    print(df.head(20).to_string(index=False))
    
    # Find how many features needed for 95% importance
    n_features_95 = (df['cumsum_importance'] <= 0.95).sum() + 1
    print(f"\nFeatures needed for 95% cumulative importance: {n_features_95} / {len(df)}")
    print(f"Top {n_features_95} features explain {df.iloc[n_features_95-1]['cumsum_importance']:.4f} of total importance")
    
    # Save full feature importance
    output_file = Path("metrics/feature_importance_named.csv")
    df.to_csv(output_file, index=False)
    print(f"\n✓ Feature importance saved to {output_file}")


if __name__ == "__main__":
    main()
