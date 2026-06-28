# REEV HAR - Random Forest Human Activity Recognition

End-to-end machine learning pipeline for smartphone-based Human Activity Recognition (HAR) with on-device deployment.

**Status:** ✅ Production Ready | **Test Accuracy:** 98% | **Model Size:** 54 KB | **Latency:** 0.03 ms

---

## Quick Start

### 1) Environment Setup

Requirements:
- `pyenv` installed
- `uv` installed (`pipx install uv`)

Setup:
```powershell
pyenv install 3.14.4
pyenv local 3.14.4
uv sync
```

### 2) Run Complete Pipeline

```powershell
uv run python code/utils/run_training_pipeline.py
```

This orchestrates all 4 steps:
1. Train Random Forest classifier (98% accuracy)
2. Export to ONNX format (54 KB)
3. Benchmark inference latency (0.03 ms per window)
4. Analyze feature importance

**Output files:**
- `model_weights/random_forest_model.pkl` — Sklearn model
- `model_weights/random_forest_model.onnx` — Deployment model
- `model_weights/random_forest_metrics.json` — Performance metrics

---

## Full Data Pipeline (if reprocessing)

### Step 1: Detect Activity Boundaries
```powershell
uv run python code/data/detect_crop_points.py \
  --data-dir data/raw \
  --output metrics/crop_config.csv \
  --class-params metrics/crop_params_per_class.json
```

### Step 2: Apply Crop Boundaries
```powershell
uv run python code/data/apply_crop.py \
  --data-dir data/raw \
  --crop-config metrics/crop_config.csv \
  --output-dir data/processed/cropped
```

### Step 3: Generate Sliding Windows
```powershell
uv run python code/data/generate_windows.py \
  --data-dir data/processed/cropped \
  --output-dir data/processed/windowed \
  --window-size 2.5 \
  --overlap-ratio 0.5 \
  --sampling-rate 100
```

### Step 4: Extract Features
```powershell
uv run python code/data/extract_features.py \
  --windows-dir data/processed/windowed \
  --output-dir data/processed/features
```

### Step 5: Split Train/Val/Test
```powershell
uv run python code/data/split_train_val_test.py \
  --features-dir data/processed/features \
  --train-ratio 0.6 \
  --val-ratio 0.2
```

### Step 6: Normalize Features
```powershell
uv run python code/data/normalize_features.py \
  --features-dir data/processed/features
```

Then run the model training pipeline (Step 2 above).

---

## Project Structure

```
REEV/
├── code/                          # Production pipeline scripts
│   ├── data/                      # Data processing (7 scripts)
│   │   ├── detect_crop_points.py
│   │   ├── apply_crop.py
│   │   ├── generate_windows.py
│   │   ├── extract_features.py
│   │   ├── split_train_val_test.py
│   │   ├── normalize_features.py
│   │   └── prepare_nanoedge_data.py
│   ├── model/                     # Model training (3 scripts)
│   │   ├── train_random_forest.py
│   │   ├── export_model_onnx.py
│   │   └── analyze_feature_importance.py
│   ├── inference/                 # Inference testing (1 script)
│   │   └── inference_onnx.py
│   └── utils/                     # Utilities (2 scripts)
│       ├── run_training_pipeline.py
│       └── visualize_interactive.py
│
├── lib/reev_har/                  # Core library (5 modules)
│   ├── data_loading.py            # Load & align sensor CSVs
│   ├── signal_detection.py        # Detect activity boundaries
│   ├── windowing.py               # Create sliding windows
│   ├── feature_extraction.py      # Extract 64 features
│   └── plotting.py                # Interactive visualizations
│
├── data/                          # Sensor data
│   ├── raw/                       # Original recordings
│   └── processed/                 # Pipeline outputs
│       ├── cropped/
│       ├── windowed/
│       └── features/
│
├── model_weights/                 # Trained models
│   ├── random_forest_model.pkl    # Sklearn model (135 KB)
│   ├── random_forest_model.onnx   # ONNX model (54 KB)
│   └── random_forest_metrics.json # Test metrics
│
├── metrics/                       # Analysis outputs
│   ├── crop_config.csv
│   ├── crop_params_per_class.json
│   ├── feature_importance.csv
│   └── feature_importance_named.csv
│
├── docs/                          # Documentation
│   ├── README.md
│   ├── QUICK_START.md
│   ├── TECHNICAL_REPORT.md
│   └── task.md
│
├── pyproject.toml                 # Dependencies (uv managed)
├── .python-version                # Python 3.14.4
└── .venv/                         # Virtual environment
```

---

## Model Performance

### Test Set Results (50 samples)
| Metric | Value |
|--------|-------|
| Overall Accuracy | 98.0% |
| Precision | 0.982 |
| Recall | 0.98 |
| F1-Score | 0.9796 |

### Per-Class Performance
| Activity | Accuracy | F1-Score | Samples |
|----------|----------|----------|---------|
| Walking | 100.0% | 1.0000 | 18 |
| Sitting | 100.0% | 1.0000 | 9 |
| Running | 100.0% | 1.0000 | 17 |
| Falling | 83.3% | 0.9091 | 6 |

### Inference Performance
- **Per-window latency:** 0.03 ms
- **Window duration:** 2.5 seconds
- **Realtime factor:** 0.0000x (1500× faster than realtime)

---

## Top 10 Most Important Features

1. **acc_y_rms** (9.82%) — RMS acceleration on Y-axis
2. **gyr_energy_high** (8.16%) — High-frequency gyroscope energy
3. **acc_energy_mid** (6.13%) — Mid-frequency acceleration energy
4. **gyr_magnitude_mean** (6.07%) — Mean gyroscope magnitude
5. **acc_y_mean** (5.89%) — Mean acceleration on Y-axis
6. **gyr_z_rms** (4.87%) — RMS gyroscope on Z-axis
7. **gyr_z_std** (3.99%) — Std dev gyroscope on Z-axis
8. **acc_y_min** (3.91%) — Min acceleration on Y-axis
9. **acc_magnitude_mean** (3.67%) — Mean acceleration magnitude
10. **acc_magnitude_std** (3.53%) — Std dev acceleration magnitude

---

## Deployment

### ONNX Model
Production-ready ONNX model for cross-platform deployment:
- Input: 64-dimensional feature vector (float32)
- Output: Activity class (int64) + confidence scores
- Size: 54 KB (suitable for edge/mobile)
- Latency: <0.1 ms on CPU

### Quick Inference
```powershell
uv run python code/inference/inference_onnx.py
```

### Optional: NanoEdge AI Export
```powershell
uv run python code/data/prepare_nanoedge_data.py
```

---

## Data Processing Pipeline

### Input Data
- 4 activity recordings (walking, sitting, running, falling)
- Accelerometer + Gyroscope at 100 Hz
- ~13,000 samples per recording

### Processing Steps
1. **Crop Detection** — Auto-detect activity start/end times
2. **Crop Application** — Remove noise from edges
3. **Windowing** — Create 2.5s overlapping windows (50% overlap)
4. **Feature Extraction** — Compute 64 time/frequency/cross-axis features
5. **Train/Val/Test Split** — Stratified 60/20/20 split
6. **Normalization** — Z-score scaling (fit on training data only)

### Output Statistics
- Total windows: 292
- Valid features: 232 (after NaN filtering)
- Train samples: 137 (59.1%)
- Val samples: 45 (19.4%)
- Test samples: 50 (21.6%)

---

## Reproducibility

The pipeline is **100% reproducible** with deterministic random seeds:
- Random seed: 42
- Test accuracy: Always 98.0%
- Model parameters: Consistent across runs
- Feature importance: Identical rankings

Verified through independent pipeline run on 2026-06-27.

---

## References

- **scikit-learn:** Random Forest classifier (n_estimators=100, max_depth=20)
- **ONNX:** Open Neural Network Exchange format
- **scipy:** Signal processing (windowing, FFT)
- **pandas:** Data manipulation and CSV I/O

---

## Authors & Contact

Technical Assessment for Reev - Data Scientist Position
Date: June 27, 2026
