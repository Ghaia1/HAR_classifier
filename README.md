# REEV HAR Classifier

**Author:** Ghaia Belaakaria

Smartphone Human Activity Recognition pipeline for 4 classes: walking, sitting, running, and falling.
The project covers raw IMU data collection, automatic boundary cropping, sliding-window preprocessing,
handcrafted feature engineering, Random Forest training, ONNX export, NanoEdgeAI cross-validation,
and embedded deployment verification on STM32U575 hardware.

---

## Key Results

| Metric | Value |
|---|---|
| Test Accuracy | 98.0% |
| Precision (weighted) | 0.982 |
| Recall (weighted) | 0.980 |
| F1 (weighted) | 0.9796 |
| ONNX model size | 54 KB |
| ONNX CPU latency (mean / P99) | 0.050 ms / 0.071 ms |
| NanoEdgeAI emulator latency | 0.021 ms/window |
| STM32U575 on-target latency | 0.306 ms/window |

---

## Context

This repository delivers a complete end-to-end HAR pipeline targeting lightweight on-device deployment.
The goal is not only to train an accurate classifier, but to produce deployable artifacts and verify
that inference remains real-time on both desktop CPU and highly constrained embedded hardware.

---

## Quick Start

Install `uv` first, then sync and run the full pipeline in one command.

```powershell
pip install uv
pyenv install 3.14.4
pyenv local 3.14.4
uv sync

# Full pipeline: train + export + benchmark
uv run python code/utils/run_training_pipeline.py
```

---

## Main Commands

Run each stage individually if needed.

```powershell
# --- Data pipeline ---

# 1. Detect crop points (removes phone pocketing artifacts at start/end of each recording)
uv run python code/data/detect_crop_points.py \
    --data-dir data/raw \
    --output metrics/crop_config.csv \
    --class-params metrics/crop_params_per_class.json

# 2. Apply crop
uv run python code/data/apply_crop.py \
    --data-dir data/raw \
    --crop-config metrics/crop_config.csv \
    --output-dir data/processed/cropped

# 3. Generate sliding windows (2.5 s, 50% overlap)
uv run python code/data/generate_windows.py \
    --data-dir data/processed/cropped \
    --output-dir data/processed/windowed \
    --window-size 2.5 \
    --overlap-ratio 0.5 \
    --sampling-rate 100

# 4. Extract features (64 features/window)
uv run python code/data/extract_features.py \
    --windows-dir data/processed/windowed \
    --output-dir data/processed/features

# 5. Stratified train/val/test split (60/20/20, seed=42)
uv run python code/data/split_train_val_test.py \
    --features-dir data/processed/features \
    --train-ratio 0.6 \
    --val-ratio 0.2

# 6. Normalize (StandardScaler fitted on train only)
uv run python code/data/normalize_features.py \
    --features-dir data/processed/features

# --- Model ---

uv run python code/model/train_random_forest.py
uv run python code/model/export_model_onnx.py

# --- Inference & benchmarks ---

uv run python code/inference/inference_onnx.py
uv run python code/inference/inference_neai.py
```

---

## Project Layout

```text
code/data/         cropping, windowing, feature extraction, splitting, normalization
code/model/        model training, ONNX export, feature importance analysis
code/inference/    ONNX and NanoEdgeAI inference and validation scripts
code/utils/        pipeline orchestration and visualization helpers
lib/reev_har/      reusable library: loading, cropping, windowing, feature logic
data/raw/          raw Phyphox CSV recordings (Accelerometer.csv + Gyroscope.csv per class)
data/processed/    cropped, windowed, and feature-extracted outputs
model_weights/     trained models, ONNX export, NanoEdgeAI export
metrics/           evaluation outputs, feature importance, NEAI results
```

---

## Data

### Collection protocol

- **App:** Phyphox (RWTH Aachen), export as CSV
- **Sampling rate:** 100 Hz
- **Signals:** 3-axis accelerometer + 3-axis gyroscope
- **Placement:** front pants pocket
- **Classes:** walking (0), sitting (1), running (2), falling (3)
- **Falling protocol:** controlled forward/lateral falls onto a soft mat; 5 s stillness after each fall; recording paused before getting up to avoid contamination

### Recorded durations

| Class | Before crop | After crop |
|---|---:|---:|
| Walking | 132.5 s | 106.0 s |
| Sitting | 122.6 s | 98.0 s |
| Running | 128.3 s | 102.7 s |
| Falling | 81.9 s | 65.5 s |

> Falling duration is shorter because Phyphox does not count paused time toward total recording duration.

### Folder structure in `data/`

```text
data/raw/
    0_walking/   Accelerometer.csv  Gyroscope.csv
    1_sitting/   Accelerometer.csv  Gyroscope.csv
    2_running/   Accelerometer.csv  Gyroscope.csv
    3_falling/   Accelerometer.csv  Gyroscope.csv
data/processed/
    cropped/     boundary-cropped recordings
    windowed/    sliding-window segments (2.5 s, 50% overlap)
    features/    64-feature tabular data, train/val/test splits, scaler artifacts
```

---

## Preprocessing & Features

| Choice | Value | Justification |
|---|---|---|
| Window size | 250 samples (2.5 s) | Captures a full gait cycle (1–2 Hz) without mixing transitions |
| Overlap | 50% (stride = 125 samples) | Approximately doubles training samples without excessive redundancy |
| Split | 60/20/20 stratified, seed=42 | Preserves class proportions; 20% val/test gives sufficient per-class support at this dataset size |
| Normalization | StandardScaler (train only) | Z-score is robust to fall spikes that would compress MinMax range; fitted on train to prevent leakage |

**64 features per window:**
- **Time domain (47):** mean, std, min, max, range, RMS, ZCR per acc axis; SMA; acc magnitude moments; gyr mean/std/RMS/max/range per axis + magnitude moments
- **Frequency domain (10):** dominant frequency, band energies (0–1 / 1–3 / 3–10 Hz), spectral entropy — for both acc and gyr
- **Cross-axis (7):** pairwise Pearson correlations for acc and gyr (6) + acc/gyr magnitude ratio (1)

> Wavelet features were considered and ruled out: FFT band energy captures the same transient structure at lower complexity for 2.5 s windows.

---

## Model

**Classifier:** `RandomForestClassifier` (scikit-learn)

| Hyperparameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 20 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |
| random_state | 42 |
| n_jobs | -1 |

**Rationale:** RF handles non-linear boundaries on small heterogeneous tabular feature sets without
careful hyperparameter tuning and provides built-in feature importance. A deep model (1D-CNN, LSTM)
would increase overfitting risk on ~230 windows without a meaningful accuracy benefit.

**NanoEdgeAI cross-check:** the same 64-feature dataset was evaluated in NanoEdgeAI Studio as an
AutoML baseline. It selected an SVM model and reached identical accuracy (98.0%, F1 = 0.9796) at
0.021 ms/window, confirming the feature set is well-conditioned and the RF result is not model-specific.

---

## Model Performance

**Overall:**

| Metric | Value |
|---|---|
| Test Accuracy | 98.0% |
| Precision (weighted) | 0.982 |
| Recall (weighted) | 0.980 |
| F1 (weighted) | 0.9796 |
| Test samples | 50 |

**Per-class (test set):**

| Activity | Precision | Recall | F1 | Samples |
|---|---:|---:|---:|---:|
| Walking | 1.00 | 1.00 | 1.00 | 18 |
| Sitting | 0.90 | 1.00 | 0.95 | 9 |
| Running | 1.00 | 1.00 | 1.00 | 17 |
| Falling | 1.00 | 0.83 | 0.91 | 6 |

**Single misclassification:** 1 falling window predicted as sitting. Post-fall stillness and seated
stillness produce nearly identical IMU profiles (low RMS, low variance, gravity-dominant acceleration)
in a short window without temporal context. Sitting precision is 0.90 because that one falling sample
was predicted as sitting, adding a false positive to the sitting column.

---

## On-Device Performance

### CPU Inference (ONNX Runtime)

Benchmark hardware: Intel Core Ultra 7 265H, 16 cores, 2.20 GHz, 31.4 GB RAM, Windows 11 Enterprise.

| Metric | Value |
|---|---|
| Model size (ONNX) | 54 KB |
| Latency (mean / P99) | 0.050 ms / 0.071 ms |
| Window duration / stride | 2500 ms / 1250 ms |
| Latency vs. 1 s classifier budget | ~20,000× faster |
| RAM per inference | < 10 KB |

Inference is one stage in a full pipeline (acquisition → feature extraction → inference → actuation).
A 1 s budget for the classifier alone leaves headroom for the rest of the stack.

### Hardware-in-the-Loop: STM32U575 (Cortex-M33 @ 160 MHz)

Deployed to NUCLEO-U575ZI-Q via ST Edge AI Core v4.0.1 (STM32CubeAI Studio).
Build: 0 errors, 0 warnings. Flash, run, and serial validation all completed successfully.

| Metric | Value |
|---|---|
| Flash (weights + runtime) | 25.5 KB (20.1 KB weights + 4.9 KB runtime) |
| RAM (activations) | 276 B |
| MACCs | 778 |
| On-target latency (mean) | 0.306 ms/window (~3,268× faster than 1 s) |
| Weight compression vs. float | −52.5% (lossless) |
| ONNX vs. C-model output | RMSE = 0, SNR = ∞ (label); SNR = 154.7 dB (probabilities) |

The C model matches the ONNX reference exactly on label output. The ONNX file (54 KB) is larger than
the on-target footprint (25.5 KB) because ONNX carries portability overhead; ST Edge AI's code
generation strips this via lossless weight compression and removal of the ONNX runtime layer.

---

## Important Outputs

| File | Description |
|---|---|
| `model_weights/random_forest_model.pkl` | Trained sklearn model |
| `model_weights/random_forest_model.onnx` | ONNX deployment artifact |
| `model_weights/random_forest_metrics.json` | Full evaluation metrics |
| `data/processed/features/scalers/scaler_global.pkl` | StandardScaler — required for inference |
| `metrics/feature_importance_named.csv` | Named feature importance ranking |
| `metrics/neai_validation_results.json` | NanoEdgeAI validation results |
| `model_weights/NEAI_libneai_project-2026-06-27-20-35_1/` | NanoEdgeAI export library |

> **Important:** inference requires both the model file and `scaler_global.pkl`.
> The scaler must be the one fitted during training — do not refit on new data.

---

## NanoEdgeAI

The repo includes a NanoEdgeAI exported library and a dedicated validation script.

- Export: `model_weights/NEAI_libneai_project-2026-06-27-20-35_1/`
- Script: `code/inference/inference_neai.py`
- Results: `metrics/neai_validation_results.json`

> NanoEdgeAI validation uses the emulator bundled in the export folder. It is not a pip dependency.

---

## Dependencies

Managed via `uv` / `pyproject.toml`. Key libraries:

```
numpy · pandas · scipy · scikit-learn · onnx · onnxruntime · skl2onnx · joblib · plotly · matplotlib · seaborn
```

---

## Notes

- Windowing: 2.5 s at 100 Hz (250 samples), 50% overlap
- Split: stratified 60/20/20, seed=42 (actual: 59.05% / 19.40% / 21.55% after rounding)
- Normalization: StandardScaler fitted on train only — no leakage
- ONNX benchmark is CPU-only (CPUExecutionProvider), no GPU