# REEV HAR Classifier

Author: Ghaia Belaakaria

Smartphone Human Activity Recognition pipeline for 4 classes: walking, sitting, running, and falling. The project covers raw data collection, preprocessing, feature engineering, Random Forest training, ONNX export, NanoEdgeAI export validation, and embedded deployment verification.

## Context

This repository was built as a practical end-to-end HAR pipeline targeting lightweight deployment. The goal is not only to train an accurate classifier, but also to produce deployable artifacts and verify that inference remains fast on CPU and on embedded hardware.

## Key Results

- Test accuracy: 98.0%
- Test weighted F1: 0.9796
- ONNX deployment model size: 54 KB
- ONNX CPU latency (fresh benchmark mean / P99): 0.050 ms / 0.071 ms
- NanoEdgeAI emulator latency: 0.021 ms/window
- STM32U575 on-target latency: 0.306 ms/window

## Quick Start

Install uv first, then sync the environment and run the full pipeline.

```powershell
pip install uv
pyenv install 3.14.4
pyenv local 3.14.4
uv sync

# Train + export + benchmark
uv run python code/utils/run_training_pipeline.py
```

## On-Device Performance

### CPU Inference (ONNX Runtime, desktop CPU)

Hardware used for benchmark:

- CPU: Intel Core Ultra 7 265H
- Cores: 16
- Logical processors: 16
- Max clock: 2.20 GHz
- RAM: 31.43 GB
- Machine: Lenovo 21RTS1PB00
- OS: Windows 11 Enterprise (10.0.26200)
- Runtime: ONNX Runtime with CPUExecutionProvider

| Metric | Value |
|---|---|
| Model size (ONNX) | 54 KB |
| Per-window latency (mean / P99) | 0.050 ms / 0.071 ms |
| Window duration / decision stride | 2500 ms / 1250 ms |
| Latency vs. 1 s target | 0.050 ms; about 20,000x faster than 1 s |
| RAM per inference | < 10 KB |

Inference is only one stage in a real pipeline (acquisition -> feature extraction -> inference -> actuation). Using a 1 s target for the classifier alone leaves headroom for the rest of the stack. At about 0.050 ms, the model is far below that threshold.

### Hardware-in-the-Loop Validation (STM32U575, Cortex-M33 @ 160 MHz)

The ONNX model was deployed to a NUCLEO-U575ZI-Q board via ST Edge AI Core v4.0.1 (STM32CubeAI Studio). This is a highly constrained embedded target: a Cortex-M33 microcontroller running at 160 MHz, with tight flash and RAM budgets, and CPU-only inference with no desktop-class runtime support and no GPU acceleration. Code generation, compilation, flashing, and on-target serial validation completed successfully.

| Metric | Value |
|---|---|
| Flash (weights + runtime) | 25.5 KB (20.1 KB weights + 4.9 KB runtime) |
| RAM (activations) | 276 B |
| MACCs | 778 |
| On-target inference latency | 0.306 ms/window mean; about 3268x faster than 1 s |
| Weight compression vs. float | -52.5% (lossless) |
| ONNX vs. C-model output | RMSE = 0, SNR = infinity (label); SNR = 154.7 dB (probabilities) |

The deployed C model matches the ONNX reference exactly on label output and with negligible floating-point rounding on probabilities. This result is especially relevant because it was achieved on highly constrained CPU-only embedded hardware rather than on a laptop or server-class processor. The ONNX file carries portability overhead; target-specific code generation reduces the footprint through lossless weight compression and removal of the ONNX runtime layer.

## Data Origin and Folder Contents

### How the raw data was collected

- Source: smartphone IMU recordings
- App/workflow: phyphox-based collection workflow
- Sampling rate: 100 Hz
- Placement: front pocket
- Signals used: accelerometer and gyroscope

### What is stored in `data/`

- `data/raw/`
	- Contains raw activity recordings committed in the repo.
	- Each class folder contains:
		- `Accelerometer.csv`
		- `Gyroscope.csv`
	- Current class folders:
		- `0_walking/`
		- `1_sitting/`
		- `2_running/`
		- `3_falling/`
- `data/processed/cropped/`
	- Same recordings after automatic boundary cropping.
- `data/processed/windowed/`
	- Sliding-window representation (2.5 s windows, 50% overlap).
- `data/processed/features/`
	- Extracted tabular features, train/val/test splits, normalized files, and scaler artifacts.

### What to expect in the data

- Raw recordings are continuous sensor traces for each activity.
- Cropped recordings remove inactive or noisy edges.
- Windowed data represents fixed-size segments used for training.
- Feature files contain 64 engineered features per window.

## Model Hyperparameters

Final classifier: `RandomForestClassifier`

| Hyperparameter | Value |
|---|---|
| n_estimators | 100 |
| max_depth | 20 |
| min_samples_split | 2 |
| min_samples_leaf | 1 |
| random_state | 42 |
| n_jobs | -1 |

## Main Commands

```powershell
# Data pipeline
uv run python code/data/detect_crop_points.py --data-dir data/raw --output metrics/crop_config.csv --class-params metrics/crop_params_per_class.json
uv run python code/data/apply_crop.py --data-dir data/raw --crop-config metrics/crop_config.csv --output-dir data/processed/cropped
uv run python code/data/generate_windows.py --data-dir data/processed/cropped --output-dir data/processed/windowed --window-size 2.5 --overlap-ratio 0.5 --sampling-rate 100
uv run python code/data/extract_features.py --windows-dir data/processed/windowed --output-dir data/processed/features
uv run python code/data/split_train_val_test.py --features-dir data/processed/features --train-ratio 0.6 --val-ratio 0.2
uv run python code/data/normalize_features.py --features-dir data/processed/features

# Model + inference
uv run python code/model/train_random_forest.py
uv run python code/model/export_model_onnx.py
uv run python code/inference/inference_onnx.py
uv run python code/inference/inference_neai.py
```

## Dependencies

Key libraries from `pyproject.toml`:

- numpy
- pandas
- scipy
- scikit-learn
- onnx
- onnxruntime
- skl2onnx
- joblib
- plotly
- matplotlib
- seaborn

## Important Outputs

- Main sklearn model: `model_weights/random_forest_model.pkl`
- Main ONNX deployment model: `model_weights/random_forest_model.onnx`
- Model metrics: `model_weights/random_forest_metrics.json`
- Global feature scaler: `data/processed/features/scalers/scaler_global.pkl`
- Named feature importance: `metrics/feature_importance_named.csv`
- NEAI validation results: `metrics/neai_validation_results.json`
- NanoEdgeAI export folder: `model_weights/NEAI_libneai_project-2026-06-27-20-35_1/`

Important: inference is not just the model file. Reproducing the trained pipeline also requires `scaler_global.pkl`, because the deployment expects the same normalization used during training.

## NanoEdgeAI

The repository includes a NanoEdgeAI exported library and a dedicated validation script.

- Library export: `model_weights/NEAI_libneai_project-2026-06-27-20-35_1/`
- Validation script: `code/inference/inference_neai.py`
- Validation results: `metrics/neai_validation_results.json`

Note: NEAI validation depends on the exported emulator/library bundled in the repo export folder; it is not a standalone pip dependency.

## Project Layout

```text
code/data/       cropping, windowing, feature extraction, splitting, normalization
code/model/      model training, ONNX export, feature importance analysis
code/inference/  ONNX and NanoEdgeAI inference/validation scripts
code/utils/      orchestration and visualization helpers
lib/reev_har/    reusable loading, detection, windowing, and feature logic
data/            raw recordings and processed pipeline outputs
model_weights/   trained models and export artifacts
metrics/         evaluation outputs and analysis files
```

## Model Performance

| Metric | Value |
|---|---|
| Test Accuracy | 98.0% |
| Test Precision | 0.982 |
| Test Recall | 0.980 |
| Test Weighted F1 | 0.9796 |

Per-class performance:

| Activity | Accuracy | F1-Score | Samples |
|---|---:|---:|---:|
| Walking | 100.0% | 1.0000 | 18 |
| Sitting | 100.0% | 1.0000 | 9 |
| Running | 100.0% | 1.0000 | 17 |
| Falling | 83.3% | 0.9091 | 6 |

## Notes

- Windowing: 2.5 s at 100 Hz (250 samples), 50% overlap
- Split strategy: stratified 60/20/20, seed=42
- Actual global split after rounding: train 59.05%, val 19.40%, test 21.55%
- Normalization: StandardScaler fitted on train only
- ONNX benchmark above is CPU-only, not GPU

