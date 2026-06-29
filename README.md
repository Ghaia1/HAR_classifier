# REEV HAR Classifier

Smartphone Human Activity Recognition pipeline (walking, sitting, running, falling) with Random Forest + ONNX + NanoEdgeAI export validation.

## Key Results

- Test accuracy: 98.0%
- Test weighted F1: 0.9796
- ONNX model size: ~54 KB
- ONNX latency: ~0.03 ms/window
- NEAI latency (emulator): ~0.021 ms/window

## Quick Start

```powershell
pyenv install 3.14.4
pyenv local 3.14.4
uv sync

# Train + export + benchmark
uv run python code/utils/run_training_pipeline.py
```

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

## Important Outputs

- model_weights/random_forest_model.pkl
- model_weights/random_forest_model.onnx
- model_weights/random_forest_metrics.json
- metrics/feature_importance_named.csv
- metrics/neai_validation_results.json
- data/processed/features/scalers/scaler_global.pkl

## Project Layout

```text
code/        data, model, inference, utility scripts
lib/         reusable HAR library
data/        raw + processed datasets
model_weights/ trained/exported models
metrics/     evaluation and analysis outputs
docs/        extended documentation
```

## Notes

- Windowing: 2.5 s at 100 Hz (250 samples), 50% overlap
- Split strategy: stratified 60/20/20, seed=42
- Normalization: StandardScaler fitted on train only (no leakage)

Additional usage notes are available in docs/QUICK_START.md.

