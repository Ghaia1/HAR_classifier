#!/usr/bin/env python3
"""
Inference script for NanoEdgeAI HAR classifier.
Evaluates validation/test accuracy and benchmarks per-window latency,
mirroring the ONNX validation workflow.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score


# Add emulator path for import
REPO_ROOT = Path(__file__).resolve().parents[2]
NEAI_EMULATOR_DIR = REPO_ROOT / "model_weights" / "NEAI_libneai_project-2026-06-27-20-35_1" / "emulators"
sys.path.insert(0, str(NEAI_EMULATOR_DIR))

from nanoedgeai_studio_emulator import NanoEdgeAIEmulator, NeaiState  # noqa: E402


def load_test_data(features_dir: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load validation and test sets from normalized feature files."""
    features_path = Path(features_dir)

    activity_map = {
        "0_walking": "walking",
        "1_sitting": "sitting",
        "2_running": "running",
        "3_falling": "falling",
    }

    X_test, y_test = [], []
    X_val, y_val = [], []

    metadata_cols = {"window_id", "activity_id", "activity_name", "recording_folder"}

    for folder_name, activity_name in activity_map.items():
        activity_dir = features_path / folder_name
        activity_id = int(folder_name[0])

        val_file = activity_dir / f"{activity_name}_val_normalized.csv"
        if val_file.exists():
            df_val = pd.read_csv(val_file)
            feature_cols = [col for col in df_val.columns if col not in metadata_cols]
            X_val.append(df_val[feature_cols].values)
            y_val.extend([activity_id] * len(df_val))

        test_file = activity_dir / f"{activity_name}_test_normalized.csv"
        if test_file.exists():
            df_test = pd.read_csv(test_file)
            feature_cols = [col for col in df_test.columns if col not in metadata_cols]
            X_test.append(df_test[feature_cols].values)
            y_test.extend([activity_id] * len(df_test))

    X_test_arr = np.vstack(X_test) if X_test else np.array([])
    X_val_arr = np.vstack(X_val) if X_val else np.array([])

    return X_val_arr, np.array(y_val), X_test_arr, np.array(y_test)


def inference_sklearn(model, X: np.ndarray) -> np.ndarray:
    """Run inference with scikit-learn model."""
    return model.predict(X)


def infer_neai(emulator: NanoEdgeAIEmulator, X: np.ndarray) -> np.ndarray:
    """Run batch inference with NanoEdgeAI emulator and return class IDs."""
    raw_preds: list[int] = []

    for row in X:
        result = emulator.detect(row.astype(np.float32).tolist())
        if result.state != NeaiState.NEAI_OK:
            raise RuntimeError(f"NEAI detect failed with state: {result.state}")
        raw_preds.append(int(result.value))

    preds = np.array(raw_preds, dtype=int)

    # Some exports may return class IDs in 1..N. Convert to 0..N-1 if needed.
    if len(preds) > 0:
        min_pred = int(preds.min())
        max_pred = int(preds.max())
        if min_pred == 1 and max_pred == emulator.class_number:
            preds = preds - 1

    return preds


def benchmark_latency(emulator: NanoEdgeAIEmulator, X: np.ndarray, n_runs: int = 1000) -> float:
    """Benchmark NEAI inference latency on one sample repeated n_runs times."""
    sample = X[0].astype(np.float32).tolist()

    # Warmup
    for _ in range(10):
        _ = emulator.detect(sample)

    start = time.perf_counter()
    for _ in range(n_runs):
        result = emulator.detect(sample)
        if result.state != NeaiState.NEAI_OK:
            raise RuntimeError(f"NEAI detect failed during benchmark: {result.state}")
    end = time.perf_counter()

    return ((end - start) / n_runs) * 1000.0


def main() -> None:
    print("Loading test data...")
    X_val, y_val, X_test, y_test = load_test_data("data/processed/features")

    activity_names = {0: "walking", 1: "sitting", 2: "running", 3: "falling"}

    print("\n=== Scikit-learn Model ===")
    model_pkl = joblib.load("model_weights/random_forest_model.pkl")

    y_pred_val_sk = inference_sklearn(model_pkl, X_val)
    y_pred_test_sk = inference_sklearn(model_pkl, X_test)

    val_acc_sk = accuracy_score(y_val, y_pred_val_sk)
    test_acc_sk = accuracy_score(y_test, y_pred_test_sk)

    print(f"Validation Accuracy: {val_acc_sk:.4f}")
    print(f"Test Accuracy:       {test_acc_sk:.4f}")

    print("\n=== NanoEdgeAI Model ===")
    neai_lib = NEAI_EMULATOR_DIR / "libneai.dll"
    if not neai_lib.exists():
        raise FileNotFoundError(f"NEAI emulator library not found: {neai_lib}")

    with NanoEdgeAIEmulator(str(neai_lib)) as emulator:
        print(f"Model type:         {emulator.model_type}")
        print(f"Input signal size:  {emulator.input_signal_size}")
        print(f"Axis number:        {emulator.axis_number}")
        print(f"Data length:        {emulator.data_length}")
        print(f"Number of classes:  {emulator.class_number}")

        y_pred_val_neai = infer_neai(emulator, X_val)
        y_pred_test_neai = infer_neai(emulator, X_test)

        val_acc_neai = accuracy_score(y_val, y_pred_val_neai)
        test_acc_neai = accuracy_score(y_test, y_pred_test_neai)

        print(f"Validation Accuracy: {val_acc_neai:.4f}")
        print(f"Test Accuracy:       {test_acc_neai:.4f}")

        print("\n=== Inference Latency Benchmark ===")
        latency_ms = benchmark_latency(emulator, X_test, n_runs=1000)
        print(f"Per-window latency: {latency_ms:.3f} ms (1000 runs)")
        print("Window duration: 2.5 s")
        print(f"Realtime factor: {(latency_ms / 2500):.4f}x")

        print("\n=== Sample Predictions (first 10 test samples) ===")
        for i in range(min(10, len(y_test))):
            true_label = activity_names[int(y_test[i])]
            pred_id = int(y_pred_test_neai[i])
            pred_label = activity_names.get(pred_id, f"class_{pred_id}")
            print(f"Sample {i}: True={true_label}, Pred={pred_label}")


if __name__ == "__main__":
    main()
