#!/usr/bin/env python3
"""
Inference script for Random Forest HAR classifier using ONNX.
Tests model on validation/test sets and benchmarks latency.
"""

import numpy as np
import pandas as pd
import joblib
import onnxruntime as ort
import time
from pathlib import Path
from sklearn.metrics import accuracy_score


def load_test_data(features_dir: str) -> tuple:
    """Load test and validation data."""
    features_path = Path(features_dir)
    
    activity_map = {
        "0_walking": "walking",
        "1_sitting": "sitting",
        "2_running": "running",
        "3_falling": "falling"
    }
    
    X_test, y_test = [], []
    X_val, y_val = [], []
    
    metadata_cols = {"window_id", "activity_id", "activity_name", "recording_folder"}
    
    for folder_name, activity_name in activity_map.items():
        activity_dir = features_path / folder_name
        activity_id = int(folder_name[0])
        
        # Load val
        val_file = activity_dir / f"{activity_name}_val_normalized.csv"
        if val_file.exists():
            df_val = pd.read_csv(val_file)
            feature_cols = [col for col in df_val.columns if col not in metadata_cols]
            X_val.append(df_val[feature_cols].values)
            y_val.extend([activity_id] * len(df_val))
        
        # Load test
        test_file = activity_dir / f"{activity_name}_test_normalized.csv"
        if test_file.exists():
            df_test = pd.read_csv(test_file)
            feature_cols = [col for col in df_test.columns if col not in metadata_cols]
            X_test.append(df_test[feature_cols].values)
            y_test.extend([activity_id] * len(df_test))
    
    X_test = np.vstack(X_test) if X_test else np.array([])
    X_val = np.vstack(X_val) if X_val else np.array([])
    
    return X_val, np.array(y_val), X_test, np.array(y_test)


def inference_sklearn(model, X: np.ndarray) -> np.ndarray:
    """Run inference with scikit-learn model."""
    return model.predict(X)


def inference_onnx(session, X: np.ndarray) -> np.ndarray:
    """Run inference with ONNX Runtime."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    X_float32 = X.astype(np.float32)
    predictions = session.run([output_name], {input_name: X_float32})[0]
    
    return predictions


def benchmark_latency(session, X: np.ndarray, n_runs: int = 1000):
    """Benchmark ONNX inference latency."""
    input_name = session.get_inputs()[0].name
    output_name = session.get_outputs()[0].name
    
    X_float32 = X.astype(np.float32)
    
    # Warmup
    for _ in range(10):
        session.run([output_name], {input_name: X_float32[:1]})
    
    # Benchmark
    start = time.perf_counter()
    for _ in range(n_runs):
        session.run([output_name], {input_name: X_float32[:1]})
    end = time.perf_counter()
    
    latency_ms = ((end - start) / n_runs) * 1000
    return latency_ms


def main():
    print("Loading test data...")
    X_val, y_val, X_test, y_test = load_test_data("data/processed/features")
    
    activity_names = {0: "walking", 1: "sitting", 2: "running", 3: "falling"}
    
    # Load sklearn model
    print("\n=== Scikit-learn Model ===")
    model_pkl = joblib.load("model_weights/random_forest_model.pkl")
    
    y_pred_val_sk = inference_sklearn(model_pkl, X_val)
    y_pred_test_sk = inference_sklearn(model_pkl, X_test)
    
    val_acc_sk = accuracy_score(y_val, y_pred_val_sk)
    test_acc_sk = accuracy_score(y_test, y_pred_test_sk)
    
    print(f"Validation Accuracy: {val_acc_sk:.4f}")
    print(f"Test Accuracy:       {test_acc_sk:.4f}")
    
    # Load ONNX model
    print("\n=== ONNX Runtime Model ===")
    session = ort.InferenceSession("model_weights/random_forest_model.onnx", 
                                   providers=['CPUExecutionProvider'])
    
    y_pred_val_onnx = inference_onnx(session, X_val)
    y_pred_test_onnx = inference_onnx(session, X_test)
    
    val_acc_onnx = accuracy_score(y_val, y_pred_val_onnx)
    test_acc_onnx = accuracy_score(y_test, y_pred_test_onnx)
    
    print(f"Validation Accuracy: {val_acc_onnx:.4f}")
    print(f"Test Accuracy:       {test_acc_onnx:.4f}")
    
    # Benchmark latency
    print("\n=== Inference Latency Benchmark ===")
    latency_ms = benchmark_latency(session, X_test, n_runs=1000)
    print(f"Per-window latency: {latency_ms:.3f} ms (1000 runs)")
    print(f"Window duration: 2.5 s")
    print(f"Realtime factor: {(latency_ms / 2500):.4f}x")
    
    # Show sample predictions
    print("\n=== Sample Predictions (first 10 test samples) ===")
    for i in range(min(10, len(y_test))):
        print(f"Sample {i}: True={activity_names[y_test[i]]}, Pred={activity_names[y_pred_test_onnx[i]]}")


if __name__ == "__main__":
    main()
