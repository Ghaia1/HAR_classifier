#!/usr/bin/env python3
"""
Export trained Random Forest model to ONNX format for on-device deployment.
"""

import joblib
import numpy as np
from pathlib import Path
import onnx
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


def export_to_onnx(model_path: str, output_path: str, n_features: int = 64):
    """Convert Random Forest model to ONNX format."""
    
    print(f"Loading model from {model_path}...")
    model = joblib.load(model_path)
    
    # Define input type: batch of 64-feature vectors
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    
    print(f"Converting to ONNX with {n_features} input features...")
    onnx_model = convert_sklearn(model, initial_types=initial_type, target_opset=12)
    
    # Validate ONNX model
    onnx.checker.check_model(onnx_model)
    print("✓ ONNX model validated")
    
    # Save
    onnx.save(onnx_model, output_path)
    print(f"✓ ONNX model saved to {output_path}")
    
    # Print model info
    print(f"\nModel Info:")
    print(f"  Inputs: {[inp.name for inp in onnx_model.graph.input]}")
    print(f"  Outputs: {[out.name for out in onnx_model.graph.output]}")
    print(f"  Nodes: {len(onnx_model.graph.node)}")


def main():
    model_path = Path("model_weights/random_forest_model.pkl")
    output_path = Path("model_weights/random_forest_model.onnx")
    
    if not model_path.exists():
        print(f"Error: Model file not found at {model_path}")
        print("Run train_random_forest.py first")
        return
    
    export_to_onnx(str(model_path), str(output_path), n_features=64)


if __name__ == "__main__":
    main()
