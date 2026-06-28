#!/usr/bin/env python3
"""
Complete HAR training pipeline runner.
Trains, evaluates, exports, and benchmarks Random Forest classifier.
"""

import subprocess
import sys
from pathlib import Path


def run_command(cmd: list, description: str) -> bool:
    """Run a shell command and report results."""
    print(f"\n{'='*60}")
    print(f"▶ {description}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"✓ {description} - SUCCESS")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ {description} - FAILED (exit code {e.returncode})")
        return False


def main():
    """Run complete pipeline."""
    print("""
================================================
  REEV HAR Random Forest Training Pipeline
================================================
    """)
    
    steps = [
        (
            [sys.executable, "code/model/train_random_forest.py"],
            "1/4 - Train Random Forest Model"
        ),
        (
            [sys.executable, "code/model/export_model_onnx.py"],
            "2/4 - Export to ONNX Format"
        ),
        (
            [sys.executable, "code/inference/inference_onnx.py"],
            "3/4 - Inference & Latency Benchmark"
        ),
        (
            [sys.executable, "code/model/analyze_feature_importance.py"],
            "4/4 - Analyze Feature Importance"
        ),
    ]
    
    results = []
    for cmd, description in steps:
        success = run_command(cmd, description)
        results.append((description, success))
        if not success:
            print(f"\n⚠️  Pipeline halted at {description}")
            break
    
    # Summary
    print(f"\n{'='*60}")
    print("PIPELINE SUMMARY")
    print(f"{'='*60}")
    
    for description, success in results:
        status = "✓ SUCCESS" if success else "✗ FAILED"
        print(f"{status:12s} - {description}")
    
    all_passed = all(success for _, success in results)
    
    if all_passed:
        print(f"\n{'='*60}")
        print("✓ ALL STEPS COMPLETED SUCCESSFULLY")
        print(f"{'='*60}")
        print(f"\nOutput files generated:")
        print(f"  Model files in 'model_weights/':")
        print(f"    - random_forest_model.pkl        (sklearn model)")
        print(f"    - random_forest_model.onnx       (ONNX model)")
        print(f"  Metrics in 'metrics/':")
        print(f"    - random_forest_metrics.json     (performance metrics)")
        print(f"    - feature_importance_named.csv   (feature rankings)")
        print(f"\nNext: Deploy ONNX model to on-device environment")
        return 0
    else:
        print(f"\n{'='*60}")
        print("✗ PIPELINE FAILED - Check errors above")
        print(f"{'='*60}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
