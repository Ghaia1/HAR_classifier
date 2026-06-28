# REEV HAR - Quick Start Guide

## ✓ Model Ready for Deployment

**Test Accuracy: 98%** | **Inference: 0.037 ms** | **Size: 54 KB**

---

## Run Training Pipeline

Train, evaluate, and export the Random Forest classifier in one command:

```bash
uv run python scripts/run_training_pipeline.py
```

**Output**:
- ✓ `outputs/random_forest_model.onnx` - ONNX model for deployment
- ✓ `outputs/random_forest_model.pkl` - Scikit-learn model
- ✓ `outputs/random_forest_metrics.json` - Performance metrics
- ✓ `outputs/feature_importance_named.csv` - Feature rankings

---

## Individual Scripts

### Train Model
```bash
uv run python scripts/train_random_forest.py
```
Trains on 137 samples, evaluates on 50 test samples

### Export to ONNX
```bash
uv run python scripts/export_model_onnx.py
```
Converts sklearn model to ONNX for on-device deployment

### Test Inference
```bash
uv run python scripts/inference_onnx.py
```
Benchmarks latency and validates ONNX model accuracy

### Analyze Features
```bash
uv run python scripts/analyze_feature_importance.py
```
Shows which features are most important for classification

---

## Using the ONNX Model

### Python (ONNX Runtime)
```python
import onnxruntime as ort
import numpy as np

# Load model
session = ort.InferenceSession("outputs/random_forest_model.onnx")
input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

# Prepare features (64-dim normalized array)
features = np.array([...], dtype=np.float32).reshape(1, 64)

# Predict
prediction = session.run([output_name], {input_name: features})[0][0]

# Map to activity
activities = ["walking", "sitting", "running", "falling"]
print(f"Predicted: {activities[prediction]}")
```

### C++ (ONNX Runtime)
```cpp
#include <onnxruntime_cxx_api.h>
#include <vector>

// Initialize session
Ort::Session session(env, "model.onnx", session_options);

// Prepare input (64 float32 values)
std::vector<float> features(64);  // Your normalized features
std::vector<int64_t> input_shape = {1, 64};

// Run inference
auto input_name = session.GetInputName(0, allocator);
Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
    memory_info, features.data(), 64, input_shape.data(), 2);

auto output_tensors = session.Run(
    Ort::RunOptions{nullptr}, &input_name, &input_tensor, 1, 
    output_names.data(), output_names.size());

// Get prediction
int64_t* output_data = output_tensors.front().GetTensorMutableData<int64_t>();
int prediction = (int)output_data[0];
```

---

## Results Summary

### Test Set Performance
```
Accuracy:  98.0%
Precision: 98.2%
Recall:    98.0%
F1-Score:  97.96%
```

### Per-Activity Results
| Activity | Accuracy | F1-Score | Samples |
|----------|----------|----------|---------|
| Walking  | 100%     | 100%     | 18      |
| Sitting  | 100%     | 100%     | 9       |
| Running  | 100%     | 100%     | 17      |
| Falling  | 83.3%    | 90.9%    | 6       |

### Inference Latency
- **Per-window**: 0.037 ms
- **Throughput**: ~27,000 windows/sec
- **Realtime**: 1500x faster than real-time

### Model Size
- **ONNX**: 54 KB
- **Scikit-learn**: 135 KB

---

## Data Format

### Input
- **Shape**: [1, 64]
- **Type**: Float32
- **Values**: Z-score normalized (-3 to +3 range typical)
- **Source**: 2.5 second sensor windows at 100 Hz

### Output
- **Type**: Integer [0-3]
- **Mapping**: 
  - 0 = walking
  - 1 = sitting
  - 2 = running
  - 3 = falling

---

## Troubleshooting

### Model not found
```bash
# Make sure you trained it first
uv run python scripts/train_random_forest.py
```

### Low accuracy on new data
- New data from different users/devices may need retraining
- Check that features are properly normalized globally
- Falling class underrepresented - collect more falling samples

### Inference errors
- Verify input shape is [1, 64]
- Ensure float32 dtype
- Check feature normalization (mean≈0, std≈1)

---

## Feature Importance

**Top 10 Features** (explain 36% of predictions):
1. acc_y_rms (9.82%)
2. gyr_energy_high (8.16%)
3. acc_energy_mid (6.13%)
4. gyr_magnitude_mean (6.07%)
5. acc_y_mean (5.89%)
6. gyr_z_rms (4.87%)
7. gyr_z_std (3.99%)
8. acc_y_min (3.91%)
9. acc_magnitude_mean (3.67%)
10. acc_magnitude_std (3.53%)

**Optimization**: Top 35 features explain 95% → consider feature selection for smaller model

---

## Files Overview

```
scripts/
├── train_random_forest.py          # Main training script
├── export_model_onnx.py             # ONNX export
├── inference_onnx.py                # Inference & benchmark
├── analyze_feature_importance.py    # Feature analysis
├── run_training_pipeline.py         # One-command pipeline
└── TRAINING_README.md               # Detailed documentation

outputs/
├── random_forest_model.pkl          # Sklearn model (production)
├── random_forest_model.onnx         # ONNX model (DEPLOYMENT)
├── random_forest_metrics.json       # Test results
├── feature_importance_named.csv     # Feature rankings
└── MODEL_SUMMARY.md                 # Full report
```

---

## Deployment Checklist

- [x] Model trained and validated (98% accuracy)
- [x] ONNX export created (54 KB)
- [x] Inference latency benchmarked (0.037 ms)
- [x] Feature importance analyzed
- [x] Performance metrics documented

**Next**: 
- [ ] Integrate ONNX model into target application
- [ ] Test on actual hardware
- [ ] Implement sliding window prediction
- [ ] Add confidence thresholds

---

**Last Updated**: 2026-06-27  
**Model**: Random Forest (100 trees, max_depth=20)  
**Framework**: Scikit-learn + ONNX Runtime
