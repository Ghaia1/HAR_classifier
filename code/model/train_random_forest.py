#!/usr/bin/env python3
"""
Train a Random Forest classifier on REEV HAR normalized features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import joblib
import json


def load_data(features_dir: str) -> tuple:
    """Load normalized training data from all activity classes."""
    features_path = Path(features_dir)
    
    activity_map = {
        "0_walking": "walking",
        "1_sitting": "sitting",
        "2_running": "running",
        "3_falling": "falling"
    }
    
    X_train, y_train = [], []
    X_val, y_val = [], []
    X_test, y_test = [], []
    
    metadata_cols = {"window_id", "activity_id", "activity_name", "recording_folder"}
    
    for folder_name, activity_name in activity_map.items():
        activity_dir = features_path / folder_name
        activity_id = int(folder_name[0])
        
        # Load train set
        train_file = activity_dir / f"{activity_name}_train_normalized.csv"
        if train_file.exists():
            df_train = pd.read_csv(train_file)
            feature_cols = [col for col in df_train.columns if col not in metadata_cols]
            X_train.append(df_train[feature_cols].values)
            y_train.extend([activity_id] * len(df_train))
        
        # Load validation set
        val_file = activity_dir / f"{activity_name}_val_normalized.csv"
        if val_file.exists():
            df_val = pd.read_csv(val_file)
            feature_cols = [col for col in df_val.columns if col not in metadata_cols]
            X_val.append(df_val[feature_cols].values)
            y_val.extend([activity_id] * len(df_val))
        
        # Load test set
        test_file = activity_dir / f"{activity_name}_test_normalized.csv"
        if test_file.exists():
            df_test = pd.read_csv(test_file)
            feature_cols = [col for col in df_test.columns if col not in metadata_cols]
            X_test.append(df_test[feature_cols].values)
            y_test.extend([activity_id] * len(df_test))
    
    X_train = np.vstack(X_train) if X_train else np.array([])
    X_val = np.vstack(X_val) if X_val else np.array([])
    X_test = np.vstack(X_test) if X_test else np.array([])
    
    y_train = np.array(y_train)
    y_val = np.array(y_val)
    y_test = np.array(y_test)
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Val set:   X={X_val.shape}, y={y_val.shape}")
    print(f"Test set:  X={X_test.shape}, y={y_test.shape}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test


def train_model(X_train: np.ndarray, y_train: np.ndarray, **kwargs) -> RandomForestClassifier:
    """Train a Random Forest classifier."""
    # Use validation set for hyperparameter tuning (optional)
    params = {
        "n_estimators": kwargs.get("n_estimators", 100),
        "max_depth": kwargs.get("max_depth", None),
        "min_samples_split": kwargs.get("min_samples_split", 2),
        "min_samples_leaf": kwargs.get("min_samples_leaf", 1),
        "random_state": 42,
        "n_jobs": -1,
    }
    
    print(f"\nTraining Random Forest with params: {params}")
    rf = RandomForestClassifier(**params)
    rf.fit(X_train, y_train)
    
    return rf


def evaluate_model(model: RandomForestClassifier, X_val: np.ndarray, y_val: np.ndarray, 
                   X_test: np.ndarray, y_test: np.ndarray) -> dict:
    """Evaluate model on validation and test sets."""
    activity_names = {0: "walking", 1: "sitting", 2: "running", 3: "falling"}
    
    # Validation metrics
    y_val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, y_val_pred)
    val_f1 = f1_score(y_val, y_val_pred, average='weighted')
    
    # Test metrics
    y_test_pred = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_test_pred)
    test_precision = precision_score(y_test, y_test_pred, average='weighted')
    test_recall = recall_score(y_test, y_test_pred, average='weighted')
    test_f1 = f1_score(y_test, y_test_pred, average='weighted')
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_test_pred)
    
    # Per-class metrics
    per_class_metrics = {}
    for activity_id in sorted(activity_names.keys()):
        mask = y_test == activity_id
        if mask.sum() > 0:
            y_test_class = y_test[mask]
            y_pred_class = y_test_pred[mask]
            per_class_metrics[activity_names[activity_id]] = {
                "accuracy": float(accuracy_score(y_test_class, y_pred_class)),
                "f1": float(f1_score(y_test_class, y_pred_class, average='weighted', zero_division=0)),
                "samples": int(mask.sum())
            }
    
    metrics = {
        "val_accuracy": float(val_acc),
        "val_f1": float(val_f1),
        "test_accuracy": float(test_acc),
        "test_precision": float(test_precision),
        "test_recall": float(test_recall),
        "test_f1": float(test_f1),
        "confusion_matrix": cm.tolist(),
        "per_class_metrics": per_class_metrics,
    }
    
    return metrics, y_test_pred


def main():
    features_dir = "data/processed/features"
    output_dir = Path("model_weights")
    output_dir.mkdir(exist_ok=True)
    # Load data
    print("Loading data...")
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(features_dir)
    
    # Train model
    model = train_model(X_train, y_train, n_estimators=100, max_depth=20)
    
    # Evaluate
    print("\nEvaluating model...")
    metrics, y_test_pred = evaluate_model(model, X_val, y_val, X_test, y_test)
    
    # Print results
    print(f"\n=== Validation Results ===")
    print(f"Accuracy: {metrics['val_accuracy']:.4f}")
    print(f"F1-Score: {metrics['val_f1']:.4f}")
    
    print(f"\n=== Test Results ===")
    print(f"Accuracy:  {metrics['test_accuracy']:.4f}")
    print(f"Precision: {metrics['test_precision']:.4f}")
    print(f"Recall:    {metrics['test_recall']:.4f}")
    print(f"F1-Score:  {metrics['test_f1']:.4f}")
    
    print(f"\n=== Confusion Matrix ===")
    activity_names = ["walking", "sitting", "running", "falling"]
    cm_df = pd.DataFrame(metrics['confusion_matrix'], 
                        index=activity_names, columns=activity_names)
    print(cm_df)
    
    print(f"\n=== Per-Class Metrics ===")
    for activity, scores in metrics['per_class_metrics'].items():
        print(f"{activity}: accuracy={scores['accuracy']:.4f}, f1={scores['f1']:.4f}, samples={scores['samples']}")
    
    # Save model
    model_path = output_dir / "random_forest_model.pkl"
    joblib.dump(model, model_path)
    print(f"\n✓ Model saved to {model_path}")
    
    # Save metrics
    metrics_path = output_dir / "random_forest_metrics.json"
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Metrics saved to {metrics_path}")
    
    # Feature importance
    feature_importance = pd.DataFrame({
        'feature_index': range(len(model.feature_importances_)),
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(f"\n=== Top 10 Most Important Features ===")
    print(feature_importance.head(10))
    
    importance_path = output_dir / "feature_importance.csv"
    feature_importance.to_csv(importance_path, index=False)
    print(f"✓ Feature importance saved to {importance_path}")


if __name__ == "__main__":
    main()
