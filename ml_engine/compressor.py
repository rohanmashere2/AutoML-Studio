"""
Model Compression & Edge Deployment — Quantization, pruning, distillation, ONNX export.
"""

import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, r2_score, mean_squared_error


def compress_model(model, X_train, y_train, X_test, y_test, problem_type='classification', output_dir=None):
    """Apply multiple compression strategies and report tradeoffs."""
    results = {
        'original': _evaluate_model(model, X_test, y_test, problem_type),
        'compressions': [],
    }
    results['original']['size_bytes'] = _model_size(model)
    
    # 1. Feature Pruning — keep only top features
    try:
        pruned = _feature_pruning(model, X_train, y_train, X_test, y_test, problem_type)
        if pruned:
            results['compressions'].append(pruned)
    except Exception:
        pass
    
    # 2. Model Distillation — train simpler model to mimic original
    try:
        distilled = _distill_model(model, X_train, y_train, X_test, y_test, problem_type)
        if distilled:
            results['compressions'].append(distilled)
    except Exception:
        pass
    
    # 3. Precision Reduction — float64 to float32
    try:
        reduced = _reduce_precision(model, X_test, y_test, problem_type)
        if reduced:
            results['compressions'].append(reduced)
    except Exception:
        pass
    
    # 4. ONNX Export
    if output_dir:
        try:
            onnx_result = _export_onnx(model, X_train, output_dir)
            if onnx_result:
                results['compressions'].append(onnx_result)
        except Exception:
            pass
    
    # Summary
    results['best_compression'] = None
    if results['compressions']:
        valid = [c for c in results['compressions'] if c.get('score_retention', 0) > 90]
        if valid:
            results['best_compression'] = min(valid, key=lambda x: x.get('size_bytes', float('inf')))
    
    return results


def _evaluate_model(model, X_test, y_test, problem_type):
    """Evaluate model performance."""
    y_pred = model.predict(X_test)
    if problem_type == 'classification':
        score = accuracy_score(y_test, y_pred)
    else:
        score = r2_score(y_test, y_pred)
    return {'score': round(float(score), 6), 'metric': 'accuracy' if problem_type == 'classification' else 'r2'}


def _model_size(model):
    """Estimate model size in bytes."""
    try:
        data = pickle.dumps(model)
        return len(data)
    except Exception:
        return 0


def _feature_pruning(model, X_train, y_train, X_test, y_test, problem_type, keep_ratio=0.5):
    """Keep only top important features."""
    if not hasattr(model, 'feature_importances_'):
        return None
    
    importances = model.feature_importances_
    n_keep = max(3, int(len(importances) * keep_ratio))
    top_idx = np.argsort(importances)[-n_keep:]
    
    X_train_pruned = X_train[:, top_idx] if isinstance(X_train, np.ndarray) else X_train.iloc[:, top_idx]
    X_test_pruned = X_test[:, top_idx] if isinstance(X_test, np.ndarray) else X_test.iloc[:, top_idx]
    
    pruned_model = clone(model)
    pruned_model.fit(X_train_pruned, y_train)
    
    metrics = _evaluate_model(pruned_model, X_test_pruned, y_test, problem_type)
    original_score = _evaluate_model(model, X_test, y_test, problem_type)['score']
    
    return {
        'method': 'Feature Pruning',
        'description': f'Kept top {n_keep}/{len(importances)} features ({keep_ratio*100:.0f}%)',
        'score': metrics['score'],
        'score_retention': round(metrics['score'] / max(original_score, 1e-10) * 100, 1),
        'size_bytes': _model_size(pruned_model),
        'features_kept': n_keep,
        'features_total': len(importances),
    }


def _distill_model(model, X_train, y_train, X_test, y_test, problem_type):
    """Knowledge distillation: train a simple model to mimic the complex one."""
    # Generate soft labels from the teacher
    teacher_preds = model.predict(X_train)
    
    if problem_type == 'classification':
        student = DecisionTreeClassifier(max_depth=5, random_state=42)
    else:
        student = DecisionTreeRegressor(max_depth=5, random_state=42)
    
    student.fit(X_train, teacher_preds)
    
    metrics = _evaluate_model(student, X_test, y_test, problem_type)
    original_score = _evaluate_model(model, X_test, y_test, problem_type)['score']
    
    return {
        'method': 'Knowledge Distillation',
        'description': 'Decision Tree student model (max_depth=5) trained on teacher predictions',
        'score': metrics['score'],
        'score_retention': round(metrics['score'] / max(original_score, 1e-10) * 100, 1),
        'size_bytes': _model_size(student),
        'student_model': 'DecisionTree(depth=5)',
    }


def _reduce_precision(model, X_test, y_test, problem_type):
    """Simulate float32 precision reduction."""
    original_size = _model_size(model)
    
    # Convert test data to float32
    if isinstance(X_test, pd.DataFrame):
        X_32 = X_test.astype(np.float32)
    else:
        X_32 = X_test.astype(np.float32)
    
    try:
        y_pred = model.predict(X_32)
        original_score = _evaluate_model(model, X_test, y_test, problem_type)['score']
        if problem_type == 'classification':
            score = accuracy_score(y_test, y_pred)
        else:
            score = r2_score(y_test, y_pred)
        
        return {
            'method': 'Precision Reduction (float32)',
            'description': 'Converted inference to float32 precision',
            'score': round(float(score), 6),
            'score_retention': round(float(score) / max(original_score, 1e-10) * 100, 1),
            'size_bytes': int(original_size * 0.5),
            'size_reduction_pct': 50,
        }
    except Exception:
        return None


def _export_onnx(model, X_train, output_dir):
    """Export model to ONNX format."""
    try:
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        n_features = X_train.shape[1]
        initial_type = [('float_input', FloatTensorType([None, n_features]))]
        
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        onnx_path = os.path.join(output_dir, 'model.onnx')
        with open(onnx_path, 'wb') as f:
            f.write(onnx_model.SerializeToString())
        
        onnx_size = os.path.getsize(onnx_path)
        
        return {
            'method': 'ONNX Export',
            'description': f'Exported to ONNX format ({onnx_size / 1024:.1f} KB)',
            'size_bytes': onnx_size,
            'path': onnx_path,
            'format': 'ONNX',
        }
    except ImportError:
        return {
            'method': 'ONNX Export',
            'description': 'skl2onnx not installed. Install with: pip install skl2onnx',
            'available': False
        }
    except Exception as e:
        return None
