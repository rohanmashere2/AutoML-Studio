"""
AutoML Problem Solver - Model Deployer
Prediction API, batch predictions, code export, and Docker packaging.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np


def predict_single(model, features, feature_names, transform_metadata=None):
    """
    Make a prediction for a single input.
    
    Args:
        model: Trained model
        features: dict of feature_name -> value
        feature_names: list of expected feature names
        transform_metadata: scaler and encoder info
    
    Returns:
        dict: prediction, probabilities, confidence
    """
    try:
        # Build input DataFrame
        row = {}
        for fname in feature_names:
            if fname in features:
                row[fname] = float(features[fname])
            else:
                row[fname] = 0.0  # Default for missing features
        
        X = pd.DataFrame([row], columns=feature_names)
        
        # Apply scaling if metadata available
        if transform_metadata and transform_metadata.get('scaler'):
            try:
                X = pd.DataFrame(
                    transform_metadata['scaler'].transform(X),
                    columns=feature_names
                )
            except Exception:
                pass
        
        # Predict
        prediction = model.predict(X)[0]
        
        result = {
            'prediction': float(prediction) if not isinstance(prediction, str) else prediction,
        }
        
        # Probabilities for classification
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(X)[0]
            result['probabilities'] = {str(i): round(float(p), 4) for i, p in enumerate(proba)}
            result['confidence'] = round(float(max(proba)), 4)
        
        # Decode target label if encoder available
        if transform_metadata and transform_metadata.get('target_encoder'):
            try:
                decoded = transform_metadata['target_encoder'].inverse_transform([int(prediction)])
                result['prediction_label'] = str(decoded[0])
            except Exception:
                pass
        
        return result
    except Exception as e:
        return {'error': str(e)}


def predict_batch(model, df, feature_names, transform_metadata=None):
    """
    Make predictions for a batch of inputs.
    
    Args:
        model: Trained model
        df: DataFrame with input data
        feature_names: list of expected feature names
    
    Returns:
        dict: predictions list, probabilities, summary
    """
    try:
        # Select and order features
        available = [f for f in feature_names if f in df.columns]
        missing = [f for f in feature_names if f not in df.columns]
        
        X = df[available].copy()
        for m in missing:
            X[m] = 0.0
        X = X[feature_names]
        
        # Ensure numeric
        X = X.apply(pd.to_numeric, errors='coerce').fillna(0)
        
        # Apply scaling
        if transform_metadata and transform_metadata.get('scaler'):
            try:
                X = pd.DataFrame(
                    transform_metadata['scaler'].transform(X),
                    columns=feature_names
                )
            except Exception:
                pass
        
        predictions = model.predict(X)
        
        result = {
            'predictions': [float(p) if not isinstance(p, str) else p for p in predictions],
            'n_predictions': len(predictions),
            'missing_features': missing,
        }
        
        if hasattr(model, 'predict_proba'):
            probas = model.predict_proba(X)
            result['probabilities'] = probas.tolist()
            result['confidence'] = [round(float(max(p)), 4) for p in probas]
        
        # Decode labels
        if transform_metadata and transform_metadata.get('target_encoder'):
            try:
                decoded = transform_metadata['target_encoder'].inverse_transform(predictions.astype(int))
                result['prediction_labels'] = [str(d) for d in decoded]
            except Exception:
                pass
        
        # Summary statistics
        if np.issubdtype(predictions.dtype, np.number):
            result['summary'] = {
                'mean': round(float(np.mean(predictions)), 4),
                'median': round(float(np.median(predictions)), 4),
                'std': round(float(np.std(predictions)), 4),
                'min': round(float(np.min(predictions)), 4),
                'max': round(float(np.max(predictions)), 4),
            }
        else:
            unique, counts = np.unique(predictions, return_counts=True)
            result['summary'] = {str(u): int(c) for u, c in zip(unique, counts)}
        
        return result
    except Exception as e:
        return {'error': str(e)}


def generate_prediction_script(model_filename, feature_names, problem_type, target_encoder_classes=None):
    """
    Generate a standalone Python prediction script.
    
    Returns:
        str: Python script code
    """
    features_list = json.dumps(feature_names, indent=4)
    
    classes_code = ""
    if target_encoder_classes:
        classes_code = f"""
# Class labels
CLASSES = {json.dumps(target_encoder_classes)}
"""
    
    script = f'''#!/usr/bin/env python3
"""
AutoML Generated Prediction Script
Problem Type: {problem_type}
Usage:
    python predict.py --input data.csv --output predictions.csv
    python predict.py --interactive
"""

import sys
import json
import joblib
import pandas as pd
import numpy as np
import argparse

# Expected features
FEATURE_NAMES = {features_list}
{classes_code}

def load_model():
    """Load the trained model."""
    return joblib.load("{model_filename}")


def predict_single(model, features):
    """Predict for a single input (dict of feature->value)."""
    row = {{fname: features.get(fname, 0.0) for fname in FEATURE_NAMES}}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    prediction = model.predict(X)[0]
    result = {{"prediction": float(prediction)}}
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["probabilities"] = {{str(i): round(float(p), 4) for i, p in enumerate(proba)}}
        result["confidence"] = round(float(max(proba)), 4)
    
    return result


def predict_batch(model, csv_path, output_path=None):
    """Predict for a CSV file."""
    df = pd.read_csv(csv_path)
    
    available = [f for f in FEATURE_NAMES if f in df.columns]
    X = df[available].copy()
    for m in [f for f in FEATURE_NAMES if f not in available]:
        X[m] = 0.0
    X = X[FEATURE_NAMES].apply(pd.to_numeric, errors="coerce").fillna(0)
    
    df["prediction"] = model.predict(X)
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        df["confidence"] = [round(float(max(p)), 4) for p in probas]
    
    if output_path:
        df.to_csv(output_path, index=False)
        print(f"Predictions saved to {{output_path}}")
    
    return df


def interactive_mode(model):
    """Interactive prediction mode."""
    print("\\n=== Interactive Prediction Mode ===")
    print(f"Enter values for {{len(FEATURE_NAMES)}} features (press Enter for default 0):\\n")
    
    while True:
        features = {{}}
        for fname in FEATURE_NAMES:
            val = input(f"  {{fname}}: ").strip()
            features[fname] = float(val) if val else 0.0
        
        result = predict_single(model, features)
        print(f"\\n  ➡️ Prediction: {{result['prediction']}}")
        if "confidence" in result:
            print(f"  📊 Confidence: {{result['confidence']:.1%}}")
        
        again = input("\\nPredict again? (y/n): ").strip().lower()
        if again != "y":
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AutoML Prediction Script")
    parser.add_argument("--input", help="Path to input CSV file")
    parser.add_argument("--output", help="Path to save predictions CSV")
    parser.add_argument("--interactive", action="store_true", help="Interactive mode")
    parser.add_argument("--json", help="JSON string of features for single prediction")
    args = parser.parse_args()
    
    model = load_model()
    print(f"Model loaded: {{type(model).__name__}}")
    
    if args.interactive:
        interactive_mode(model)
    elif args.json:
        features = json.loads(args.json)
        result = predict_single(model, features)
        print(json.dumps(result, indent=2))
    elif args.input:
        predict_batch(model, args.input, args.output or "predictions.csv")
    else:
        parser.print_help()
'''
    return script


def generate_dockerfile(model_filename, python_version="3.11"):
    """Generate a Dockerfile for model serving."""
    return f'''FROM python:{python_version}-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY {model_filename} .
COPY predict.py .
COPY serve.py .

EXPOSE 8000

CMD ["python", "serve.py"]
'''


def generate_serve_script(model_filename, feature_names, port=8000):
    """Generate a Flask-based model serving script."""
    features_list = json.dumps(feature_names)
    
    return f'''#!/usr/bin/env python3
"""AutoML Model Serving API — Generated automatically."""

from flask import Flask, request, jsonify
import joblib
import pandas as pd
import numpy as np

app = Flask(__name__)
model = joblib.load("{model_filename}")
FEATURE_NAMES = {features_list}


@app.route("/predict", methods=["POST"])
def predict():
    """Single prediction endpoint."""
    data = request.json
    if not data:
        return jsonify({{"error": "No JSON data provided"}}), 400
    
    row = {{f: data.get(f, 0.0) for f in FEATURE_NAMES}}
    X = pd.DataFrame([row], columns=FEATURE_NAMES)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    prediction = model.predict(X)[0]
    result = {{"prediction": float(prediction)}}
    
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(X)[0]
        result["probabilities"] = {{str(i): round(float(p), 4) for i, p in enumerate(proba)}}
        result["confidence"] = round(float(max(proba)), 4)
    
    return jsonify(result)


@app.route("/batch-predict", methods=["POST"])
def batch_predict():
    """Batch prediction endpoint (JSON array)."""
    data = request.json
    if not data or not isinstance(data, list):
        return jsonify({{"error": "Expected JSON array"}}), 400
    
    df = pd.DataFrame(data)
    X = df.reindex(columns=FEATURE_NAMES, fill_value=0.0)
    X = X.apply(pd.to_numeric, errors="coerce").fillna(0)
    
    predictions = model.predict(X)
    results = [{{"prediction": float(p)}} for p in predictions]
    
    if hasattr(model, "predict_proba"):
        probas = model.predict_proba(X)
        for i, p in enumerate(probas):
            results[i]["confidence"] = round(float(max(p)), 4)
    
    return jsonify({{"predictions": results, "count": len(results)}})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({{"status": "healthy", "model": type(model).__name__}})


@app.route("/info", methods=["GET"])
def info():
    return jsonify({{
        "model_type": type(model).__name__,
        "features": FEATURE_NAMES,
        "n_features": len(FEATURE_NAMES),
    }})


if __name__ == "__main__":
    print(f"Serving model: {{type(model).__name__}}")
    print(f"Features: {{len(FEATURE_NAMES)}}")
    print(f"Endpoints: /predict, /batch-predict, /health, /info")
    app.run(host="0.0.0.0", port={port})
'''


def generate_requirements():
    """Generate requirements.txt for deployment."""
    return """flask>=2.0
pandas>=1.5
numpy>=1.21
scikit-learn>=1.2
xgboost>=1.7
lightgbm>=3.3
joblib>=1.2
"""


def generate_api_docs(feature_names, problem_type, model_name):
    """Generate API documentation in Markdown."""
    features_json = json.dumps({f: "0.0" for f in feature_names[:5]}, indent=4)
    
    return f"""# AutoML Prediction API Documentation

## Model Info
- **Model**: {model_name}
- **Problem Type**: {problem_type}
- **Features**: {len(feature_names)}

## Endpoints

### POST /predict
Single prediction.

**Request:**
```json
{features_json}
```

**Response:**
```json
{{
    "prediction": 1.0,
    "confidence": 0.95,
    "probabilities": {{"0": 0.05, "1": 0.95}}
}}
```

### POST /batch-predict
Batch predictions.

**Request:** JSON array of feature objects.

### GET /health
Health check.

### GET /info
Model and feature information.

## cURL Examples

```bash
# Single prediction
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{features_json}'

# Health check
curl http://localhost:8000/health
```
"""


def export_deployment_package(model_path, feature_names, problem_type, model_name, output_dir):
    """
    Generate a complete deployment package.
    
    Returns:
        dict: paths to all generated files
    """
    deploy_dir = os.path.join(output_dir, 'deployment')
    os.makedirs(deploy_dir, exist_ok=True)
    
    model_filename = 'model.pkl'
    
    # Copy model
    import shutil
    dst_model = os.path.join(deploy_dir, model_filename)
    if os.path.exists(model_path):
        shutil.copy2(model_path, dst_model)
    
    files = {}
    
    # Prediction script
    script = generate_prediction_script(model_filename, feature_names, problem_type)
    predict_path = os.path.join(deploy_dir, 'predict.py')
    with open(predict_path, 'w') as f:
        f.write(script)
    files['predict_script'] = predict_path
    
    # Serve script
    serve = generate_serve_script(model_filename, feature_names)
    serve_path = os.path.join(deploy_dir, 'serve.py')
    with open(serve_path, 'w') as f:
        f.write(serve)
    files['serve_script'] = serve_path
    
    # Dockerfile
    docker = generate_dockerfile(model_filename)
    docker_path = os.path.join(deploy_dir, 'Dockerfile')
    with open(docker_path, 'w') as f:
        f.write(docker)
    files['dockerfile'] = docker_path
    
    # Requirements
    reqs = generate_requirements()
    reqs_path = os.path.join(deploy_dir, 'requirements.txt')
    with open(reqs_path, 'w') as f:
        f.write(reqs)
    files['requirements'] = reqs_path
    
    # API docs
    docs = generate_api_docs(feature_names, problem_type, model_name)
    docs_path = os.path.join(deploy_dir, 'API_DOCS.md')
    with open(docs_path, 'w') as f:
        f.write(docs)
    files['api_docs'] = docs_path
    
    return {
        'deploy_dir': deploy_dir,
        'files': files,
        'model_filename': model_filename,
    }
