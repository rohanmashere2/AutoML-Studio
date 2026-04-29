"""
Anomaly Detection Engine — Ensemble-based anomaly detection with consensus voting.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM


def detect_anomalies(X, contamination=0.05):
    """Run ensemble anomaly detection and return consensus-voted anomalies."""
    if isinstance(X, pd.DataFrame):
        feature_names = X.columns.tolist()
        X_arr = X.values
    else:
        feature_names = [f'f_{i}' for i in range(X.shape[1])]
        X_arr = X
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(np.nan_to_num(X_arr, nan=0.0))
    n = X_scaled.shape[0]
    
    # Collect predictions from multiple detectors
    votes = np.zeros(n, dtype=int)
    detector_results = {}
    
    # 1. Isolation Forest
    try:
        iso = IsolationForest(contamination=contamination, random_state=42, n_estimators=200)
        preds = iso.fit_predict(X_scaled)
        scores = iso.decision_function(X_scaled)
        anomaly_mask = preds == -1
        votes[anomaly_mask] += 1
        detector_results['Isolation Forest'] = {
            'n_anomalies': int(anomaly_mask.sum()),
            'pct': round(anomaly_mask.mean() * 100, 2)
        }
    except Exception:
        pass
    
    # 2. Local Outlier Factor
    try:
        lof = LocalOutlierFactor(n_neighbors=min(20, n - 1), contamination=contamination)
        preds = lof.fit_predict(X_scaled)
        lof_scores = -lof.negative_outlier_factor_
        anomaly_mask = preds == -1
        votes[anomaly_mask] += 1
        detector_results['Local Outlier Factor'] = {
            'n_anomalies': int(anomaly_mask.sum()),
            'pct': round(anomaly_mask.mean() * 100, 2)
        }
    except Exception:
        pass
    
    # 3. One-Class SVM (for smaller datasets)
    if n <= 10000:
        try:
            svm = OneClassSVM(nu=contamination, kernel='rbf', gamma='scale')
            preds = svm.fit_predict(X_scaled)
            anomaly_mask = preds == -1
            votes[anomaly_mask] += 1
            detector_results['One-Class SVM'] = {
                'n_anomalies': int(anomaly_mask.sum()),
                'pct': round(anomaly_mask.mean() * 100, 2)
            }
        except Exception:
            pass
    
    # 4. Statistical (Z-score based)
    try:
        z_scores = np.abs(X_scaled)
        max_z = z_scores.max(axis=1)
        anomaly_mask = max_z > 3.0
        votes[anomaly_mask] += 1
        detector_results['Z-Score'] = {
            'n_anomalies': int(anomaly_mask.sum()),
            'pct': round(anomaly_mask.mean() * 100, 2)
        }
    except Exception:
        pass
    
    # 5. Mahalanobis distance
    try:
        if X_scaled.shape[1] < X_scaled.shape[0]:
            cov = np.cov(X_scaled.T)
            if np.linalg.det(cov) > 1e-10:
                cov_inv = np.linalg.inv(cov)
                mean = X_scaled.mean(axis=0)
                mahal = np.array([np.sqrt((x - mean) @ cov_inv @ (x - mean).T) for x in X_scaled])
                threshold = np.percentile(mahal, (1 - contamination) * 100)
                anomaly_mask = mahal > threshold
                votes[anomaly_mask] += 1
                detector_results['Mahalanobis'] = {
                    'n_anomalies': int(anomaly_mask.sum()),
                    'pct': round(anomaly_mask.mean() * 100, 2)
                }
    except Exception:
        pass
    
    # 6. HDBSCAN (density-based, handles complex shapes)
    try:
        import hdbscan
        hdb = hdbscan.HDBSCAN(min_cluster_size=max(5, n // 50), prediction_data=False)
        hdb.fit(X_scaled)
        # Points labeled -1 are outliers; outlier_scores_ gives a probability
        hdb_labels = hdb.labels_
        anomaly_mask = hdb_labels == -1
        if anomaly_mask.sum() > 0:
            votes[anomaly_mask] += 1
            detector_results['HDBSCAN'] = {
                'n_anomalies': int(anomaly_mask.sum()),
                'pct': round(anomaly_mask.mean() * 100, 2)
            }
    except ImportError:
        pass
    except Exception:
        pass
    
    # Consensus: anomaly if detected by 2+ methods
    n_detectors = len(detector_results)
    consensus_threshold = max(2, n_detectors // 2)
    is_anomaly = votes >= consensus_threshold
    anomaly_score = votes / max(n_detectors, 1)
    
    # Top anomalous rows
    anomaly_indices = np.where(is_anomaly)[0]
    top_anomalies = []
    sorted_idx = np.argsort(-anomaly_score)[:min(50, len(sorted_idx) if 'sorted_idx' in dir() else 50)]
    sorted_idx = np.argsort(-anomaly_score)[:50]
    
    for idx in sorted_idx:
        if anomaly_score[idx] > 0:
            row_data = {feature_names[j]: round(float(X_arr[idx, j]), 4) for j in range(min(len(feature_names), 10))} if len(feature_names) > 0 else {}
            top_anomalies.append({
                'index': int(idx),
                'score': round(float(anomaly_score[idx]), 3),
                'votes': int(votes[idx]),
                'values': row_data
            })
    
    # Score distribution
    score_dist = {
        'clean': int((anomaly_score == 0).sum()),
        'suspicious': int(((anomaly_score > 0) & (~is_anomaly)).sum()),
        'anomaly': int(is_anomaly.sum())
    }
    
    return {
        'n_anomalies': int(is_anomaly.sum()),
        'anomaly_pct': round(is_anomaly.mean() * 100, 2),
        'detectors': detector_results,
        'consensus_threshold': consensus_threshold,
        'top_anomalies': top_anomalies[:20],
        'score_distribution': score_dist,
        'anomaly_labels': is_anomaly.astype(int).tolist(),
        'anomaly_scores': anomaly_score.tolist(),
    }
