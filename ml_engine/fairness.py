"""
AI Fairness & Bias Auditing Engine — Auto-detect bias and compute fairness metrics.
"""

import re
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix


SENSITIVE_PATTERNS = {
    'gender': re.compile(r'(gender|sex|male|female)', re.I),
    'race': re.compile(r'(race|ethnicity|ethnic|color)', re.I),
    'age': re.compile(r'(^age$|age_group|age_range)', re.I),
    'religion': re.compile(r'(religion|faith|religious)', re.I),
    'nationality': re.compile(r'(nationality|country|nation|citizen)', re.I),
    'disability': re.compile(r'(disability|disabled|handicap)', re.I),
    'marital': re.compile(r'(marital|married|spouse|marriage)', re.I),
}


def detect_sensitive_columns(df):
    """Auto-detect likely sensitive/protected attribute columns."""
    sensitive = []
    for col in df.columns:
        col_str = str(col)
        for category, pattern in SENSITIVE_PATTERNS.items():
            if pattern.search(col_str):
                sensitive.append({'column': col, 'category': category, 'n_groups': int(df[col].nunique())})
                break
    return sensitive


def audit_fairness(y_true, y_pred, sensitive_features, feature_names=None):
    """Compute comprehensive fairness metrics for each sensitive attribute."""
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    
    reports = []
    
    for i, sens_col in enumerate(sensitive_features.T if hasattr(sensitive_features, 'T') else [sensitive_features]):
        col_name = feature_names[i] if feature_names and i < len(feature_names) else f'sensitive_{i}'
        sens = np.array(sens_col)
        
        groups = np.unique(sens[~pd.isna(sens)])
        if len(groups) < 2 or len(groups) > 20:
            continue
        
        group_metrics = []
        group_positive_rates = {}
        group_tprs = {}
        group_fprs = {}
        
        for group in groups:
            mask = sens == group
            n_group = mask.sum()
            if n_group < 5:
                continue
            
            yt = y_true[mask]
            yp = y_pred[mask]
            
            # Basic rates
            positive_rate = yp.mean() if len(yp) > 0 else 0
            accuracy = (yt == yp).mean() if len(yt) > 0 else 0
            
            # TPR, FPR
            tp = ((yt == 1) & (yp == 1)).sum()
            fn = ((yt == 1) & (yp == 0)).sum()
            fp = ((yt == 0) & (yp == 1)).sum()
            tn = ((yt == 0) & (yp == 0)).sum()
            
            tpr = tp / max(tp + fn, 1)
            fpr = fp / max(fp + tn, 1)
            precision = tp / max(tp + fp, 1)
            
            group_positive_rates[str(group)] = positive_rate
            group_tprs[str(group)] = tpr
            group_fprs[str(group)] = fpr
            
            group_metrics.append({
                'group': str(group),
                'size': int(n_group),
                'positive_rate': round(positive_rate, 4),
                'accuracy': round(accuracy, 4),
                'tpr': round(tpr, 4),
                'fpr': round(fpr, 4),
                'precision': round(precision, 4),
            })
        
        if len(group_metrics) < 2:
            continue
        
        # Compute fairness metrics
        pos_rates = [g['positive_rate'] for g in group_metrics]
        tprs = [g['tpr'] for g in group_metrics]
        fprs = [g['fpr'] for g in group_metrics]
        
        # Demographic Parity Difference
        dp_diff = max(pos_rates) - min(pos_rates)
        
        # Disparate Impact Ratio (80% rule)
        min_rate = min(pos_rates) if min(pos_rates) > 0 else 0.001
        max_rate = max(pos_rates) if max(pos_rates) > 0 else 0.001
        disparate_impact = min_rate / max_rate
        
        # Equalized Odds Difference
        eq_odds_tpr = max(tprs) - min(tprs) if tprs else 0
        eq_odds_fpr = max(fprs) - min(fprs) if fprs else 0
        
        # Overall fairness score (0-100, higher = fairer)
        fairness_score = 100
        if dp_diff > 0.1: fairness_score -= 25
        if dp_diff > 0.2: fairness_score -= 25
        if disparate_impact < 0.8: fairness_score -= 20
        if eq_odds_tpr > 0.1: fairness_score -= 15
        if eq_odds_fpr > 0.1: fairness_score -= 15
        fairness_score = max(0, fairness_score)
        
        # Generate recommendations
        mitigations = []
        if dp_diff > 0.1:
            mitigations.append({
                'issue': 'Demographic Parity Violation',
                'severity': 'high' if dp_diff > 0.2 else 'medium',
                'suggestion': 'Consider reweighting training samples or adjusting prediction thresholds per group.'
            })
        if disparate_impact < 0.8:
            mitigations.append({
                'issue': '80% Rule Violation (Disparate Impact)',
                'severity': 'high',
                'suggestion': 'The model may have discriminatory impact. Consider removing proxy features or applying fairness constraints.'
            })
        if eq_odds_tpr > 0.15:
            mitigations.append({
                'issue': 'Unequal True Positive Rates',
                'severity': 'medium',
                'suggestion': 'The model is better at detecting positives for some groups. Consider calibration or equalized odds post-processing.'
            })
        
        grade = 'A' if fairness_score >= 90 else 'B' if fairness_score >= 75 else 'C' if fairness_score >= 60 else 'D' if fairness_score >= 40 else 'F'
        
        reports.append({
            'feature': col_name,
            'n_groups': len(group_metrics),
            'groups': group_metrics,
            'demographic_parity_diff': round(dp_diff, 4),
            'disparate_impact_ratio': round(disparate_impact, 4),
            'equalized_odds_tpr_diff': round(eq_odds_tpr, 4),
            'equalized_odds_fpr_diff': round(eq_odds_fpr, 4),
            'fairness_score': fairness_score,
            'grade': grade,
            'status': 'fair' if fairness_score >= 75 else 'warning' if fairness_score >= 50 else 'unfair',
            'mitigations': mitigations,
        })
    
    overall_score = np.mean([r['fairness_score'] for r in reports]) if reports else 100
    
    return {
        'reports': reports,
        'overall_fairness_score': round(overall_score, 1),
        'overall_grade': 'A' if overall_score >= 90 else 'B' if overall_score >= 75 else 'C' if overall_score >= 60 else 'D' if overall_score >= 40 else 'F',
        'n_sensitive_features': len(reports),
        'has_bias': any(r['status'] == 'unfair' for r in reports),
    }
