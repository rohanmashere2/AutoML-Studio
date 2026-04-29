"""
Data Quality Scoring Engine — Compute 0-100 quality score per column and overall.
"""

import numpy as np
import pandas as pd


def compute_data_quality(df, target_column=None):
    """Compute data quality scores for each column and overall dataset."""
    column_scores = []
    
    for col in df.columns:
        if col == target_column:
            continue
        
        score = _score_column(df[col], col)
        column_scores.append(score)
    
    # Overall score
    if column_scores:
        overall = np.mean([s['quality_score'] for s in column_scores])
    else:
        overall = 100.0
    
    return {
        'overall_score': round(overall, 1),
        'overall_grade': _grade(overall),
        'column_scores': sorted(column_scores, key=lambda x: x['quality_score']),
        'total_columns': len(column_scores),
        'high_quality': sum(1 for s in column_scores if s['quality_score'] >= 80),
        'medium_quality': sum(1 for s in column_scores if 50 <= s['quality_score'] < 80),
        'low_quality': sum(1 for s in column_scores if s['quality_score'] < 50),
    }


def _score_column(series, col_name):
    """Score a single column on multiple quality dimensions."""
    n = len(series)
    if n == 0:
        return {'column': col_name, 'quality_score': 0, 'issues': ['Empty column']}
    
    # 1. Completeness (0-100): % non-null
    completeness = (1 - series.isnull().mean()) * 100
    
    # 2. Uniqueness (0-100): penalize if too many or too few unique values
    nunique = series.nunique()
    if pd.api.types.is_numeric_dtype(series):
        uniqueness = min(100, (nunique / max(n, 1)) * 200)  # more unique = better for numeric
    else:
        ratio = nunique / max(n, 1)
        if ratio > 0.95:  # Almost all unique = likely ID
            uniqueness = 30
        elif ratio < 0.001 and n > 100:  # Almost no variance
            uniqueness = 40
        else:
            uniqueness = 80
    
    # 3. Consistency (0-100): type uniformity
    consistency = 100
    if series.dtype == 'object':
        non_null = series.dropna().astype(str)
        if len(non_null) > 0:
            numeric_count = sum(1 for v in non_null.head(50) if _is_numeric(v))
            if 0.2 < numeric_count / min(len(non_null), 50) < 0.8:
                consistency = 40  # Mixed types
    
    # 4. Validity (0-100): check for suspicious values
    validity = 100
    issues = []
    
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 10:
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_count = ((non_null < q1 - 3 * iqr) | (non_null > q3 + 3 * iqr)).sum()
                outlier_pct = outlier_count / len(non_null)
                if outlier_pct > 0.05:
                    validity -= 20
                    issues.append(f'{outlier_count} extreme outliers')
            
            # Check for negative values in likely-positive columns
            if non_null.median() > 0 and (non_null < 0).sum() > 0:
                neg_pct = (non_null < 0).sum() / len(non_null)
                if neg_pct < 0.05:
                    issues.append('Suspicious negative values')
    
    if completeness < 100:
        missing_pct = 100 - completeness
        if missing_pct > 50:
            issues.append(f'{missing_pct:.0f}% missing')
        elif missing_pct > 10:
            issues.append(f'{missing_pct:.0f}% missing')
    
    # Weighted composite score
    quality_score = (
        completeness * 0.35 +
        consistency * 0.25 +
        validity * 0.25 +
        uniqueness * 0.15
    )
    
    # Outlier statistics & skewness for numeric columns
    outlier_pct = 0.0
    skewness = None
    distribution = None
    if pd.api.types.is_numeric_dtype(series):
        non_null = series.dropna()
        if len(non_null) > 10:
            q1, q3 = non_null.quantile(0.25), non_null.quantile(0.75)
            iqr = q3 - q1
            if iqr > 0:
                outlier_mask = (non_null < q1 - 1.5 * iqr) | (non_null > q3 + 1.5 * iqr)
                outlier_pct = round(float(outlier_mask.mean() * 100), 2)
            skewness = round(float(non_null.skew()), 3)
            kurtosis_val = float(non_null.kurtosis())
            if abs(skewness) < 0.5:
                distribution = 'normal'
            elif skewness > 1.5:
                distribution = 'highly_right_skewed'
            elif skewness > 0.5:
                distribution = 'right_skewed'
            elif skewness < -1.5:
                distribution = 'highly_left_skewed'
            else:
                distribution = 'left_skewed'
    
    # Generate actionable recommendations
    recommendations = []
    if completeness < 70:
        recommendations.append({'action': 'drop_or_impute', 'reason': f'{100-completeness:.0f}% missing — consider dropping or advanced imputation (KNN/MICE)'})
    elif completeness < 95:
        recommendations.append({'action': 'impute', 'reason': f'{100-completeness:.0f}% missing — impute with median/mode'})
    if outlier_pct > 5:
        recommendations.append({'action': 'clip_outliers', 'reason': f'{outlier_pct:.1f}% outliers detected — consider winsorizing'})
    if skewness is not None and abs(skewness) > 1.5:
        recommendations.append({'action': 'log_transform', 'reason': f'High skewness ({skewness:.2f}) — apply log/sqrt transform'})
    if uniqueness < 40 and series.dtype == 'object':
        recommendations.append({'action': 'review_column', 'reason': 'Possible ID column or near-constant — may not add predictive value'})
    
    return {
        'column': col_name,
        'dtype': str(series.dtype),
        'quality_score': round(min(100, max(0, quality_score)), 1),
        'grade': _grade(quality_score),
        'completeness': round(completeness, 1),
        'consistency': round(consistency, 1),
        'validity': round(validity, 1),
        'uniqueness': round(uniqueness, 1),
        'outlier_pct': outlier_pct,
        'skewness': skewness,
        'distribution': distribution,
        'issues': issues,
        'recommendations': recommendations,
        'n_missing': int(series.isnull().sum()),
        'n_unique': int(nunique),
    }


def _grade(score):
    if score >= 90: return 'A'
    if score >= 80: return 'B'
    if score >= 70: return 'C'
    if score >= 50: return 'D'
    return 'F'


def _is_numeric(val):
    try:
        float(str(val).replace(',', ''))
        return True
    except (ValueError, TypeError):
        return False
