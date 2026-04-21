"""
Smart AutoEDA Dashboard — Comprehensive automated exploratory data analysis
with AI-driven natural language narratives and actionable recommendations.
"""

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def run_auto_eda(df, target_column=None):
    """Generate comprehensive EDA report with AI-driven insights."""
    report = {
        'overview': _dataset_overview(df),
        'distributions': _column_distributions(df, target_column),
        'correlations': _correlation_analysis(df),
        'missing_patterns': _missing_analysis(df),
        'outliers': _outlier_analysis(df),
        'insights': [],
    }

    if target_column and target_column in df.columns:
        report['target_analysis'] = _target_analysis(df, target_column)

    report['insights'] = _generate_insights(df, report, target_column)
    report['narrative'] = generate_narrative_report(df, report, target_column)
    report['actionable_recommendations'] = _actionable_recommendations(report)

    return report


# ── Narrative Report Generator ────────────────────────────────

def generate_narrative_report(df, report=None, target_column=None):
    """Generate a natural language narrative summary of the dataset.

    Returns a multi-paragraph string that reads like a data scientist's notes.
    """
    if report is None:
        report = run_auto_eda(df, target_column)

    overview = report.get('overview', {})
    n_rows = overview.get('n_rows', len(df))
    n_cols = overview.get('n_cols', len(df.columns))
    n_numeric = overview.get('n_numeric', 0)
    n_categorical = overview.get('n_categorical', 0)
    missing_pct = overview.get('total_missing_pct', 0)
    dup_pct = overview.get('duplicated_pct', 0)

    paragraphs = []

    # Paragraph 1: Dataset Overview
    p1 = (
        f"This dataset contains **{n_rows:,} records** across **{n_cols} features** "
        f"({n_numeric} numeric, {n_categorical} categorical). "
    )
    if missing_pct > 0:
        missing_info = report.get('missing_patterns', {})
        worst_cols = missing_info.get('columns', [])
        if worst_cols:
            top3 = ', '.join(f"`{c['column']}` ({c['pct']}%)" for c in worst_cols[:3])
            p1 += f"Missing data affects {missing_pct:.1f}% of cells overall, concentrated in {top3}. "
        else:
            p1 += f"Missing data is minimal at {missing_pct:.1f}%. "
    else:
        p1 += "The dataset is fully complete with no missing values. "
    if dup_pct > 1:
        p1 += f"**{dup_pct:.1f}% duplicate rows** were detected — these should be removed before training."
    paragraphs.append(p1)

    # Paragraph 2: Key Findings (correlations, skew, outliers)
    findings = []
    corrs = report.get('correlations', {})
    top_corrs = corrs.get('top_correlations', [])
    if top_corrs:
        strongest = top_corrs[0]
        findings.append(
            f"`{strongest['feature_1']}` and `{strongest['feature_2']}` show a "
            f"{'strong' if abs(strongest['correlation']) > 0.7 else 'moderate'} correlation "
            f"of **{strongest['correlation']:.2f}**"
        )

    # Skewed features
    skewed = [d for d in report.get('distributions', [])
              if d.get('type') == 'numeric' and abs(d.get('skewness', 0)) > 2]
    if skewed:
        names = ', '.join(f"`{d['column']}`" for d in skewed[:3])
        findings.append(f"{names} {'is' if len(skewed) == 1 else 'are'} heavily skewed and may benefit from log/sqrt transformation")

    # Outliers
    outliers = report.get('outliers', {})
    outlier_cols = outliers.get('columns', [])
    high_outlier = [o for o in outlier_cols if o['pct'] > 5]
    if high_outlier:
        names = ', '.join(f"`{o['column']}`" for o in high_outlier[:3])
        findings.append(f"{names} {'has' if len(high_outlier) == 1 else 'have'} significant outliers (>5% of values)")

    if findings:
        p2 = "**Key findings:** " + "; ".join(findings) + "."
        paragraphs.append(p2)

    # Paragraph 3: Target analysis
    target = report.get('target_analysis', {})
    if target:
        if target.get('type') == 'classification':
            classes = target.get('class_distribution', [])
            if classes:
                class_str = ', '.join(f"{c['label']} ({c.get('pct', 0):.0f}%)" for c in classes[:5])
                p3 = f"The target `{target.get('column', '')}` is a **{target['type']}** problem with classes: {class_str}. "
                if target.get('is_imbalanced'):
                    p3 += f"The classes are **imbalanced** (ratio {target.get('imbalance_ratio', '?')}:1) — consider SMOTE, class weighting, or stratified sampling."
                else:
                    p3 += "Class distribution is reasonably balanced."
                paragraphs.append(p3)
        elif target.get('type') == 'regression':
            stats = target.get('stats', {})
            p3 = (
                f"The target `{target.get('column', '')}` is a **continuous variable** "
                f"(mean={stats.get('mean', 0):.2f}, std={stats.get('std', 0):.2f}). "
            )
            skew = stats.get('skew', 0)
            if abs(skew) > 1:
                p3 += f"It is {'right' if skew > 0 else 'left'}-skewed (skewness={skew:.2f}) — a log transform may help normalize it."
            paragraphs.append(p3)

        # Top correlated features with target
        target_corrs = target.get('top_correlations', [])
        if target_corrs:
            top3 = target_corrs[:3]
            feat_str = ', '.join(f"`{f['feature']}` (r={f['correlation']:.2f})" for f in top3)
            paragraphs.append(f"**Most predictive features:** {feat_str} — these are likely to be the strongest drivers in your model.")

    return "\n\n".join(paragraphs)


# ── Actionable Recommendations ────────────────────────────────

def _actionable_recommendations(report):
    """Generate specific, executable recommendations from insights."""
    recs = []

    # Missing data recommendations
    missing = report.get('missing_patterns', {})
    for col_info in missing.get('columns', [])[:5]:
        col = col_info['column']
        pct = col_info['pct']
        if pct > 60:
            recs.append({
                'priority': 'high',
                'category': 'Missing Data',
                'action': f"Drop column `{col}` — {pct}% missing values make reliable imputation impossible.",
                'code': f"df = df.drop(columns=['{col}'])",
            })
        elif pct > 5:
            recs.append({
                'priority': 'medium',
                'category': 'Missing Data',
                'action': f"Impute `{col}` — {pct}% missing. Use median for skewed data, mean for normal.",
                'code': f"df['{col}'] = df['{col}'].fillna(df['{col}'].median())",
            })

    # Correlation-based recommendations
    corrs = report.get('correlations', {})
    for pair in corrs.get('top_correlations', [])[:3]:
        if abs(pair['correlation']) > 0.95:
            recs.append({
                'priority': 'high',
                'category': 'Multicollinearity',
                'action': f"Remove `{pair['feature_2']}` — it's {pair['correlation']:.0%} correlated with `{pair['feature_1']}`, adding redundancy.",
                'code': f"df = df.drop(columns=['{pair['feature_2']}'])",
            })
        elif abs(pair['correlation']) > 0.8:
            recs.append({
                'priority': 'medium',
                'category': 'Multicollinearity',
                'action': f"Consider removing either `{pair['feature_1']}` or `{pair['feature_2']}` (correlation={pair['correlation']:.2f}).",
            })

    # Outlier recommendations
    for o in report.get('outliers', {}).get('columns', [])[:3]:
        if o['pct'] > 5:
            recs.append({
                'priority': 'medium',
                'category': 'Outliers',
                'action': f"Cap outliers in `{o['column']}` — {o['n_outliers']} values ({o['pct']}%) outside IQR bounds [{o['lower_bound']}, {o['upper_bound']}].",
                'code': f"df['{o['column']}'] = df['{o['column']}'].clip(lower={o['lower_bound']}, upper={o['upper_bound']})",
            })

    # Skewness recommendations
    for dist in report.get('distributions', []):
        if dist.get('type') == 'numeric' and abs(dist.get('skewness', 0)) > 2:
            col = dist['column']
            recs.append({
                'priority': 'low',
                'category': 'Distribution',
                'action': f"Apply log transform to `{col}` — skewness={dist['skewness']:.2f} may hurt distance-based models.",
                'code': f"df['{col}'] = np.log1p(df['{col}'])",
            })

    # Target recommendations
    target = report.get('target_analysis', {})
    if target.get('is_imbalanced'):
        recs.append({
            'priority': 'high',
            'category': 'Class Imbalance',
            'action': f"Target `{target.get('column', '')}` is imbalanced (ratio {target.get('imbalance_ratio', '?')}:1). Use SMOTE or class_weight='balanced'.",
        })

    return recs


# ── Core Analysis Functions ───────────────────────────────────

def _dataset_overview(df):
    """General dataset statistics."""
    mem = df.memory_usage(deep=True).sum()
    return {
        'n_rows': len(df), 'n_cols': len(df.columns),
        'memory_mb': round(mem / 1024 / 1024, 2),
        'n_numeric': len(df.select_dtypes(include='number').columns),
        'n_categorical': len(df.select_dtypes(include='object').columns),
        'n_datetime': len(df.select_dtypes(include='datetime').columns),
        'total_missing': int(df.isnull().sum().sum()),
        'total_missing_pct': round(df.isnull().mean().mean() * 100, 2),
        'duplicated_rows': int(df.duplicated().sum()),
        'duplicated_pct': round(df.duplicated().mean() * 100, 2),
    }


def _column_distributions(df, target_column=None):
    """Compute distribution stats for each column."""
    distributions = []

    for col in df.columns:
        info = {
            'column': col, 'dtype': str(df[col].dtype),
            'n_missing': int(df[col].isnull().sum()),
            'missing_pct': round(df[col].isnull().mean() * 100, 2),
            'n_unique': int(df[col].nunique()),
            'is_target': col == target_column,
        }

        if pd.api.types.is_numeric_dtype(df[col]):
            vals = df[col].dropna()
            if len(vals) > 0:
                info.update({
                    'type': 'numeric',
                    'mean': round(float(vals.mean()), 4),
                    'median': round(float(vals.median()), 4),
                    'std': round(float(vals.std()), 4),
                    'min': round(float(vals.min()), 4),
                    'max': round(float(vals.max()), 4),
                    'skewness': round(float(vals.skew()), 4),
                    'kurtosis': round(float(vals.kurtosis()), 4),
                    'q25': round(float(vals.quantile(0.25)), 4),
                    'q75': round(float(vals.quantile(0.75)), 4),
                })
                # Histogram bins
                try:
                    counts, edges = np.histogram(vals, bins=min(30, int(np.sqrt(len(vals)))))
                    info['histogram'] = {
                        'counts': counts.tolist(),
                        'edges': [round(float(e), 4) for e in edges],
                    }
                except Exception:
                    pass

                # Normality test
                if len(vals) >= 20 and len(vals) <= 5000:
                    try:
                        _, p_value = sp_stats.normaltest(vals)
                        info['is_normal'] = bool(p_value > 0.05)
                        info['normality_p'] = round(float(p_value), 6)
                    except Exception:
                        pass

                # Near-zero variance check
                if vals.std() < 1e-6:
                    info['near_zero_variance'] = True

                # Bimodal detection (Hartigan's dip test approximation)
                try:
                    if len(vals) > 50:
                        hist_counts, _ = np.histogram(vals, bins=20)
                        peaks = sum(1 for i in range(1, len(hist_counts) - 1)
                                    if hist_counts[i] > hist_counts[i-1] and hist_counts[i] > hist_counts[i+1])
                        if peaks >= 2:
                            info['is_multimodal'] = True
                            info['n_modes'] = peaks
                except Exception:
                    pass
        else:
            info['type'] = 'categorical'
            vc = df[col].value_counts().head(15)
            info['top_values'] = [{'value': str(v), 'count': int(c), 'pct': round(c / len(df) * 100, 2)} for v, c in vc.items()]

            # High-cardinality check
            if df[col].nunique() > 50:
                info['high_cardinality'] = True
            # Potential ID column
            if df[col].nunique() == len(df):
                info['potential_id'] = True
            # Potential datetime string
            try:
                sample = df[col].dropna().head(20)
                parsed = pd.to_datetime(sample, errors='coerce', infer_datetime_format=True)
                if parsed.notna().sum() > 15:
                    info['potential_datetime'] = True
            except Exception:
                pass

        distributions.append(info)

    return distributions


def _correlation_analysis(df):
    """Compute correlation matrix and find top correlations."""
    numeric = df.select_dtypes(include='number')
    if numeric.shape[1] < 2:
        return {'matrix': [], 'top_correlations': [], 'columns': []}

    # Use at most 50 columns
    if numeric.shape[1] > 50:
        numeric = numeric.iloc[:, :50]

    corr = numeric.corr()

    # Find top correlations
    top_corrs = []
    cols = corr.columns.tolist()
    for i in range(len(cols)):
        for j in range(i + 1, len(cols)):
            val = corr.iloc[i, j]
            if not np.isnan(val) and abs(val) > 0.3:
                top_corrs.append({
                    'feature_1': cols[i], 'feature_2': cols[j],
                    'correlation': round(float(val), 4),
                    'strength': 'strong' if abs(val) > 0.7 else 'moderate'
                })

    top_corrs.sort(key=lambda x: abs(x['correlation']), reverse=True)

    # Convert matrix to serializable format
    matrix_data = []
    display_cols = cols[:30]
    for col in display_cols:
        row = [round(float(corr.loc[col, c]), 3) if not np.isnan(corr.loc[col, c]) else 0 for c in display_cols]
        matrix_data.append(row)

    return {
        'matrix': matrix_data,
        'columns': display_cols,
        'top_correlations': top_corrs[:20],
        'n_strong': sum(1 for c in top_corrs if c['strength'] == 'strong'),
        'n_moderate': sum(1 for c in top_corrs if c['strength'] == 'moderate'),
    }


def _missing_analysis(df):
    """Analyze missing value patterns."""
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    columns = []
    for col, count in missing.items():
        columns.append({
            'column': col, 'count': int(count),
            'pct': round(count / len(df) * 100, 2)
        })

    # Check for systematic patterns (columns missing together)
    patterns = []
    if len(missing) >= 2:
        missing_cols = missing.index.tolist()[:10]
        for i in range(len(missing_cols)):
            for j in range(i + 1, len(missing_cols)):
                c1, c2 = missing_cols[i], missing_cols[j]
                both_missing = (df[c1].isnull() & df[c2].isnull()).sum()
                either_missing = (df[c1].isnull() | df[c2].isnull()).sum()
                if either_missing > 0:
                    overlap = both_missing / either_missing
                    if overlap > 0.5:
                        patterns.append({
                            'columns': [c1, c2], 'overlap': round(overlap, 3),
                            'both_missing': int(both_missing)
                        })

    return {
        'total_missing_cells': int(df.isnull().sum().sum()),
        'columns_with_missing': len(columns),
        'columns': columns[:20],
        'patterns': patterns[:10],
    }


def _outlier_analysis(df):
    """Detect outliers using IQR method."""
    outliers = []
    for col in df.select_dtypes(include='number').columns:
        vals = df[col].dropna()
        if len(vals) < 10:
            continue
        q1, q3 = vals.quantile(0.25), vals.quantile(0.75)
        iqr = q3 - q1
        if iqr == 0:
            continue
        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr
        n_outliers = ((vals < lower) | (vals > upper)).sum()
        if n_outliers > 0:
            outliers.append({
                'column': col, 'n_outliers': int(n_outliers),
                'pct': round(n_outliers / len(vals) * 100, 2),
                'lower_bound': round(float(lower), 4),
                'upper_bound': round(float(upper), 4),
                'min': round(float(vals.min()), 4),
                'max': round(float(vals.max()), 4),
                'q1': round(float(q1), 4), 'q3': round(float(q3), 4),
            })

    outliers.sort(key=lambda x: x['n_outliers'], reverse=True)
    return {'columns': outliers[:20], 'total_outlier_columns': len(outliers)}


def _target_analysis(df, target):
    """Analyze the target variable in depth."""
    result = {'column': target}

    if pd.api.types.is_numeric_dtype(df[target]):
        vals = df[target].dropna()
        if df[target].nunique() <= 20:
            result['type'] = 'classification'
            vc = df[target].value_counts()
            result['class_distribution'] = [{'label': str(k), 'count': int(v), 'pct': round(v / len(df) * 100, 2)} for k, v in vc.items()]
            result['is_imbalanced'] = bool((vc.max() / vc.min()) > 3) if vc.min() > 0 else True
            result['imbalance_ratio'] = round(float(vc.max() / max(vc.min(), 1)), 2)
        else:
            result['type'] = 'regression'
            result['stats'] = {
                'mean': round(float(vals.mean()), 4),
                'median': round(float(vals.median()), 4),
                'std': round(float(vals.std()), 4),
                'skew': round(float(vals.skew()), 4),
            }

        # Correlation with other features
        correlations = []
        for col in df.select_dtypes(include='number').columns:
            if col == target:
                continue
            corr = df[target].corr(df[col])
            if not np.isnan(corr):
                correlations.append({'feature': col, 'correlation': round(float(corr), 4)})
        correlations.sort(key=lambda x: abs(x['correlation']), reverse=True)
        result['top_correlations'] = correlations[:15]

        # Check for potential data leakage (suspiciously high correlation)
        if correlations and abs(correlations[0]['correlation']) > 0.95:
            result['leakage_warning'] = {
                'feature': correlations[0]['feature'],
                'correlation': correlations[0]['correlation'],
                'message': f"Column '{correlations[0]['feature']}' has suspiciously high correlation with target ({correlations[0]['correlation']:.3f}). This may indicate data leakage."
            }
    else:
        result['type'] = 'classification'
        vc = df[target].value_counts()
        result['class_distribution'] = [{'label': str(k), 'count': int(v)} for k, v in vc.items()]

    return result


def _generate_insights(df, report, target_column):
    """Generate AI-like natural language insights from the data."""
    insights = []

    # Missing data insights
    missing = report.get('missing_patterns', {})
    if missing.get('total_missing_cells', 0) > 0:
        cols = missing.get('columns', [])
        worst = cols[0] if cols else None
        if worst and worst['pct'] > 30:
            insights.append({
                'icon': '🩹', 'category': 'Missing Data', 'severity': 'high',
                'text': f"Column '{worst['column']}' has {worst['pct']}% missing values. Consider dropping it or using advanced imputation."
            })
        # Co-missing patterns
        patterns = missing.get('patterns', [])
        if patterns:
            p = patterns[0]
            insights.append({
                'icon': '🔗', 'category': 'Co-Missing Pattern', 'severity': 'medium',
                'text': f"Columns {p['columns'][0]} and {p['columns'][1]} tend to be missing together ({p['overlap']*100:.0f}% overlap) — they may share a common data source issue."
            })

    # Correlation insights
    corrs = report.get('correlations', {})
    top = corrs.get('top_correlations', [])
    if top:
        if abs(top[0]['correlation']) > 0.9:
            insights.append({
                'icon': '🔗', 'category': 'Multicollinearity', 'severity': 'high',
                'text': f"'{top[0]['feature_1']}' and '{top[0]['feature_2']}' are highly correlated ({top[0]['correlation']:.2f}). Consider removing one to reduce redundancy."
            })

    # Distribution insights
    for dist in report.get('distributions', []):
        if dist.get('type') == 'numeric':
            skew = dist.get('skewness', 0)
            if abs(skew) > 2:
                direction = "right" if skew > 0 else "left"
                insights.append({
                    'icon': '📐', 'category': 'Skewed Distribution', 'severity': 'medium',
                    'text': f"'{dist['column']}' is heavily {direction}-skewed (skewness: {skew:.2f}). A log transform may help."
                })

            # Near-zero variance
            if dist.get('near_zero_variance'):
                insights.append({
                    'icon': '📉', 'category': 'Near-Zero Variance', 'severity': 'medium',
                    'text': f"'{dist['column']}' has near-zero variance — it provides almost no information for modeling. Consider removing it."
                })

            # Multimodal distribution
            if dist.get('is_multimodal'):
                insights.append({
                    'icon': '🏔️', 'category': 'Multimodal Distribution', 'severity': 'low',
                    'text': f"'{dist['column']}' appears to have {dist.get('n_modes', 2)} modes — this may indicate distinct sub-populations in your data."
                })

        elif dist.get('type') == 'categorical':
            # High cardinality
            if dist.get('high_cardinality'):
                insights.append({
                    'icon': '🏷️', 'category': 'High Cardinality', 'severity': 'medium',
                    'text': f"'{dist['column']}' has {dist['n_unique']} unique values — consider target encoding or embedding instead of one-hot encoding."
                })

            # Potential ID column
            if dist.get('potential_id'):
                insights.append({
                    'icon': '🆔', 'category': 'Potential ID Column', 'severity': 'high',
                    'text': f"'{dist['column']}' has all unique values — likely an ID column. Remove it before training."
                })

            # Potential datetime string
            if dist.get('potential_datetime'):
                insights.append({
                    'icon': '📅', 'category': 'Datetime String', 'severity': 'medium',
                    'text': f"'{dist['column']}' looks like datetime stored as text. Parse it to extract temporal features (day_of_week, month, etc.)."
                })

    # Outlier insights
    outliers = report.get('outliers', {})
    for o in outliers.get('columns', [])[:3]:
        if o['pct'] > 5:
            insights.append({
                'icon': '📊', 'category': 'Outliers', 'severity': 'medium',
                'text': f"'{o['column']}' has {o['n_outliers']} outliers ({o['pct']}%). Range: [{o['min']}, {o['max']}] vs IQR bounds [{o['lower_bound']}, {o['upper_bound']}]."
            })

    # Target insights
    target = report.get('target_analysis', {})
    if target.get('is_imbalanced'):
        insights.append({
            'icon': '⚖️', 'category': 'Class Imbalance', 'severity': 'high',
            'text': f"Target '{target.get('column')}' is imbalanced (ratio: {target.get('imbalance_ratio')}:1). SMOTE or class weighting recommended."
        })

    # Data leakage warning
    if target.get('leakage_warning'):
        lw = target['leakage_warning']
        insights.append({
            'icon': '⚠️', 'category': 'Potential Data Leakage', 'severity': 'critical',
            'text': lw['message']
        })

    # Size insights
    overview = report.get('overview', {})
    if overview.get('n_rows', 0) < 100:
        insights.append({
            'icon': '📏', 'category': 'Small Dataset', 'severity': 'medium',
            'text': f"Only {overview['n_rows']} rows detected. Consider synthetic data augmentation for better model performance."
        })

    if overview.get('duplicated_pct', 0) > 5:
        insights.append({
            'icon': '🔄', 'category': 'Duplicates', 'severity': 'medium',
            'text': f"{overview['duplicated_pct']}% duplicate rows detected. These will be removed during cleaning."
        })

    return insights
