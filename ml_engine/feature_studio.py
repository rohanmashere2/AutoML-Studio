"""
AutoML Studio — No-Code Feature Engineering Studio
Validates expressions, previews feature distributions, and checks
correlation with target before adding features to the dataset.
"""

import re
import numpy as np
import pandas as pd


class FeatureStudio:
    """No-code feature engineering with validation and preview."""

    SAFE_FUNCTIONS = {
        'log': np.log1p,
        'sqrt': np.sqrt,
        'abs': np.abs,
        'square': np.square,
        'exp': np.exp,
        'round': np.round,
    }

    SAFE_OPERATORS = {'+', '-', '*', '/', '**'}

    def validate_expression(self, expression, df):
        """
        Validate a feature expression against the dataframe.

        Returns:
            dict with is_valid, error message, and parsed info
        """
        try:
            # Check for dangerous patterns
            dangerous = ['import', 'exec', 'eval', 'os.', 'sys.', 'open(', '__', 'lambda']
            expr_lower = expression.lower()
            for d in dangerous:
                if d in expr_lower:
                    return {'is_valid': False, 'error': f'Expression contains forbidden pattern: {d}'}

            # Try parsing
            result = self._evaluate_expression(expression, df.head(5))

            if result is None:
                return {'is_valid': False, 'error': 'Expression returned None'}

            return {
                'is_valid': True,
                'output_dtype': str(result.dtype),
                'sample_values': result.head(5).tolist(),
            }
        except Exception as e:
            return {'is_valid': False, 'error': str(e)}

    def preview_feature(self, expression, df, name=None, target_column=None):
        """
        Compute the new feature and return distribution + correlation stats.

        Returns:
            dict with preview data, statistics, histogram, and target correlation
        """
        try:
            new_col = self._evaluate_expression(expression, df)

            if new_col is None:
                return {'error': 'Expression returned None'}

            result = {
                'name': name or expression,
                'expression': expression,
                'preview': [round(float(v), 4) if pd.notna(v) else None for v in new_col.head(20)],
                'dtype': str(new_col.dtype),
            }

            # Statistics
            if pd.api.types.is_numeric_dtype(new_col):
                vals = new_col.dropna()
                if len(vals) > 0:
                    result['stats'] = {
                        'mean': round(float(vals.mean()), 4),
                        'std': round(float(vals.std()), 4),
                        'min': round(float(vals.min()), 4),
                        'max': round(float(vals.max()), 4),
                        'missing': int(new_col.isnull().sum()),
                        'missing_pct': round(float(new_col.isnull().mean() * 100), 2),
                        'n_unique': int(new_col.nunique()),
                    }

                    # Histogram
                    try:
                        counts, edges = np.histogram(vals, bins=min(30, int(np.sqrt(len(vals)))))
                        result['histogram'] = {
                            'counts': counts.tolist(),
                            'edges': [round(float(e), 4) for e in edges],
                        }
                    except Exception:
                        pass

                    # Target correlation
                    if target_column and target_column in df.columns:
                        try:
                            target = df[target_column]
                            if pd.api.types.is_numeric_dtype(target):
                                corr = float(new_col.corr(target))
                                if not np.isnan(corr):
                                    result['target_correlation'] = round(corr, 4)
                                    result['usefulness'] = (
                                        'strongly_useful' if abs(corr) > 0.3
                                        else 'likely_useful' if abs(corr) > 0.1
                                        else 'low_signal'
                                    )
                                    result['usefulness_label'] = {
                                        'strongly_useful': '🟢 Strongly correlated with target',
                                        'likely_useful': '🟡 Moderately correlated with target',
                                        'low_signal': '⚪ Weak correlation with target',
                                    }.get(result['usefulness'], '')
                        except Exception:
                            pass

            return result
        except Exception as e:
            return {'error': str(e)}

    def add_feature(self, expression, name, df):
        """
        Add the validated feature to the dataframe.

        Returns:
            tuple: (modified_df, feature_info)
        """
        try:
            new_col = self._evaluate_expression(expression, df)
            df = df.copy()
            df[name] = new_col

            return df, {
                'success': True,
                'name': name,
                'expression': expression,
                'dtype': str(new_col.dtype),
                'new_shape': list(df.shape),
            }
        except Exception as e:
            return df, {'success': False, 'error': str(e)}

    def suggest_features(self, df, target_column=None):
        """
        Suggest common feature engineering transformations.

        Returns:
            list of suggested features with expressions
        """
        suggestions = []
        numeric_cols = df.select_dtypes(include='number').columns.tolist()

        # Log transforms for skewed features
        for col in numeric_cols:
            if df[col].min() >= 0 and abs(df[col].skew()) > 1:
                suggestions.append({
                    'name': f'{col}_log',
                    'expression': f'log({col})',
                    'rationale': f'{col} is skewed (skew={df[col].skew():.2f}) — log transform may help.',
                    'category': 'transform',
                })

        # Ratios between numeric pairs (top 3 by variance)
        if len(numeric_cols) >= 2:
            # Pick pairs with highest variance in their ratio
            pairs_tried = 0
            for i in range(len(numeric_cols)):
                for j in range(i + 1, len(numeric_cols)):
                    if pairs_tried >= 3:
                        break
                    c1, c2 = numeric_cols[i], numeric_cols[j]
                    if (df[c2] != 0).all():
                        suggestions.append({
                            'name': f'{c1}_per_{c2}',
                            'expression': f'{c1} / {c2}',
                            'rationale': f'Ratio of {c1} to {c2} may capture relative relationships.',
                            'category': 'ratio',
                        })
                        pairs_tried += 1
                if pairs_tried >= 3:
                    break

        # Date feature extraction
        for col in df.columns:
            if pd.api.types.is_datetime64_any_dtype(df[col]):
                suggestions.append({
                    'name': f'{col}_dayofweek',
                    'expression': f'day_of_week({col})',
                    'rationale': f'Day-of-week from {col} may capture weekly patterns.',
                    'category': 'temporal',
                })
                suggestions.append({
                    'name': f'{col}_month',
                    'expression': f'month({col})',
                    'rationale': f'Month from {col} may capture seasonal patterns.',
                    'category': 'temporal',
                })

        return suggestions

    def _evaluate_expression(self, expression, df):
        """Safely evaluate a feature engineering expression."""
        # Replace column names with df['col'] references
        namespace = {
            'df': df,
            'np': np,
            'pd': pd,
            'log': np.log1p,
            'sqrt': np.sqrt,
            'abs': np.abs,
            'square': np.square,
            'exp': np.exp,
        }

        # Add date extraction functions
        def day_of_week(col_name):
            return pd.to_datetime(df[col_name]).dt.dayofweek

        def month(col_name):
            return pd.to_datetime(df[col_name]).dt.month

        def year(col_name):
            return pd.to_datetime(df[col_name]).dt.year

        def hour(col_name):
            return pd.to_datetime(df[col_name]).dt.hour

        namespace['day_of_week'] = day_of_week
        namespace['month'] = month
        namespace['year'] = year
        namespace['hour'] = hour

        # Replace column references with df['col']
        processed = expression
        for col in sorted(df.columns, key=len, reverse=True):
            # Use word boundary matching to avoid partial replacements
            pattern = r'\b' + re.escape(col) + r'\b'
            if re.search(pattern, processed):
                # Don't replace if it's already inside a function call
                processed = re.sub(pattern, f"df['{col}']", processed)

        try:
            result = eval(processed, {"__builtins__": {}}, namespace)
            if isinstance(result, pd.Series):
                return result
            elif isinstance(result, np.ndarray):
                return pd.Series(result, index=df.index)
            elif isinstance(result, (int, float)):
                return pd.Series([result] * len(df), index=df.index)
            else:
                return pd.Series(result, index=df.index)
        except Exception as e:
            raise ValueError(f"Expression evaluation failed: {str(e)}")

    def auto_feature_crossing(self, df, target=None, max_pairs=10):
        """Automatically generate interaction features from numeric column pairs.
        
        Filters by mutual information with target if available.
        
        Returns:
            dict with new columns added and their MI scores
        """
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        
        if len(numeric_cols) < 2:
            return {'features': [], 'message': 'Need at least 2 numeric columns'}
        
        # Limit to top columns by variance
        variances = df[numeric_cols].var().sort_values(ascending=False)
        top_cols = variances.head(min(8, len(numeric_cols))).index.tolist()
        
        # Generate interaction features
        new_features = []
        df_new = df.copy()
        
        from itertools import combinations
        for col_a, col_b in combinations(top_cols, 2):
            name = f"{col_a}_x_{col_b}"
            df_new[name] = df[col_a] * df[col_b]
            new_features.append({'name': name, 'type': 'interaction', 'parents': [col_a, col_b]})
        
        # Score by mutual information with target
        if target and target in df.columns:
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                y = df[target].dropna()
                X_new = df_new[[f['name'] for f in new_features]].loc[y.index].fillna(0)
                
                if df[target].dtype == 'object' or df[target].nunique() <= 20:
                    mi_scores = mutual_info_classif(X_new, y, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X_new, y, random_state=42)
                
                for i, feat in enumerate(new_features):
                    feat['mi_score'] = round(float(mi_scores[i]), 4)
                
                new_features.sort(key=lambda x: x.get('mi_score', 0), reverse=True)
                new_features = new_features[:max_pairs]
            except Exception:
                new_features = new_features[:max_pairs]
        else:
            new_features = new_features[:max_pairs]
        
        return {
            'features': new_features,
            'n_generated': len(new_features),
        }
    
    def auto_polynomial_features(self, df, target=None, degree=2, max_features=10):
        """Generate polynomial features and filter by MI with target.
        
        Returns:
            dict with polynomial feature names and scores
        """
        numeric_cols = df.select_dtypes(include='number').columns.tolist()
        if target and target in numeric_cols:
            numeric_cols.remove(target)
        
        top_cols = numeric_cols[:min(6, len(numeric_cols))]
        
        new_features = []
        df_new = df.copy()
        
        for col in top_cols:
            for d in range(2, degree + 1):
                name = f"{col}_pow{d}"
                df_new[name] = df[col] ** d
                new_features.append({'name': name, 'type': 'polynomial', 'parent': col, 'degree': d})
        
        # Filter by MI with target
        if target and target in df.columns and new_features:
            try:
                from sklearn.feature_selection import mutual_info_classif, mutual_info_regression
                y = df[target].dropna()
                X_new = df_new[[f['name'] for f in new_features]].loc[y.index].fillna(0).replace([np.inf, -np.inf], 0)
                
                if df[target].dtype == 'object' or df[target].nunique() <= 20:
                    mi_scores = mutual_info_classif(X_new, y, random_state=42)
                else:
                    mi_scores = mutual_info_regression(X_new, y, random_state=42)
                
                for i, feat in enumerate(new_features):
                    feat['mi_score'] = round(float(mi_scores[i]), 4)
                
                new_features.sort(key=lambda x: x.get('mi_score', 0), reverse=True)
            except Exception:
                pass
        
        return {
            'features': new_features[:max_features],
            'n_generated': len(new_features[:max_features]),
        }
