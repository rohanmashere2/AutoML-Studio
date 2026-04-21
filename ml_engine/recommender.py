"""
AutoML Problem Solver - Recommendation Engine
Analyzes training results and generates actionable recommendations for improvement.
"""

import numpy as np


def generate_recommendations(profile, clean_report, transform_report, training_results):
    """
    Analyze the full pipeline context and generate prioritized recommendations.
    
    Returns:
        list: Recommendations sorted by impact level, each with:
              - title, description, impact, category, action (config changes for retrain)
    """
    recommendations = []
    
    problem_type = training_results.get('problem_type', 'classification')
    leaderboard = training_results.get('leaderboard', [])
    context = training_results.get('training_context', {})
    feature_importance = training_results.get('feature_importance', [])
    cv_results = context.get('cv_results', {})
    best_score = training_results.get('best_score', 0)
    
    # 1. Data Quality Diagnostics
    recommendations += _check_data_quality(profile, clean_report)
    
    # 2. Class Imbalance Analysis
    if problem_type == 'classification':
        recommendations += _check_imbalance(profile, transform_report, leaderboard)
    
    # 3. Feature Diagnostics
    recommendations += _check_features(feature_importance, context, transform_report)
    
    # 4. Overfitting / Underfitting Detection
    recommendations += _check_fitting(leaderboard, problem_type, best_score)
    
    # 5. CV Variance Analysis
    recommendations += _check_cv_variance(cv_results)
    
    # 6. Model-Specific Insights
    recommendations += _check_model_insights(leaderboard, problem_type)
    
    # 7. General Performance Recommendations
    recommendations += _check_general_performance(best_score, problem_type, context)
    
    # Sort by impact level (high first)
    impact_order = {'high': 0, 'medium': 1, 'low': 2}
    recommendations.sort(key=lambda x: impact_order.get(x.get('impact', 'low'), 3))
    
    return recommendations


def _check_data_quality(profile, clean_report):
    """Check for data quality issues that may affect performance."""
    recs = []
    
    total_missing_pct = profile.get('total_missing_pct', 0)
    
    if total_missing_pct > 20:
        recs.append({
            'title': 'High Missing Value Rate',
            'description': f'{total_missing_pct}% of your data was missing. Median/mode imputation was used, but this can introduce bias. Consider using KNN imputation or iterative imputation for better results.',
            'impact': 'high',
            'category': 'data_quality',
            'icon': '🩹',
            'action': {'imputation': 'advanced'},
        })
    elif total_missing_pct > 5:
        recs.append({
            'title': 'Moderate Missing Values',
            'description': f'{total_missing_pct}% of data had missing values. The imputation strategy might affect model accuracy. Advanced imputation methods could help.',
            'impact': 'medium',
            'category': 'data_quality',
            'icon': '🩹',
            'action': {'imputation': 'advanced'},
        })
    
    # Check outlier impact
    if clean_report and 'steps' in clean_report:
        for step in clean_report['steps']:
            if step.get('name') == 'Handle Outliers' and step.get('count', 0) > 0:
                outlier_count = step['count']
                recs.append({
                    'title': 'Significant Outliers Detected',
                    'description': f'{outlier_count} outliers were capped. If the data truly has extreme values that are valid, tree-based models handle these better than linear models.',
                    'impact': 'medium',
                    'category': 'data_quality',
                    'icon': '📊',
                    'action': {'prefer_tree_models': True},
                })
    
    # Check duplicates
    dup_pct = profile.get('duplicates_pct', 0)
    if dup_pct > 10:
        recs.append({
            'title': 'High Duplicate Rate',
            'description': f'{dup_pct}% of rows were duplicates. This may indicate data collection issues. Verify if duplicates are intentional.',
            'impact': 'medium',
            'category': 'data_quality',
            'icon': '🔄',
            'action': {},
        })
    
    return recs


def _check_imbalance(profile, transform_report, leaderboard):
    """Check for class imbalance issues."""
    recs = []
    
    class_dist = profile.get('class_distribution', {})
    if not class_dist:
        return recs
    
    counts = list(class_dist.values())
    if len(counts) < 2:
        return recs
    
    ratio = max(counts) / max(min(counts), 1)
    
    if ratio > 3:
        # Check if accuracy is high but F1 is low
        if leaderboard:
            best = leaderboard[0].get('metrics', {})
            acc = best.get('accuracy', 0)
            f1 = best.get('f1', 0)
            
            if acc - f1 > 0.1:
                recs.append({
                    'title': 'Accuracy-F1 Gap Detected',
                    'description': f'Accuracy ({acc:.1%}) is much higher than F1 ({f1:.1%}). The model may be biased toward the majority class. Optimizing for F1 score instead of accuracy would give better results.',
                    'impact': 'high',
                    'category': 'imbalance',
                    'icon': '⚖️',
                    'action': {'optimize_metric': 'f1', 'class_weight': 'balanced'},
                })
            else:
                recs.append({
                    'title': 'Severe Class Imbalance',
                    'description': f'Class ratio is {ratio:.1f}:1. SMOTE was applied but more aggressive balancing with class_weight="balanced" could improve minority class detection.',
                    'impact': 'high',
                    'category': 'imbalance',
                    'icon': '⚖️',
                    'action': {'class_weight': 'balanced', 'smote_strategy': 'aggressive'},
                })
    elif ratio > 2:
        recs.append({
            'title': 'Moderate Class Imbalance',
            'description': f'Class ratio is {ratio:.1f}:1. Using class_weight="balanced" during training could improve performance on the minority class.',
            'impact': 'medium',
            'category': 'imbalance',
            'icon': '⚖️',
            'action': {'class_weight': 'balanced'},
        })
    
    return recs


def _check_features(feature_importance, context, transform_report):
    """Check for feature-related issues."""
    recs = []
    n_features = context.get('n_features', 0)
    
    # Low feature count
    if n_features < 5:
        recs.append({
            'title': 'Very Few Features',
            'description': f'Only {n_features} features available. Consider adding polynomial feature interactions to create more predictive signals.',
            'impact': 'high',
            'category': 'features',
            'icon': '🔧',
            'action': {'polynomial_features': True, 'degree': 2},
        })
    
    # Check feature importance spread
    if feature_importance and len(feature_importance) > 2:
        importances = [f['importance'] for f in feature_importance]
        top_importance = importances[0]
        
        # If top feature dominates
        if top_importance > 0.5:
            recs.append({
                'title': f'Feature Dominance: "{feature_importance[0]["feature"]}"',
                'description': f'One feature carries {top_importance:.0%} of importance. The model relies too heavily on a single feature. Adding more predictive features would improve robustness.',
                'impact': 'medium',
                'category': 'features',
                'icon': '📊',
                'action': {'polynomial_features': True},
            })
        
        # If all features have very low importance
        if max(importances) < 0.1 and len(importances) > 5:
            recs.append({
                'title': 'Low Feature Predictive Power',
                'description': 'No single feature has strong predictive power. The existing features may not capture the underlying patterns well. Consider feature engineering or collecting additional data.',
                'impact': 'high',
                'category': 'features',
                'icon': '🔍',
                'action': {'polynomial_features': True, 'feature_engineering': True},
            })
    
    # Check correlation removal from transform
    if transform_report and 'steps' in transform_report:
        for step in transform_report['steps']:
            if step.get('name') == 'Remove High Correlation' and step.get('count', 0) > 2:
                recs.append({
                    'title': 'Multicollinearity Issues',
                    'description': f'{step["count"]} features were removed due to high correlation. Consider applying PCA to capture variance more efficiently.',
                    'impact': 'medium',
                    'category': 'features',
                    'icon': '🔗',
                    'action': {'apply_pca': True},
                })
    
    return recs


def _check_fitting(leaderboard, problem_type, best_score):
    """Detect overfitting and underfitting."""
    recs = []
    
    if not leaderboard:
        return recs
    
    best = leaderboard[0].get('metrics', {})
    
    if problem_type == 'classification':
        train_score = best.get('train_accuracy', 0)
        test_score = best.get('accuracy', 0)
        threshold_low = 0.65
    else:
        train_score = best.get('train_r2', 0)
        test_score = best.get('r2', 0)
        threshold_low = 0.3
    
    gap = train_score - test_score
    
    # Overfitting: high train, low test
    if gap > 0.1 and train_score > 0.85:
        recs.append({
            'title': 'Overfitting Detected',
            'description': f'Training score ({train_score:.1%}) is significantly higher than test score ({test_score:.1%}). The model is memorizing the training data. Adding regularization, reducing model complexity, or getting more data would help.',
            'impact': 'high',
            'category': 'fitting',
            'icon': '📈',
            'action': {'regularization': True, 'reduce_complexity': True},
        })
    
    # Underfitting: both low
    if test_score < threshold_low and train_score < threshold_low + 0.1:
        recs.append({
            'title': 'Underfitting Detected',
            'description': f'Both training ({train_score:.1%}) and test ({test_score:.1%}) scores are low. The models are too simple to capture patterns. Try more complex models, polynomial features, or adding more features.',
            'impact': 'high',
            'category': 'fitting',
            'icon': '📉',
            'action': {'increase_complexity': True, 'polynomial_features': True},
        })
    
    return recs


def _check_cv_variance(cv_results):
    """Check for high variance across CV folds."""
    recs = []
    
    for model_name, cv in cv_results.items():
        if cv.get('std', 0) > 0.05:
            recs.append({
                'title': f'High Variance in {model_name}',
                'description': f'{model_name} has high CV standard deviation ({cv["std"]:.4f}). This means performance varies significantly across data splits. More data or stronger regularization would help stabilize results.',
                'impact': 'medium',
                'category': 'variance',
                'icon': '📊',
                'action': {'regularization': True},
            })
            break  # Only report once
    
    return recs


def _check_model_insights(leaderboard, problem_type):
    """Generate model comparison insights."""
    recs = []
    
    if len(leaderboard) < 2:
        return recs
    
    best_score = leaderboard[0]['primary_metric']
    worst_score = leaderboard[-1]['primary_metric']
    
    # If all models perform similarly
    if best_score - worst_score < 0.03 and best_score > 0:
        recs.append({
            'title': 'All Models Perform Similarly',
            'description': 'The performance gap between the best and worst model is very small. This suggests the data itself is the bottleneck. Collecting more features or more samples would likely help more than switching models.',
            'impact': 'high',
            'category': 'model',
            'icon': '🤔',
            'action': {'feature_engineering': True},
        })
    
    # If linear models ≈ tree models
    linear_models = ['Logistic Regression', 'Linear Regression', 'Ridge', 'Lasso']
    tree_models = ['Random Forest', 'Gradient Boosting', 'XGBoost', 'LightGBM']
    
    linear_scores = [e['primary_metric'] for e in leaderboard if e['model'] in linear_models and e['primary_metric'] > -999]
    tree_scores = [e['primary_metric'] for e in leaderboard if e['model'] in tree_models and e['primary_metric'] > -999]
    
    if linear_scores and tree_scores:
        avg_linear = np.mean(linear_scores)
        avg_tree = np.mean(tree_scores)
        
        if abs(avg_linear - avg_tree) < 0.02:
            recs.append({
                'title': 'Linear Relationships Detected',
                'description': 'Linear and tree-based models perform similarly. The data likely has strong linear patterns. A simpler linear model may be preferable for interpretability.',
                'impact': 'low',
                'category': 'model',
                'icon': '📐',
                'action': {},
            })
    
    return recs


def _check_general_performance(best_score, problem_type, context):
    """General performance-based recommendations."""
    recs = []
    
    if problem_type == 'classification' and best_score < 0.7:
        recs.append({
            'title': 'Low Overall Accuracy',
            'description': f'Best accuracy is {best_score:.1%}. This is below typical expectations. The dataset may lack sufficient signal for the target, or the features may need significant engineering.',
            'impact': 'high',
            'category': 'performance',
            'icon': '⚠️',
            'action': {'polynomial_features': True, 'feature_engineering': True},
        })
    elif problem_type == 'regression' and best_score < 0.5:
        recs.append({
            'title': 'Low R² Score',
            'description': f'Best R² is {best_score:.4f}. The model explains less than 50% of variance. Consider adding more relevant features or transforming existing ones.',
            'impact': 'high',
            'category': 'performance',
            'icon': '⚠️',
            'action': {'polynomial_features': True, 'feature_engineering': True},
        })
    
    n_features = context.get('n_features', 0)
    n_samples = context.get('X_train_shape', (0, 0))
    if isinstance(n_samples, (tuple, list)):
        n_samples = n_samples[0]
    
    if n_samples < 100:
        recs.append({
            'title': 'Small Dataset Warning',
            'description': f'Only {n_samples} training samples available. Models may not generalize well. If possible, collect more data for better results.',
            'impact': 'high',
            'category': 'data_quality',
            'icon': '📦',
            'action': {'regularization': True},
        })
    
    return recs
