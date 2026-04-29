"""
Model Card Generator — Automatically produces structured, human-readable
model documentation following Google's Model Card framework.
Includes: intended use, performance metrics, bias analysis, limitations.
"""

import json
from datetime import datetime


def generate_model_card(session_data, profile=None, training_results=None,
                        explainability=None, fairness_report=None):
    """
    Auto-generate a Model Card from pipeline session data.

    Args:
        session_data: dict or PipelineSession with full pipeline results
        (remaining args are overrides)

    Returns:
        dict with structured model card sections + rendered markdown
    """
    prof = profile or (session_data.get('profile') if isinstance(session_data, dict) else getattr(session_data, 'profile', {})) or {}
    train = training_results or (session_data.get('training_results') if isinstance(session_data, dict) else getattr(session_data, 'training_results', {})) or {}
    expl = explainability or (session_data.get('explainability') if isinstance(session_data, dict) else getattr(session_data, 'explainability', {})) or {}
    fair = fairness_report or (session_data.get('fairness_report') if isinstance(session_data, dict) else getattr(session_data, 'fairness_report', {})) or {}

    target = prof.get('target_column', 'unknown')
    problem_type = prof.get('problem_type', 'unknown')
    best_model = train.get('best_model', 'Unknown')
    best_score = train.get('best_score', 0)
    metric_name = train.get('primary_metric_name', 'score')

    # Top features
    top_features = []
    if expl and 'global_importance' in expl:
        top_features = [f['feature'] for f in expl['global_importance'][:5]]
    elif train and 'feature_importance' in train:
        top_features = [f['feature'] for f in train['feature_importance'][:5]]

    # Performance breakdown
    leaderboard = train.get('leaderboard', [])
    perf_summary = []
    for entry in leaderboard[:5]:
        perf_summary.append({
            'model': entry.get('model', ''),
            'score': entry.get('primary_metric', 0),
        })

    # Bias / fairness
    bias_section = None
    if fair and 'audit_results' in fair:
        bias_section = {
            'sensitive_columns': fair.get('sensitive_columns', []),
            'fairness_verdict': fair.get('overall_verdict', 'Not assessed'),
            'details': fair.get('audit_results', []),
        }

    # Limitations
    limitations = []
    n_rows = prof.get('n_rows', 0)
    if n_rows < 500:
        limitations.append('Small dataset — model may not generalize well.')
    if prof.get('total_missing_pct', 0) > 10:
        limitations.append(f"{prof['total_missing_pct']}% missing data was imputed — predictions for edge cases may be unreliable.")
    if problem_type == 'classification':
        cd = prof.get('class_distribution', {})
        if cd:
            vals = list(cd.values())
            if vals and max(vals) / max(min(vals), 1) > 5:
                limitations.append('Significant class imbalance — minority class predictions may be less reliable.')
    limitations.append('Model trained on historical data — may not capture future distribution shifts.')
    limitations.append('Always validate with domain experts before production deployment.')

    card = {
        'model_details': {
            'name': best_model,
            'type': problem_type,
            'target': target,
            'framework': 'scikit-learn / AutoML Studio',
            'created': datetime.now().isoformat(),
        },
        'intended_use': {
            'primary': f'Predict "{target}" ({problem_type})',
            'users': 'Data scientists, ML engineers, business analysts',
            'out_of_scope': 'Should not be used for life-critical decisions without human oversight.',
        },
        'training_data': {
            'n_rows': n_rows,
            'n_features': prof.get('n_cols', 0),
            'missing_pct': prof.get('total_missing_pct', 0),
            'class_distribution': prof.get('class_distribution'),
        },
        'performance': {
            'primary_metric': metric_name,
            'best_score': round(best_score, 4),
            'model_comparison': perf_summary,
        },
        'explainability': {
            'top_features': top_features,
            'method': 'SHAP' if expl else 'feature_importances',
        },
        'bias_analysis': bias_section,
        'limitations': limitations,
    }

    # Render as markdown
    card['markdown'] = _render_markdown(card)

    return card


def _render_markdown(card):
    """Render a model card dict as human-readable markdown."""
    md = []
    md.append(f"# Model Card: {card['model_details']['name']}")
    md.append(f"*Generated: {card['model_details']['created'][:10]}*\n")

    md.append("## Model Details")
    md.append(f"- **Algorithm**: {card['model_details']['name']}")
    md.append(f"- **Task**: {card['model_details']['type']}")
    md.append(f"- **Target**: {card['model_details']['target']}")
    md.append(f"- **Framework**: {card['model_details']['framework']}\n")

    md.append("## Intended Use")
    md.append(f"- **Primary**: {card['intended_use']['primary']}")
    md.append(f"- **Users**: {card['intended_use']['users']}")
    md.append(f"- **Out of Scope**: {card['intended_use']['out_of_scope']}\n")

    md.append("## Training Data")
    td = card['training_data']
    md.append(f"- **Records**: {td['n_rows']:,}")
    md.append(f"- **Features**: {td['n_features']}")
    md.append(f"- **Missing**: {td['missing_pct']}%\n")

    md.append("## Performance")
    perf = card['performance']
    md.append(f"- **{perf['primary_metric']}**: **{perf['best_score']:.4f}**")
    for comp in perf.get('model_comparison', []):
        md.append(f"  - {comp['model']}: {comp['score']:.4f}")
    md.append("")

    md.append("## Key Features")
    for feat in card['explainability']['top_features']:
        md.append(f"- {feat}")
    md.append("")

    if card.get('bias_analysis'):
        md.append("## Bias Analysis")
        ba = card['bias_analysis']
        md.append(f"- **Verdict**: {ba['fairness_verdict']}")
        md.append(f"- **Sensitive columns checked**: {', '.join(ba.get('sensitive_columns', []))}\n")

    md.append("## Limitations & Risks")
    for lim in card['limitations']:
        md.append(f"- ⚠️ {lim}")

    return "\n".join(md)
