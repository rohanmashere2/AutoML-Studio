"""
AutoML Problem Solver - Report Generator
Generates comprehensive HTML reports with embedded charts.
"""

import os
import json
from datetime import datetime


def generate_html_report(session_data, output_dir):
    """
    Generate a comprehensive HTML report for an AutoML session.
    
    Args:
        session_data: dict containing all pipeline results
        output_dir: directory to save the report
    
    Returns:
        str: path to the generated HTML file
    """
    profile = session_data.get('profile', {})
    clean_report = session_data.get('clean_report', {})
    transform_report = session_data.get('transform_report', {})
    training_results = session_data.get('training_results', {})
    retrain_results = session_data.get('retrain_results', {})
    explainability = session_data.get('explainability', {})
    diagnostics = session_data.get('diagnostics', {})
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>AutoML Report — {profile.get('target_column', 'Unknown')} | {datetime.now().strftime('%Y-%m-%d')}</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.0/dist/chart.umd.min.js"></script>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, -apple-system, sans-serif; background: #0a0e1a; color: #e0e6f0; line-height: 1.6; padding: 40px; }}
        .container {{ max-width: 1100px; margin: 0 auto; }}
        
        h1 {{ font-size: 2rem; background: linear-gradient(135deg, #00d4ff, #7b2ffc); -webkit-background-clip: text; -webkit-text-fill-color: transparent; margin-bottom: 8px; }}
        h2 {{ font-size: 1.4rem; color: #00d4ff; margin: 32px 0 16px; padding-bottom: 8px; border-bottom: 1px solid rgba(0,212,255,0.2); }}
        h3 {{ font-size: 1.1rem; color: #c0c8e0; margin: 20px 0 8px; }}
        
        .card {{ background: rgba(16,20,40,0.95); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; margin-bottom: 20px; }}
        .card-accent {{ border-left: 3px solid #00d4ff; }}
        .card-success {{ border-left: 3px solid #00e676; }}
        .card-warning {{ border-left: 3px solid #ff9800; }}
        
        .metric-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 16px; margin: 16px 0; }}
        .metric {{ background: rgba(0,212,255,0.05); border-radius: 12px; padding: 16px; text-align: center; }}
        .metric .value {{ font-size: 1.8rem; font-weight: 700; color: #00d4ff; }}
        .metric .label {{ font-size: 0.8rem; color: #8892b0; text-transform: uppercase; letter-spacing: 1px; }}
        
        table {{ width: 100%; border-collapse: collapse; margin: 12px 0; }}
        th, td {{ padding: 10px 14px; text-align: left; border-bottom: 1px solid rgba(255,255,255,0.05); }}
        th {{ color: #00d4ff; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; }}
        tr:hover {{ background: rgba(0,212,255,0.03); }}
        
        .badge {{ display: inline-block; padding: 2px 10px; border-radius: 20px; font-size: 0.75rem; font-weight: 600; }}
        .badge-green {{ background: rgba(0,230,118,0.15); color: #00e676; }}
        .badge-yellow {{ background: rgba(255,152,0,0.15); color: #ff9800; }}
        .badge-red {{ background: rgba(255,82,82,0.15); color: #ff5252; }}
        .badge-blue {{ background: rgba(0,212,255,0.15); color: #00d4ff; }}
        
        .chart-container {{ position: relative; width: 100%; height: 300px; margin: 16px 0; }}
        
        .step {{ display: flex; align-items: flex-start; gap: 12px; padding: 12px 0; border-bottom: 1px solid rgba(255,255,255,0.03); }}
        .step-icon {{ font-size: 1.2rem; min-width: 30px; }}
        .step-content {{ flex: 1; }}
        .step-title {{ font-weight: 600; color: #e0e6f0; }}
        .step-desc {{ font-size: 0.9rem; color: #8892b0; }}
        
        .rec {{ background: rgba(123,47,252,0.08); border-left: 3px solid #7b2ffc; border-radius: 8px; padding: 16px; margin: 8px 0; }}
        .rec-title {{ font-weight: 600; color: #c4a0ff; }}
        .rec-desc {{ font-size: 0.9rem; color: #8892b0; margin-top: 4px; }}
        
        .trophy {{ font-size: 2rem; text-align: center; margin-bottom: 8px; }}
        .best-model {{ text-align: center; padding: 20px; background: linear-gradient(135deg, rgba(255,215,0,0.1), rgba(255,215,0,0.02)); border-radius: 16px; }}
        .best-model .name {{ font-size: 1.5rem; font-weight: 700; color: #ffd700; }}
        .best-model .score {{ font-size: 2.5rem; font-weight: 800; color: #00e676; margin: 8px 0; }}
        
        .footer {{ text-align: center; margin-top: 40px; padding: 20px; color: #8892b0; font-size: 0.85rem; }}
        
        @media print {{
            body {{ background: white; color: #333; padding: 20px; }}
            .card {{ border: 1px solid #ddd; }}
            .metric .value {{ color: #2196f3; }}
            h1 {{ color: #1a237e; -webkit-text-fill-color: #1a237e; }}
            h2 {{ color: #1565c0; }}
        }}
    </style>
</head>
<body>
<div class="container">
    <h1>⚡ AutoML Problem Solver — Report</h1>
    <p style="color:#8892b0; margin-bottom:32px;">Generated on {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>
    
    {_render_dataset_section(profile)}
    {_render_cleaning_section(clean_report, transform_report)}
    {_render_training_section(training_results)}
    {_render_recommendations_section(training_results)}
    {_render_retrain_section(retrain_results)}
    {_render_explainability_section(explainability)}
    
    <div class="footer">
        <p>Generated by AutoML Problem Solver</p>
    </div>
</div>
</body>
</html>'''
    
    os.makedirs(output_dir, exist_ok=True)
    report_path = os.path.join(output_dir, 'automl_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)
    
    return report_path


def _render_dataset_section(profile):
    if not profile:
        return ''
    
    return f'''
    <h2>📊 Dataset Overview</h2>
    <div class="metric-grid">
        <div class="metric">
            <div class="value">{profile.get('n_rows', 0):,}</div>
            <div class="label">Rows</div>
        </div>
        <div class="metric">
            <div class="value">{profile.get('n_cols', 0)}</div>
            <div class="label">Columns</div>
        </div>
        <div class="metric">
            <div class="value">{profile.get('target_column', 'N/A')}</div>
            <div class="label">Target Column</div>
        </div>
        <div class="metric">
            <div class="value">{profile.get('problem_type', 'N/A').title()}</div>
            <div class="label">Problem Type</div>
        </div>
        <div class="metric">
            <div class="value">{profile.get('total_missing_pct', 0)}%</div>
            <div class="label">Missing Data</div>
        </div>
        <div class="metric">
            <div class="value">{profile.get('duplicates', 0)}</div>
            <div class="label">Duplicates</div>
        </div>
    </div>'''


def _render_cleaning_section(clean_report, transform_report):
    if not clean_report and not transform_report:
        return ''
    
    steps_html = ''
    
    if clean_report and 'steps' in clean_report:
        for step in clean_report['steps']:
            applied = '✅' if step.get('applied') else '⬜'
            steps_html += f'''
            <div class="step">
                <div class="step-icon">{step.get('icon', '🔧')}</div>
                <div class="step-content">
                    <div class="step-title">{applied} {step.get('name', '')}</div>
                    <div class="step-desc">{step.get('description', '')}</div>
                </div>
            </div>'''
    
    if transform_report and 'steps' in transform_report:
        for step in transform_report['steps']:
            applied = '✅' if step.get('applied') else '⬜'
            steps_html += f'''
            <div class="step">
                <div class="step-icon">{step.get('icon', '🔧')}</div>
                <div class="step-content">
                    <div class="step-title">{applied} {step.get('name', '')}</div>
                    <div class="step-desc">{step.get('description', '')}</div>
                </div>
            </div>'''
    
    summary = clean_report.get('summary', {})
    
    return f'''
    <h2>🧹 Data Cleaning & Transformation</h2>
    <div class="card card-accent">
        {steps_html}
    </div>
    <div class="metric-grid">
        <div class="metric">
            <div class="value">{summary.get('original_rows', 0):,} → {summary.get('cleaned_rows', 0):,}</div>
            <div class="label">Rows</div>
        </div>
        <div class="metric">
            <div class="value">{summary.get('original_cols', 0)} → {summary.get('cleaned_cols', 0)}</div>
            <div class="label">Columns</div>
        </div>
    </div>'''


def _render_training_section(training_results):
    if not training_results:
        return ''
    
    best_model = training_results.get('best_model', 'N/A')
    best_score = training_results.get('best_score', 0)
    metric_name = training_results.get('primary_metric_name', 'score')
    leaderboard = training_results.get('leaderboard', [])
    
    # Leaderboard table
    rows_html = ''
    for entry in leaderboard:
        rank = entry.get('rank', '-')
        medal = '🥇' if rank == 1 else '🥈' if rank == 2 else '🥉' if rank == 3 else f'#{rank}'
        score = entry.get('primary_metric', 0)
        if score > -999:
            score_str = f'{score:.4f}'
        else:
            score_str = 'Error'
        
        rows_html += f'''
            <tr>
                <td>{medal}</td>
                <td><strong>{entry.get("model", "")}</strong></td>
                <td>{score_str}</td>
            </tr>'''
    
    # Feature importance
    fi_html = ''
    for fi in training_results.get('feature_importance', [])[:10]:
        pct = fi['importance'] * 100
        fi_html += f'''
            <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                <span style="min-width:140px; font-size:0.85rem;">{fi['feature']}</span>
                <div style="flex:1; background:rgba(0,212,255,0.1); border-radius:4px; height:20px;">
                    <div style="width:{min(pct*2, 100)}%; height:100%; background:linear-gradient(90deg,#00d4ff,#7b2ffc); border-radius:4px;"></div>
                </div>
                <span style="min-width:50px; text-align:right; font-size:0.85rem;">{pct:.1f}%</span>
            </div>'''
    
    return f'''
    <h2>🤖 Model Training Results</h2>
    <div class="best-model">
        <div class="trophy">🏆</div>
        <div class="name">{best_model}</div>
        <div class="score">{best_score:.4f}</div>
        <div class="label" style="color:#8892b0;">{metric_name.upper()}</div>
    </div>
    
    <div class="card" style="margin-top:20px;">
        <h3>📋 Model Leaderboard</h3>
        <table>
            <thead><tr><th>Rank</th><th>Model</th><th>{metric_name.title()}</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    
    <div class="card">
        <h3>🎯 Feature Importance (Top 10)</h3>
        {fi_html if fi_html else '<p style="color:#8892b0;">No feature importance available.</p>'}
    </div>'''


def _render_recommendations_section(training_results):
    if not training_results:
        return ''
    
    recommendations = training_results.get('recommendations', [])
    if not recommendations:
        return ''
    
    recs_html = ''
    for rec in recommendations:
        impact_badge = {
            'high': '<span class="badge badge-red">HIGH IMPACT</span>',
            'medium': '<span class="badge badge-yellow">MEDIUM</span>',
            'low': '<span class="badge badge-blue">LOW</span>',
        }.get(rec.get('impact'), '')
        
        recs_html += f'''
        <div class="rec">
            <div class="rec-title">{rec.get('icon', '💡')} {rec.get('title', '')} {impact_badge}</div>
            <div class="rec-desc">{rec.get('description', '')}</div>
        </div>'''
    
    return f'''
    <h2>🧠 AI Recommendations</h2>
    {recs_html}'''


def _render_retrain_section(retrain_results):
    if not retrain_results:
        return ''
    
    improvement = retrain_results.get('improvement', 0)
    pct = retrain_results.get('improvement_pct', 0)
    
    retrain_report = retrain_results.get('retrain_report', {})
    verdict = retrain_report.get('verdict_text', '')
    
    # Model comparison table
    comp_html = ''
    for comp in retrain_report.get('model_comparison', []):
        delta = comp.get('delta', 0)
        delta_color = '#00e676' if delta > 0 else '#ff5252' if delta < 0 else '#8892b0'
        comp_html += f'''
            <tr>
                <td>{comp.get("model", "")}</td>
                <td>{comp.get("before", 0):.4f}</td>
                <td>{comp.get("after", 0):.4f}</td>
                <td style="color:{delta_color}">{delta:+.4f}</td>
            </tr>'''
    
    return f'''
    <h2>🔄 Retrain Results</h2>
    <div class="card card-success">
        <p style="font-size:1.1rem;">{verdict}</p>
    </div>
    
    {"<div class='card'><h3>Per-Model Comparison</h3><table><thead><tr><th>Model</th><th>Before</th><th>After</th><th>Change</th></tr></thead><tbody>" + comp_html + "</tbody></table></div>" if comp_html else ""}'''


def _render_explainability_section(explainability):
    if not explainability or 'error' in explainability:
        return ''
    
    global_importance = explainability.get('global_importance', [])
    
    fi_html = ''
    for fi in global_importance[:15]:
        pct = fi['importance'] * 100
        fi_html += f'''
            <div style="display:flex; align-items:center; gap:8px; margin:4px 0;">
                <span style="min-width:160px; font-size:0.85rem;">{fi['feature']}</span>
                <div style="flex:1; background:rgba(123,47,252,0.1); border-radius:4px; height:20px;">
                    <div style="width:{min(pct*2, 100)}%; height:100%; background:linear-gradient(90deg,#7b2ffc,#ff6ec7); border-radius:4px;"></div>
                </div>
                <span style="min-width:50px; text-align:right; font-size:0.85rem;">{pct:.1f}%</span>
            </div>'''
    
    return f'''
    <h2>🔍 Model Explainability (SHAP)</h2>
    <div class="card">
        <h3>Global Feature Importance (SHAP Values)</h3>
        <p style="color:#8892b0; font-size:0.9rem; margin-bottom:12px;">
            SHAP values show the average impact of each feature on model predictions.
        </p>
        {fi_html}
    </div>'''


def generate_executive_summary(session_data, llm_agent=None):
    """
    Generate a non-technical executive summary report.
    Uses LLM for natural language if available, otherwise template-based.

    Args:
        session_data: dict with profile, training_results, explainability, etc.
        llm_agent: optional AutoMLChatAgent instance for LLM-powered summaries

    Returns:
        dict with executive_summary HTML and key findings
    """
    profile = session_data.get('profile', {})
    training = session_data.get('training_results', {})
    explain = session_data.get('explainability', {})
    recs = session_data.get('recommendations', [])

    target = profile.get('target_column', 'the target')
    problem_type = profile.get('problem_type', 'unknown')
    best_model = training.get('best_model', 'Unknown')
    best_score = training.get('best_score', 0)
    metric_name = training.get('primary_metric_name', 'score')
    n_rows = profile.get('n_rows', 0)
    n_cols = profile.get('n_cols', 0)

    # Top features
    top_features = []
    if explain and 'global_importance' in explain:
        top_features = explain['global_importance'][:3]

    # Try LLM-powered summary
    if llm_agent and hasattr(llm_agent, 'llm_provider') and llm_agent.llm_provider == 'openai':
        try:
            prompt = (
                f"Write a 200-word executive summary for a non-technical stakeholder about this ML model:\n"
                f"- Task: Predict '{target}' ({problem_type})\n"
                f"- Data: {n_rows:,} records, {n_cols} features\n"
                f"- Best model: {best_model} with {metric_name}={best_score:.4f}\n"
                f"- Top drivers: {', '.join(f['feature'] for f in top_features)}\n"
                f"Write in plain business English. Include: what it does, accuracy, key drivers, "
                f"recommended actions, and risks/limitations."
            )
            response = llm_agent.openai_client.chat.completions.create(
                model='gpt-4o-mini',
                messages=[{'role': 'user', 'content': prompt}],
                max_tokens=500,
                temperature=0.7,
            )
            llm_summary = response.choices[0].message.content
        except Exception:
            llm_summary = None
    else:
        llm_summary = None

    # Template-based fallback
    if not llm_summary:
        sections = []

        # Section 1: What does this model do?
        if problem_type == 'classification':
            sections.append(
                f"**What does this model do?** This AI model predicts '{target}' "
                f"by analyzing patterns in {n_rows:,} historical records across {n_cols} data points. "
                f"It categorizes each record into one of the known outcome groups."
            )
        else:
            sections.append(
                f"**What does this model do?** This AI model estimates the value of '{target}' "
                f"by analyzing patterns in {n_rows:,} historical records across {n_cols} data points."
            )

        # Section 2: How accurate is it?
        score_pct = best_score * 100 if best_score <= 1 else best_score
        if score_pct > 90:
            quality = "excellent"
        elif score_pct > 80:
            quality = "good"
        elif score_pct > 70:
            quality = "moderate"
        else:
            quality = "developing"

        sections.append(
            f"**How accurate is it?** The model achieves {quality} performance with a "
            f"{metric_name} of {best_score:.1%} using {best_model}. "
            f"This means it correctly handles approximately {score_pct:.0f} out of every 100 cases."
        )

        # Section 3: Key drivers
        if top_features:
            driver_text = []
            for i, f in enumerate(top_features, 1):
                driver_text.append(f"{i}. **{f['feature']}** ({f['importance']*100:.0f}% influence)")
            sections.append(
                "**What drives the predictions?** The top factors influencing outcomes are:\n" +
                "\n".join(driver_text)
            )

        # Section 4: Recommended actions
        if recs and isinstance(recs, list):
            action_items = [f"• {r.get('title', r.get('description', ''))}" for r in recs[:3]]
            sections.append(
                "**Recommended next steps:**\n" + "\n".join(action_items)
            )

        # Section 5: Risks
        sections.append(
            "**Risks & Limitations:** This model is trained on historical data and may not "
            "account for recent changes in patterns. Regular monitoring is recommended to ensure "
            "predictions remain accurate. The model should be validated by domain experts before "
            "deploying in production."
        )

        llm_summary = "\n\n".join(sections)

    # Build executive report HTML
    html = _build_executive_html(llm_summary, profile, training, top_features)

    return {
        'summary_text': llm_summary,
        'html': html,
        'key_metrics': {
            'model': best_model,
            'score': round(best_score, 4),
            'metric': metric_name,
            'top_features': [f['feature'] for f in top_features],
        },
    }


def _build_executive_html(summary_text, profile, training, top_features):
    """Build clean executive HTML report."""
    target = profile.get('target_column', '')
    best = training.get('best_model', 'Unknown')
    score = training.get('best_score', 0)

    features_html = ''
    for f in top_features:
        pct = f['importance'] * 100
        features_html += f'''
        <div style="display:flex; align-items:center; gap:12px; margin:8px 0;">
            <span style="min-width:150px; font-weight:600;">{f['feature']}</span>
            <div style="flex:1; height:24px; background:rgba(255,255,255,0.05); border-radius:6px; overflow:hidden;">
                <div style="width:{min(pct*2, 100)}%; height:100%; background:linear-gradient(90deg,#00d4ff,#7b2ffc); border-radius:6px;"></div>
            </div>
            <span style="min-width:50px; text-align:right;">{pct:.0f}%</span>
        </div>'''

    # Convert markdown-style bold to HTML
    formatted = summary_text.replace('**', '<strong>', 1)
    while '**' in formatted:
        formatted = formatted.replace('**', '</strong>', 1)
        if '**' in formatted:
            formatted = formatted.replace('**', '<strong>', 1)
    formatted = formatted.replace('\n', '<br>')

    return f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Executive Summary — {target}</title>
    <style>
        * {{ margin: 0; padding: 0; box-sizing: border-box; }}
        body {{ font-family: 'Segoe UI', system-ui, sans-serif; background: #0a0e1a; color: #e0e6f0; padding: 40px; }}
        .container {{ max-width: 800px; margin: 0 auto; }}
        h1 {{ font-size: 1.8rem; color: #00d4ff; margin-bottom: 8px; }}
        .subtitle {{ color: #8892b0; margin-bottom: 32px; }}
        .card {{ background: rgba(16,20,40,0.95); border: 1px solid rgba(255,255,255,0.06); border-radius: 16px; padding: 24px; margin-bottom: 20px; }}
        .metric {{ display: inline-block; background: rgba(0,212,255,0.1); border: 1px solid rgba(0,212,255,0.3); border-radius: 12px; padding: 16px 24px; margin: 8px; text-align: center; }}
        .metric-value {{ font-size: 1.8rem; font-weight: 700; color: #00d4ff; }}
        .metric-label {{ font-size: 0.85rem; color: #8892b0; }}
        .summary {{ line-height: 1.8; font-size: 1.05rem; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 Executive Summary</h1>
        <p class="subtitle">Generated {datetime.now().strftime('%B %d, %Y at %I:%M %p')}</p>

        <div style="display:flex; flex-wrap:wrap; gap:8px; margin-bottom:24px;">
            <div class="metric">
                <div class="metric-value">{score:.1%}</div>
                <div class="metric-label">Model Accuracy</div>
            </div>
            <div class="metric">
                <div class="metric-value">{best}</div>
                <div class="metric-label">Best Algorithm</div>
            </div>
            <div class="metric">
                <div class="metric-value">{profile.get('n_rows', 0):,}</div>
                <div class="metric-label">Records Analyzed</div>
            </div>
        </div>

        <div class="card">
            <div class="summary">{formatted}</div>
        </div>

        <div class="card">
            <h3 style="color:#00d4ff; margin-bottom:16px;">Key Drivers</h3>
            {features_html}
        </div>
    </div>
</body>
</html>'''
