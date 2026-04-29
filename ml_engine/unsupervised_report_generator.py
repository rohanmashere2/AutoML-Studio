"""
Unsupervised Report Generator
Creates a standalone HTML report for unsupervised analysis only.
"""

import os
from datetime import datetime


def _safe(v, default='—'):
    return default if v is None else v


def _render_clustering(clustering):
    if not clustering:
        return '<div class="card"><p>No clustering results available.</p></div>'

    algo = _safe(clustering.get('best_algorithm'))
    optimal_k = _safe(clustering.get('optimal_k'))
    silhouette = clustering.get('best_silhouette')
    silhouette_txt = f"{silhouette:.4f}" if isinstance(silhouette, (int, float)) else '—'

    rows = ''
    for name, info in (clustering.get('algorithms') or {}).items():
        sil = info.get('silhouette')
        sil_txt = f"{sil:.4f}" if isinstance(sil, (int, float)) else '—'
        rows += f"<tr><td>{name}</td><td>{_safe(info.get('n_clusters'))}</td><td>{sil_txt}</td></tr>"

    table_html = (
        '<table><thead><tr><th>Algorithm</th><th>Clusters</th><th>Silhouette</th></tr></thead>'
        f'<tbody>{rows}</tbody></table>'
    ) if rows else '<p>No algorithm comparison available.</p>'

    return f'''
    <div class="card">
        <h3>Clustering</h3>
        <div class="grid">
            <div class="metric"><span>Best Algorithm</span><strong>{algo}</strong></div>
            <div class="metric"><span>Optimal K</span><strong>{optimal_k}</strong></div>
            <div class="metric"><span>Best Silhouette</span><strong>{silhouette_txt}</strong></div>
        </div>
        {table_html}
    </div>
    '''


def _render_anomaly(anomaly):
    if not anomaly:
        return '<div class="card"><p>No anomaly results available.</p></div>'

    n_anom = _safe(anomaly.get('n_anomalies'), 0)
    anom_pct = anomaly.get('anomaly_pct')
    anom_pct_txt = f"{anom_pct:.2f}%" if isinstance(anom_pct, (int, float)) else '—'

    return f'''
    <div class="card">
        <h3>Anomaly Detection</h3>
        <div class="grid">
            <div class="metric"><span>Anomalies</span><strong>{n_anom}</strong></div>
            <div class="metric"><span>Anomaly Rate</span><strong>{anom_pct_txt}</strong></div>
            <div class="metric"><span>Detectors</span><strong>{len(anomaly.get('detectors') or {})}</strong></div>
        </div>
    </div>
    '''


def _render_dim_reduction(dimred):
    if not dimred:
        return '<div class="card"><p>No dimensionality reduction results available.</p></div>'

    method = _safe(dimred.get('method'))
    components = dimred.get('components') or []
    return f'''
    <div class="card">
        <h3>Dimensionality Reduction</h3>
        <div class="grid">
            <div class="metric"><span>Method</span><strong>{method}</strong></div>
            <div class="metric"><span>Points Projected</span><strong>{len(components)}</strong></div>
        </div>
    </div>
    '''


def _render_association(assoc):
    if not assoc:
        return '<div class="card"><p>No association rules available.</p></div>'

    rules = assoc.get('rules') or []
    return f'''
    <div class="card">
        <h3>Association Rules</h3>
        <div class="grid">
            <div class="metric"><span>Rules</span><strong>{len(rules)}</strong></div>
        </div>
    </div>
    '''


def _render_topics(topics):
    if not topics:
        return '<div class="card"><p>No topic modeling results available.</p></div>'

    topic_list = topics.get('topics') or []
    items = ''.join([f'<li>{t}</li>' for t in topic_list[:15]])
    return f'''
    <div class="card">
        <h3>Topic Modeling</h3>
        <p>Topics discovered: <strong>{len(topic_list)}</strong></p>
        <ul>{items}</ul>
    </div>
    '''


def generate_unsupervised_html_report(session_data, output_dir):
    """Generate an HTML report for unsupervised analysis only."""
    os.makedirs(output_dir, exist_ok=True)

    unsup = session_data.get('unsupervised_results') or {}
    profile = session_data.get('profile') or {}
    session_id = session_data.get('session_id', 'unknown')

    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Unsupervised Report - {session_id}</title>
  <style>
    body {{ font-family: Arial, sans-serif; background:#f7f9fc; color:#1f2937; margin:0; padding:24px; }}
    .container {{ max-width:1000px; margin:0 auto; }}
    h1 {{ margin:0 0 8px; }}
    .sub {{ color:#6b7280; margin-bottom:16px; }}
    .card {{ background:#fff; border:1px solid #e5e7eb; border-radius:10px; padding:16px; margin-bottom:14px; }}
    .grid {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(180px,1fr)); gap:10px; margin-bottom:10px; }}
    .metric {{ background:#f8fafc; border:1px solid #e5e7eb; border-radius:8px; padding:10px; }}
    .metric span {{ display:block; color:#6b7280; font-size:12px; }}
    .metric strong {{ font-size:18px; }}
    table {{ width:100%; border-collapse:collapse; }}
    th, td {{ border-bottom:1px solid #e5e7eb; padding:8px; text-align:left; font-size:13px; }}
    th {{ color:#374151; }}
  </style>
</head>
<body>
  <div class="container">
    <h1>Unsupervised Analysis Report</h1>
    <div class="sub">Session: {session_id} · Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</div>

    <div class="card">
      <h3>Dataset Snapshot</h3>
      <div class="grid">
        <div class="metric"><span>Rows</span><strong>{_safe(profile.get('n_rows'), 0)}</strong></div>
        <div class="metric"><span>Columns</span><strong>{_safe(profile.get('n_cols'), 0)}</strong></div>
        <div class="metric"><span>Missing Data</span><strong>{_safe(profile.get('total_missing_pct'), 0)}%</strong></div>
      </div>
    </div>

    {_render_clustering(unsup.get('clustering') or {})}
    {_render_anomaly(unsup.get('anomaly') or {})}
    {_render_dim_reduction(unsup.get('dim_reduction') or {})}
    {_render_association(unsup.get('association') or {})}
    {_render_topics(unsup.get('topics') or {})}
  </div>
</body>
</html>'''

    report_path = os.path.join(output_dir, 'unsupervised_report.html')
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(html)

    return report_path
