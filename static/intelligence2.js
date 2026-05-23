// ═══════════════════════════════════════════════════════════
// AutoML Studio — Advanced Analysis Features JS (Part 2)
// ═══════════════════════════════════════════════════════════

// ── Feature #4: Model Shelf-Life ──
async function runShelflife() {
    const el = document.getElementById('shelflifeContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnShelflife');
    btn.disabled = true; btn.textContent = '⏰ Predicting…';
    _ilLoad(el, '⏰ Analyzing feature stability and predicting model shelf-life…');
    try {
        const r = await fetch(`/api/v5/shelf-life/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const days = d.estimated_days || d.shelf_life_days || '—';
        const expiry = d.expiry_date || '—';
        const risk = d.risk_level || 'medium';
        const riskColor = risk === 'low' ? '#22C55E' : risk === 'high' ? '#EF4444' : '#F59E0B';
        h += `<div style="text-align:center;padding:24px;background:linear-gradient(135deg,rgba(6,182,212,.08),rgba(99,102,241,.05));border-radius:var(--r8);border:1px solid rgba(6,182,212,.2);margin-bottom:14px">`;
        h += `<div style="font-size:48px;font-weight:900;color:var(--accent)">${days}</div><div style="font-size:12px;color:var(--txt2);text-transform:uppercase">Estimated Days Until Retraining Needed</div>`;
        if (expiry !== '—') h += `<div style="margin-top:8px;font-size:13px;font-weight:600">📅 Retrain by: ${expiry}</div>`;
        h += `<div style="margin-top:6px"><span class="tag" style="background:${riskColor}20;color:${riskColor};border:1px solid ${riskColor}40">${risk} risk</span></div></div>`;
        const feats = d.feature_stability || d.drifting_features || [];
        if (feats.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Feature Drift Velocity</span></div><table><thead><tr><th>Feature</th><th>Stability</th><th>Drift Speed</th></tr></thead><tbody>';
            feats.slice(0, 10).forEach(f => { h += `<tr><td style="font-weight:600">${f.feature || f.name || '—'}</td><td>${f.stability || '—'}</td><td style="color:${f.speed === 'fast' ? 'var(--danger)' : 'var(--txt2)'}">${f.speed || f.velocity || '—'}</td></tr>`; });
            h += '</tbody></table></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '⏰ Predict Shelf-Life';
}

// ── Feature #6: Self-Healing ──
async function runSelfheal() {
    const el = document.getElementById('selfhealContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    _ilLoad(el, '🧪 Loading self-healing status…');
    try {
        const r = await fetch(`/api/v5/self-heal/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const fixes = d.fixes_applied || d.healing_log || [];
        const total = d.total_fixes || fixes.length;
        h += `<div class="kpi-row">${_kpi('Total Auto-Fixes', total, '#22C55E')}${_kpi('Pipeline Status', d.status || 'Healthy', '#6366F1')}${_kpi('Uptime', d.uptime || '100%', '#06B6D4')}${_kpi('Errors Caught', d.errors_caught || 0, '#EF4444')}</div>`;
        if (fixes.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Healing Log</span></div><table><thead><tr><th>Time</th><th>Error</th><th>Auto-Fix Applied</th><th>Result</th></tr></thead><tbody>';
            fixes.forEach(f => { h += `<tr><td style="font-size:10px">${f.time || '—'}</td><td style="color:var(--danger)">${f.error || '—'}</td><td style="color:var(--success);font-weight:600">${f.fix || '—'}</td><td><span class="tag tg">${f.result || 'Fixed'}</span></td></tr>`; });
            h += '</tbody></table></div>';
        } else { h += '<div class="alert al-s"><span class="al-ico">✓</span>No errors encountered. Pipeline is running clean.</div>'; }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
}

// ── Feature #10: Feature X-Ray ──
async function runXray() {
    const el = document.getElementById('xrayContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnXray');
    btn.disabled = true; btn.textContent = '🧮 Scanning…';
    _ilLoad(el, '🧮 Discovering feature interactions (H-statistic + SHAP)…');
    try {
        const r = await fetch(`/api/v5/interaction-xray/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const pairs = d.interactions || d.pairs || d.pairwise_interactions || [];
        const summary = d.summary || d.recommendation || (d.total_pairs_tested != null ? `${d.significant_interactions || 0} significant interactions found across ${d.total_pairs_tested} tested pairs.` : '');
        if (summary) h += `<div class="alert al-i mb"><span class="al-ico">🧮</span>${summary}</div>`;
        const totalPairs = d.total_pairs_tested != null ? d.total_pairs_tested : '—';
        const sigCount = d.significant_interactions != null ? d.significant_interactions : pairs.length;
        h += `<div class="kpi-row">${_kpi('Interactions', sigCount, '#6366F1')}${_kpi('Pairs Tested', totalPairs, '#06B6D4')}${_kpi('Top Matches', pairs.length || '—', '#22C55E')}</div>`;
        if (pairs.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Top Feature Interactions</span></div><table><thead><tr><th>#</th><th>Feature Pair</th><th>Interaction Strength</th><th>Description</th></tr></thead><tbody>';
            pairs.slice(0, 15).forEach((p, i) => {
                const strength = p.strength ?? p.interaction_strength ?? p.h_statistic ?? 0;
                const pct = Math.min(strength * 100, 100);
                const f1 = p.feature_1 || p.feature_a || p.feature || '';
                const f2 = p.feature_2 || p.feature_b || p.other_feature || '';
                const description = p.description || p.narrative || '';
                h += `<tr><td>${i + 1}</td><td style="font-weight:600">${f1} × ${f2}</td><td><div style="display:flex;align-items:center;gap:6px"><div class="pb" style="width:80px"><div class="pb-fill" style="width:${pct}%;background:var(--accent)"></div></div><span style="font-weight:700;font-size:11px">${Number.isFinite(strength) ? strength.toFixed(3) : '—'}</span></div></td><td style="font-size:11px;color:var(--txt2)">${description}</td></tr>`;
            });
            h += '</tbody></table></div>';
        } else {
            h += '<div class="alert al-i"><span class="al-ico">ℹ</span>No strong interactions were detected for this model and dataset.</div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🧮 Scan Interactions';
}

// ── Feature #12: Sample Difficulty ──
async function runDifficulty() {
    const el = document.getElementById('difficultyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnDifficulty');
    btn.disabled = true; btn.textContent = '🎯 Scoring…';
    _ilLoad(el, '🎯 Scoring sample difficulty across multiple models…');
    try {
        const r = await fetch(`/api/v5/sample-difficulty/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        h += `<div class="kpi-row">${_kpi('Easy Samples', d.easy_count || '—', '#22C55E')}${_kpi('Medium', d.medium_count || '—', '#F59E0B')}${_kpi('Hard Samples', d.hard_count || '—', '#EF4444')}${_kpi('Mislabel Candidates', d.mislabel_count || '—', '#8B5CF6')}</div>`;
        const hardest = d.hardest_samples || d.hard_samples || [];
        if (hardest.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Hardest Samples (Potential Mislabels)</span></div><table><thead><tr><th>Index</th><th>Difficulty</th><th>Category</th><th>Details</th></tr></thead><tbody>';
            hardest.slice(0, 20).forEach(s => { h += `<tr><td>${s.index ?? '—'}</td><td style="font-weight:700;color:var(--danger)">${typeof s.difficulty === 'number' ? s.difficulty.toFixed(3) : s.difficulty || '—'}</td><td><span class="tag ${s.category === 'mislabel' ? 'tr' : s.category === 'hard' ? 'ta' : 'tb'}">${s.category || 'hard'}</span></td><td style="font-size:11px">${s.reason || ''}</td></tr>`; });
            h += '</tbody></table></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🎯 Score Samples';
}

// ── Feature #18: Error Slicing ──
async function runErrslice() {
    const el = document.getElementById('errsliceContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnErrslice');
    btn.disabled = true; btn.textContent = '🔪 Slicing…';
    _ilLoad(el, '🔪 Discovering error slices across feature subgroups…');
    try {
        const r = await fetch(`/api/v5/error-slices/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const slices = d.slices || d.error_slices || [];
        if (slices.length) {
            h += '<table><thead><tr><th>Slice</th><th>Size</th><th>Error Rate</th><th>vs Overall</th><th>Severity</th></tr></thead><tbody>';
            slices.forEach(s => {
                const errRate = s.error_rate != null ? (s.error_rate * 100).toFixed(1) + '%' : '—';
                const sevCls = s.severity === 'critical' ? 'tr' : s.severity === 'high' ? 'ta' : 'tb';
                h += `<tr><td style="font-weight:600">${s.slice_name || s.condition || '—'}</td><td>${s.size || '—'}</td><td style="font-weight:700;color:var(--danger)">${errRate}</td><td>${s.relative || ''}</td><td><span class="tag ${sevCls}">${s.severity || '—'}</span></td></tr>`;
            });
            h += '</tbody></table>';
        } else { h += '<div class="alert al-s"><span class="al-ico">✓</span>No significant error slices found. Model performs consistently.</div>'; }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🔪 Slice Errors';
}

// ── Feature #11: Conformal Prediction ──
async function runConformal() {
    const el = document.getElementById('conformalContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnConformal');
    btn.disabled = true; btn.textContent = '📐 Building…';
    _ilLoad(el, '📐 Building conformal predictor with mathematical guarantees…');
    try {
        const r = await fetch(`/api/v5/conformal/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        h += `<div class="kpi-row">${_kpi('Coverage', d.coverage ?? (d.actual_coverage || '—'), '#22C55E')}${_kpi('Avg Set Size', d.avg_set_size || d.avg_interval_width || '—', '#6366F1')}${_kpi('Target Coverage', d.target_coverage || '90%', '#06B6D4')}${_kpi('Calibration', d.calibrated ? '✓ Yes' : '— Pending', '#F59E0B')}</div>`;
        if (d.samples && d.samples.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Sample Predictions with Guaranteed Bounds</span></div><table><thead><tr><th>#</th><th>Prediction</th><th>Lower / Set</th><th>Upper</th><th>Actual</th></tr></thead><tbody>';
            d.samples.slice(0, 15).forEach((s, i) => { h += `<tr><td>${i + 1}</td><td style="font-weight:600">${s.prediction || '—'}</td><td>${s.lower || s.prediction_set || '—'}</td><td>${s.upper || ''}</td><td>${s.actual || '—'}</td></tr>`; });
            h += '</tbody></table></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '📐 Build Conformal Predictor';
}

// ── Feature #13: Learning Curve ──
async function runLearncurve() {
    const el = document.getElementById('learncurveContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnLearncurve');
    btn.disabled = true; btn.textContent = '📈 Computing…';
    _ilLoad(el, '📈 Training on increasing data fractions and fitting power-law curve…');
    try {
        const r = await fetch(`/api/v5/learning-curve/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const ext = d.extrapolations || d.predictions || {};
        h += `<div class="kpi-row">${_kpi('Current Score', d.current_score?.toFixed(4) || '—', '#6366F1')}${_kpi('At 2x Data', ext['2x']?.toFixed(4) || '—', '#10B981')}${_kpi('At 5x Data', ext['5x']?.toFixed(4) || '—', '#06B6D4')}${_kpi('At 10x Data', ext['10x']?.toFixed(4) || '—', '#22C55E')}</div>`;
        const pts = d.curve_points || d.learning_curve || [];
        if (pts.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Learning Curve Data</span></div><table><thead><tr><th>Data Fraction</th><th>Train Score</th><th>Val Score</th></tr></thead><tbody>';
            pts.forEach(p => { h += `<tr><td>${p.fraction || p.train_size || '—'}</td><td>${p.train_score?.toFixed(4) || '—'}</td><td style="font-weight:700;color:var(--accent)">${p.val_score?.toFixed(4) || p.test_score?.toFixed(4) || '—'}</td></tr>`; });
            h += '</tbody></table></div>';
        }
        if (d.verdict) h += `<div class="alert al-i" style="margin-top:10px"><span class="al-ico">📈</span>${d.verdict}</div>`;
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '📈 Predict Learning Curve';
}

// ── Feature #19: CV Stability ──
async function runCvstability() {
    const el = document.getElementById('cvstabilityContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnCvstability');
    btn.disabled = true; btn.textContent = '🔄 Analyzing…';
    _ilLoad(el, '🔄 Tracking per-sample prediction stability across CV folds…');
    try {
        const r = await fetch(`/api/v5/cv-stability/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        h += `<div class="kpi-row">${_kpi('Stable Samples', d.stable_count || '—', '#22C55E')}${_kpi('Unstable Samples', d.unstable_count || '—', '#EF4444')}${_kpi('Avg Stability', d.avg_stability?.toFixed(3) || '—', '#6366F1')}${_kpi('Flip-Flop Rate', d.flipflop_rate || '—', '#F59E0B')}</div>`;
        const unstable = d.unstable_samples || [];
        if (unstable.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Most Unstable Samples (Flip-Flop Across Folds)</span></div><table><thead><tr><th>Index</th><th>Stability</th><th>Folds Correct</th><th>Category</th></tr></thead><tbody>';
            unstable.slice(0, 20).forEach(s => { h += `<tr><td>${s.index ?? '—'}</td><td style="font-weight:700;color:var(--danger)">${typeof s.stability === 'number' ? s.stability.toFixed(3) : s.stability || '—'}</td><td>${s.folds_correct || '—'}</td><td><span class="tag tr">${s.category || 'unstable'}</span></td></tr>`; });
            h += '</tbody></table></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🔄 Analyze CV Stability';
}

// ── Feature #25: Data Sufficiency ──
async function runSufficiency() {
    const el = document.getElementById('sufficiencyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnSufficiency');
    btn.disabled = true; btn.textContent = '📊 Checking…';
    _ilLoad(el, '📊 Estimating data sufficiency for reliable modeling…');
    try {
        const r = await fetch(`/api/v5/data-sufficiency/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const verdict = d.verdict || d.overall_verdict || 'unknown';
        const vColor = verdict.toLowerCase().includes('sufficient') ? '#22C55E' : verdict.toLowerCase().includes('marginal') ? '#F59E0B' : '#EF4444';
        h += `<div style="text-align:center;padding:24px;background:${vColor}10;border:2px solid ${vColor}40;border-radius:var(--r8);margin-bottom:14px"><div style="font-size:28px;font-weight:900;color:${vColor}">${verdict.toUpperCase()}</div>`;
        if (d.recommended_samples) h += `<div style="margin-top:6px;font-size:13px">Recommended: <strong>${d.recommended_samples}</strong> samples (you have ${d.current_samples || '—'})</div>`;
        h += '</div>';
        const checks = d.checks || d.details || [];
        if (checks.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">Sufficiency Checks</span></div><table><thead><tr><th>Check</th><th>Result</th><th>Details</th></tr></thead><tbody>';
            checks.forEach(c => {
                const pass = c.passed || c.status === 'pass';
                h += `<tr><td style="font-weight:600">${c.name || c.check || '—'}</td><td><span class="tag ${pass ? 'tg' : 'tr'}">${pass ? 'PASS' : 'FAIL'}</span></td><td style="font-size:11px">${c.details || c.message || ''}</td></tr>`;
            });
            h += '</tbody></table></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '📊 Check Data Sufficiency';
}

// ── Feature #7: Research Paper ──
async function runPaper() {
    const el = document.getElementById('paperContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnPaper');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Generating Paper…';
    _ilLoad(el, '📖 Generating structured research paper (Abstract, Methodology, Results, Discussion)…');
    try {
        const r = await fetch(`/api/v5/paper/${_getSessionId()}`);
        if (!r.ok) {
            const txt = await r.text();
            _ilErr(el, txt || `HTTP ${r.status}`);
            return;
        }
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const paper = d.paper || d;
        const sections = paper.sections || [];
        h += '<div style="max-width:800px;margin:0 auto;background:var(--bg0);border:1px solid var(--bdr);border-radius:var(--r8);padding:30px;box-shadow:0 2px 20px rgba(0,0,0,.05)">';
        h += `<div style="text-align:center;margin-bottom:24px;border-bottom:2px solid var(--bdr);padding-bottom:20px"><h2 style="font-size:18px;font-weight:800;margin-bottom:6px">${paper.title || 'AutoML Experiment Report'}</h2>`;
        h += `<div style="font-size:11px;color:var(--txt2)">${paper.authors || 'AutoML Studio'} — ${paper.date || new Date().toLocaleDateString()}</div></div>`;
        if (sections.length) {
            sections.forEach(sec => {
                h += `<div style="margin-bottom:18px"><h3 style="font-size:14px;font-weight:700;margin-bottom:6px;color:var(--accent)">${sec.title || ''}</h3><div style="font-size:12px;line-height:1.7;color:var(--txt2)">${sec.content || ''}</div></div>`;
            });
        } else if (paper.html) { h += paper.html; }
        else if (paper.markdown || paper.text) { h += `<div style="font-size:12px;line-height:1.7;white-space:pre-wrap">${paper.markdown || paper.text}</div>`; }
        h += '</div>';
        if (paper.download_url || d.download_url) { h += `<div style="text-align:center;margin-top:14px"><a href="${paper.download_url || d.download_url}" class="btn primary" download>⬇ Download Paper</a></div>`; }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.innerHTML = '📖 Generate Research Paper';
}

// ═══════════════════════════════════════════════════════════
// Auto-load hooks (called by nav system when page is shown)
// ═══════════════════════════════════════════════════════════
function loadShelflife() {
    const el = document.getElementById('shelflifeContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Predict Shelf-Life</b> to estimate when retraining will be needed.</div>';
    }
}

function loadSelfheal() {
    const el = document.getElementById('selfhealContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    // Auto-fetch healing status since it's lightweight
    runSelfheal();
}

function loadXray() {
    const el = document.getElementById('xrayContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Scan Interactions</b> to discover feature interactions.</div>';
    }
}

function loadDifficulty() {
    const el = document.getElementById('difficultyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Score Samples</b> to rate every sample\'s difficulty.</div>';
    }
}

function loadErrslice() {
    const el = document.getElementById('errsliceContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Slice Errors</b> to discover where your model fails.</div>';
    }
}

function loadConformal() {
    const el = document.getElementById('conformalContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Build Conformal Predictor</b> for mathematically guaranteed predictions.</div>';
    }
}

function loadLearncurve() {
    const el = document.getElementById('learncurveContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Predict Learning Curve</b> to see how more data improves performance.</div>';
    }
}

function loadCvstability() {
    const el = document.getElementById('cvstabilityContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Analyze CV Stability</b> to find flip-flop predictions.</div>';
    }
}

function loadSufficiency() {
    const el = document.getElementById('sufficiencyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Upload a dataset')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Check Data Sufficiency</b> to verify your dataset size.</div>';
    }
}

function loadPaper() {
    const el = document.getElementById('paperContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Complete a full')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Generate Research Paper</b> to create a full academic paper from your experiment.</div>';
    }
}
