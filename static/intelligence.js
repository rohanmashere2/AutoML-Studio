// ═══════════════════════════════════════════════════════════
// AutoML Studio — Intelligence Lab & Advanced Analysis JS
// ═══════════════════════════════════════════════════════════

// Session ID helper — reads from global S or falls back to localStorage
function _getSessionId() {
    if (window.S && window.S.sessionId) return window.S.sessionId;
    // Fallback: read directly from localStorage (same key used by dashboard)
    return localStorage.getItem('aml_sid') || null;
}

function _ilLoad(el, msg) { el.innerHTML = `<div style="text-align:center;padding:30px;color:var(--txt2)">${msg}</div>`; }
function _ilErr(el, e) { el.innerHTML = `<div class="alert al-d"><span class="al-ico">❌</span>${e}</div>`; }
function _ilNoSession(el) { _ilLoad(el, '⚠️ No active session. Upload a dataset first.'); }
function _kpi(label, value, color) { return `<div class="kpi-card"><div class="kpi-lbl">${label}</div><div class="kpi-val" style="color:${color || 'var(--accent)'}">${value}</div></div>`; }

// ── Feature #1: Model Prophecy ──
async function runProphecy() {
    const el = document.getElementById('prophecyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnProphecy');
    btn.disabled = true; btn.textContent = '🔮 Analyzing…';
    _ilLoad(el, '🔮 Computing dataset DNA (50+ meta-features)…');
    try {
        const r = await fetch(`/api/v5/dataset-dna/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const dna = d.dna || d.dataset_dna || d.meta_features || {};
        const pred = d.prophecy || d.predictions || {};
        const winner = pred.predicted_winner || pred.best_algorithm || '—';
        const score = pred.estimated_score != null ? pred.estimated_score.toFixed(4) : '—';
        const conf = pred.confidence != null ? (pred.confidence * 100).toFixed(0) + '%' : '—';
        const time = pred.estimated_time || '—';
        h += `<div class="kpi-row">${_kpi('🏆 Predicted Winner', winner, '#6366F1')}${_kpi('📊 Est. Score', score, '#10B981')}${_kpi('🎯 Confidence', conf, '#F59E0B')}${_kpi('⏱ Est. Time', time, '#06B6D4')}</div>`;
        if (pred.ranking && pred.ranking.length) {
            h += '<div class="card mb"><div class="card-hdr"><span class="card-title">Algorithm Ranking Prediction</span></div><table><thead><tr><th>Rank</th><th>Algorithm</th><th>Est. Score</th><th>Confidence</th></tr></thead><tbody>';
            pred.ranking.forEach((r, i) => { h += `<tr${i === 0 ? ' style="background:rgba(99,102,241,.08)"' : ''}><td><strong>${i + 1}</strong></td><td style="font-weight:600">${r.algorithm || r.model || '—'}</td><td style="font-weight:700;color:var(--success)">${(r.score || 0).toFixed(4)}</td><td>${r.confidence ? ((r.confidence * 100).toFixed(0) + '%') : '—'}</td></tr>`; });
            h += '</tbody></table></div>';
        }
        const dnaKeys = Object.keys(dna);
        if (dnaKeys.length) {
            h += '<div class="card"><div class="card-hdr"><span class="card-title">🧬 Dataset DNA (Meta-Features)</span></div><div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:8px">';
            dnaKeys.slice(0, 20).forEach(k => { h += `<div style="padding:8px;background:var(--bg1);border-radius:var(--r4);border:1px solid var(--bdr)"><div style="font-size:9px;color:var(--txt2);text-transform:uppercase">${k}</div><div style="font-size:13px;font-weight:700">${typeof dna[k] === 'number' ? dna[k].toFixed(3) : dna[k]}</div></div>`; });
            h += '</div></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🔮 Analyze Dataset DNA';
}

// ── Feature #5: Prediction Autopsy ──
async function runAutopsy() {
    const el = document.getElementById('autopsyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const idx = parseInt(document.getElementById('autopsyIdx').value) || 0;
    const btn = document.getElementById('btnAutopsy');
    btn.disabled = true; btn.textContent = '🔬 Analyzing…';
    _ilLoad(el, '🔬 Performing prediction autopsy on sample #' + idx + '…');
    try {
        const r = await fetch(`/api/v5/prediction-autopsy/${_getSessionId()}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sample_index: idx }) });
        if (!r.ok) {
            const txt = await r.text();
            _ilErr(el, txt || `HTTP ${r.status}`);
            return;
        }
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '<div style="font-family:var(--mono);font-size:12px;background:#1a1a2e;color:#e0e0e0;padding:20px;border-radius:var(--r8);line-height:1.8;white-space:pre-wrap">';
        h += `<span style="color:#FFD700;font-size:14px;font-weight:700">📋 PREDICTION AUTOPSY — Sample #${idx}</span>\n`;
        h += `<span style="color:#888">━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━</span>\n`;
        const pred = d.prediction ?? d.predicted_class ?? '—';
        const actual = d.actual ?? d.actual_class ?? '—';
        const conf2 = d.confidence != null ? (d.confidence * 100).toFixed(1) + '%' : '—';
        const correct = d.correct != null ? (d.correct ? '✓' : '✗') : '';
        h += `Prediction: <span style="color:#60A5FA;font-weight:700">${pred}</span> (${conf2} confidence)\n`;
        h += `Actual:     <span style="color:${d.correct ? '#22C55E' : '#EF4444'};font-weight:700">${actual} ${correct}</span>\n\n`;
        if (d.decision_path && d.decision_path.length) {
            h += `<span style="color:#FFD700">🧠 Decision Path:</span>\n`;
            d.decision_path.forEach((s, i) => { h += `  Step ${i + 1}: <span style="color:#818CF8">${s.feature || ''}</span>=${s.value || ''} → ${s.effect || s.description || ''}\n`; });
            h += '\n';
        }
        if (d.similar_samples) { h += `<span style="color:#FFD700">📊 Similar Samples:</span>\n  ${d.similar_samples.count || 0} similar in training data\n  ${d.similar_samples.agreement || '—'}% agree with prediction\n\n`; }
        if (d.counterfactuals && d.counterfactuals.length) {
            h += `<span style="color:#FFD700">🔄 What Would Change the Prediction:</span>\n`;
            d.counterfactuals.forEach(cf => { h += `  If <span style="color:#F59E0B">${cf.feature}</span> → ${cf.value} then prediction flips to <span style="color:#22C55E">${cf.new_prediction || 'opposite'}</span>\n`; });
        }
        h += '</div>';
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '🔬 Run Autopsy';
}

// ── Feature #2: Data Prescription ──
async function runPrescription() {
    const el = document.getElementById('prescriptionContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnPrescription');
    btn.disabled = true; btn.textContent = '💊 Analyzing…';
    _ilLoad(el, '💊 Generating data collection prescriptions…');
    try {
        const r = await fetch(`/api/v5/data-prescription/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const rx = d.prescriptions || d.recommendations || [];
        if (d.summary) h += `<div class="alert al-i mb"><span class="al-ico">💊</span>${d.summary}</div>`;
        if (rx.length) {
            rx.forEach((p, i) => {
                const impact = p.impact || p.priority || 'medium';
                const impCls = impact === 'high' ? 'tr' : impact === 'medium' ? 'ta' : 'tg';
                h += `<div class="card mb" style="border-left:3px solid ${impact === 'high' ? 'var(--danger)' : impact === 'medium' ? 'var(--warn)' : 'var(--success)'}"><div class="card-hdr"><span class="card-title">Rx #${i + 1}: ${p.title || p.action || 'Collect More Data'}</span><span class="tag ${impCls}">${impact} impact</span></div><div style="font-size:12px;color:var(--txt2);line-height:1.6">${p.description || p.details || ''}</div>`;
                if (p.estimated_improvement) h += `<div style="margin-top:8px;font-weight:700;color:var(--success)">📈 Estimated improvement: ${p.estimated_improvement}</div>`;
                if (p.samples_needed) h += `<div style="font-size:11px;color:var(--txt2)">Samples needed: ~${p.samples_needed}</div>`;
                h += '</div>';
            });
        } else { h += '<div class="alert al-i"><span class="al-ico">ℹ</span>No specific prescriptions generated. Your data coverage may be sufficient.</div>'; }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.textContent = '💊 Generate Prescription';
}

// ── Feature #3: Adversarial Stress Test ──
async function runAdversarial() {
    const el = document.getElementById('adversarialContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnAdversarial');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Attacking…';
    _ilLoad(el, '🛡️ Running adversarial attacks (perturbation, boundary, distribution shift, missing data)…');
    try {
        const r = await fetch(`/api/v5/adversarial-test/${_getSessionId()}`);
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const grade = d.overall_grade || d.robustness_grade || 'C';
        const score = d.robustness_score != null ? d.robustness_score : 50;
        const gradeColor = { A: '#22C55E', B: '#10B981', C: '#F59E0B', D: '#F97316', F: '#EF4444' }[grade[0]] || '#F59E0B';
        h += `<div style="text-align:center;padding:20px;margin-bottom:14px"><div style="display:inline-flex;align-items:center;justify-content:center;width:100px;height:100px;border-radius:50%;border:5px solid ${gradeColor};font-size:42px;font-weight:900;color:${gradeColor}">${grade}</div><div style="margin-top:8px;font-size:14px;font-weight:700">Robustness Score: ${score}/100</div></div>`;
        const attacks = d.attacks || d.results || [];
        if (attacks.length) {
            h += '<table><thead><tr><th>Attack Type</th><th>Result</th><th>Severity</th><th>Details</th></tr></thead><tbody>';
            attacks.forEach(a => {
                const sev = a.severity || 'medium';
                const sevCls = sev === 'high' || sev === 'critical' ? 'tr' : sev === 'medium' ? 'ta' : 'tg';
                h += `<tr><td style="font-weight:600">${a.attack_type || a.name || '—'}</td><td>${a.result || '—'}</td><td><span class="tag ${sevCls}">${sev}</span></td><td style="font-size:11px;color:var(--txt2)">${a.details || a.description || ''}</td></tr>`;
            });
            h += '</tbody></table>';
        }
        if (d.vulnerabilities && d.vulnerabilities.length) {
            h += '<div class="alert al-w" style="margin-top:12px"><span class="al-ico">⚠</span><div><strong>Vulnerabilities Found:</strong><ul style="margin:4px 0 0 16px">';
            d.vulnerabilities.forEach(v => { h += `<li>${v}</li>`; });
            h += '</ul></div></div>';
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.innerHTML = '🛡️ Launch Stress Test';
}

// ── Feature #9: Tournament ──
async function runTournament() {
    const el = document.getElementById('tournamentContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    const btn = document.getElementById('btnTournament');
    btn.disabled = true; btn.innerHTML = '<span class="spinner"></span> Running Tournament…';
    _ilLoad(el, '⚔️ Running model tournament (this may take a few minutes)…');
    try {
        const r = await fetch(`/api/v5/tournament/${_getSessionId()}`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: '{}' });
        const d = await r.json();
        if (d.error) { _ilErr(el, d.error); return; }
        let h = '';
        const champion = d.champion || d.winner || {};
        h += `<div style="text-align:center;padding:20px;background:linear-gradient(135deg,rgba(245,158,11,.1),rgba(217,119,6,.05));border-radius:var(--r8);border:1px solid rgba(245,158,11,.3);margin-bottom:14px"><div style="font-size:40px">🏆</div><div style="font-size:18px;font-weight:800;margin:4px 0">${champion.model || champion.name || 'Champion'}</div><div style="font-size:24px;font-weight:900;color:var(--accent)">${champion.score != null ? champion.score.toFixed(4) : '—'}</div><div style="font-size:10px;color:var(--txt2);text-transform:uppercase">Tournament Champion</div></div>`;
        const rounds = d.rounds || d.bracket || [];
        if (rounds.length) {
            rounds.forEach((round, ri) => {
                h += `<div class="card mb"><div class="card-hdr"><span class="card-title">Round ${ri + 1}: ${round.name || 'Round ' + (ri + 1)}</span><span class="badge bg-blue">${(round.models || round.competitors || []).length} competitors</span></div>`;
                h += '<table><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Status</th></tr></thead><tbody>';
                (round.models || round.competitors || round.results || []).forEach((m, mi) => {
                    const adv = m.advanced || m.passed || mi < (round.advance_count || 5);
                    h += `<tr style="${adv ? '' : 'opacity:0.5'}"><td>${mi + 1}</td><td style="font-weight:600">${m.model || m.name || '—'}</td><td style="font-weight:700;color:${adv ? 'var(--success)' : 'var(--txt2)'}">${m.score != null ? m.score.toFixed(4) : '—'}</td><td>${adv ? '<span class="tag tg">Advanced</span>' : '<span class="tag tgy">Eliminated</span>'}</td></tr>`;
                });
                h += '</tbody></table></div>';
            });
        }
        el.innerHTML = h;
    } catch (e) { _ilErr(el, e.message); }
    btn.disabled = false; btn.innerHTML = '⚔️ Start Tournament';
}

// ═══════════════════════════════════════════════════════════
// Auto-load hooks (called by nav system when page is shown)
// ═══════════════════════════════════════════════════════════
function loadProphecy() {
    const el = document.getElementById('prophecyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    // Don't auto-run heavy computation, just show ready state
    if (el && el.innerHTML.includes('Upload a dataset')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Session active. Click <b>Analyze Dataset DNA</b> to predict the best algorithm.</div>';
    }
}

function loadAutopsy() {
    const el = document.getElementById('autopsyContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Enter a sample')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Session active. Enter a sample index and click <b>Run Autopsy</b>.</div>';
    }
}

function loadPrescription() {
    const el = document.getElementById('prescriptionContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Generate Prescription</b> to get data collection recommendations.</div>';
    }
}

function loadAdversarial() {
    const el = document.getElementById('adversarialContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Train a model')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Launch Stress Test</b> to attack your model and find vulnerabilities.</div>';
    }
}

function loadTournament() {
    const el = document.getElementById('tournamentContent');
    if (!_getSessionId()) { _ilNoSession(el); return; }
    if (el && el.innerHTML.includes('Upload a dataset')) {
        el.innerHTML = '<div style="text-align:center;padding:40px;color:var(--txt2)">✅ Ready. Click <b>Start Tournament</b> to run a 50+ model bracket competition.</div>';
    }
}
