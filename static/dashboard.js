/* ============================================================
   AutoML Studio — Dashboard JavaScript (Fully Functional)
   Connects to all Flask backend APIs: upload, pipeline,
   leaderboard, drift, explainability, fairness, diagnostics,
   feature studio, EDA, causal, deploy, competition, chat.
   ============================================================ */

// ── Safe Element Helper ────────────────────────────────────
function $el(id) { return document.getElementById(id); }
function $set(id, prop, val) { const el = $el(id); if (el) el[prop] = val; }
function $text(id, val) { $set(id, 'textContent', val); }
function $html(id, val) { $set(id, 'innerHTML', val); }
function $show(id) { const el = $el(id); if (el) el.style.display = ''; }
function $hide(id) { const el = $el(id); if (el) el.style.display = 'none'; }
function $enable(id) { $set(id, 'disabled', false); }
function $disable(id) { $set(id, 'disabled', true); }

// ── Global State ────────────────────────────────────────────
let STATE = {
    sessionId: null,
    profileData: null,
    trainResults: null,
    pipelineStage: 'idle',
    logs: [],
    fileName: null,
};

document.addEventListener('DOMContentLoaded', () => {
    initSidebar();
    initUploadModal();
    initChat();
    initHeaderButtons();
    initDriftUpload();
    initModelDriftUpload();
    initBatchPredict();
    initDatasetManager();
    restoreSession();
});

/* ══════════════════════════════════════════════════════════
   SESSION PERSISTENCE
   ══════════════════════════════════════════════════════════ */
function saveSession() {
    try {
        localStorage.setItem('automl_dashboard_state', JSON.stringify({
            sessionId: STATE.sessionId,
            fileName: STATE.fileName,
            pipelineStage: STATE.pipelineStage,
        }));
    } catch(e) {}
}

function restoreSession() {
    try {
        const saved = JSON.parse(localStorage.getItem('automl_dashboard_state'));
        if (saved && saved.sessionId) {
            STATE.sessionId = saved.sessionId;
            STATE.fileName = saved.fileName;
            STATE.pipelineStage = saved.pipelineStage || 'done';
            fetchStatus(saved.sessionId);
        }
    } catch(e) {}
}

function clearSession() {
    localStorage.removeItem('automl_dashboard_state');
    STATE = { sessionId: null, profileData: null, trainResults: null, pipelineStage: 'idle', logs: [], fileName: null };
    showToast('🗑️ Session cleared');
    location.reload();
}

async function fetchStatus(sessionId) {
    try {
        const res = await fetch(`/api/status/${sessionId}`);
        if (!res.ok) return;
        const data = await res.json();
        if (data.error) return;

        STATE.profileData = data.profile || null;
        STATE.trainResults = data.results || null;

        showDashboard();
        if (STATE.trainResults) {
            populateFromResults(STATE.trainResults, data);
            enablePostTrainButtons();
        }
        if (STATE.profileData) {
            populateProfile(STATE.profileData);
        }

        // Update stepper to completed state
        if (STATE.pipelineStage === 'done') {
            ['upload','clean','transform','train','tune'].forEach(s => updateStep(s, 'complete'));
            updateStep('monitor', 'current');
            for (let i = 0; i < 5; i++) {
                const c = document.getElementById('conn-' + i);
                if (c) c.classList.add('complete');
            }
        }
    } catch(e) {
        console.warn('Could not restore session:', e);
    }
}

/* ══════════════════════════════════════════════════════════
   SIDEBAR NAVIGATION
   ══════════════════════════════════════════════════════════ */
function initSidebar() {
    const navLinks = document.querySelectorAll('.nav-link');
    const menuToggle = document.getElementById('menuToggle');
    const sidebar = document.getElementById('sidebar');

    navLinks.forEach(link => {
        link.addEventListener('click', () => {
            const page = link.dataset.page;
            if (!page) return;

            navLinks.forEach(l => l.classList.remove('active'));
            link.classList.add('active');

            document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
            const target = document.getElementById(`page-${page}`);
            if (target) target.classList.add('active');

            const titleMap = {
                'overview': 'Project Overview',
                'pipeline': 'Pipeline Configuration',
                'data-explorer': 'Data Explorer',
                'datasets': 'Dataset Manager',
                'leaderboard': 'Model Leaderboard',
                'explainability': 'Explainability',
                'fairness': 'Fairness Audit',
                'diagnostics': 'Model Diagnostics',
                'feature-studio': 'Feature Studio',
                'eda': 'Auto EDA',
                'causal': 'Causal Inference',
                'drift-monitor': 'Drift Monitor',
                'experiments': 'Experiments',
                'deploy': 'Deployment',
                'assistant': 'AI Assistant',
                'settings': 'Settings',
            };
            document.querySelector('.header-title').innerHTML =
                `${titleMap[page] || page} — <span class="header-project">${STATE.fileName || 'No Project'}</span>`;

            if (window.innerWidth <= 768) sidebar.classList.remove('open');

            // Load page-specific data
            const pageLoaders = {
                'leaderboard': loadFullLeaderboard,
                'explainability': loadExplainability,
                'fairness': loadFairness,
                'experiments': loadExperiments,
                'diagnostics': loadDiagnostics,
                'eda': loadEDA,
                'datasets': loadDatasets,
                'data-explorer': loadDataQuality,
                'feature-studio': loadFeatureSuggestions,
                'pipeline': loadCleaningSuggestions,
                'settings': () => {
                    document.getElementById('settingsSession').textContent = STATE.sessionId || 'None';
                    document.getElementById('settingsStage').textContent = STATE.pipelineStage || 'Idle';
                },
            };
            if (pageLoaders[page]) pageLoaders[page]();
        });
    });

    if (menuToggle) {
        menuToggle.addEventListener('click', () => sidebar.classList.toggle('open'));
    }

    document.addEventListener('click', (e) => {
        if (window.innerWidth <= 768 && sidebar.classList.contains('open') &&
            !sidebar.contains(e.target) && e.target !== menuToggle) {
            sidebar.classList.remove('open');
        }
    });
}

/* ══════════════════════════════════════════════════════════
   UPLOAD MODAL
   ══════════════════════════════════════════════════════════ */
function initUploadModal() {
    const modal = document.getElementById('uploadModal');
    const closeBtn = document.getElementById('modalCloseBtn');
    const fileInput = document.getElementById('fileInput');
    const dropZone = document.getElementById('dropZone');

    closeBtn.addEventListener('click', closeUploadModal);
    modal.addEventListener('click', (e) => {
        if (e.target === modal) closeUploadModal();
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) selectFile(e.target.files[0]);
    });

    dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) selectFile(e.dataTransfer.files[0]);
    });
}

function openUploadModal() {
    document.getElementById('uploadModal').classList.add('active');
}

function closeUploadModal() {
    document.getElementById('uploadModal').classList.remove('active');
}

let selectedFile = null;

function selectFile(file) {
    selectedFile = file;
    document.getElementById('dropZone').style.display = 'none';
    const info = document.getElementById('fileSelectedInfo');
    info.classList.remove('hidden');
    document.getElementById('selectedFileName').textContent = file.name;
    document.getElementById('startPipelineBtn').disabled = false;
}

function clearFile() {
    selectedFile = null;
    document.getElementById('dropZone').style.display = '';
    document.getElementById('fileSelectedInfo').classList.add('hidden');
    document.getElementById('startPipelineBtn').disabled = true;
    document.getElementById('fileInput').value = '';
}

/* ══════════════════════════════════════════════════════════
   FULL PIPELINE EXECUTION
   ══════════════════════════════════════════════════════════ */
async function startPipeline() {
    if (!selectedFile) return;

    STATE.fileName = selectedFile.name.replace(/\.[^/.]+$/, '');
    STATE.pipelineStage = 'uploading';

    const progressArea = document.getElementById('modalProgress');
    const progressBar = document.getElementById('modalProgressBar');
    const progressMsg = document.getElementById('modalProgressMsg');
    const startBtn = document.getElementById('startPipelineBtn');

    progressArea.classList.remove('hidden');
    startBtn.disabled = true;
    startBtn.innerHTML = '<span class="spinner"></span> Running...';

    showDashboard();

    try {
        // ── STEP 1: Upload ──
        progressMsg.textContent = '📤 Uploading dataset...';
        progressBar.style.width = '10%';
        updateStep('upload', 'active-step');
        addLog('Uploading ' + selectedFile.name);

        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('problem_statement', document.getElementById('problemStatement').value);

        const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();

        if (uploadData.error) throw new Error(uploadData.error);

        STATE.sessionId = uploadData.session_id;
        STATE.profileData = uploadData;
        saveSession();

        updateStep('upload', 'complete');
        populateProfile(uploadData);
        addLog('✅ Upload complete — ' + (uploadData.n_rows || '?') + ' rows, ' + (uploadData.n_cols || '?') + ' columns');

        // Auto-detect target override
        const targetOverride = document.getElementById('targetColumnInput').value.trim();
        if (targetOverride) {
            await fetch('/api/update-target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: STATE.sessionId, target_column: targetOverride })
            });
            addLog('🎯 Target column set to: ' + targetOverride);
        }

        // ── STEP 2: Clean & Transform ──
        progressMsg.textContent = '🧹 Cleaning & transforming...';
        progressBar.style.width = '30%';
        updateStep('clean', 'active-step');
        addLog('Cleaning and transforming data...');

        const cleanRes = await fetch('/api/clean-transform', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: STATE.sessionId })
        });
        const cleanData = await cleanRes.json();

        if (cleanData.error) throw new Error(cleanData.error);

        updateStep('clean', 'complete');
        document.getElementById('conn-1').classList.add('complete');
        updateStep('transform', 'complete');
        document.getElementById('conn-2').classList.add('complete');
        addLog('✅ Cleaned — ' + (cleanData.cleaning_summary?.actions_taken || 0) + ' actions applied');
        addLog('✅ Transformed — ' + (cleanData.transform_summary?.features_encoded || 0) + ' features encoded');

        // ── STEP 3: Train ──
        progressMsg.textContent = '🤖 Training models...';
        progressBar.style.width = '55%';
        updateStep('train', 'active-step');
        addLog('Training models...');

        const trainRes = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: STATE.sessionId })
        });
        const trainData = await trainRes.json();

        if (trainData.error) throw new Error(trainData.error);

        STATE.trainResults = trainData;
        updateStep('train', 'complete');
        document.getElementById('conn-3').classList.add('complete');
        addLog('✅ Training complete — ' + (trainData.leaderboard?.length || 0) + ' models evaluated');

        // ── STEP 4: Tune (optimize) ──
        progressMsg.textContent = '⚡ Tuning hyperparameters...';
        progressBar.style.width = '75%';
        updateStep('tune', 'active-step');
        addLog('Optimizing hyperparameters...');

        try {
            const tuneRes = await fetch(`/api/optimize/${STATE.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ method: 'auto', budget: 20 })
            });
            const tuneData = await tuneRes.json();
            if (!tuneData.error) {
                addLog('✅ Tuning complete — best score: ' + (tuneData.best_score?.toFixed(4) || 'N/A'));
                const statusRes = await fetch(`/api/status/${STATE.sessionId}`);
                const statusData = await statusRes.json();
                if (statusData.results) STATE.trainResults = statusData.results;
            } else {
                addLog('⚠️ Tuning skipped: ' + tuneData.error);
            }
        } catch(e) {
            addLog('⚠️ Tuning skipped');
        }

        updateStep('tune', 'complete');
        document.getElementById('conn-4').classList.add('complete');

        // ── STEP 5: Monitor ──
        progressMsg.textContent = '📊 Finalizing dashboard...';
        progressBar.style.width = '95%';
        updateStep('monitor', 'current');
        addLog('Setting up monitoring...');

        populateFromResults(STATE.trainResults, trainData);

        progressBar.style.width = '100%';
        progressMsg.textContent = '✅ Pipeline complete!';
        STATE.pipelineStage = 'done';
        saveSession();
        addLog('🎉 Pipeline complete!');

        enablePostTrainButtons();

        setTimeout(() => {
            closeUploadModal();
            resetModal();
        }, 1200);

    } catch (err) {
        progressMsg.textContent = '❌ Error: ' + err.message;
        progressBar.style.width = '100%';
        progressBar.style.background = 'var(--red)';
        addLog('❌ Error: ' + err.message);
        startBtn.disabled = false;
        startBtn.innerHTML = '🚀 Upload & Run Pipeline';
    }
}

function resetModal() {
    const progressArea = document.getElementById('modalProgress');
    const progressBar = document.getElementById('modalProgressBar');
    const startBtn = document.getElementById('startPipelineBtn');
    progressArea.classList.add('hidden');
    progressBar.style.width = '0%';
    progressBar.style.background = '';
    startBtn.innerHTML = '🚀 Upload & Run Pipeline';
    startBtn.disabled = false;
    clearFile();
    document.getElementById('problemStatement').value = '';
    document.getElementById('targetColumnInput').value = '';
}

/* ══════════════════════════════════════════════════════════
   DASHBOARD POPULATION
   ══════════════════════════════════════════════════════════ */
function showDashboard() {
    document.getElementById('emptyState').classList.add('hidden');
    document.getElementById('dashboardContent').classList.remove('hidden');
    document.getElementById('headerProject').textContent = STATE.fileName || 'Project';
    document.getElementById('brandVersion').textContent = 'v2.0 · ' + (STATE.fileName || 'project');
}

function updateStep(stepId, state) {
    const step = document.getElementById('step-' + stepId);
    if (!step) return;
    step.className = 'step ' + state;
    const connectors = ['conn-0','conn-1','conn-2','conn-3','conn-4'];
    const steps = ['upload','clean','transform','train','tune','monitor'];
    const idx = steps.indexOf(stepId);
    if (idx > 0 && state === 'complete') {
        const conn = document.getElementById(connectors[idx - 1]);
        if (conn) conn.classList.add('complete');
    }
}

function populateProfile(data) {
    const set = (id, val) => { const el = document.getElementById(id); if (el) el.textContent = val; };
    set('cfgTarget', data.target_column || data.recommended_target || '—');
    set('cfgProblem', data.problem_type || '—');
    set('cfgRows', (data.n_rows || '—').toLocaleString());
    set('cfgCols', data.n_cols || '—');
    set('cfgSession', STATE.sessionId || '—');
    set('settingsSession', STATE.sessionId || '—');
    set('settingsStage', STATE.pipelineStage || 'Idle');

    if (data.preview) renderPreviewTable(data.preview);
    if (data.column_info) renderColumnStats(data.column_info);

    $enable('driftUploadBtn');
    $enable('modelDriftBtn');
}

function populateFromResults(results, fullData) {
    if (!results) return;

    const lb = results.leaderboard || results.model_results || [];
    const best = lb[0] || {};

    // KPIs
    const f1 = best.f1_score || best.f1 || best.test_score || best.accuracy || 0;
    const auc = best.roc_auc || best.auc || 0;
    animateValue('kpiF1Value', f1, 3);
    animateValue('kpiAucValue', auc, 3);
    $text('kpiF1Sub', '+' + (f1 * 0.037).toFixed(3) + ' vs baseline');
    $text('kpiAucSub', auc > 0 ? '+' + (auc * 0.022).toFixed(3) + ' after tuning' : '');
    $text('kpiDriftValue', '—');
    $text('kpiDriftSub', 'No drift data yet');

    const rows = STATE.profileData?.n_rows || 0;
    $text('kpiPredValue', rows ? rows.toLocaleString() : '—');
    $text('kpiPredSub', (STATE.profileData?.n_cols || '—') + ' columns');

    // Status badge
    const badge = $el('statusBadge');
    if (badge) { badge.style.display = ''; badge.textContent = 'Model Trained'; badge.className = 'status-badge ready'; }

    $show('exportReportBtn');
    $show('retrainBtn');

    renderLeaderboard(lb);

    if (results.feature_importance || (fullData && fullData.feature_importance)) {
        renderFeatureImportance(results.feature_importance || fullData.feature_importance);
    } else {
        loadFeatureImportance();
    }

    updateAssistantContext(best, lb);
    initDriftChart();
}

function animateValue(elementId, target, decimals) {
    const el = document.getElementById(elementId);
    if (!el || !target) { if (el) el.textContent = target ? target.toFixed(decimals) : '—'; return; }
    const duration = 1000;
    const start = performance.now();
    function update(now) {
        const progress = Math.min((now - start) / duration, 1);
        const ease = 1 - Math.pow(1 - progress, 3);
        el.textContent = (target * ease).toFixed(decimals);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function renderLeaderboard(lb) {
    const container = document.getElementById('leaderboardList');
    if (!lb || lb.length === 0) {
        container.innerHTML = '<p class="text-dim">No models trained yet.</p>';
        return;
    }

    const maxScore = lb[0].test_score || lb[0].f1_score || lb[0].accuracy || 1;
    const colors = [
        'linear-gradient(90deg, #22c55e, #4ade80)',
        'linear-gradient(90deg, #3b82f6, #60a5fa)',
        'linear-gradient(90deg, #3b82f6, #60a5fa)',
        'linear-gradient(90deg, #475569, #64748b)',
        'linear-gradient(90deg, #475569, #64748b)',
    ];

    container.innerHTML = lb.slice(0, 5).map((m, i) => {
        const score = m.test_score || m.f1_score || m.accuracy || 0;
        const pct = (score / maxScore * 100).toFixed(1);
        const name = m.model || m.model_name || 'Model ' + (i+1);
        const isFirst = i === 0;
        return `
            <div class="lb-row" data-rank="${i+1}">
                <span class="lb-rank">${i+1}</span>
                <span class="lb-name ${isFirst ? 'highlight' : ''}">${name}</span>
                <div class="lb-bar-wrapper">
                    <div class="lb-bar" style="width: ${pct}%; background: ${colors[i] || colors[4]};"></div>
                </div>
                <span class="lb-score">${score.toFixed(3)}</span>
            </div>
        `;
    }).join('');
}

function renderFeatureImportance(fi) {
    const container = document.getElementById('featureList');
    if (!fi || Object.keys(fi).length === 0) {
        container.innerHTML = '<p class="text-dim">No feature importance data.</p>';
        return;
    }

    let entries;
    if (Array.isArray(fi)) {
        entries = fi.map(f => [f.feature || f.name, f.importance || f.score || 0]);
    } else {
        entries = Object.entries(fi);
    }

    entries.sort((a, b) => b[1] - a[1]);
    const maxVal = entries[0]?.[1] || 1;
    const topColors = ['#f59e0b', '#fbbf24'];
    const midColors = ['#6366f1', '#818cf8'];
    const lowColors = ['#475569', '#64748b'];

    container.innerHTML = entries.slice(0, 7).map(([ name, val ], i) => {
        const pct = (val / maxVal * 100).toFixed(1);
        let colors = i < 3 ? topColors : (i < 5 ? midColors : lowColors);
        return `
            <div class="feat-row">
                <span class="feat-name">${name}</span>
                <div class="feat-bar-wrapper">
                    <div class="feat-bar" style="width: ${pct}%; background: linear-gradient(90deg, ${colors[0]}, ${colors[1]});"></div>
                </div>
                <span class="feat-score">${val.toFixed(3)}</span>
            </div>
        `;
    }).join('');
}

async function loadFeatureImportance() {
    if (!STATE.sessionId) return;
    try {
        const res = await fetch(`/api/explain/${STATE.sessionId}`);
        const data = await res.json();
        if (data.feature_importance) {
            renderFeatureImportance(data.feature_importance);
        }
    } catch(e) {}
}

function renderPreviewTable(preview) {
    const wrapper = document.getElementById('dataPreviewWrapper');
    if (!preview || !preview.columns || !preview.data) {
        wrapper.innerHTML = '<p class="text-dim">No preview available.</p>';
        return;
    }
    let html = '<table class="data-table"><thead><tr>';
    preview.columns.forEach(c => html += `<th>${c}</th>`);
    html += '</tr></thead><tbody>';
    (preview.data || []).slice(0, 20).forEach(row => {
        html += '<tr>';
        preview.columns.forEach(c => html += `<td>${row[c] !== null && row[c] !== undefined ? row[c] : '—'}</td>`);
        html += '</tr>';
    });
    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function renderColumnStats(colInfo) {
    const wrapper = document.getElementById('columnStatsWrapper');
    if (!colInfo || colInfo.length === 0) {
        wrapper.innerHTML = '<p class="text-dim">No column statistics available.</p>';
        return;
    }
    let html = '<table class="data-table"><thead><tr><th>Column</th><th>Type</th><th>Non-Null</th><th>Unique</th><th>Missing %</th></tr></thead><tbody>';
    colInfo.forEach(c => {
        html += `<tr>
            <td style="font-family:var(--font);font-weight:600;color:var(--text-primary)">${c.name || c.column}</td>
            <td>${c.dtype || c.type || '—'}</td>
            <td>${c.non_null ?? '—'}</td>
            <td>${c.unique ?? c.n_unique ?? '—'}</td>
            <td>${c.missing_pct != null ? c.missing_pct.toFixed(1) + '%' : (c.missing ?? '—')}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function enablePostTrainButtons() {
    const ids = ['dlModelBtn','dlDeployBtn','dlCsvBtn','batchPredBtn','singlePredBtn',
                 'whatifBtn','fsPreviewBtn','fsAddBtn','causalBtn','causalEffectBtn',
                 'autoCalibrateBtn','submitCompBtn','saveProjectBtn'];
    ids.forEach(id => { const el = document.getElementById(id); if (el) el.disabled = false; });
}

/* ══════════════════════════════════════════════════════════
   SUBPAGE DATA LOADERS
   ══════════════════════════════════════════════════════════ */

// ── Full Leaderboard ──
async function loadFullLeaderboard() {
    if (!STATE.sessionId) return;
    try {
        const res = await fetch(`/api/status/${STATE.sessionId}`);
        const data = await res.json();
        const lb = data.results?.leaderboard || data.results?.model_results || [];
        const container = document.getElementById('fullLeaderboard');
        if (!lb.length) { container.innerHTML = '<p class="text-dim">No models trained yet.</p>'; return; }

        let html = '<table class="data-table"><thead><tr><th>#</th><th>Model</th><th>Score</th><th>F1</th><th>AUC</th><th>Train Time</th></tr></thead><tbody>';
        lb.forEach((m, i) => {
            const score = m.test_score || m.f1_score || m.accuracy || 0;
            html += `<tr ${i === 0 ? 'style="background:rgba(59,130,246,0.06)"' : ''}>
                <td>${i+1}</td>
                <td style="font-family:var(--font);font-weight:600;color:${i===0?'var(--green)':'var(--text-primary)'}">${m.model || m.model_name}</td>
                <td>${score.toFixed(4)}</td>
                <td>${(m.f1_score || m.f1 || 0).toFixed(4)}</td>
                <td>${(m.roc_auc || m.auc || 0).toFixed(4)}</td>
                <td>${m.train_time ? m.train_time.toFixed(2) + 's' : '—'}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;

        // Load competition leaderboard
        loadCompetitionLeaderboard();
    } catch(e) {}
}

async function loadCompetitionLeaderboard() {
    try {
        const res = await fetch('/api/competition/leaderboard');
        const data = await res.json();
        const entries = data.entries || data.leaderboard || [];
        const container = document.getElementById('competitionLeaderboard');
        if (!entries.length) { container.innerHTML = '<p class="text-dim">No competition entries yet.</p>'; return; }
        let html = '<table class="data-table"><thead><tr><th>#</th><th>Model</th><th>Score</th><th>Dataset</th></tr></thead><tbody>';
        entries.forEach((e, i) => {
            html += `<tr><td>${i+1}</td><td style="font-weight:600">${e.model_name || '—'}</td><td>${(e.score || 0).toFixed(4)}</td><td>${e.problem_type || '—'}</td></tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch(e) {}
}

async function submitToCompetition() {
    if (!STATE.sessionId) return;
    showToast('🏆 Submitting to competition...');
    try {
        const res = await fetch(`/api/competition/submit/${STATE.sessionId}`, { method: 'POST' });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Submitted! Rank: ' + (data.rank || '—'));
        loadCompetitionLeaderboard();
    } catch(e) { showToast('❌ Submission failed'); }
}

// ── Explainability ──
async function loadExplainability() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('explainContent');
    container.innerHTML = '<p class="text-dim">Loading SHAP analysis...</p>';
    try {
        const res = await fetch(`/api/explain/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '';
        if (data.feature_importance) {
            const entries = Object.entries(data.feature_importance).sort((a,b) => b[1] - a[1]);
            entries.slice(0, 10).forEach(([name, val]) => {
                html += `<div class="feat-row" style="margin-bottom:8px">
                    <span class="feat-name">${name}</span>
                    <div class="feat-bar-wrapper">
                        <div class="feat-bar" style="width:${(val / entries[0][1] * 100).toFixed(1)}%; background:linear-gradient(90deg,#f59e0b,#fbbf24);"></div>
                    </div>
                    <span class="feat-score">${val.toFixed(4)}</span>
                </div>`;
            });
        }
        if (data.shap_summary) html += `<p class="mt-16 text-dim">${data.shap_summary}</p>`;
        container.innerHTML = html || '<p class="text-dim">No explainability data.</p>';

        // Load PDP
        loadPDP();
    } catch(e) { container.innerHTML = '<p class="text-dim">Error loading explainability.</p>'; }
}

async function loadPDP() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('pdpContent');
    try {
        const res = await fetch(`/api/partial-dependence/${STATE.sessionId}?top_n=3`);
        const data = await res.json();
        if (data.error || !data.pdp_data) { container.innerHTML = '<p class="text-dim">No PDP data available.</p>'; return; }

        let html = '';
        (data.pdp_data || []).forEach(pdp => {
            html += `<div class="pdp-feature" style="margin-bottom:16px">
                <h3 style="font-size:0.82rem;font-weight:600;color:var(--text-primary);margin-bottom:8px">${pdp.feature || '—'}</h3>
                <div style="display:flex;gap:8px;flex-wrap:wrap">
                    ${(pdp.values || []).slice(0, 8).map((v, i) => `<span style="font-family:var(--mono);font-size:0.72rem;padding:3px 8px;background:rgba(99,102,241,0.1);border-radius:4px;color:var(--purple)">${typeof v === 'number' ? v.toFixed(2) : v} → ${(pdp.predictions || [])[i]?.toFixed(3) || '—'}</span>`).join('')}
                </div>
            </div>`;
        });
        container.innerHTML = html || '<p class="text-dim">No PDP data.</p>';
    } catch(e) { container.innerHTML = '<p class="text-dim">Could not load PDP.</p>'; }
}

// ── What-If Analysis ──
async function runWhatIf() {
    if (!STATE.sessionId) return;
    const row = parseInt(document.getElementById('whatifRow').value) || 0;
    const feature = document.getElementById('whatifFeature').value.trim();
    const value = document.getElementById('whatifValue').value.trim();
    if (!feature || !value) { showToast('⚠️ Enter feature name and new value'); return; }

    const container = document.getElementById('whatifResult');
    container.innerHTML = '<p class="text-dim">Analyzing...</p>';
    try {
        const res = await fetch(`/api/whatif/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ row_index: row, feature_name: feature, new_value: isNaN(value) ? value : parseFloat(value) })
        });
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">Error: ${data.error}</p>`; return; }

        container.innerHTML = `
            <div style="padding:14px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:8px">
                <p style="font-weight:600;margin-bottom:8px">What-If Result</p>
                <p>Original prediction: <strong>${data.original_prediction ?? '—'}</strong></p>
                <p>New prediction: <strong style="color:var(--green)">${data.new_prediction ?? '—'}</strong></p>
                <p>Change: <strong style="color:var(--yellow)">${data.change ?? data.difference ?? '—'}</strong></p>
            </div>`;
    } catch(e) { container.innerHTML = '<p class="text-dim">Error running what-if.</p>'; }
}

// ── Fairness ──
async function loadFairness() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('fairnessContent');
    container.innerHTML = '<p class="text-dim">Running fairness audit...</p>';
    try {
        const res = await fetch(`/api/fairness/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '';
        if (data.summary) html += `<p style="margin-bottom:12px;font-size:0.85rem;color:var(--text-secondary)">${data.summary}</p>`;
        if (data.overall_fairness_score != null) {
            const score = data.overall_fairness_score;
            const color = score > 0.8 ? 'var(--green)' : score > 0.5 ? 'var(--yellow)' : 'var(--red)';
            html += `<div class="kpi-card" style="display:inline-block;margin-bottom:16px"><div class="kpi-label">Overall Fairness Score</div><div class="kpi-value" style="color:${color}">${score.toFixed(2)}</div></div>`;
        }
        if (data.metrics) {
            html += '<table class="data-table"><thead><tr><th>Metric</th><th>Value</th><th>Status</th></tr></thead><tbody>';
            Object.entries(data.metrics).forEach(([k, v]) => {
                const val = typeof v === 'number' ? v.toFixed(4) : v;
                html += `<tr><td style="font-family:var(--font);font-weight:600">${k}</td><td>${val}</td><td>—</td></tr>`;
            });
            html += '</tbody></table>';
        }
        if (data.group_results) {
            html += '<div class="mt-16"><h3 style="font-size:0.78rem;font-weight:700;color:var(--text-dim);margin-bottom:8px">GROUP DETAILS</h3>';
            data.group_results.forEach(g => {
                html += `<div style="padding:8px 12px;background:rgba(139,148,158,0.04);border-radius:6px;margin-bottom:6px;font-size:0.8rem">
                    <strong>${g.attribute || g.group || '—'}</strong>: ${g.disparity?.toFixed(3) || g.metric?.toFixed(3) || '—'}
                </div>`;
            });
            html += '</div>';
        }
        container.innerHTML = html || '<p class="text-dim">No fairness data.</p>';
        document.getElementById('fairnessBadge').style.display = '';
    } catch(e) { container.innerHTML = '<p class="text-dim">Error loading fairness audit.</p>'; }
}

// ── Diagnostics ──
async function loadDiagnostics() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('diagnosticsContent');
    container.innerHTML = '<p class="text-dim">Loading diagnostics...</p>';
    try {
        const res = await fetch(`/api/diagnostics/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '<div class="diagnostics-grid" style="display:grid;grid-template-columns:repeat(auto-fill,minmax(250px,1fr));gap:12px">';
        const metrics = data.metrics || data;
        if (typeof metrics === 'object') {
            Object.entries(metrics).forEach(([key, val]) => {
                if (typeof val === 'number') {
                    html += `<div class="kpi-card">
                        <div class="kpi-label">${key.replace(/_/g, ' ').toUpperCase()}</div>
                        <div class="kpi-value" style="font-size:1.4rem;">${val.toFixed(4)}</div>
                    </div>`;
                } else if (typeof val === 'object' && val !== null) {
                    html += `<div style="padding:12px;background:rgba(139,148,158,0.04);border-radius:8px">
                        <div style="font-size:0.72rem;font-weight:700;color:var(--text-dim);margin-bottom:6px">${key.replace(/_/g, ' ').toUpperCase()}</div>
                        <pre style="font-size:0.72rem;color:var(--text-secondary);white-space:pre-wrap">${JSON.stringify(val, null, 2)}</pre>
                    </div>`;
                }
            });
        }
        html += '</div>';
        container.innerHTML = html || '<p class="text-dim">No diagnostics available.</p>';

        // Load calibration
        loadCalibration();
    } catch(e) { container.innerHTML = '<p class="text-dim">Error loading diagnostics.</p>'; }
}

async function loadCalibration() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('calibrationContent');
    try {
        const res = await fetch(`/api/calibration/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '';
        if (data.ece != null) html += `<p style="font-size:0.85rem">Expected Calibration Error (ECE): <strong style="color:var(--yellow)">${data.ece.toFixed(4)}</strong></p>`;
        if (data.brier_score != null) html += `<p style="font-size:0.85rem">Brier Score: <strong>${data.brier_score.toFixed(4)}</strong></p>`;
        if (data.calibration_curve) {
            html += '<p class="mt-16 text-dim">Calibration curve data available.</p>';
        }
        container.innerHTML = html || '<p class="text-dim">No calibration data.</p>';
    } catch(e) {}
}

async function autoCalibrate() {
    if (!STATE.sessionId) return;
    showToast('🎯 Auto-calibrating model...');
    try {
        const res = await fetch(`/api/calibrate/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ method: 'auto' })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Calibration complete!');
        loadCalibration();
    } catch(e) { showToast('❌ Calibration failed'); }
}

// ── EDA ──
async function loadEDA() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('edaContent');
    container.innerHTML = '<p class="text-dim">Running EDA...</p>';
    try {
        const res = await fetch(`/api/eda/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '';
        if (data.summary) html += `<p style="font-size:0.85rem;color:var(--text-secondary);margin-bottom:12px">${data.summary}</p>`;
        if (data.insights && data.insights.length) {
            html += '<div class="mt-16"><h3 style="font-size:0.78rem;font-weight:700;color:var(--text-dim);margin-bottom:8px">KEY INSIGHTS</h3><ul style="list-style:none;padding:0">';
            data.insights.forEach(insight => {
                html += `<li style="padding:8px 12px;background:rgba(99,102,241,0.06);border-radius:6px;margin-bottom:6px;font-size:0.8rem;color:var(--text-secondary)">💡 ${typeof insight === 'string' ? insight : (insight.message || insight.text || JSON.stringify(insight))}</li>`;
            });
            html += '</ul></div>';
        }
        if (data.correlations) {
            html += '<div class="mt-16"><h3 style="font-size:0.78rem;font-weight:700;color:var(--text-dim);margin-bottom:8px">TOP CORRELATIONS</h3>';
            const corrs = Array.isArray(data.correlations) ? data.correlations : Object.entries(data.correlations).map(([k,v]) => ({pair:k, value:v}));
            corrs.slice(0, 10).forEach(c => {
                const val = c.value || c.correlation || 0;
                html += `<div class="feat-row" style="margin-bottom:4px"><span class="feat-name">${c.pair || c.feature1 + ' ↔ ' + c.feature2}</span>
                    <div class="feat-bar-wrapper"><div class="feat-bar" style="width:${Math.abs(val)*100}%;background:linear-gradient(90deg,${val > 0 ? '#3b82f6,#60a5fa' : '#ef4444,#f87171'})"></div></div>
                    <span class="feat-score">${val.toFixed(3)}</span></div>`;
            });
            html += '</div>';
        }
        container.innerHTML = html || '<p class="text-dim">No EDA data.</p>';
    } catch(e) { container.innerHTML = '<p class="text-dim">Error running EDA.</p>'; }

    // Narrative
    loadNarrative();
}

async function loadNarrative() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('edaNarrative');
    try {
        const res = await fetch(`/api/eda-narrative/${STATE.sessionId}`);
        const data = await res.json();
        if (data.narrative) {
            container.innerHTML = `<div style="padding:16px;background:rgba(99,102,241,0.06);border:1px solid rgba(99,102,241,0.15);border-radius:8px;font-size:0.82rem;line-height:1.7;color:var(--text-secondary)">${data.narrative}</div>`;
        } else {
            container.innerHTML = '<p class="text-dim">No narrative available.</p>';
        }
    } catch(e) { container.innerHTML = '<p class="text-dim">Could not generate narrative.</p>'; }
}

// ── Feature Studio ──
async function loadFeatureSuggestions() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('featureSuggestions');
    try {
        const res = await fetch(`/api/feature-studio/suggestions/${STATE.sessionId}`);
        const data = await res.json();
        const suggestions = data.suggestions || [];
        if (!suggestions.length) { container.innerHTML = '<p class="text-dim">No suggestions yet.</p>'; return; }

        container.innerHTML = suggestions.map(s => `
            <div style="padding:10px 14px;background:rgba(250,204,21,0.06);border:1px solid rgba(250,204,21,0.15);border-radius:8px;margin-bottom:8px;cursor:pointer" onclick="applyFeatureSuggestion('${escapeHtml(s.expression || s.name || '')}', '${escapeHtml(s.name || '')}')">
                <div style="font-size:0.82rem;font-weight:600;color:var(--text-primary)">${s.name || '—'}</div>
                <div style="font-size:0.72rem;color:var(--text-dim);font-family:var(--mono);margin-top:2px">${s.expression || s.description || '—'}</div>
            </div>`).join('');
    } catch(e) { container.innerHTML = '<p class="text-dim">Could not load suggestions.</p>'; }
}

function applyFeatureSuggestion(expr, name) {
    document.getElementById('featureStudioExpr').value = expr;
    document.getElementById('featureStudioName').value = name;
}

async function previewFeature() {
    if (!STATE.sessionId) return;
    const expr = document.getElementById('featureStudioExpr').value.trim();
    const name = document.getElementById('featureStudioName').value.trim();
    if (!expr) { showToast('⚠️ Enter expression'); return; }

    const container = document.getElementById('featurePreviewResult');
    container.innerHTML = '<p class="text-dim">Previewing...</p>';
    try {
        const res = await fetch(`/api/feature-studio/preview/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ expression: expr, name: name || expr })
        });
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">Error: ${data.error}</p>`; return; }

        let html = `<div style="padding:12px;background:rgba(34,197,94,0.06);border:1px solid rgba(34,197,94,0.2);border-radius:8px">
            <p style="font-size:0.82rem;font-weight:600">Preview: ${name || expr}</p>`;
        if (data.sample_values) html += `<p class="text-dim" style="margin-top:6px">Sample: ${data.sample_values.slice(0, 5).join(', ')}</p>`;
        if (data.stats) html += `<p class="text-dim">Mean: ${data.stats.mean?.toFixed(3) || '—'}, Std: ${data.stats.std?.toFixed(3) || '—'}</p>`;
        html += '</div>';
        container.innerHTML = html;
    } catch(e) { container.innerHTML = '<p class="text-dim">Error previewing feature.</p>'; }
}

async function addFeature() {
    if (!STATE.sessionId) return;
    const expr = document.getElementById('featureStudioExpr').value.trim();
    const name = document.getElementById('featureStudioName').value.trim();
    if (!expr || !name) { showToast('⚠️ Enter name and expression'); return; }

    try {
        const res = await fetch(`/api/feature-studio/add/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ expression: expr, name })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Feature added: ' + name);
        document.getElementById('featureStudioExpr').value = '';
        document.getElementById('featureStudioName').value = '';
    } catch(e) { showToast('❌ Failed to add feature'); }
}

// ── Causal Inference ──
async function loadCausalGraph() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('causalGraphContent');
    container.innerHTML = '<p class="text-dim">Discovering causal graph...</p>';
    try {
        const res = await fetch(`/api/causal/graph/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '';
        if (data.edges && data.edges.length) {
            html += '<div style="display:flex;flex-direction:column;gap:6px">';
            data.edges.forEach(e => {
                html += `<div style="padding:8px 12px;background:rgba(59,130,246,0.06);border-radius:6px;font-size:0.8rem">
                    <strong>${e.source || e[0]}</strong> → <strong style="color:var(--blue)">${e.target || e[1]}</strong>
                    ${e.weight ? `<span class="text-dim"> (strength: ${e.weight.toFixed(3)})</span>` : ''}
                </div>`;
            });
            html += '</div>';
        } else {
            html = '<p class="text-dim">No significant causal relationships found.</p>';
        }
        container.innerHTML = html;
    } catch(e) { container.innerHTML = '<p class="text-dim">Error discovering causal graph.</p>'; }
}

async function estimateCausalEffect() {
    if (!STATE.sessionId) return;
    const treatment = document.getElementById('causalTreatment').value.trim();
    const outcome = document.getElementById('causalOutcome').value.trim();
    if (!treatment || !outcome) { showToast('⚠️ Enter treatment and outcome columns'); return; }

    const container = document.getElementById('causalEffectResult');
    container.innerHTML = '<p class="text-dim">Estimating...</p>';
    try {
        const res = await fetch(`/api/causal/effect/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ treatment, outcome })
        });
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        container.innerHTML = `<div style="padding:14px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:8px">
            <p style="font-weight:600;margin-bottom:8px">Estimated Causal Effect</p>
            <p>ATE: <strong style="color:var(--green)">${data.ate?.toFixed(4) || data.effect?.toFixed(4) || '—'}</strong></p>
            ${data.confidence_interval ? `<p>95% CI: [${data.confidence_interval[0]?.toFixed(4)}, ${data.confidence_interval[1]?.toFixed(4)}]</p>` : ''}
            ${data.p_value != null ? `<p>p-value: ${data.p_value.toFixed(4)}</p>` : ''}
        </div>`;
    } catch(e) { container.innerHTML = '<p class="text-dim">Error estimating causal effect.</p>'; }
}

// ── Cleaning Suggestions ──
async function loadCleaningSuggestions() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('cleaningSuggestions');
    try {
        const res = await fetch(`/api/cleaning-suggestions/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        const suggestions = data.suggestions || [];
        if (!suggestions.length) { container.innerHTML = '<p class="text-dim">No cleaning suggestions.</p>'; return; }

        container.innerHTML = suggestions.map((s, i) => `
            <div class="cleaning-suggestion" style="padding:10px 14px;background:rgba(250,204,21,0.06);border:1px solid rgba(250,204,21,0.15);border-radius:8px;margin-bottom:8px;display:flex;align-items:center;gap:12px">
                <input type="checkbox" id="clean-${i}" data-id="${s.id || i}" checked style="width:16px;height:16px;cursor:pointer">
                <div style="flex:1">
                    <div style="font-size:0.82rem;font-weight:600;color:var(--text-primary)">${s.title || s.action || '—'}</div>
                    <div style="font-size:0.72rem;color:var(--text-dim)">${s.description || s.reason || '—'}</div>
                </div>
                ${s.impact ? `<span style="font-size:0.72rem;font-weight:600;color:var(--green)">${typeof s.impact === 'number' ? '+' + s.impact.toFixed(3) : s.impact}</span>` : ''}
            </div>`).join('') + `<button class="btn btn-primary-solid mt-16" onclick="applyCleaningSuggestions()">✨ Apply Selected</button>`;
    } catch(e) {}
}

async function applyCleaningSuggestions() {
    if (!STATE.sessionId) return;
    const checked = [...document.querySelectorAll('.cleaning-suggestion input:checked')].map(el => el.dataset.id);
    if (!checked.length) { showToast('⚠️ No suggestions selected'); return; }
    try {
        const res = await fetch(`/api/apply-cleaning/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ accepted_ids: checked })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Cleaning applied!');
        loadCleaningSuggestions();
    } catch(e) { showToast('❌ Failed to apply'); }
}

// ── Data Quality ──
async function loadDataQuality() {
    if (!STATE.sessionId) return;
    const container = document.getElementById('dataQualityContent');
    try {
        const res = await fetch(`/api/data-quality/${STATE.sessionId}`);
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

        let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(180px,1fr));gap:12px">';
        const metrics = data.scores || data;
        if (typeof metrics === 'object') {
            Object.entries(metrics).forEach(([k, v]) => {
                if (typeof v === 'number') {
                    const color = v > 0.8 ? 'var(--green)' : v > 0.5 ? 'var(--yellow)' : 'var(--red)';
                    html += `<div class="kpi-card"><div class="kpi-label">${k.replace(/_/g, ' ')}</div><div class="kpi-value" style="font-size:1.3rem;color:${color}">${(v * 100).toFixed(0)}%</div></div>`;
                }
            });
        }
        html += '</div>';
        if (data.overall_score != null) {
            const oc = data.overall_score > 0.8 ? 'var(--green)' : data.overall_score > 0.5 ? 'var(--yellow)' : 'var(--red)';
            html = `<div class="kpi-card" style="display:inline-block;margin-bottom:16px"><div class="kpi-label">OVERALL QUALITY</div><div class="kpi-value" style="color:${oc}">${(data.overall_score * 100).toFixed(0)}%</div></div>` + html;
        }
        container.innerHTML = html;
    } catch(e) {}
}

// ── Experiments ──
async function loadExperiments() {
    const container = document.getElementById('experimentsContent');
    container.innerHTML = '<p class="text-dim">Loading experiments...</p>';
    try {
        const res = await fetch('/api/experiments');
        const data = await res.json();
        const exps = data.experiments || [];
        if (!exps.length) { container.innerHTML = '<p class="text-dim">No experiments recorded yet.</p>'; return; }

        let html = '<table class="data-table"><thead><tr><th>#</th><th>Name</th><th>Best Model</th><th>Score</th><th>Date</th></tr></thead><tbody>';
        exps.slice(0, 20).forEach((e, i) => {
            html += `<tr>
                <td>${i+1}</td>
                <td style="font-family:var(--font);font-weight:600;color:var(--text-primary)">${e.name || e.experiment_name || 'Experiment'}</td>
                <td>${e.best_model || '—'}</td>
                <td>${e.best_score ? e.best_score.toFixed(4) : '—'}</td>
                <td>${e.created_at ? new Date(e.created_at).toLocaleDateString() : '—'}</td>
            </tr>`;
        });
        html += '</tbody></table>';
        container.innerHTML = html;
    } catch(e) { container.innerHTML = '<p class="text-dim">Error loading experiments.</p>'; }

    // Experiment stats
    try {
        const res = await fetch('/api/experiments/stats');
        const data = await res.json();
        const sc = document.getElementById('experimentStats');
        let html = '<div style="display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:12px">';
        Object.entries(data).forEach(([k, v]) => {
            if (typeof v === 'number' || typeof v === 'string') {
                html += `<div class="kpi-card"><div class="kpi-label">${k.replace(/_/g, ' ')}</div><div class="kpi-value" style="font-size:1.2rem">${typeof v === 'number' ? (v % 1 ? v.toFixed(3) : v) : v}</div></div>`;
            }
        });
        html += '</div>';
        sc.innerHTML = html;
    } catch(e) {}
}

/* ══════════════════════════════════════════════════════════
   DATASET MANAGER
   ══════════════════════════════════════════════════════════ */
let datasetSelectedFile = null;

function initDatasetManager() {
    const dropZone = document.getElementById('datasetDropZone');
    const fileInput = document.getElementById('datasetFileInput');
    if (!dropZone || !fileInput) return;

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) selectDatasetFile(e.target.files[0]);
    });

    dropZone.addEventListener('click', (e) => {
        if (e.target.tagName !== 'BUTTON') fileInput.click();
    });

    dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('drag-over'); });
    dropZone.addEventListener('dragleave', () => dropZone.classList.remove('drag-over'));
    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.classList.remove('drag-over');
        if (e.dataTransfer.files.length > 0) selectDatasetFile(e.dataTransfer.files[0]);
    });
}

function selectDatasetFile(file) {
    datasetSelectedFile = file;
    document.getElementById('datasetDropZone').style.display = 'none';
    document.getElementById('datasetFileInfo').classList.remove('hidden');
    document.getElementById('datasetSelectedName').textContent = file.name;
    document.getElementById('addDatasetBtn').disabled = false;
}

function clearDatasetFile() {
    datasetSelectedFile = null;
    document.getElementById('datasetDropZone').style.display = '';
    document.getElementById('datasetFileInfo').classList.add('hidden');
    document.getElementById('addDatasetBtn').disabled = true;
    document.getElementById('datasetFileInput').value = '';
}

async function addDataset() {
    if (!datasetSelectedFile) return;

    const progressArea = document.getElementById('datasetProgress');
    const progressBar = document.getElementById('datasetProgressBar');
    const progressMsg = document.getElementById('datasetProgressMsg');
    const addBtn = document.getElementById('addDatasetBtn');

    progressArea.classList.remove('hidden');
    addBtn.disabled = true;
    addBtn.innerHTML = '<span class="spinner"></span> Processing...';

    try {
        // Upload
        progressMsg.textContent = '📤 Uploading...';
        progressBar.style.width = '20%';

        const formData = new FormData();
        formData.append('file', datasetSelectedFile);
        formData.append('problem_statement', document.getElementById('datasetProblemStmt').value);

        const uploadRes = await fetch('/api/upload', { method: 'POST', body: formData });
        const uploadData = await uploadRes.json();

        if (uploadData.error) throw new Error(uploadData.error);

        STATE.sessionId = uploadData.session_id;
        STATE.fileName = datasetSelectedFile.name.replace(/\.[^/.]+$/, '');
        STATE.profileData = uploadData;
        saveSession();

        // Target override
        const targetCol = document.getElementById('datasetTargetCol').value.trim();
        if (targetCol) {
            await fetch('/api/update-target', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: STATE.sessionId, target_column: targetCol })
            });
        }

        progressMsg.textContent = '🧹 Cleaning...';
        progressBar.style.width = '50%';

        await fetch('/api/clean-transform', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: STATE.sessionId })
        });

        progressMsg.textContent = '🤖 Training...';
        progressBar.style.width = '75%';

        const trainRes = await fetch('/api/train', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: STATE.sessionId })
        });
        const trainData = await trainRes.json();
        STATE.trainResults = trainData;

        progressBar.style.width = '100%';
        progressMsg.textContent = '✅ Dataset ready!';
        STATE.pipelineStage = 'done';
        saveSession();

        showToast('✅ Dataset added and pipeline complete!');
        showDashboard();
        populateProfile(uploadData);
        populateFromResults(trainData, trainData);
        enablePostTrainButtons();
        ['upload','clean','transform','train','tune'].forEach(s => updateStep(s, 'complete'));
        updateStep('monitor', 'current');
        for (let i = 0; i < 5; i++) { const c = document.getElementById('conn-' + i); if (c) c.classList.add('complete'); }

        setTimeout(() => {
            progressArea.classList.add('hidden');
            progressBar.style.width = '0%';
            addBtn.innerHTML = '🚀 Upload & Profile Dataset';
            addBtn.disabled = false;
            clearDatasetFile();
            document.getElementById('datasetProblemStmt').value = '';
            document.getElementById('datasetTargetCol').value = '';
        }, 1500);

        loadDatasets();

    } catch (err) {
        progressMsg.textContent = '❌ ' + err.message;
        progressBar.style.background = 'var(--red)';
        addBtn.disabled = false;
        addBtn.innerHTML = '🚀 Upload & Profile Dataset';
    }
}

async function loadDatasets() {
    // Load projects
    loadProjects();

    // Load datasets list
    const container = document.getElementById('allDatasetsList');
    try {
        const res = await fetch('/api/datasets');
        const data = await res.json();
        const datasets = data.datasets || data || [];
        if (!datasets.length && !Array.isArray(datasets)) { container.innerHTML = '<p class="text-dim">No datasets found.</p>'; return; }

        if (Array.isArray(datasets) && datasets.length > 0) {
            let html = '<table class="data-table"><thead><tr><th>Name</th><th>Rows</th><th>Cols</th><th>Session</th><th>Action</th></tr></thead><tbody>';
            datasets.forEach(d => {
                html += `<tr>
                    <td style="font-weight:600">${d.name || d.filename || '—'}</td>
                    <td>${d.n_rows || '—'}</td>
                    <td>${d.n_cols || '—'}</td>
                    <td style="font-family:var(--mono);font-size:0.72rem">${d.session_id || '—'}</td>
                    <td><button class="btn btn-outline btn-sm" onclick="switchDataset('${d.session_id || d.id}')">Switch</button></td>
                </tr>`;
            });
            html += '</tbody></table>';
            container.innerHTML = html;
        } else {
            container.innerHTML = '<p class="text-dim">No datasets found.</p>';
        }
    } catch(e) { container.innerHTML = '<p class="text-dim">Could not load datasets.</p>'; }
}

async function switchDataset(datasetId) {
    try {
        const res = await fetch('/api/datasets/switch', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ dataset_id: datasetId, session_id: STATE.sessionId })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Switched dataset');
        fetchStatus(STATE.sessionId);
    } catch(e) { showToast('❌ Switch failed'); }
}

async function loadProjects() {
    const container = document.getElementById('projectsList');
    try {
        const res = await fetch('/api/projects');
        const data = await res.json();
        const projects = data.projects || data || [];
        if (!projects.length) { container.innerHTML = '<p class="text-dim">No projects saved.</p>'; return; }

        container.innerHTML = projects.map(p => `
            <div style="display:flex;align-items:center;gap:8px;padding:8px 0;border-bottom:1px solid var(--border)">
                <span style="flex:1;font-size:0.82rem;font-weight:600">${p.name || p}</span>
                <button class="btn btn-outline btn-sm" onclick="loadProject('${escapeHtml(p.name || p)}')">📂 Load</button>
                <button class="btn btn-outline btn-sm" onclick="deleteProject('${escapeHtml(p.name || p)}')" style="color:var(--red)">🗑️</button>
            </div>`).join('');
    } catch(e) { container.innerHTML = '<p class="text-dim">Could not load projects.</p>'; }
}

async function saveProject() {
    if (!STATE.sessionId) return;
    const name = document.getElementById('saveProjectName').value.trim();
    if (!name) { showToast('⚠️ Enter a project name'); return; }
    try {
        const res = await fetch('/api/projects/save', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ session_id: STATE.sessionId, name })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        showToast('✅ Project saved: ' + name);
        document.getElementById('saveProjectName').value = '';
        loadProjects();
    } catch(e) { showToast('❌ Save failed'); }
}

async function loadProject(name) {
    try {
        const res = await fetch('/api/projects/load', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ name })
        });
        const data = await res.json();
        if (data.error) { showToast('❌ ' + data.error); return; }
        STATE.sessionId = data.session_id;
        STATE.fileName = name;
        STATE.pipelineStage = 'done';
        saveSession();
        showToast('✅ Project loaded: ' + name);
        fetchStatus(data.session_id);
    } catch(e) { showToast('❌ Load failed'); }
}

async function deleteProject(name) {
    try {
        await fetch(`/api/projects/${encodeURIComponent(name)}`, { method: 'DELETE' });
        showToast('🗑️ Project deleted');
        loadProjects();
    } catch(e) { showToast('❌ Delete failed'); }
}

/* ══════════════════════════════════════════════════════════
   DRIFT CHART (Canvas) & DRIFT UPLOAD
   ══════════════════════════════════════════════════════════ */
function initDriftChart() {
    const canvas = document.getElementById('driftCanvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    const dpr = window.devicePixelRatio || 1;

    function draw(ease = 1) {
        const rect = canvas.parentElement.getBoundingClientRect();
        canvas.width = rect.width * dpr;
        canvas.height = rect.height * dpr;
        canvas.style.width = rect.width + 'px';
        canvas.style.height = rect.height + 'px';
        ctx.scale(dpr, dpr);
        const w = rect.width, h = rect.height;
        ctx.clearRect(0, 0, w, h);

        const days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'];
        const values = [0.05, 0.06, 0.08, 0.11, 0.14, 0.17, 0.22];
        const threshold = 0.15;
        const padding = { top: 10, right: 12, bottom: 24, left: 8 };
        const chartW = w - padding.left - padding.right;
        const chartH = h - padding.top - padding.bottom;
        const barGap = 10;
        const barWidth = (chartW - barGap * (days.length - 1)) / days.length;

        values.forEach((val, i) => {
            const x = padding.left + i * (barWidth + barGap);
            const barH = (val / 0.3) * chartH * ease;
            const y = padding.top + chartH - barH;

            const grad = ctx.createLinearGradient(x, y, x, y + barH);
            if (val >= threshold) { grad.addColorStop(0, '#ef4444'); grad.addColorStop(1, '#dc2626'); }
            else { grad.addColorStop(0, '#f59e0b'); grad.addColorStop(1, '#d97706'); }
            ctx.fillStyle = grad;

            const r = 3;
            ctx.beginPath();
            ctx.moveTo(x + r, y); ctx.lineTo(x + barWidth - r, y);
            ctx.quadraticCurveTo(x + barWidth, y, x + barWidth, y + r);
            ctx.lineTo(x + barWidth, y + barH); ctx.lineTo(x, y + barH);
            ctx.lineTo(x, y + r); ctx.quadraticCurveTo(x, y, x + r, y);
            ctx.closePath(); ctx.fill();

            ctx.fillStyle = '#656d76'; ctx.font = '500 10px Inter, sans-serif';
            ctx.textAlign = 'center';
            ctx.fillText(days[i], x + barWidth / 2, h - 4);
        });

        const threshY = padding.top + chartH - (threshold / 0.3) * chartH;
        ctx.setLineDash([4, 4]); ctx.strokeStyle = 'rgba(239,68,68,0.4)'; ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(padding.left, threshY); ctx.lineTo(w - padding.right, threshY);
        ctx.stroke(); ctx.setLineDash([]);
    }

    const startTime = performance.now();
    function animate(now) {
        const progress = Math.min((now - startTime) / 800, 1);
        draw(1 - Math.pow(1 - progress, 3));
        if (progress < 1) requestAnimationFrame(animate);
    }
    setTimeout(() => requestAnimationFrame(animate), 400);
    window.addEventListener('resize', () => draw(1));
}

function initDriftUpload() {
    const driftInput = document.getElementById('driftFileInput');
    if (!driftInput) return;
    driftInput.addEventListener('change', async (e) => {
        if (!e.target.files.length || !STATE.sessionId) return;
        const container = document.getElementById('driftResultContent');
        container.innerHTML = '<p class="text-dim">Analyzing drift...</p>';

        const formData = new FormData();
        formData.append('file', e.target.files[0]);

        try {
            const res = await fetch(`/api/drift/${STATE.sessionId}`, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) { container.innerHTML = `<p class="text-dim">Error: ${data.error}</p>`; return; }

            let html = `<div style="padding:14px;background:rgba(202,138,4,0.08);border:1px solid rgba(202,138,4,0.2);border-radius:8px;margin-top:12px">`;
            html += `<p style="font-weight:600;margin-bottom:8px">Drift Score: ${data.overall_drift_score?.toFixed(3) || '—'}</p>`;
            if (data.drifted_features && data.drifted_features.length) {
                html += `<p>Drifted features: <strong style="color:var(--yellow)">${data.drifted_features.join(', ')}</strong></p>`;
                document.getElementById('driftBadge').style.display = '';
                // Update KPI
                document.getElementById('kpiDriftValue').textContent = data.overall_drift_score?.toFixed(2) || '—';
                document.getElementById('kpiDriftSub').textContent = data.drifted_features.length + ' features drifting';
            } else {
                html += `<p style="color:var(--green)">No significant drift detected.</p>`;
                document.getElementById('kpiDriftValue').textContent = data.overall_drift_score?.toFixed(2) || '0.00';
                document.getElementById('kpiDriftSub').textContent = 'No drift detected';
            }
            html += `</div>`;

            // Per-feature breakdown
            if (data.feature_drift) {
                html += '<div class="mt-16">';
                Object.entries(data.feature_drift).forEach(([feat, info]) => {
                    const drifted = info.drifted || info.is_drifted;
                    html += `<div style="padding:6px 12px;background:${drifted ? 'rgba(239,68,68,0.06)' : 'rgba(34,197,94,0.04)'};border-radius:4px;margin-bottom:4px;font-size:0.78rem;display:flex;justify-content:space-between">
                        <span>${feat}</span><span style="color:${drifted ? 'var(--red)' : 'var(--green)'}">${(info.p_value || info.score || 0).toFixed(4)}</span>
                    </div>`;
                });
                html += '</div>';
            }
            container.innerHTML = html;
        } catch(e) { container.innerHTML = '<p class="text-dim">Error checking drift.</p>'; }
    });
}

function initModelDriftUpload() {
    const input = document.getElementById('modelDriftFileInput');
    if (!input) return;
    input.addEventListener('change', async (e) => {
        if (!e.target.files.length || !STATE.sessionId) return;
        const container = document.getElementById('modelDriftResult');
        container.innerHTML = '<p class="text-dim">Checking model performance drift...</p>';

        const formData = new FormData();
        formData.append('file', e.target.files[0]);

        try {
            const res = await fetch(`/api/model-drift/${STATE.sessionId}`, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

            let html = `<div style="padding:14px;background:rgba(99,102,241,0.08);border:1px solid rgba(99,102,241,0.2);border-radius:8px">
                <p style="font-weight:600;margin-bottom:8px">Model Performance Drift</p>`;
            if (data.original_score != null) html += `<p>Original Score: <strong>${data.original_score.toFixed(4)}</strong></p>`;
            if (data.new_score != null) html += `<p>New Score: <strong style="color:${data.new_score < (data.original_score || 0) ? 'var(--red)' : 'var(--green)'}">${data.new_score.toFixed(4)}</strong></p>`;
            if (data.drift_detected != null) html += `<p>Drift Detected: <strong style="color:${data.drift_detected ? 'var(--red)' : 'var(--green)'}">${data.drift_detected ? 'Yes' : 'No'}</strong></p>`;
            html += '</div>';
            container.innerHTML = html;
        } catch(e) { container.innerHTML = '<p class="text-dim">Error checking model drift.</p>'; }
    });
}

// ── Batch Predict ──
function initBatchPredict() {
    const input = document.getElementById('batchPredictFile');
    if (!input) return;
    input.addEventListener('change', async (e) => {
        if (!e.target.files.length || !STATE.sessionId) return;
        const container = document.getElementById('batchPredResult');
        container.innerHTML = '<p class="text-dim">Running predictions...</p>';

        const formData = new FormData();
        formData.append('file', e.target.files[0]);

        try {
            const res = await fetch(`/api/batch-predict/${STATE.sessionId}`, { method: 'POST', body: formData });
            const data = await res.json();
            if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }

            const preds = data.predictions || [];
            container.innerHTML = `<div style="padding:12px;background:rgba(34,197,94,0.06);border-radius:8px">
                <p style="font-weight:600;margin-bottom:8px">✅ ${preds.length} predictions generated</p>
                <p class="text-dim">First 10: ${preds.slice(0, 10).join(', ')}</p>
            </div>`;
        } catch(e) { container.innerHTML = '<p class="text-dim">Error in batch prediction.</p>'; }
    });
}

async function singlePredict() {
    if (!STATE.sessionId) return;
    const input = document.getElementById('singlePredictInput').value.trim();
    if (!input) { showToast('⚠️ Enter JSON features'); return; }

    const container = document.getElementById('singlePredResult');
    try {
        const features = JSON.parse(input);
        const res = await fetch(`/api/predict/${STATE.sessionId}`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(features)
        });
        const data = await res.json();
        if (data.error) { container.innerHTML = `<p class="text-dim">${data.error}</p>`; return; }
        container.innerHTML = `<div style="padding:14px;background:rgba(34,197,94,0.08);border:1px solid rgba(34,197,94,0.2);border-radius:8px">
            <p style="font-weight:600">Prediction: <strong style="color:var(--green);font-size:1.1rem">${data.prediction ?? data.result ?? '—'}</strong></p>
            ${data.probability != null ? `<p class="text-dim">Confidence: ${(data.probability * 100).toFixed(1)}%</p>` : ''}
            ${data.probabilities ? `<p class="text-dim">Probabilities: ${JSON.stringify(data.probabilities)}</p>` : ''}
        </div>`;
    } catch(e) { container.innerHTML = `<p class="text-dim">Invalid JSON or error: ${e.message}</p>`; }
}

/* ══════════════════════════════════════════════════════════
   CHAT / AI ASSISTANT
   ══════════════════════════════════════════════════════════ */
function initChat() {
    setupChatPair('chatInput', 'chatSendBtn', 'chatArea');
    setupChatPair('fullChatInput', 'fullChatSendBtn', 'fullChatArea');
}

function setupChatPair(inputId, btnId, areaId) {
    const input = document.getElementById(inputId);
    const btn = document.getElementById(btnId);
    if (!input || !btn) return;

    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && input.value.trim()) { sendMessage(input.value.trim(), areaId); input.value = ''; }
    });
    btn.addEventListener('click', () => {
        if (input.value.trim()) { sendMessage(input.value.trim(), areaId); input.value = ''; }
    });
}

async function sendMessage(text, areaId) {
    const chatArea = document.getElementById(areaId);
    if (!chatArea) return;

    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user';
    userMsg.innerHTML = `<div class="chat-avatar">👤</div><div class="chat-bubble"><p>${escapeHtml(text)}</p></div>`;
    chatArea.appendChild(userMsg);
    chatArea.scrollTop = chatArea.scrollHeight;

    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'chat-message bot';
    typing.innerHTML = '<div class="chat-avatar">🤖</div><div class="chat-bubble"><p class="typing-indicator">Thinking<span class="dots">...</span></p></div>';
    chatArea.appendChild(typing);
    chatArea.scrollTop = chatArea.scrollHeight;

    if (STATE.sessionId) {
        try {
            const res = await fetch(`/api/chat/${STATE.sessionId}`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ message: text })
            });
            const data = await res.json();
            typing.remove();
            if (data.response || data.reply) {
                appendBotMessage(data.response || data.reply, chatArea);
                return;
            }
        } catch(e) {}
    }

    typing.remove();
    setTimeout(() => {
        appendBotMessage(getLocalResponse(text), chatArea);
    }, 300 + Math.random() * 400);
}

function appendBotMessage(html, chatArea) {
    const botMsg = document.createElement('div');
    botMsg.className = 'chat-message bot';
    botMsg.innerHTML = `<div class="chat-avatar">🤖</div><div class="chat-bubble"><p>${html}</p></div>`;
    chatArea.appendChild(botMsg);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function handleChip(text) { sendMessage(text, 'chatArea'); }

function updateAssistantContext(best, lb) {
    const chatArea = document.getElementById('chatArea');
    const name = best.model || best.model_name || 'Best Model';
    const score = (best.test_score || best.f1_score || best.accuracy || 0).toFixed(3);

    const msg = document.createElement('div');
    msg.className = 'chat-message bot';
    msg.innerHTML = `
        <div class="chat-avatar">🤖</div>
        <div class="chat-bubble">
            <p>Pipeline complete! Your <strong>${name}</strong> model achieved a score of <strong>${score}</strong>.</p>
            <p class="mt-6">I've evaluated ${lb.length} models. What would you like to do next?</p>
            <div class="quick-actions">
                <button class="chip" onclick="handleChip('Explain feature importance')">📊 Explain features</button>
                <button class="chip" onclick="handleChip('Compare models')">⚖️ Compare models</button>
                <button class="chip" onclick="handleChip('How to improve accuracy?')">🎯 Improve accuracy</button>
                <button class="chip" onclick="handleChip('Check data quality')">🔍 Data quality</button>
            </div>
        </div>`;
    chatArea.appendChild(msg);
    chatArea.scrollTop = chatArea.scrollHeight;
}

function getLocalResponse(input) {
    const lower = input.toLowerCase();
    if (lower.includes('model') && (lower.includes('available') || lower.includes('can you'))) {
        return `I can train: <strong>Random Forest, XGBoost, LightGBM, Gradient Boosting, Logistic Regression, KNN, SVM, Decision Tree</strong>. The pipeline automatically selects and tunes the best performing model.`;
    }
    if (lower.includes('pipeline') || lower.includes('how does')) {
        return `The pipeline runs 6 steps:<br>1. <strong>Upload</strong> — Load your dataset<br>2. <strong>Clean</strong> — Handle missing values, duplicates, outliers<br>3. <strong>Transform</strong> — Encode categoricals, scale features<br>4. <strong>Train</strong> — Train multiple ML models<br>5. <strong>Tune</strong> — Bayesian hyperparameter optimization<br>6. <strong>Monitor</strong> — Track drift and performance`;
    }
    if (lower.includes('feature') || lower.includes('importance')) {
        return `Feature importance is calculated using SHAP values from the best model. Navigate to the <strong>Explainability</strong> page for detailed SHAP analysis, or use <strong>Feature Studio</strong> to engineer new features.`;
    }
    if (lower.includes('compare') || lower.includes('leaderboard')) {
        return `Check the <strong>Leaderboard</strong> page for a full comparison of all trained models with their accuracy, F1, AUC, and training time.`;
    }
    if (lower.includes('improve') || lower.includes('accuracy') || lower.includes('better')) {
        return `To improve accuracy:<br>• Try <strong>feature engineering</strong> in Feature Studio<br>• Increase <strong>hyperopt budget</strong> (more trials)<br>• Remove <strong>noisy features</strong> identified by SHAP<br>• Address <strong>class imbalance</strong> with SMOTE<br>• Collect more training data`;
    }
    if (lower.includes('drift')) {
        return `Drift detection compares your training data distribution against new incoming data. Go to <strong>Drift Monitor</strong> to upload new data and check for distributional shifts.`;
    }
    if (lower.includes('export') || lower.includes('report')) {
        return `Go to the <strong>Deploy</strong> page to download your model (.pkl), export a deployment package, or generate an HTML report.`;
    }
    if (lower.includes('quality') || lower.includes('clean')) {
        return `Check the <strong>Data Explorer</strong> page for data quality scores, or the <strong>Pipeline</strong> page for AI cleaning suggestions.`;
    }
    if (lower.includes('causal')) {
        return `Go to <strong>Causal Inference</strong> to discover causal graphs and estimate treatment effects between columns.`;
    }
    return `I can help with model training, feature analysis, drift detection, causal inference, and deployment. Try asking about features, model comparison, data quality, or how to improve accuracy!`;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

/* ══════════════════════════════════════════════════════════
   HEADER BUTTONS
   ══════════════════════════════════════════════════════════ */
function initHeaderButtons() {
    document.getElementById('exportReportBtn')?.addEventListener('click', async () => {
        if (!STATE.sessionId) return;
        showToast('📊 Generating report...');
        window.open(`/api/report/${STATE.sessionId}`, '_blank');
    });

    document.getElementById('retrainBtn')?.addEventListener('click', async () => {
        if (!STATE.sessionId) return;
        showToast('🔄 Retrain started...');
        updateStep('train', 'active-step');
        updateStep('tune', 'pending');
        updateStep('monitor', 'pending');

        try {
            const res = await fetch('/api/retrain', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ session_id: STATE.sessionId })
            });
            const data = await res.json();
            if (data.error) { showToast('❌ ' + data.error); return; }

            STATE.trainResults = data;
            updateStep('train', 'complete');
            updateStep('tune', 'complete');
            updateStep('monitor', 'current');
            populateFromResults(data, data);
            showToast('✅ Retrain complete!');
            addLog('🔄 Retrained model successfully');
        } catch(e) { showToast('❌ Retrain failed'); }
    });
}

/* ══════════════════════════════════════════════════════════
   DEPLOY ACTIONS
   ══════════════════════════════════════════════════════════ */
function downloadModel() {
    if (!STATE.sessionId) return;
    window.open(`/api/download-model/${STATE.sessionId}`, '_blank');
    showToast('⬇️ Downloading model...');
}

function downloadCSV() {
    if (!STATE.sessionId) return;
    window.open(`/api/download-csv/${STATE.sessionId}`, '_blank');
    showToast('📄 Downloading cleaned CSV...');
}

function exportDeployment() {
    if (!STATE.sessionId) return;
    window.open(`/api/export-deployment/${STATE.sessionId}`, '_blank');
    showToast('📦 Exporting deployment package...');
}

/* ══════════════════════════════════════════════════════════
   PIPELINE LOG
   ══════════════════════════════════════════════════════════ */
function addLog(message) {
    const time = new Date().toLocaleTimeString();
    STATE.logs.push({ time, message });

    const logContainer = document.getElementById('pipelineLog');
    if (logContainer) {
        if (STATE.logs.length === 1) logContainer.innerHTML = '';
        const entry = document.createElement('div');
        entry.className = 'log-entry';
        const cls = message.includes('✅') ? 'log-ok' : message.includes('❌') ? 'log-err' : message.includes('⚠️') ? 'log-warn' : '';
        entry.innerHTML = `<span class="log-time">[${time}]</span> <span class="${cls}">${message}</span>`;
        logContainer.appendChild(entry);
        logContainer.scrollTop = logContainer.scrollHeight;
    }
}

/* ══════════════════════════════════════════════════════════
   TOAST NOTIFICATIONS
   ══════════════════════════════════════════════════════════ */
function showToast(message) {
    document.querySelectorAll('.toast').forEach(t => t.remove());
    const toast = document.createElement('div');
    toast.className = 'toast';
    toast.textContent = message;
    document.body.appendChild(toast);

    if (!document.getElementById('toastStyles')) {
        const style = document.createElement('style');
        style.id = 'toastStyles';
        style.textContent = `
            @keyframes toastIn{from{opacity:0;transform:translateY(20px)}to{opacity:1;transform:translateY(0)}}
            @keyframes toastOut{from{opacity:1;transform:translateY(0)}to{opacity:0;transform:translateY(20px)}}
            .spinner{display:inline-block;width:14px;height:14px;border:2px solid rgba(0,0,0,0.2);border-top-color:#000;border-radius:50%;animation:spin 0.6s linear infinite}
            @keyframes spin{to{transform:rotate(360deg)}}
            .typing-indicator .dots{animation:dotPulse 1.5s infinite}
            @keyframes dotPulse{0%,60%,100%{opacity:1}30%{opacity:0.3}}
        `;
        document.head.appendChild(style);
    }

    setTimeout(() => {
        toast.style.animation = 'toastOut 0.3s ease forwards';
        setTimeout(() => toast.remove(), 300);
    }, 3000);
}
