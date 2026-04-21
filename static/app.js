/**
 * AutoML Studio — Frontend Application
 * 9-page SPA with hash-based routing, centralized state, and all 18 features.
 */

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// STATE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
let sessionId = localStorage.getItem('automl_session_id') || null;
let profileData = null;
let trainingResults = null;
let diagnosticsData = null;
let explainData = null;
let recommendations = null;
let activityLog = JSON.parse(localStorage.getItem('automl_activity') || '[]');
let charts = {};
let pendingFile = null;
let experimentStats = { total: 0, best_score: 0 };

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// ROUTER
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
const PAGE_TITLES = {
    dashboard: 'Dashboard',
    dataset: 'Dataset',
    analysis: 'Analysis',
    automl: 'AutoML Pipeline',
    models: 'Models',
    visualization: 'Visualization',
    insights: 'AI Insights',
    assistant: 'AI Assistant',
    deployment: 'Deployment',
};

function navigateTo(page) {
    window.location.hash = page;
}

function handleRoute() {
    const hash = window.location.hash.replace('#', '') || 'dashboard';
    // Hide all pages, show target
    document.querySelectorAll('.page').forEach(p => p.classList.remove('active'));
    const target = document.getElementById(`page-${hash}`);
    if (target) target.classList.add('active');
    // Update nav
    document.querySelectorAll('.nav-item').forEach(n => n.classList.remove('active'));
    const navItem = document.querySelector(`.nav-item[data-page="${hash}"]`);
    if (navItem) navItem.classList.add('active');
    // Update topbar title
    document.getElementById('topbarTitle').textContent = PAGE_TITLES[hash] || 'Dashboard';
    // Lazy init pages
    if (hash === 'dashboard') initDashboard();
    if (hash === 'deployment') loadProjectsList();
    if (hash === 'competition') loadCompetitionBoard();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INIT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
document.addEventListener('DOMContentLoaded', () => {
    // Router
    window.addEventListener('hashchange', handleRoute);
    handleRoute();

    // Nav clicks
    document.querySelectorAll('.nav-item').forEach(item => {
        item.addEventListener('click', (e) => {
            e.preventDefault();
            navigateTo(item.dataset.page);
        });
    });

    // Sidebar toggle (mobile)
    document.getElementById('sidebarToggle').addEventListener('click', () => {
        document.getElementById('sidebar').classList.toggle('open');
    });

    // File upload handlers
    setupFileUploads();

    // Session badge
    updateSessionBadge();

    // Validate stored session against backend
    validateSession();

    // Init dashboard stats
    initDashboard();

    // Hyperopt range listener
    const budgetSlider = document.getElementById('hyperoptBudget');
    if (budgetSlider) {
        budgetSlider.addEventListener('input', () => {
            document.getElementById('hyperoptBudgetLabel').textContent = budgetSlider.value;
        });
    }
});

async function validateSession() {
    if (!sessionId) return;
    try {
        const res = await fetch(`/api/status/${sessionId}`);
        const data = await res.json();
        if (data.error) {
            // Session no longer exists on backend — clear stale reference
            console.log('Stale session cleared:', sessionId);
            sessionId = null;
            localStorage.removeItem('automl_session_id');
            updateSessionBadge();
        } else {
            // Restore state from session
            if (data.profile) {
                profileData = data.profile;
                renderProfile(profileData);
                enablePostUploadButtons();
            }
            if (data.training_results) {
                trainingResults = data.training_results;
                recommendations = data.recommendations;
                renderAllResults();
            }
        }
    } catch (e) {
        // Server not reachable — clear session
        sessionId = null;
        localStorage.removeItem('automl_session_id');
        updateSessionBadge();
    }
}



function updateSessionBadge() {
    document.getElementById('sessionBadge').textContent = sessionId ? `Session: ${sessionId.substring(0, 8)}...` : 'No session';
    document.getElementById('saveProjectBtn').style.display = sessionId ? '' : 'none';
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// FILE UPLOAD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function setupFileUploads() {
    const uploadArea = document.getElementById('uploadArea');
    const fileInput = document.getElementById('fileInput');
    const globalInput = document.getElementById('globalFileInput');
    const driftInput = document.getElementById('driftFileInput');
    const modelDriftInput = document.getElementById('modelDriftFileInput');

    // Click to upload
    uploadArea.addEventListener('click', () => fileInput.click());

    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => { e.preventDefault(); uploadArea.classList.add('drag-over'); });
    uploadArea.addEventListener('dragleave', () => uploadArea.classList.remove('drag-over'));
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('drag-over');
        if (e.dataTransfer.files.length) handleFileSelect(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', (e) => { if (e.target.files[0]) handleFileSelect(e.target.files[0]); });
    globalInput.addEventListener('change', (e) => {
        if (e.target.files[0]) {
            handleFileSelect(e.target.files[0]);
            navigateTo('dataset');
        }
    });

    if (driftInput) driftInput.addEventListener('change', (e) => { if (e.target.files[0]) runDriftCheck(e.target.files[0]); });
    if (modelDriftInput) modelDriftInput.addEventListener('change', (e) => { if (e.target.files[0]) runModelDriftCheck(e.target.files[0]); });
}

function handleFileSelect(file) {
    pendingFile = file;
    document.getElementById('uploadPrompt').classList.add('hidden');
    document.getElementById('fileSelectedInfo').classList.remove('hidden');
    document.getElementById('selectedFileName').textContent = `${file.name} (${formatBytes(file.size)})`;
    document.getElementById('analyzeBtn').disabled = false;
    document.getElementById('analyzeBtn').onclick = () => uploadDataset();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// API HELPERS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function apiFetch(url, options = {}) {
    try {
        const res = await fetch(url, options);
        return await res.json();
    } catch (err) {
        return { error: err.message };
    }
}

function formatBytes(b) {
    if (b < 1024) return b + ' B';
    if (b < 1048576) return (b / 1024).toFixed(1) + ' KB';
    return (b / 1048576).toFixed(1) + ' MB';
}

function addActivity(text) {
    const now = new Date();
    const timeStr = now.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    activityLog.unshift({ text, time: timeStr });
    if (activityLog.length > 20) activityLog.pop();
    localStorage.setItem('automl_activity', JSON.stringify(activityLog));
    renderActivityFeed();
}

function renderActivityFeed() {
    const el = document.getElementById('activityFeed');
    if (!el) return;
    el.innerHTML = activityLog.slice(0, 10).map(a =>
        `<div class="activity-item"><span class="activity-dot"></span><span class="activity-text">${a.text}</span><span class="activity-time">${a.time}</span></div>`
    ).join('');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DASHBOARD
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function initDashboard() {
    renderActivityFeed();
    // Update stats from localStorage
    const storedModels = parseInt(localStorage.getItem('automl_models_count') || '0');
    const storedBest = localStorage.getItem('automl_best_score') || '—';
    const storedDatasets = parseInt(localStorage.getItem('automl_dataset_count') || '0');
    const storedTime = localStorage.getItem('automl_last_run') || '—';
    document.getElementById('statModels').textContent = storedModels;
    document.getElementById('statBestAcc').textContent = storedBest;
    document.getElementById('statDatasets').textContent = storedDatasets;
    document.getElementById('statLastRun').textContent = storedTime;

    // Fetch experiment stats
    loadExperimentStats();
}

async function loadExperimentStats() {
    const data = await apiFetch('/api/experiments/stats');
    if (!data.error) {
        experimentStats = data;
        if (data.total_experiments) {
            document.getElementById('statModels').textContent = data.total_experiments;
            localStorage.setItem('automl_models_count', data.total_experiments);
        }
    }
}

function updateDashboardStats() {
    if (trainingResults) {
        const leaderboard = trainingResults.leaderboard || [];
        document.getElementById('statModels').textContent = leaderboard.length;
        localStorage.setItem('automl_models_count', leaderboard.length);
        if (trainingResults.best_score) {
            const s = (trainingResults.best_score * 100).toFixed(1) + '%';
            document.getElementById('statBestAcc').textContent = s;
            localStorage.setItem('automl_best_score', s);
        }
        const now = new Date().toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
        document.getElementById('statLastRun').textContent = now;
        localStorage.setItem('automl_last_run', now);
        renderPerformanceChart(leaderboard);
    }
    const dc = parseInt(localStorage.getItem('automl_dataset_count') || '0');
    document.getElementById('statDatasets').textContent = dc;
}

function renderPerformanceChart(leaderboard) {
    if (!leaderboard || !leaderboard.length) return;
    const ctx = document.getElementById('dashPerformanceChart');
    if (!ctx) return;
    if (charts.dashPerf) charts.dashPerf.destroy();

    const names = leaderboard.map(m => m.model || m.name || '?');
    const scores = leaderboard.map(m => parseFloat(m.primary_metric ?? m.score) || 0);
    const colors = scores.map((s, i) => i === 0 ? '#3B82F6' : 'rgba(148,163,184,0.4)');

    charts.dashPerf = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: names,
            datasets: [{ label: 'Score', data: scores, backgroundColor: colors, borderRadius: 6 }]
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false } },
            scales: {
                x: { min: 0, max: 1, ticks: { color: '#64748B', font: { size: 10 } }, grid: { color: 'rgba(148,163,184,0.06)' } },
                y: { ticks: { color: '#94A3B8', font: { size: 11 } }, grid: { display: false } },
            },
        },
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DATASET PAGE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function uploadDataset() {
    if (!pendingFile) return;
    const btn = document.getElementById('analyzeBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Analyzing...';

    const formData = new FormData();
    formData.append('file', pendingFile);
    formData.append('problem_statement', document.getElementById('problemStatement').value);

    const data = await apiFetch('/api/upload', { method: 'POST', body: formData });

    if (data.error) {
        btn.disabled = false;
        btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Dataset';
        alert('Error: ' + data.error);
        return;
    }

    sessionId = data.session_id;
    profileData = data.profile || data;
    localStorage.setItem('automl_session_id', sessionId);

    const dc = parseInt(localStorage.getItem('automl_dataset_count') || '0') + 1;
    localStorage.setItem('automl_dataset_count', dc);

    updateSessionBadge();
    renderProfile(profileData);
    enablePostUploadButtons();
    addActivity(`📂 Uploaded: ${pendingFile.name}`);

    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🔍</span> Analyze Dataset';
}

function renderProfile(profile) {
    const section = document.getElementById('profileResults');
    section.classList.remove('hidden');

    // Detection cards
    const grid = document.getElementById('detectionGrid');
    grid.innerHTML = `
        <div class="detection-card"><div class="det-icon">📋</div><div class="det-label">Rows × Cols</div><div class="det-value">${(profile.n_rows || 0).toLocaleString()} × ${profile.n_cols || 0}</div></div>
        <div class="detection-card"><div class="det-icon">🎯</div><div class="det-label">Target</div><div class="det-value purple">${profile.target_column || '—'}</div></div>
        <div class="detection-card"><div class="det-icon">🧠</div><div class="det-label">Problem Type</div><div class="det-value green">${profile.problem_type || '—'}</div></div>
        <div class="detection-card"><div class="det-icon">🩹</div><div class="det-label">Missing</div><div class="det-value amber">${profile.total_missing_pct || 0}%</div></div>
    `;

    // Target select
    const targetSel = document.getElementById('targetSelect');
    const pipeTargetSel = document.getElementById('pipelineTargetSelect');
    targetSel.innerHTML = '';
    pipeTargetSel.innerHTML = '';
    (profile.columns || []).forEach(c => {
        const name = c.name || c;
        targetSel.innerHTML += `<option value="${name}" ${name === profile.target_column ? 'selected' : ''}>${name}</option>`;
        pipeTargetSel.innerHTML += `<option value="${name}" ${name === profile.target_column ? 'selected' : ''}>${name}</option>`;
    });

    // Problem type
    const ptSel = document.getElementById('problemTypeSelect');
    ptSel.value = profile.problem_type || 'classification';
    document.getElementById('problemTypeBadge').textContent = profile.problem_type || '—';

    // Preview table
    if (profile.preview) {
        renderPreviewTable(profile.preview);
    }

    // Column info
    if (profile.columns) {
        renderColumnInfo(profile.columns);
    }
}

function renderPreviewTable(preview) {
    const wrapper = document.getElementById('previewTableWrapper');
    if (!preview || !preview.length) { wrapper.innerHTML = '<p class="text-dim">No preview available.</p>'; return; }
    const cols = Object.keys(preview[0]);
    let html = `<table class="data-table"><thead><tr>${cols.map(c => `<th>${c}</th>`).join('')}</tr></thead><tbody>`;
    preview.slice(0, 10).forEach(row => {
        html += `<tr>${cols.map(c => `<td>${row[c] !== null ? row[c] : '<span class="text-dim">null</span>'}</td>`).join('')}</tr>`;
    });
    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function renderColumnInfo(columns) {
    const wrapper = document.getElementById('columnInfoWrapper');
    let html = `<table class="data-table"><thead><tr><th>Column</th><th>Type</th><th>Missing %</th><th>Unique</th></tr></thead><tbody>`;
    columns.forEach(c => {
        const missing = c.missing_pct || 0;
        const color = missing > 20 ? 'color:var(--red)' : missing > 5 ? 'color:var(--amber)' : 'color:var(--green)';
        html += `<tr>
            <td style="color:var(--text-primary);font-weight:500;font-family:var(--font-sans)">${c.name || c}</td>
            <td><span class="tag tag-cyan">${c.dtype || '—'}</span></td>
            <td style="${color};font-weight:600">${missing}%</td>
            <td>${c.n_unique || '—'}</td>
        </tr>`;
    });
    html += '</tbody></table>';
    wrapper.innerHTML = html;
}

function enablePostUploadButtons() {
    document.getElementById('runAutoMLBtn').disabled = false;
    document.getElementById('runEdaBtn').disabled = false;
    document.getElementById('driftUploadBtn').disabled = false;
    document.getElementById('modelDriftUploadBtn').disabled = false;
    document.getElementById('cleaningSuggestionsPanel').classList.remove('hidden');
    document.getElementById('similarityPanel').classList.remove('hidden');
    document.getElementById('reframingPanel').classList.remove('hidden');
    getCleaningSuggestions();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AUTOML PIPELINE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function runFullPipeline() {
    if (!sessionId) { alert('Upload a dataset first!'); return; }
    const btn = document.getElementById('runAutoMLBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Training...';
    const progressEl = document.getElementById('automlProgress');
    progressEl.classList.remove('hidden');

    addActivity('🚀 Started AutoML pipeline');
    updatePipelineFlow('clean');

    // Step 1: Clean & Transform
    setProgress(10, 'Cleaning & transforming data...');
    const cleanResult = await apiFetch('/api/clean-transform', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    });
    if (cleanResult.error) { handlePipelineError(cleanResult.error, btn); return; }
    updatePipelineFlow('train');
    setProgress(30, 'Training models...');

    // Step 2: Train
    const trainResult = await apiFetch('/api/train', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId }),
    });
    if (trainResult.error) { handlePipelineError(trainResult.error, btn); return; }
    updatePipelineFlow('evaluate');
    setProgress(70, 'Evaluating models...');

    // Poll for completion
    trainingResults = await pollForResults();
    setProgress(90, 'Generating explanations...');

    // Fetch extra data
    explainData = await apiFetch(`/api/explain/${sessionId}`);
    diagnosticsData = await apiFetch(`/api/diagnostics/${sessionId}`);

    updatePipelineFlow('deploy');
    setProgress(100, 'Pipeline complete!');

    addActivity(`✅ Training complete${trainingResults?.best_model ? ' — Best: ' + trainingResults.best_model : ''}`);
    renderAllResults();

    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🚀</span> Run AutoML';
}

async function pollForResults() {
    let maxPolls = 120;
    while (maxPolls-- > 0) {
        const status = await apiFetch(`/api/status/${sessionId}`);
        if (status.status === 'complete' && status.training_results) {
            trainingResults = status.training_results;
            recommendations = status.recommendations;
            return trainingResults;
        }
        if (status.status === 'error') {
            return null;
        }
        // Update progress from session
        if (status.progress) {
            setProgress(30 + (status.progress * 0.4), status.progress_message || 'Training...');
        }
        await new Promise(r => setTimeout(r, 2000));
    }
    return null;
}

function setProgress(pct, msg) {
    document.getElementById('automlProgressBar').style.width = pct + '%';
    document.getElementById('automlProgressPct').textContent = pct + '%';
    document.getElementById('automlProgressMsg').textContent = msg || '';
}

function updatePipelineFlow(currentStep) {
    const steps = ['upload', 'clean', 'transform', 'train', 'evaluate', 'deploy'];
    const idx = steps.indexOf(currentStep);
    document.querySelectorAll('.pipeline-node').forEach((node, i) => {
        node.classList.remove('complete', 'active', 'pending');
        if (i < idx) node.classList.add('complete');
        else if (i === idx) node.classList.add('active');
        else node.classList.add('pending');
    });
}

function handlePipelineError(error, btn) {
    setProgress(0, `Error: ${error}`);
    btn.disabled = false;
    btn.innerHTML = '<span class="btn-icon">🚀</span> Run AutoML';
    addActivity(`❌ Pipeline error: ${error}`);
}

function toggleSemiAutoPanel() {
    document.getElementById('semiAutoPanel').classList.toggle('hidden');
}

async function applyCustomPipeline() {
    const models = [...document.querySelectorAll('#modelCheckboxes input:checked')].map(c => c.value);
    const preprocessing = [...document.querySelectorAll('#preprocessCheckboxes input:checked')].map(c => c.value);
    const data = await apiFetch(`/api/custom-pipeline/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ config: { models, preprocessing } }),
    });
    if (data.success) addActivity('🔧 Custom pipeline configured');
    document.getElementById('semiAutoPanel').classList.add('hidden');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// RENDER ALL RESULTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderAllResults() {
    updateDashboardStats();
    renderModelsPage();
    renderVisualizationPage();
    renderInsightsPage();
    enableDeploymentButtons();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// MODELS PAGE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderModelsPage() {
    const el = document.getElementById('modelsContent');
    if (!trainingResults) { el.innerHTML = '<div class="glass-card"><p class="text-dim text-center" style="padding:40px">Run autoML pipeline first.</p></div>'; return; }

    const leaderboard = trainingResults.leaderboard || [];
    const bestModel = trainingResults.best_model || '';
    const bestScore = trainingResults.best_score || 0;
    const metricName = trainingResults.primary_metric_name || 'Score';

    // Normalize leaderboard keys (backend: model/primary_metric, frontend: name/score)
    const normalizedLB = leaderboard.map(m => ({
        name: m.model || m.name || '?',
        score: m.primary_metric ?? m.score ?? 0,
        metrics: m.metrics || {},
        rank: m.rank,
    }));

    let html = '';

    // Best model card
    html += `<div class="best-model-card mb-16">
        <div class="content">
            <div class="trophy">🏆</div>
            <div class="best-name">${bestModel}</div>
            <div class="best-score">${(bestScore * 100).toFixed(2)}%</div>
            <div class="best-metric-label">${metricName}</div>
        </div>
    </div>`;

    // Leaderboard table with key metrics
    const metricKeys = normalizedLB.length > 0 ? Object.keys(normalizedLB[0].metrics).filter(k => !['tuned'].includes(k)).slice(0, 5) : [];

    html += '<div class="glass-card"><div class="section-title"><div class="icon cyan">📊</div><span>Model Leaderboard</span></div>';
    html += '<div class="table-wrapper"><table class="leaderboard-table"><thead><tr><th>#</th><th>Model</th><th>Score</th>';
    metricKeys.forEach(m => html += `<th>${m.replace(/_/g, ' ').toUpperCase()}</th>`);
    html += '</tr></thead><tbody>';

    normalizedLB.forEach((m, i) => {
        const rankCls = i === 0 ? 'rank-1' : i === 1 ? 'rank-2' : i === 2 ? 'rank-3' : 'rank-other';
        const isBest = m.name === bestModel;
        const score = parseFloat(m.score) || 0;
        const scoreCls = score >= 0.9 ? 'metric-good' : score >= 0.7 ? 'metric-ok' : 'metric-bad';
        html += `<tr class="${isBest ? 'best-model' : ''}">
            <td><span class="rank-badge ${rankCls}">${i + 1}</span></td>
            <td class="model-name">${isBest ? '⭐ ' : ''}${m.name}</td>
            <td class="metric-value ${scoreCls}">${(score * 100).toFixed(2)}%</td>`;
        metricKeys.forEach(k => {
            const v = m.metrics[k];
            if (v === undefined || v === null) { html += '<td>—</td>'; return; }
            html += `<td class="metric-value">${typeof v === 'number' ? (Math.abs(v) < 1 ? (v * 100).toFixed(2) + '%' : v.toFixed(4)) : v}</td>`;
        });
        html += '</tr>';
    });
    html += '</tbody></table></div></div>';

    // Model comparison chart
    html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon purple">📈</div><span>Score Comparison</span></div><div class="chart-container"><canvas id="modelCompChart"></canvas></div></div>';

    el.innerHTML = html;

    // Comparison chart
    if (normalizedLB.length) {
        const ctx = document.getElementById('modelCompChart');
        if (charts.modelComp) charts.modelComp.destroy();
        charts.modelComp = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: normalizedLB.map(m => m.name),
                datasets: [{
                    label: metricName,
                    data: normalizedLB.map(m => parseFloat(m.score) || 0),
                    backgroundColor: normalizedLB.map((_, i) => i === 0 ? 'rgba(59,130,246,0.7)' : 'rgba(148,163,184,0.3)'),
                    borderRadius: 6,
                }],
            },
            options: {
                responsive: true, maintainAspectRatio: false,
                plugins: { legend: { display: false }, tooltip: { callbacks: { label: ctx => (ctx.raw * 100).toFixed(2) + '%' } } },
                scales: {
                    y: { ticks: { callback: v => (v * 100) + '%', color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } },
                    x: { ticks: { color: '#94A3B8', font: { size: 10 } }, grid: { display: false } },
                },
            },
        });
    }

    // Show panels
    document.getElementById('hyperoptPanel').classList.remove('hidden');
    document.getElementById('calibrationPanel').classList.remove('hidden');
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// VISUALIZATION PAGE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderVisualizationPage() {
    const el = document.getElementById('vizContent');
    if (!trainingResults && !diagnosticsData) { return; }

    document.getElementById('explainerPanel').classList.remove('hidden');

    let html = '';

    // Confusion Matrix
    const diag = diagnosticsData || {};
    if (diag.confusion_matrix) {
        html += '<div class="glass-card"><div class="section-title"><div class="icon cyan">🔢</div><span>Confusion Matrix</span></div>';
        html += renderConfusionMatrix(diag.confusion_matrix, diag.class_labels || []);
        html += '</div>';
    }

    // ROC Curve
    if (diag.roc_curve || diag.roc_auc) {
        html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon purple">📈</div><span>ROC Curve</span></div>';
        html += '<div class="chart-container"><canvas id="rocChart"></canvas></div></div>';
    }

    // Feature Importance Chart
    if (explainData && explainData.global_importance) {
        html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon green">🎯</div><span>Feature Importance</span></div>';
        html += '<div class="chart-container"><canvas id="featureImpChart"></canvas></div></div>';
    }

    // Learning Curves
    if (diag.learning_curve) {
        html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon magenta">📉</div><span>Learning Curve</span></div>';
        html += '<div class="chart-container"><canvas id="learningChart"></canvas></div></div>';
    }

    el.innerHTML = html || '<div class="glass-card"><p class="text-dim text-center" style="padding:40px">No visualization data yet.</p></div>';

    // Render charts
    if (diag.roc_curve) renderROCChart(diag.roc_curve);
    if (explainData && explainData.global_importance) renderFeatureImpChart(explainData.global_importance);
    if (diag.learning_curve) renderLearningChart(diag.learning_curve);
}

function renderConfusionMatrix(matrix, labels) {
    if (!matrix || !matrix.length) return '<p class="text-dim">No confusion matrix data.</p>';
    const n = matrix.length;
    let html = '<div style="display:flex;justify-content:center;margin:16px 0"><div class="confusion-matrix" style="grid-template-columns: 40px repeat(' + n + ', 1fr);">';
    // Header row
    html += '<div class="cm-label"></div>';
    for (let j = 0; j < n; j++) html += `<div class="cm-label">${labels[j] || j}</div>`;
    // Data
    for (let i = 0; i < n; i++) {
        html += `<div class="cm-label">${labels[i] || i}</div>`;
        for (let j = 0; j < n; j++) {
            const cls = i === j ? 'diagonal' : 'off-diagonal';
            html += `<div class="cm-cell ${cls}" title="Actual: ${labels[i] || i}, Predicted: ${labels[j] || j}">${matrix[i][j]}</div>`;
        }
    }
    html += '</div></div>';
    return html;
}

function renderROCChart(roc) {
    const ctx = document.getElementById('rocChart');
    if (!ctx) return;
    if (charts.roc) charts.roc.destroy();
    const fpr = roc.fpr || [];
    const tpr = roc.tpr || [];
    charts.roc = new Chart(ctx, {
        type: 'line',
        data: {
            labels: fpr.map(v => v.toFixed(2)),
            datasets: [
                { label: `ROC (AUC=${roc.auc ? roc.auc.toFixed(3) : '?'})`, data: tpr, borderColor: '#3B82F6', backgroundColor: 'rgba(59,130,246,0.1)', fill: true, tension: 0.3, pointRadius: 0 },
                { label: 'Baseline', data: fpr, borderColor: 'rgba(148,163,184,0.3)', borderDash: [5, 5], pointRadius: 0 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94A3B8' } }, zoom: { zoom: { wheel: { enabled: true }, pinch: { enabled: true }, mode: 'xy' }, pan: { enabled: true, mode: 'xy' } } },
            scales: { x: { title: { display: true, text: 'FPR', color: '#64748B' }, ticks: { color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } }, y: { title: { display: true, text: 'TPR', color: '#64748B' }, ticks: { color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } } },
        },
    });
}

function renderFeatureImpChart(importance) {
    const ctx = document.getElementById('featureImpChart');
    if (!ctx) return;
    if (charts.featImp) charts.featImp.destroy();
    const sorted = [...importance].sort((a, b) => b.importance - a.importance).slice(0, 15);
    charts.featImp = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sorted.map(f => f.feature),
            datasets: [{ label: 'Importance', data: sorted.map(f => f.importance), backgroundColor: 'rgba(139,92,246,0.5)', borderRadius: 4 }],
        },
        options: {
            indexAxis: 'y',
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false }, zoom: { zoom: { wheel: { enabled: true }, mode: 'y' }, pan: { enabled: true, mode: 'y' } } },
            scales: { x: { ticks: { color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } }, y: { ticks: { color: '#94A3B8', font: { size: 10 } }, grid: { display: false } } },
        },
    });
}

function renderLearningChart(lc) {
    const ctx = document.getElementById('learningChart');
    if (!ctx) return;
    if (charts.learning) charts.learning.destroy();
    charts.learning = new Chart(ctx, {
        type: 'line',
        data: {
            labels: lc.train_sizes || [],
            datasets: [
                { label: 'Train Score', data: lc.train_scores_mean || [], borderColor: '#3B82F6', tension: 0.3, pointRadius: 2 },
                { label: 'Val Score', data: lc.val_scores_mean || [], borderColor: '#22C55E', tension: 0.3, pointRadius: 2 },
            ],
        },
        options: {
            responsive: true, maintainAspectRatio: false,
            plugins: { legend: { labels: { color: '#94A3B8' } } },
            scales: { x: { title: { display: true, text: 'Training Size', color: '#64748B' }, ticks: { color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } }, y: { title: { display: true, text: 'Score', color: '#64748B' }, ticks: { color: '#64748B' }, grid: { color: 'rgba(148,163,184,0.06)' } } },
        },
    });
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// INSIGHTS PAGE
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function renderInsightsPage() {
    // Feature ranking
    if (explainData && explainData.global_importance) {
        const panel = document.getElementById('featureRankingPanel');
        panel.classList.remove('hidden');
        const content = document.getElementById('featureRankingContent');
        const sorted = [...explainData.global_importance].sort((a, b) => b.importance - a.importance);
        const maxImp = sorted[0]?.importance || 1;
        content.innerHTML = sorted.slice(0, 15).map((f, i) =>
            `<div class="feature-rank-item">
                <div class="feature-rank-num">${i + 1}</div>
                <div class="feature-rank-name">${f.feature}</div>
                <div class="feature-rank-bar"><div class="feature-rank-bar-fill" style="width:${(f.importance / maxImp * 100).toFixed(0)}%"></div></div>
                <div class="feature-rank-score">${f.importance.toFixed(4)}</div>
            </div>`
        ).join('');
    }

    // AI Recommendations
    if (recommendations && recommendations.length) {
        const el = document.getElementById('aiInsightsContent');
        el.innerHTML = recommendations.map(r => {
            const icon = r.category === 'model' ? '🤖' : r.category === 'data' ? '📊' : r.category === 'feature' ? '🔧' : '💡';
            const impact = r.priority === 'high' ? 'high' : r.priority === 'medium' ? 'medium' : 'low';
            return `<div class="rec-card">
                <div class="rec-icon">${icon}</div>
                <div class="rec-content">
                    <div class="rec-header">
                        <span class="rec-title">${r.title || r.category || 'Suggestion'}</span>
                        <span class="impact-badge impact-${impact}">${impact}</span>
                    </div>
                    <div class="rec-description">${r.description || r.text || ''}</div>
                </div>
            </div>`;
        }).join('');
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// HYPERPARAMETER OPTIMIZATION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function runHyperopt() {
    if (!sessionId) return;
    const btn = document.getElementById('runHyperoptBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Optimizing...';
    const el = document.getElementById('hyperoptResults');
    el.innerHTML = '<p class="text-dim">⏳ Running optimization... This may take a few minutes.</p>';

    const method = document.getElementById('hyperoptMethod').value;
    const budget = parseInt(document.getElementById('hyperoptBudget').value);

    addActivity(`⚡ Started hyperparameter optimization (${method})`);

    const data = await apiFetch(`/api/optimize/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method, budget }),
    });

    btn.disabled = false;
    btn.innerHTML = '⚡ Optimize Hyperparameters';

    if (data.error) {
        el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`;
        return;
    }

    addActivity(`✅ Optimization complete — ${data.method}, ${data.optimized_count} models`);

    let html = `<div class="result-card"><h4>⚡ Optimization Results — ${data.method}</h4>
        <span class="metric good">${data.optimized_count} models optimized</span>
        <span class="metric warn">${data.skipped_count} skipped</span>
        ${data.best_model ? `<span class="metric good">Best: ${data.best_model} (${(data.best_score * 100).toFixed(2)}%)</span>` : ''}
    </div>`;

    if (data.models) {
        Object.entries(data.models).forEach(([name, m]) => {
            if (m.skipped) {
                html += `<div class="result-card"><h4>${name}</h4><span class="metric warn">Skipped: ${m.reason}</span></div>`;
            } else {
                const improvement = m.improvement ? `<span class="metric ${m.improvement > 0 ? 'good' : 'warn'}">${m.improvement > 0 ? '+' : ''}${(m.improvement * 100).toFixed(2)}%</span>` : '';
                html += `<div class="result-card"><h4>${name}</h4>
                    <span class="metric good">Optimized: ${(m.optimized_score * 100).toFixed(2)}%</span>
                    ${m.original_score ? `<span class="metric warn">Original: ${(m.original_score * 100).toFixed(2)}%</span>` : ''}
                    ${improvement}
                    <span class="metric">${m.time_seconds}s</span>
                </div>`;
            }
        });
    }

    el.innerHTML = html;

    // Re-fetch results to update leaderboard
    const status = await apiFetch(`/api/status/${sessionId}`);
    if (status.training_results) {
        trainingResults = status.training_results;
        renderModelsPage();
        updateDashboardStats();
    }
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CLEANING SUGGESTIONS & IMPACT (v3.0)
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function getCleaningSuggestions() {
    if (!sessionId) return;
    const el = document.getElementById('cleaningSuggestionsList');
    el.innerHTML = '<p class="text-dim">⏳ Benchmarking cleaning alternatives and analyzing impact...</p>';

    const data = await apiFetch(`/api/cleaning-impact/${sessionId}`);
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }

    const suggestions = data.benchmarks || [];
    if (!suggestions.length) { el.innerHTML = '<p class="text-dim">✅ No cleaning required — data looks great!</p>'; return; }

    el.innerHTML = suggestions.map(s => {
        const titleParts = s.strategy.split(':');
        const title = titleParts.length > 1 ? titleParts[1].trim() : s.strategy;
        const icon = title.includes('impute') ? '🩹' : title.includes('encode') ? '🔤' : title.includes('drop') ? '🗑️' : '🔧';
        return `<div class="suggestion-card" id="sug-${s.id}">
            <input type="checkbox" class="suggestion-check" value="${s.id}" onchange="this.parentElement.classList.toggle('selected')">
            <div class="suggestion-icon">${icon}</div>
            <div class="suggestion-body">
                <div class="suggestion-title">${title} (${s.strategy})</div>
                <div class="suggestion-desc" style="display:flex;gap:12px;margin-top:8px">
                    <span class="metric warn">Before: ${(s.performance_before * 100).toFixed(1)}%</span>
                    <span class="metric good">After: ${(s.performance_after * 100).toFixed(1)}%</span>
                    <span class="metric ${s.impact_delta >= 0 ? 'good' : 'bad'}">Δ: ${(s.impact_delta * 100).toFixed(2)}%</span>
                </div>
            </div>
        </div>`;
    }).join('');
}

async function applySelectedSuggestions() {
    const checks = [...document.querySelectorAll('.suggestion-check:checked')];
    if (!checks.length) { alert('Select at least one suggestion to apply.'); return; }
    const ids = checks.map(c => parseInt(c.value));

    const btn = document.getElementById('applyCleaningBtn');
    btn.disabled = true;
    btn.innerHTML = '<span class="spinner"></span> Applying...';

    const data = await apiFetch(`/api/apply-cleaning/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ accepted_ids: ids }),
    });

    btn.disabled = false;
    btn.innerHTML = '✅ Apply Selected';

    if (data.error) { alert('Error: ' + data.error); return; }
    addActivity(`🧹 Applied ${data.n_applied} cleaning suggestions`);
    // Refresh suggestions
    getCleaningSuggestions();
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// AUTO EDA
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function runAutoEDA() {
    if (!sessionId) return;
    const el = document.getElementById('edaResults');
    el.innerHTML = '<div class="glass-card"><p class="text-dim text-center" style="padding:40px">⏳ Running comprehensive EDA...</p></div>';

    const data = await apiFetch(`/api/eda/${sessionId}`);
    if (data.error) { el.innerHTML = `<div class="glass-card"><p class="text-dim">❌ ${data.error}</p></div>`; return; }

    addActivity('📊 Completed AutoEDA');
    renderEDA(data);

    // Also fetch the narrative
    document.getElementById('narrativeEdaPanel').classList.remove('hidden');
    document.getElementById('narrativeEdaContent').innerHTML = '<p class="text-dim">🤖 Generating executive narrative...</p>';
    
    const narrativeData = await apiFetch(`/api/eda-narrative/${sessionId}`);
    if (!narrativeData.error) {
        document.getElementById('narrativeEdaContent').innerHTML = `
            <div style="font-size: 1.05rem; line-height: 1.6; color: var(--text-primary);">
                ${narrativeData.narrative.replace(/\n\n/g, '<br><br>').replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')}
            </div>
            <div style="margin-top:16px; font-size: 0.85rem; color: var(--text-dim);">
                <span class="impact-badge impact-${narrativeData.data_health_score >= 80 ? 'low' : narrativeData.data_health_score >= 50 ? 'medium' : 'high'}">Health Score: ${narrativeData.data_health_score}/100</span>
            </div>
        `;
    }
}

function renderEDA(eda) {
    const el = document.getElementById('edaResults');
    const ov = eda.overview || {};
    let html = `<div class="glass-card">
        <div class="section-title"><div class="icon cyan">📋</div><span>Overview</span></div>
        <div class="detection-grid">
            <div class="detection-card"><div class="det-icon">📋</div><div class="det-label">Rows × Cols</div><div class="det-value">${(ov.n_rows || 0).toLocaleString()} × ${ov.n_cols || 0}</div></div>
            <div class="detection-card"><div class="det-icon">🩹</div><div class="det-label">Missing</div><div class="det-value amber">${ov.total_missing_pct || 0}%</div></div>
            <div class="detection-card"><div class="det-icon">🔄</div><div class="det-label">Duplicates</div><div class="det-value">${ov.duplicated_pct || 0}%</div></div>
            <div class="detection-card"><div class="det-icon">💾</div><div class="det-label">Memory</div><div class="det-value">${ov.memory_mb || 0} MB</div></div>
        </div>
    </div>`;

    // AI Insights
    if (eda.insights && eda.insights.length) {
        html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon purple">🧠</div><span>AI Insights</span></div>';
        eda.insights.forEach(i => {
            html += `<div class="insight-card ${i.severity || 'info'}"><span class="insight-icon">${i.icon || '💡'}</span><div><div class="insight-cat">${i.category || ''}</div><div class="insight-text">${i.text}</div></div></div>`;
        });
        html += '</div>';
    }

    // Correlations
    const corrs = eda.correlations?.top_correlations || [];
    if (corrs.length) {
        html += '<div class="glass-card mt-16"><div class="section-title"><div class="icon green">🔗</div><span>Top Correlations</span></div>';
        corrs.slice(0, 10).forEach(c => {
            const abs = Math.abs(c.correlation);
            const color = abs > 0.7 ? 'var(--red)' : abs > 0.5 ? 'var(--amber)' : 'var(--green)';
            html += `<div class="quality-bar"><span class="label">${c.feature_1} ↔ ${c.feature_2}</span><div class="bar"><div class="bar-fill" style="width:${abs * 100}%;background:${color}"></div></div><span class="score" style="color:${color}">${c.correlation}</span></div>`;
        });
        html += '</div>';
    }

    el.innerHTML = html;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// CHAT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function sendQuickChat(msg) {
    document.getElementById('chatInput').value = msg;
    sendChat();
}

async function sendChat() {
    const input = document.getElementById('chatInput');
    const msg = input.value.trim();
    if (!msg) return;
    input.value = '';

    addChatMsg('user', msg);

    if (!sessionId) { addChatMsg('assistant', '📤 Please upload a dataset first.'); return; }

    try {
        const data = await apiFetch(`/api/chat/${sessionId}`, {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message: msg }),
        });
        addChatMsg('assistant', data.response || data.error || 'No response');
    } catch (err) {
        addChatMsg('assistant', '❌ ' + err.message);
    }
}

function addChatMsg(role, text) {
    const el = document.getElementById('chatMessages');
    const avatar = role === 'user' ? '🧑' : '🤖';
    el.innerHTML += `<div class="chat-msg ${role}"><div class="chat-avatar">${avatar}</div><div class="chat-bubble">${text}</div></div>`;
    el.scrollTop = el.scrollHeight;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DRIFT DETECTION
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function runDriftCheck(file) {
    const el = document.getElementById('driftResult');
    el.innerHTML = '<p class="text-dim">⏳ Checking data drift...</p>';

    const formData = new FormData();
    formData.append('file', file);

    const data = await apiFetch(`/api/drift/${sessionId}`, { method: 'POST', body: formData });
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }

    addActivity(`🔍 Data drift check: ${data.overall_status}`);
    const status = data.overall_status || 'unknown';
    const cls = status === 'healthy' ? 'healthy' : status === 'critical' ? 'critical' : 'warning';
    let html = `<div class="drift-card ${cls}">
        <h4 style="margin-bottom:8px">${status === 'healthy' ? '🟢' : status === 'critical' ? '🔴' : '🟡'} ${status.toUpperCase()}</h4>
        <p style="font-size:0.88rem;color:var(--text-secondary)">${data.recommendation || ''}</p>
    </div>`;

    if (data.feature_drift) {
        html += '<div class="mt-16">';
        data.feature_drift.slice(0, 10).forEach(f => {
            const color = f.status === 'high_drift' ? 'var(--red)' : f.status === 'moderate_drift' ? 'var(--amber)' : 'var(--green)';
            html += `<div class="quality-bar"><span class="label">${f.feature}</span><div class="bar"><div class="bar-fill" style="width:${Math.min(f.psi * 200, 100)}%;background:${color}"></div></div><span class="score" style="color:${color}">${f.psi}</span></div>`;
        });
        html += '</div>';
    }
    el.innerHTML = html;
}

async function runModelDriftCheck(file) {
    const el = document.getElementById('modelDriftResult');
    el.innerHTML = '<p class="text-dim">⏳ Checking model drift...</p>';

    const formData = new FormData();
    formData.append('file', file);

    const data = await apiFetch(`/api/model-drift/${sessionId}`, { method: 'POST', body: formData });
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }

    addActivity(`📉 Model drift check: ${data.status}`);
    const cls = data.status === 'healthy' ? 'healthy' : data.status === 'critical' ? 'critical' : 'warning';
    el.innerHTML = `<div class="drift-card ${cls}">
        <h4 style="margin-bottom:8px">${data.status === 'healthy' ? '🟢' : data.status === 'critical' ? '🔴' : '🟡'} ${data.message}</h4>
        <p style="font-size:0.85rem;color:var(--text-secondary)">${data.action}</p>
        <div style="margin-top:12px;display:flex;gap:16px">
            <span class="metric ${data.degradation <= 0 ? 'good' : 'warn'}">Original: ${(data.original_score * 100).toFixed(1)}%</span>
            <span class="metric ${data.new_score >= data.original_score ? 'good' : 'bad'}">New: ${(data.new_score * 100).toFixed(1)}%</span>
            <span class="metric ${data.degradation_pct <= 0 ? 'good' : 'bad'}">Δ: ${data.degradation_pct}%</span>
        </div>
    </div>`;
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// DEPLOYMENT
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
function enableDeploymentButtons() {
    ['dlModelBtn', 'exportDeployBtn', 'dlReportBtn', 'dlCsvBtn'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.disabled = false;
    });
}

function downloadModel() { if (sessionId) window.location.href = `/api/download-model/${sessionId}`; }
function downloadCSV() { if (sessionId) window.location.href = `/api/download-csv/${sessionId}`; }
function downloadReport() { if (sessionId) window.location.href = `/api/report/${sessionId}`; }
function exportDeployment() { if (sessionId) window.location.href = `/api/export-deployment/${sessionId}`; }

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// PROJECTS
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
async function saveCurrentProject() {
    if (!sessionId) { alert('No active session to save.'); return; }
    const name = prompt('Project name:', `project_${sessionId.substring(0, 8)}`);
    if (!name) return;

    const data = await apiFetch('/api/projects/save', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ session_id: sessionId, name }),
    });

    if (data.success) {
        addActivity(`💾 Saved project: ${name}`);
        alert('Project saved!');
        loadProjectsList();
    } else {
        alert('Error saving: ' + (data.error || 'Unknown'));
    }
}

async function loadProjectsList() {
    const el = document.getElementById('projectsList');
    const data = await apiFetch('/api/projects');
    const projects = data.projects || [];

    if (!projects.length) { el.innerHTML = '<p class="text-dim">No saved projects yet.</p>'; return; }

    el.innerHTML = projects.map(p =>
        `<div class="result-card" style="display:flex;align-items:center;gap:12px">
            <div style="flex:1">
                <strong style="color:var(--text-primary)">${p.name}</strong>
                <div style="font-size:0.78rem;color:var(--text-dim)">
                    ${p.profile_summary?.problem_type || '?'} · ${p.profile_summary?.n_rows || 0} rows · ${p.size_mb} MB · ${p.saved_at ? new Date(p.saved_at).toLocaleDateString() : ''}
                </div>
            </div>
            <button class="btn btn-sm btn-primary" onclick="loadProject('${p.name}')">Load</button>
            <button class="btn btn-sm btn-secondary" onclick="deleteProject('${p.name}')" style="color:var(--red)">🗑️</button>
        </div>`
    ).join('');
}

async function loadProject(name) {
    const data = await apiFetch('/api/projects/load', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name }),
    });

    if (data.success) {
        sessionId = data.session_id;
        localStorage.setItem('automl_session_id', sessionId);
        updateSessionBadge();
        addActivity(`📂 Loaded project: ${name}`);
        if (data.metadata?.profile) {
            profileData = data.metadata.profile;
            renderProfile(profileData);
        }
        if (data.metadata?.training_results) {
            trainingResults = data.metadata.training_results;
            recommendations = data.metadata?.recommendations;
            renderAllResults();
        }
        navigateTo('dashboard');
    } else {
        alert('Error loading: ' + (data.error || 'Unknown'));
    }
}

async function deleteProject(name) {
    if (!confirm(`Delete project "${name}"?`)) return;
    await apiFetch(`/api/projects/${name}`, { method: 'DELETE' });
    loadProjectsList();
    addActivity(`🗑️ Deleted project: ${name}`);
}

// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
// V3.0 ADVANCED FEATURES
// ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

// --- Dataset Similarity ---
async function findSimilarDatasets() {
    if (!sessionId) return;
    const el = document.getElementById('similarityContent');
    el.innerHTML = '<p class="text-dim">⏳ Searching database for similar datasets...</p>';
    const data = await apiFetch(`/api/dataset-similarity/${sessionId}`);
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    if (!data.matches || !data.matches.length) { el.innerHTML = '<p class="text-dim">No highly similar datasets found.</p>'; return; }
    
    el.innerHTML = data.matches.map(m => `
        <div class="result-card">
            <h4>${m.dataset} <span style="font-size:0.8rem;color:var(--cyan)">(Similarity: ${(m.similarity * 100).toFixed(1)}%)</span></h4>
            <div style="font-size:0.85rem;color:var(--text-secondary);margin-top:8px">
                Recommended Config: <strong>${m.recommended_config || 'Auto'}</strong> 
                <br>Best Model Historically: <strong>${m.historical_best || 'Unknown'}</strong>
            </div>
        </div>
    `).join('');
}

// --- Problem Reframing ---
async function getReframingSuggestions() {
    if (!sessionId) return;
    const el = document.getElementById('reframingContent');
    el.innerHTML = '<p class="text-dim">⏳ Generating reframing ideas...</p>';
    const data = await apiFetch(`/api/problem-reframing/${sessionId}`);
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    
    el.innerHTML = data.alternatives.map(a => `
        <div class="insight-card info" style="margin-bottom:8px">
            <span class="insight-icon">📐</span>
            <div>
                <div class="insight-cat">${a.new_task}</div>
                <div class="insight-text">${a.reasoning}</div>
                <div style="font-size:0.8rem;margin-top:4px;color:var(--green)">Target: ${a.new_target} | Method: ${a.method}</div>
            </div>
        </div>
    `).join('');
}

// --- Counterfactuals ---
async function generateCounterfactuals() {
    if (!sessionId) return;
    const rowIndex = document.getElementById('cfRowIndex').value;
    const desired = document.getElementById('cfDesiredOutcome').value;
    const el = document.getElementById('cfResults');
    el.innerHTML = '<p class="text-dim">⏳ Generating Counterfactuals (this may take a moment)...</p>';
    
    const data = await apiFetch(`/api/counterfactuals/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ row_index: parseInt(rowIndex), desired_outcome: desired }),
    });
    
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    
    let html = '<div class="table-wrapper"><table class="data-table"><thead><tr><th>Original</th><th>Counterfactual</th></tr></thead><tbody>';
    for (let i = 0; i < data.counterfactuals.length; i++) {
        html += `<tr><td><pre style="margin:0;font-size:0.8rem">${JSON.stringify(data.original, null, 2)}</pre></td>
                     <td><pre style="margin:0;font-size:0.8rem">${JSON.stringify(data.counterfactuals[i], null, 2)}</pre></td></tr>`;
    }
    html += '</tbody></table></div>';
    el.innerHTML = html;
}

// --- Calibration ---
async function calibrateModel() {
    if (!sessionId) return;
    const method = document.getElementById('calibrationMethod').value;
    const el = document.getElementById('calibrationResults');
    el.innerHTML = '<p class="text-dim">⏳ Calibrating model probabilities...</p>';
    
    const data = await apiFetch(`/api/calibration/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ method: method }),
    });
    
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    
    el.innerHTML = `
        <div class="result-card">
            <h4>Calibration Complete</h4>
            <p>Original ECE: <span class="metric warn">${data.original_ece.toFixed(4)}</span></p>
            <p>Calibrated ECE: <span class="metric good">${data.calibrated_ece.toFixed(4)}</span></p>
        </div>
    `;
    addActivity(`⚖️ Applied ${method} calibration`);
}

// --- Feature Studio ---
async function previewFeature() {
    if (!sessionId) return;
    const name = document.getElementById('fsFeatureName').value;
    const expr = document.getElementById('fsExpression').value;
    const el = document.getElementById('fsPreviewResult');
    
    if (!name || !expr) { el.innerHTML = '<p class="text-dim">Please provide name and expression.</p>'; return; }
    el.innerHTML = '<p class="text-dim">⏳ Previewing feature...</p>';
    
    const data = await apiFetch(`/api/feature-studio/preview/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, expression: expr }),
    });
    
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    el.innerHTML = `<div class="result-card" style="border: 1px solid var(--green)"><p class="text-dim mb-16">✅ Expression is valid.</p> <pre style="margin:0">${JSON.stringify(data.preview.slice(0,5), null, 2)}</pre></div>`;
}

async function addFeature() {
    if (!sessionId) return;
    const name = document.getElementById('fsFeatureName').value;
    const expr = document.getElementById('fsExpression').value;
    if (!name || !expr) { alert('Provide name and expression.'); return; }
    
    const data = await apiFetch(`/api/feature-studio/add/${sessionId}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name: name, expression: expr }),
    });
    if (data.error) { alert('Error: ' + data.error); return; }
    alert('Feature added successfully!');
    addActivity(`➕ Added feature: ${name}`);
    document.getElementById('fsPreviewResult').innerHTML = '';
}

async function getFeatureSuggestions() {
    if (!sessionId) return;
    const el = document.getElementById('fsSuggestionsContent');
    el.innerHTML = '<p class="text-dim">⏳ Analyzing columns for feature combinations...</p>';
    const data = await apiFetch(`/api/feature-studio/suggest/${sessionId}`);
    
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    if (!data.suggestions.length) { el.innerHTML = '<p class="text-dim">No suggestions at this time.</p>'; return; }
    
    el.innerHTML = data.suggestions.map(s => `
        <div class="result-card mb-16">
            <h4>${s.name}</h4>
            <code>${s.expression}</code>
            <p style="font-size:0.8rem;color:var(--text-secondary);margin-top:8px">${s.reason}</p>
            <button class="btn btn-sm btn-secondary mt-16" onclick="document.getElementById('fsFeatureName').value='${s.name}';document.getElementById('fsExpression').value='${s.expression}';previewFeature();">Use This</button>
        </div>
    `).join('');
}

// --- Competition Engine ---
async function submitToCompetition() {
    if (!sessionId || !trainingResults?.best_model) { alert('Train a model first!'); return; }
    const btn = document.getElementById('compSubmitBtn');
    btn.disabled = true;
    
    const data = await apiFetch(`/api/competition/submit/${sessionId}`, { method: 'POST' });
    if (data.error) { alert('Error: ' + data.error); btn.disabled = false; return; }
    
    alert(`Model submitted! New Rank: ${data.rank}`);
    addActivity(`🏆 Submitted model to competition (Rank ${data.rank})`);
    loadCompetitionBoard();
}

async function loadCompetitionBoard() {
    const el = document.getElementById('competitionBoardContent');
    const taskType = document.getElementById('compProblemType').value || 'all';
    const data = await apiFetch(`/api/competition/leaderboard?problem_type=${taskType}`);
    
    if (data.error) { el.innerHTML = `<p class="text-dim">❌ ${data.error}</p>`; return; }
    if (!data.leaderboard || !data.leaderboard.length) { el.innerHTML = '<p class="text-dim text-center">No models on the leaderboard.</p>'; return; }
    
    let html = '<div class="table-wrapper"><table class="leaderboard-table"><thead><tr><th>Rank</th><th>Model</th><th>Dataset</th><th>Score</th><th>Date</th></tr></thead><tbody>';
    data.leaderboard.forEach(r => {
        const dateStr = new Date(r.timestamp).toLocaleDateString();
        const scoreStr = (r.best_score * 100).toFixed(2) + '%';
        html += `<tr>
            <td><span class="rank-badge ${r.rank === 1 ? 'rank-1' : r.rank === 2 ? 'rank-2' : r.rank === 3 ? 'rank-3' : 'rank-other'}">${r.rank}</span></td>
            <td style="font-weight:600;color:var(--text-primary)">${r.model}</td>
            <td style="color:var(--text-secondary)">${r.dataset_hash.substring(0,8)}</td>
            <td class="metric-good">${scoreStr}</td>
            <td style="font-size:0.8rem;color:var(--text-dim)">${dateStr}</td>
        </tr>`;
    });
    html += '</tbody></table></div>';
    el.innerHTML = html;
}

// --- Executive Report ---
async function generateExecutiveReport() {
    if (!sessionId) return;
    const btn = document.getElementById('dlExecReportBtn');
    btn.disabled = true;
    btn.innerHTML = '⏳ Generating...';
    
    const data = await apiFetch(`/api/executive-report/${sessionId}`);
    btn.disabled = false;
    btn.innerHTML = '📊 Generate Executive Report';
    
    if (data.error) { alert('Error: ' + data.error); return; }
    
    // Create a Blob and trigger download
    const blob = new Blob([data.report_html], { type: 'text/html' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `executive_summary_${sessionId.substring(0,8)}.html`;
    a.click();
    URL.revokeObjectURL(url);
    addActivity('📊 Downloaded Executive Report');
}

// Expose buttons on load completion
const originalEnableDeployment = enableDeploymentButtons;
enableDeploymentButtons = function() {
    originalEnableDeployment();
    document.getElementById('dlExecReportBtn').disabled = false;
    document.getElementById('compSubmitBtn').disabled = false;
};
