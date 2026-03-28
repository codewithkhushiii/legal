// ==========================================
// ⚖️ CITATION AUDITOR v2.1 — APP ENGINE
// Full courtroom logic with RAG quote verification
// ==========================================

const API_BASE = window.location.origin;

// ==========================================
// GLOBAL STATE
// ==========================================
let selectedFile = null;
let auditData = null;
let allResults = [];
let chatHistory = [];
let useAuditContext = false;
let bulkFiles = [];

// ==========================================
// BOOT SEQUENCE
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    // Generate session ID
    const sid = 'SCI-' + Math.random().toString(36).substring(2, 8).toUpperCase();
    document.getElementById('session-id').textContent = sid;

    // Start clock
    updateClock();
    setInterval(updateClock, 1000);

    // Check server
    checkServer();
    setInterval(checkServer, 30000);

    // Dismiss oath screen after animation
    setTimeout(() => {
        const oath = document.getElementById('oath-screen');
        oath.classList.add('dismissed');
        setTimeout(() => {
            oath.style.display = 'none';
            document.getElementById('app').classList.remove('hidden');
        }, 1000);
    }, 4000);

    // Setup file upload
    setupFileUpload();
    setupBulkUpload();
    setupFilterTabs();
    setupModal();
    loadHistory();
});

function updateClock() {
    const el = document.getElementById('bench-clock');
    if (el) {
        const now = new Date();
        el.textContent = now.toLocaleTimeString('en-IN', {
            hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
        });
    }
}

async function checkServer() {
    try {
        const res = await fetch(`${API_BASE}/db-stats`);
        const data = await res.json();

        setIndicator('ind-api', true);
        setIndicator('ind-llm', true);
        setIndicator('ind-db', data.loaded);
        setIndicator('ind-rag', true);

        const regEl = document.getElementById('registry-records');
        if (regEl && data.record_count) {
            regEl.textContent = `${data.record_count.toLocaleString()} CASES LOADED`;
        }
    } catch (e) {
        setIndicator('ind-api', false);
        setIndicator('ind-llm', false);
        setIndicator('ind-db', false);
        setIndicator('ind-rag', false);
    }
}

function setIndicator(id, online) {
    const el = document.getElementById(id);
    if (!el) return;
    el.classList.toggle('online', online);
    el.classList.toggle('offline', !online);
}

// ==========================================
// TAB NAVIGATION
// ==========================================
function switchTab(tab) {
    const tabs = ['audit', 'search', 'bulk', 'history'];
    tabs.forEach(t => {
        const el = document.getElementById(`tab-${t}`);
        const btn = document.getElementById(`nav-${t}`);
        if (el) el.classList.toggle('hidden', t !== tab);
        if (btn) btn.classList.toggle('active', t === tab);
    });
}

// ==========================================
// FILE UPLOAD
// ==========================================
function setupFileUpload() {
    const zone = document.getElementById('seal-dropzone');
    const input = document.getElementById('file-input');
    const removeBtn = document.getElementById('filed-remove');

    zone.addEventListener('click', () => input.click());
    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
    });

    input.addEventListener('change', e => {
        if (e.target.files[0]) handleFile(e.target.files[0]);
    });

    removeBtn.addEventListener('click', clearFile);

    document.getElementById('gavel-btn').addEventListener('click', runAudit);
}

function handleFile(file) {
    if (file.type !== 'application/pdf') {
        showToast('Only PDF files are accepted', 'error');
        return;
    }
    selectedFile = file;
    document.getElementById('seal-dropzone').classList.add('hidden');
    document.getElementById('filed-doc').classList.remove('hidden');
    document.getElementById('filed-name').textContent = file.name;
    document.getElementById('filed-size').textContent = formatSize(file.size);
    document.getElementById('gavel-btn').disabled = false;
    showToast('Document filed successfully', 'success');
}

function clearFile() {
    selectedFile = null;
    document.getElementById('seal-dropzone').classList.remove('hidden');
    document.getElementById('filed-doc').classList.add('hidden');
    document.getElementById('gavel-btn').disabled = true;
    document.getElementById('file-input').value = '';
}

function formatSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

// ==========================================
// MAIN AUDIT
// ==========================================
async function runAudit() {
    if (!selectedFile) return;

    // Show deliberation
    document.getElementById('chamber-idle').classList.add('hidden');
    document.getElementById('chamber-results').classList.add('hidden');
    document.getElementById('chamber-deliberation').classList.remove('hidden');
    document.getElementById('gavel-btn').disabled = true;

    // Reset steps
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`ds-${i}`);
        step.classList.remove('active', 'done');
    }
    activateStep(1);

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);
        formData.append('language', localStorage.getItem('lexai_language') || 'english');

        activateStep(2);
        const res = await fetch(`${API_BASE}/audit-document`, {
            method: 'POST',
            body: formData
        });

        if (!res.ok) throw new Error(`Server error: ${res.status}`);

        activateStep(3);
        await sleep(500);
        activateStep(4);
        await sleep(500);
        activateStep(5);

        const data = await res.json();
        activateStep(6);
        await sleep(500);

        auditData = data;
        allResults = data.results || [];

        // Save to history
        saveToHistory(data);

        // Render
        renderVerdictSummary(data);
        renderResults(allResults);

        // Show results
        document.getElementById('chamber-deliberation').classList.add('hidden');
        document.getElementById('chamber-results').classList.remove('hidden');

        // Show post-audit actions
        document.getElementById('post-audit-actions').classList.remove('hidden');
        document.getElementById('court-breakdown').classList.remove('hidden');

        showToast('Judgment pronounced!', 'success');

    } catch (e) {
        showToast(`Audit failed: ${e.message}`, 'error');
        document.getElementById('chamber-deliberation').classList.add('hidden');
        document.getElementById('chamber-idle').classList.remove('hidden');
    }

    document.getElementById('gavel-btn').disabled = false;
}

function activateStep(n) {
    for (let i = 1; i <= 6; i++) {
        const step = document.getElementById(`ds-${i}`);
        if (i < n) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (i === n) {
            step.classList.add('active');
            step.classList.remove('done');
        } else {
            step.classList.remove('active', 'done');
        }
    }
}

function sleep(ms) { return new Promise(r => setTimeout(r, ms)); }

// ==========================================
// RENDER VERDICT SUMMARY
// ==========================================
function renderVerdictSummary(data) {
    const results = data.results || [];

    let verified = 0, fabricated = 0, skipped = 0, unverified = 0;
    let qVerified = 0, qContradicted = 0, qUnsupported = 0, qFabricated = 0;

    results.forEach(r => {
        const status = (r.verification || {}).status || '';
        if (status.includes('VERIFIED') && !status.includes('HC-')) verified++;
        else if (status.includes('HALLUCINATION')) fabricated++;
        else if (status.includes('SKIPPED') || status.includes('HC-')) skipped++;
        else unverified++;

        // Quote verification
        const qStatus = (r.quote_verification || {}).status || '';
        const qVerdict = (r.quote_verification || {}).verdict || '';
        if (qVerdict === 'SUPPORTED' || qStatus.includes('VERIFIED')) qVerified++;
        else if (qVerdict === 'CONTRADICTED' || qStatus.includes('CONTRADICTED')) qContradicted++;
        else if (qVerdict === 'UNSUPPORTED') qUnsupported++;
        if (qStatus.includes('SKIPPED') && status.includes('HALLUCINATION')) qFabricated++;
    });

    document.getElementById('vc-total').textContent = results.length;
    document.getElementById('vc-upheld').textContent = verified;
    document.getElementById('vc-overruled').textContent = fabricated;
    document.getElementById('vc-skipped').textContent = skipped;
    document.getElementById('vc-unheard').textContent = unverified;

    // Quote summary
    document.getElementById('vc-quote-verified').textContent = qVerified;
    document.getElementById('vc-quote-contradicted').textContent = qContradicted;
    document.getElementById('vc-quote-unsupported').textContent = qUnsupported;
    document.getElementById('vc-quote-fabricated').textContent = qFabricated;
    if (qVerified + qContradicted + qUnsupported + qFabricated > 0) {
        document.getElementById('quote-summary').classList.remove('hidden');
    }

    // Court breakdown bars
    const sc = data.supreme_court_count || 0;
    const hc = data.high_court_count || 0;
    const total = sc + hc || 1;
    document.getElementById('cb-sc-fill').style.width = `${(sc / total) * 100}%`;
    document.getElementById('cb-hc-fill').style.width = `${(hc / total) * 100}%`;
    document.getElementById('cb-sc-count').textContent = sc;
    document.getElementById('cb-hc-count').textContent = hc;

    // Filter counts
    document.getElementById('jf-all').textContent = results.length;
    document.getElementById('jf-upheld').textContent = verified;
    document.getElementById('jf-fabricated').textContent = fabricated;
    document.getElementById('jf-skipped').textContent = skipped;
    document.getElementById('jf-unverified').textContent = unverified;
    document.getElementById('jf-quote-issues').textContent = qContradicted + qUnsupported;
}

// ==========================================
// RENDER RESULT CARDS
// ==========================================
function renderResults(results, filter = 'all') {
    const roll = document.getElementById('judgment-roll');
    roll.innerHTML = '';

    const filtered = results.filter(r => {
        if (filter === 'all') return true;
        const cat = categorize(r);
        if (filter === 'quote-issue') {
            const qv = (r.quote_verification || {}).verdict || '';
            return qv === 'CONTRADICTED' || qv === 'UNSUPPORTED';
        }
        return cat === filter;
    });

    if (filtered.length === 0) {
        roll.innerHTML = '<div style="text-align:center;padding:3rem;color:var(--text-muted);"><p>No results match this filter.</p></div>';
        return;
    }

    filtered.forEach((r, i) => {
        const card = createJudgmentCard(r, i);
        roll.appendChild(card);
    });
}

function categorize(r) {
    const status = (r.verification || {}).status || '';
    if (status.includes('VERIFIED') && !status.includes('HC-')) return 'verified';
    if (status.includes('HALLUCINATION')) return 'hallucinated';
    if (status.includes('SKIPPED') || status.includes('HC-')) return 'skipped';
    return 'no-match';
}

function createJudgmentCard(r, index) {
    const cat = categorize(r);
    const v = r.verification || {};
    const q = r.quote_verification || {};
    const confidence = v.confidence || 0;

    // Verdict label
    const verdictLabels = {
        'verified': 'UPHELD',
        'hallucinated': 'FABRICATED',
        'skipped': 'HIGH COURT',
        'no-match': 'UNVERIFIED'
    };

    // Quote verification badge
    let qvBadge = '';
    const qVerdict = q.verdict || '';
    const qStatus = q.status || '';
    if (qStatus.includes('VERIFIED') || qVerdict === 'SUPPORTED') {
        qvBadge = '<span class="qv-badge qv-ok"><i class="fas fa-check"></i> Quote OK</span>';
    } else if (qVerdict === 'CONTRADICTED' || qStatus.includes('CONTRADICTED')) {
        qvBadge = '<span class="qv-badge qv-bad"><i class="fas fa-times"></i> Contradicted</span>';
    } else if (qVerdict === 'PARTIALLY_SUPPORTED') {
        qvBadge = '<span class="qv-badge qv-warn"><i class="fas fa-exclamation"></i> Partial</span>';
    } else if (qVerdict === 'UNSUPPORTED') {
        qvBadge = '<span class="qv-badge qv-warn"><i class="fas fa-question"></i> Unsupported</span>';
    } else if (qStatus.includes('SKIPPED')) {
        qvBadge = '<span class="qv-badge qv-skip"><i class="fas fa-forward"></i> Skipped</span>';
    }

    // Has quote issue?
    const hasQuoteIssue = qVerdict === 'CONTRADICTED' || qVerdict === 'UNSUPPORTED';

    // Confidence bar
    const confClass = confidence >= 70 ? 'high' : confidence >= 40 ? 'mid' : 'low';

    const card = document.createElement('div');
    card.className = `j-card ${cat} ${hasQuoteIssue ? 'quote-issue' : ''}`;
    card.style.animationDelay = `${index * 0.05}s`;
    card.onclick = () => openOrderModal(r, index);

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${r.target_citation || 'Unknown Citation'}</div>
            <div class="j-badges">
                <span class="j-verdict-badge">${verdictLabels[cat] || 'UNKNOWN'}</span>
                ${qvBadge}
            </div>
        </div>
        ${confidence > 0 ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">CONFIDENCE</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confClass}" style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : ''}
        <div class="j-card-details">
            ${v.matched_name ? `<div class="j-detail"><i class="fas fa-check"></i><span class="j-label">MATCH</span><span class="j-value">${v.matched_name}</span></div>` : ''}
            ${v.reason ? `<div class="j-detail"><i class="fas fa-comment"></i><span class="j-label">REASON</span><span class="j-value">${truncate(v.reason, 80)}</span></div>` : ''}
            ${v.message ? `<div class="j-detail"><i class="fas fa-info-circle"></i><span class="j-label">NOTE</span><span class="j-value">${truncate(v.message, 80)}</span></div>` : ''}
        </div>
        <div class="j-card-foot">
            <span class="j-read-order"><i class="fas fa-file-alt"></i> Read Full Order</span>
            <span class="j-serial">#${String(index + 1).padStart(3, '0')}</span>
        </div>
    `;

    return card;
}

function truncate(str, len) {
    if (!str) return '';
    return str.length > len ? str.substring(0, len) + '...' : str;
}

// ==========================================
// FILTER TABS
// ==========================================
function setupFilterTabs() {
    document.addEventListener('click', e => {
        const tab = e.target.closest('.jf-tab');
        if (!tab) return;

        document.querySelectorAll('.jf-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');

        const filter = tab.dataset.filter;
        renderResults(allResults, filter);
    });
}

// ==========================================
// ORDER MODAL
// ==========================================
function setupModal() {
    document.getElementById('order-close').addEventListener('click', () => {
        document.getElementById('order-overlay').classList.add('hidden');
    });

    document.getElementById('order-overlay').addEventListener('click', e => {
        if (e.target === e.currentTarget) {
            e.currentTarget.classList.add('hidden');
        }
    });
}

function openOrderModal(r, index) {
    const body = document.getElementById('order-body');
    const v = r.verification || {};
    const q = r.quote_verification || {};
    const cat = categorize(r);

    const verdictText = {
        'verified': '🟢 CITATION UPHELD — Verified in Supreme Court Registry',
        'hallucinated': '🔴 CITATION FABRICATED — No matching case found',
        'skipped': '🟡 HIGH COURT CITATION — Not verified against SC registry',
        'no-match': '⚪ UNVERIFIED — Could not determine validity'
    };

    let qvSection = '';
    if (q.status && !q.status.includes('SKIPPED')) {
        qvSection = `
            <div class="order-section order-quote-section">
                <div class="order-section-title">QUOTE VERIFICATION (RAG ANALYSIS)</div>
                <div class="order-field">
                    <div class="of-label">VERDICT</div>
                    <div class="of-value" style="font-weight:700;">${q.verdict || q.status || 'N/A'}</div>
                </div>
                ${q.reason ? `<div class="order-field"><div class="of-label">ANALYSIS</div><div class="of-value">${q.reason}</div></div>` : ''}
                ${q.similarity_score ? `<div class="order-field"><div class="of-label">SIMILARITY SCORE</div><div class="of-value">${(q.similarity_score * 100).toFixed(1)}%</div></div>` : ''}
                ${q.chunks_analyzed ? `<div class="order-field"><div class="of-label">CHUNKS ANALYZED</div><div class="of-value">${q.chunks_analyzed}</div></div>` : ''}
                ${q.found_paragraph ? `
                    <div class="order-field">
                        <div class="of-label">SOURCE PARAGRAPH FROM JUDGMENT</div>
                        <div class="order-source-paragraph">${q.found_paragraph}</div>
                    </div>` : ''}
            </div>`;
    }

    body.innerHTML = `
        <div class="order-section">
            <div class="order-section-title">MATTER BEFORE THE COURT</div>
            <div class="order-field">
                <div class="of-label">CITED AUTHORITY</div>
                <div class="of-value" style="font-family:var(--font-legal);font-size:1rem;font-weight:600;">${r.target_citation || 'Unknown'}</div>
            </div>
            <div class="order-field">
                <div class="of-label">COURT CLASSIFICATION</div>
                <div class="of-value">${r.court_type || 'Unknown'}</div>
            </div>
            <div class="order-field">
                <div class="of-label">SERIAL NUMBER</div>
                <div class="of-value">#${String(index + 1).padStart(3, '0')}</div>
            </div>
        </div>

        <div class="order-verdict-banner ${cat}">
            ${verdictText[cat] || 'UNKNOWN STATUS'}
        </div>

        <div class="order-section">
            <div class="order-section-title">VERIFICATION DETAILS</div>
            ${v.matched_name ? `<div class="order-field"><div class="of-label">MATCHED CASE</div><div class="of-value">${v.matched_name}</div></div>` : ''}
            ${v.matched_citation ? `<div class="order-field"><div class="of-label">DATABASE CITATION</div><div class="of-value" style="font-family:var(--font-mono);font-size:0.8rem;">${v.matched_citation}</div></div>` : ''}
            ${v.reason ? `<div class="order-field"><div class="of-label">REASONING</div><div class="of-value">${v.reason}</div></div>` : ''}
            ${v.message ? `<div class="order-field"><div class="of-label">NOTE</div><div class="of-value">${v.message}</div></div>` : ''}
            ${v.confidence ? `<div class="order-field"><div class="of-label">CONFIDENCE</div><div class="of-value">${v.confidence}%</div></div>` : ''}
        </div>

        ${qvSection}
    `;

    document.getElementById('order-overlay').classList.remove('hidden');
}

// ==========================================
// MANUAL SEARCH
// ==========================================
function fillSearch(text) {
    document.getElementById('manual-search-input').value = text;
}

async function runManualSearch() {
    const input = document.getElementById('manual-search-input');
    const area = document.getElementById('search-results-area');
    const citation = input.value.trim();

    if (!citation) {
        showToast('Please enter a citation to verify', 'warning');
        return;
    }

    area.innerHTML = '<div class="search-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Verifying citation...</p></div>';

    try {
        const res = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation, language: localStorage.getItem('lexai_language') || 'english' })
        });

        const data = await res.json();
        const v = data.verification || {};
        const status = v.status || '';

        let cat = 'no-match';
        if (status.includes('VERIFIED')) cat = 'verified';
        else if (status.includes('HALLUCINATION')) cat = 'hallucinated';

        const verdictLabels = { verified: '🟢 UPHELD', hallucinated: '🔴 FABRICATED', 'no-match': '⚪ UNVERIFIED' };

        area.innerHTML = `
            <div class="search-result-card ${cat}">
                <div class="src-header">
                    <div class="src-citation">${citation}</div>
                    <div class="src-verdict">${verdictLabels[cat] || status}</div>
                </div>
                <div class="src-details">
                    ${v.matched_name ? `<div class="src-field"><span class="src-label">MATCH</span><span>${v.matched_name}</span></div>` : ''}
                    ${v.matched_citation ? `<div class="src-field"><span class="src-label">CITATION</span><span style="font-family:var(--font-mono);">${v.matched_citation}</span></div>` : ''}
                    ${v.reason ? `<div class="src-field"><span class="src-label">REASON</span><span>${v.reason}</span></div>` : ''}
                    ${v.message ? `<div class="src-field"><span class="src-label">NOTE</span><span>${v.message}</span></div>` : ''}
                    ${v.confidence ? `<div class="src-field"><span class="src-label">CONFIDENCE</span><span>${v.confidence}%</span></div>` : ''}
                    <div class="src-field"><span class="src-label">COURT</span><span>${data.court_type || 'Unknown'}</span></div>
                </div>
            </div>
        `;
    } catch (e) {
        area.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle" style="font-size:2rem;"></i><p>Error: ${e.message}</p></div>`;
    }
}

// ==========================================
// BULK AUDIT
// ==========================================
function setupBulkUpload() {
    const zone = document.getElementById('bulk-dropzone');
    const input = document.getElementById('bulk-file-input');

    zone.addEventListener('dragover', e => { e.preventDefault(); zone.classList.add('drag-over'); });
    zone.addEventListener('dragleave', () => zone.classList.remove('drag-over'));
    zone.addEventListener('drop', e => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        addBulkFiles(e.dataTransfer.files);
    });

    input.addEventListener('change', e => addBulkFiles(e.target.files));
}

function addBulkFiles(fileList) {
    for (const file of fileList) {
        if (file.type !== 'application/pdf') continue;
        if (bulkFiles.some(f => f.name === file.name)) continue;
        bulkFiles.push(file);
    }
    renderBulkFileList();
    document.getElementById('bulk-audit-btn').disabled = bulkFiles.length === 0;
}

function renderBulkFileList() {
    const list = document.getElementById('bulk-file-list');
    list.innerHTML = bulkFiles.map((f, i) => `
        <div class="bulk-file-item">
            <i class="fas fa-file-pdf" style="color:var(--red);"></i>
            <span class="bfi-name">${f.name}</span>
            <span class="bfi-size">${formatSize(f.size)}</span>
            <button class="bfi-remove" onclick="removeBulkFile(${i})"><i class="fas fa-times"></i></button>
        </div>
    `).join('');
}

function removeBulkFile(index) {
    bulkFiles.splice(index, 1);
    renderBulkFileList();
    document.getElementById('bulk-audit-btn').disabled = bulkFiles.length === 0;
}

async function runBulkAudit() {
    if (bulkFiles.length === 0) return;

    const progress = document.getElementById('bulk-progress');
    const fill = document.getElementById('bulk-progress-fill');
    const text = document.getElementById('bulk-progress-text');
    const area = document.getElementById('bulk-results-area');
    const btn = document.getElementById('bulk-audit-btn');

    progress.classList.remove('hidden');
    btn.disabled = true;
    fill.style.width = '10%';
    text.textContent = 'Uploading documents...';

    try {
        const formData = new FormData();
        bulkFiles.forEach(f => formData.append('files', f));
        formData.append('language', localStorage.getItem('lexai_language') || 'english');

        fill.style.width = '30%';
        text.textContent = 'Processing citations...';

        const res = await fetch(`${API_BASE}/audit-multiple`, {
            method: 'POST',
            body: formData
        });

        fill.style.width = '80%';
        text.textContent = 'Analyzing results...';

        const data = await res.json();

        fill.style.width = '100%';
        text.textContent = 'Complete!';

        // Count quote issues
        let totalQuoteIssues = 0;
        (data.documents || []).forEach(doc => {
            (doc.results || []).forEach(r => {
                const qv = (r.quote_verification || {}).verdict || '';
                if (qv === 'CONTRADICTED' || qv === 'UNSUPPORTED') totalQuoteIssues++;
            });
        });

        area.innerHTML = `
            <div class="bulk-summary-header">
                <div class="bulk-stat"><div class="bs-num">${data.total_documents || 0}</div><div class="bs-label">DOCUMENTS</div></div>
                <div class="bulk-stat"><div class="bs-num">${(data.total_sc_citations || 0) + (data.total_hc_citations || 0)}</div><div class="bs-label">CITATIONS</div></div>
                <div class="bulk-stat upheld"><div class="bs-num">${data.total_verified || 0}</div><div class="bs-label">UPHELD</div></div>
                <div class="bulk-stat fabricated"><div class="bs-num">${data.total_fabricated || 0}</div><div class="bs-label">FABRICATED</div></div>
                <div class="bulk-stat quote-issues"><div class="bs-num">${totalQuoteIssues}</div><div class="bs-label">QUOTE ISSUES</div></div>
            </div>
            ${(data.documents || []).map(doc => {
                const hasError = doc.error;
                return `
                <div class="bulk-doc-card ${hasError ? 'error' : ''}">
                    <div class="bdc-header">
                        <i class="fas fa-file-pdf" style="color:${hasError ? 'var(--red)' : 'var(--gold)'};"></i>
                        <span class="bdc-name">${doc.filename}</span>
                        <span class="bdc-badge ${hasError ? 'error' : ''}">${hasError ? 'ERROR' : `${doc.citations_found || 0} citations`}</span>
                    </div>
                    ${hasError ? `<p style="font-size:0.7rem;color:var(--red);margin-top:0.3rem;">${doc.error}</p>` : `
                    <div class="bdc-stats">
                        <span>SC: ${doc.sc_count || 0}</span>
                        <span>HC: ${doc.hc_count || 0}</span>
                        <span style="color:var(--green);">Upheld: ${(doc.results || []).filter(r => (r.verification||{}).status?.includes('VERIFIED')).length}</span>
                        <span style="color:var(--red);">Fabricated: ${(doc.results || []).filter(r => (r.verification||{}).status?.includes('HALLUCINATION')).length}</span>
                    </div>`}
                </div>`;
            }).join('')}
        `;

        showToast('Bulk audit complete!', 'success');
    } catch (e) {
        showToast(`Bulk audit failed: ${e.message}`, 'error');
        area.innerHTML = `<div class="search-error"><p>Error: ${e.message}</p></div>`;
    }

    btn.disabled = false;
}

// ==========================================
// AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!auditData) return showToast('Run an audit first', 'warning');

    const overlay = document.getElementById('summary-overlay');
    const body = document.getElementById('summary-body');
    overlay.classList.remove('hidden');
    body.innerHTML = '<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary...</p></div>';

    try {
        const res = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: auditData.results || [],
                total: auditData.total_citations_found || 0,
                sc_count: auditData.supreme_court_count || 0,
                hc_count: auditData.high_court_count || 0,
                language: localStorage.getItem('lexai_language') || 'english'
            })
        });

        const data = await res.json();

        body.innerHTML = `
            <div class="order-section">
                <div class="order-section-title">RISK ASSESSMENT</div>
                <div class="order-verdict-banner ${data.risk_level === 'High' ? 'hallucinated' : data.risk_level === 'Medium' ? 'skipped' : 'verified'}">
                    RISK LEVEL: ${data.risk_level || 'Unknown'}
                </div>
            </div>
            <div class="order-section">
                <div class="order-section-title">PROFESSIONAL SUMMARY</div>
                <div class="of-value" style="white-space:pre-wrap;line-height:1.8;">${data.summary || 'No summary available.'}</div>
            </div>
            <div class="order-section">
                <div class="order-section-title">STATISTICS</div>
                <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;">
                    <div class="verdict-card upheld"><div class="vc-number">${data.stats?.verified || 0}</div><div class="vc-label">VERIFIED</div></div>
                    <div class="verdict-card overruled"><div class="vc-number">${data.stats?.fabricated || 0}</div><div class="vc-label">FABRICATED</div></div>
                    <div class="verdict-card skipped"><div class="vc-number">${data.stats?.skipped || 0}</div><div class="vc-label">SKIPPED</div></div>
                    <div class="verdict-card unheard"><div class="vc-number">${data.stats?.unverified || 0}</div><div class="vc-label">UNVERIFIED</div></div>
                </div>
            </div>
        `;
    } catch (e) {
        body.innerHTML = `<div class="search-error"><p>Error: ${e.message}</p></div>`;
    }
}

// ==========================================
// EXPORT
// ==========================================
function exportCSV() {
    if (!allResults.length) return showToast('No results to export', 'warning');

    let csv = 'Citation,Court Type,Verification Status,Confidence,Matched Case,Quote Verdict,Quote Reason\n';
    allResults.forEach(r => {
        const v = r.verification || {};
        const q = r.quote_verification || {};
        csv += `"${(r.target_citation || '').replace(/"/g, '""')}","${r.court_type || ''}","${v.status || ''}","${v.confidence || ''}","${(v.matched_name || '').replace(/"/g, '""')}","${q.verdict || q.status || ''}","${(q.reason || '').replace(/"/g, '""')}"\n`;
    });

    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'citation_audit_report.csv';
    a.click();
    URL.revokeObjectURL(url);
    showToast('CSV exported', 'success');
}

function exportPDF() {
    if (!allResults.length) return showToast('No results to export', 'warning');

    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();

        doc.setFontSize(16);
        doc.text('Legal Citation Audit Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString()}`, 20, 28);
        doc.text(`Total Citations: ${allResults.length}`, 20, 34);

        let y = 45;
        allResults.forEach((r, i) => {
            if (y > 270) { doc.addPage(); y = 20; }
            const v = r.verification || {};
            const q = r.quote_verification || {};
            doc.setFontSize(9);
            doc.text(`${i + 1}. ${(r.target_citation || 'Unknown').substring(0, 80)}`, 20, y);
            y += 5;
            doc.setFontSize(7);
            doc.text(`   Status: ${v.status || 'Unknown'} | Confidence: ${v.confidence || 'N/A'}% | Quote: ${q.verdict || q.status || 'N/A'}`, 20, y);
            y += 8;
        });

        doc.save('citation_audit_report.pdf');
        showToast('PDF exported', 'success');
    } catch (e) {
        showToast('PDF export failed: ' + e.message, 'error');
    }
}

function exportSummaryPDF() {
    showToast('Summary download coming soon', 'info');
}

// ==========================================
// HISTORY
// ==========================================
function saveToHistory(data) {
    const history = JSON.parse(localStorage.getItem('audit_history') || '[]');
    history.unshift({
        filename: data.filename || selectedFile?.name || 'Unknown',
        date: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified: (data.results || []).filter(r => (r.verification || {}).status?.includes('VERIFIED')).length,
        fabricated: (data.results || []).filter(r => (r.verification || {}).status?.includes('HALLUCINATION')).length,
    });

    // Keep last 50
    if (history.length > 50) history.length = 50;
    localStorage.setItem('audit_history', JSON.stringify(history));
    loadHistory();
}

function loadHistory() {
    const history = JSON.parse(localStorage.getItem('audit_history') || '[]');
    const list = document.getElementById('history-list');
    const count = document.getElementById('history-count');

    count.textContent = `${history.length} records`;

    if (history.length === 0) {
        list.innerHTML = '<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>';
        return;
    }

    list.innerHTML = history.map(h => `
        <div class="history-item">
            <div class="hi-top">
                <span class="hi-file"><i class="fas fa-file-pdf" style="color:var(--red);margin-right:0.3rem;"></i>${h.filename}</span>
                <span class="hi-date">${new Date(h.date).toLocaleDateString('en-IN')}</span>
            </div>
            <div class="hi-stats">
                <span>Total: ${h.total}</span>
                <span>SC: ${h.sc}</span>
                <span>HC: ${h.hc}</span>
                <span style="color:var(--green);">✓ ${h.verified}</span>
                <span style="color:var(--red);">✗ ${h.fabricated}</span>
            </div>
        </div>
    `).join('');
}

function clearHistory() {
    localStorage.removeItem('audit_history');
    loadHistory();
    showToast('History cleared', 'info');
}

// ==========================================
// TOAST NOTIFICATIONS
// ==========================================
function showToast(message, type = 'info') {
    const container = document.getElementById('toast-container');
    const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };

    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${message}</span>`;
    container.appendChild(toast);

    requestAnimationFrame(() => toast.classList.add('show'));

    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// CHATBOT
// ==========================================
function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    document.getElementById('chat-notification').style.display = 'none';
}

function toggleAuditContext() {
    useAuditContext = !useAuditContext;
    const btn = document.getElementById('ctx-btn');
    const bar = document.getElementById('chat-context-bar');
    btn.classList.toggle('active', useAuditContext);
    bar.classList.toggle('hidden', !useAuditContext);
}

function clearChat() {
    chatHistory = [];
    const msgs = document.getElementById('chat-messages');
    msgs.innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><p>Chat cleared. How can I help you?</p></div>
        </div>
    `;
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const msgs = document.getElementById('chat-messages');
    const message = input.value.trim();
    if (!message) return;

    // Add user message
    msgs.innerHTML += `
        <div class="chat-msg user">
            <div class="msg-avatar">👤</div>
            <div class="msg-bubble"><p>${escapeHtml(message)}</p></div>
        </div>
    `;

    input.value = '';
    input.style.height = 'auto';
    document.getElementById('chat-suggestions').style.display = 'none';

    // Typing indicator
    const typingId = 'typing-' + Date.now();
    msgs.innerHTML += `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble typing-indicator"><span></span><span></span><span></span></div>
        </div>
    `;
    msgs.scrollTop = msgs.scrollHeight;

    // Build context
    let auditContext = null;
    if (useAuditContext && auditData) {
        const verified = (auditData.results || []).filter(r => (r.verification || {}).status?.includes('VERIFIED')).length;
        const fabricated = (auditData.results || []).filter(r => (r.verification || {}).status?.includes('HALLUCINATION')).length;
        auditContext = `Audit of "${auditData.filename}": ${auditData.total_citations_found} citations, ${verified} verified, ${fabricated} fabricated.`;
    }

    try {
        const res = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: chatHistory,
                audit_context: auditContext,
                language: localStorage.getItem('lexai_language') || 'english'
            })
        });

        const data = await res.json();

        // Remove typing
        document.getElementById(typingId)?.remove();

        // Add response
        msgs.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble"><p>${formatChat(data.reply)}</p></div>
            </div>
        `;

        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.reply });
        if (chatHistory.length > 20) chatHistory = chatHistory.slice(-20);

    } catch (e) {
        document.getElementById(typingId)?.remove();
        msgs.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble"><p style="color:var(--red);">Error: ${e.message}</p></div>
            </div>
        `;
    }

    msgs.scrollTop = msgs.scrollHeight;
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatChat(text) {
    let html = escapeHtml(text);
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    html = html.replace(/^[-•]\s+(.+)$/gm, '<br>• $1');
    html = html.replace(/\n/g, '<br>');
    return html;
}