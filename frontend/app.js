// ==========================================
// ⚖️ LEGAL CITATION AUDITOR v2.0 — ENGINE
// Features: Audit, Search, Bulk, History, Export, Chatbot, Summary
// ==========================================

const API_BASE = '';

// ===== DOM REFS — AUDIT =====
const oathScreen       = document.getElementById('oath-screen');
const appEl            = document.getElementById('app');
const sealDropzone     = document.getElementById('seal-dropzone');
const fileInput        = document.getElementById('file-input');
const filedDoc         = document.getElementById('filed-doc');
const filedName        = document.getElementById('filed-name');
const filedSize        = document.getElementById('filed-size');
const filedRemove      = document.getElementById('filed-remove');
const gavelBtn         = document.getElementById('gavel-btn');
const chamberIdle      = document.getElementById('chamber-idle');
const chamberDelib     = document.getElementById('chamber-deliberation');
const chamberResults   = document.getElementById('chamber-results');
const judgmentRoll     = document.getElementById('judgment-roll');
const orderOverlay     = document.getElementById('order-overlay');
const orderBody        = document.getElementById('order-body');
const orderClose       = document.getElementById('order-close');
const benchClock       = document.getElementById('bench-clock');
const sessionId        = document.getElementById('session-id');
const toastContainer   = document.getElementById('toast-container');

// Stats
const vcTotal      = document.getElementById('vc-total');
const vcUpheld     = document.getElementById('vc-upheld');
const vcOverruled  = document.getElementById('vc-overruled');
const vcSkipped    = document.getElementById('vc-skipped');
const vcUnheard    = document.getElementById('vc-unheard');

// Filter counts
const jfAll        = document.getElementById('jf-all');
const jfUpheld     = document.getElementById('jf-upheld');
const jfFabricated = document.getElementById('jf-fabricated');
const jfSkipped    = document.getElementById('jf-skipped');
const jfUnverified = document.getElementById('jf-unverified');

// Court breakdown
const cbScFill     = document.getElementById('cb-sc-fill');
const cbHcFill     = document.getElementById('cb-hc-fill');
const cbScCount    = document.getElementById('cb-sc-count');
const cbHcCount    = document.getElementById('cb-hc-count');
const courtBreakdown = document.getElementById('court-breakdown');

let selectedFile = null;
let auditData    = [];
let lastAuditResponse = null;

// ==========================================
// 1. OATH SCREEN (Boot)
// ==========================================
function runOathSequence() {
    setTimeout(() => {
        oathScreen.classList.add('dismissed');
        appEl.classList.remove('hidden');
        setTimeout(() => { oathScreen.style.display = 'none'; }, 1000);
    }, 4000);
}

// ==========================================
// 2. CLOCK
// ==========================================
function updateClock() {
    const now = new Date();
    const opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const dateStr = now.toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric' });
    benchClock.textContent = `${dateStr}  ${now.toLocaleTimeString('en-IN', opts)}`;
}
setInterval(updateClock, 1000);
updateClock();

sessionId.textContent = `SCI-${Date.now().toString(36).toUpperCase().slice(-6)}`;

// ==========================================
// 3. TAB NAVIGATION
// ==========================================
function switchTab(tab) {
    ['audit', 'search', 'bulk', 'history'].forEach(t => {
        const tabEl = document.getElementById(`tab-${t}`);
        const navEl = document.getElementById(`nav-${t}`);
        if (t === tab) {
            tabEl.classList.remove('hidden');
            navEl.classList.add('active');
        } else {
            tabEl.classList.add('hidden');
            navEl.classList.remove('active');
        }
    });
    if (tab === 'history') renderHistory();
}

// ==========================================
// 4. FILE HANDLING
// ==========================================
sealDropzone.addEventListener('click', () => fileInput.click());

sealDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    sealDropzone.classList.add('drag-over');
});
sealDropzone.addEventListener('dragleave', () => sealDropzone.classList.remove('drag-over'));
sealDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    sealDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});
fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});
filedRemove.addEventListener('click', (e) => { e.stopPropagation(); clearFile(); });

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('The Court only accepts PDF documents.', 'error'); return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showToast('Document exceeds the 50 MB filing limit.', 'error'); return;
    }
    selectedFile = file;
    filedName.textContent = file.name;
    filedSize.textContent = formatBytes(file.size);
    filedDoc.classList.remove('hidden');
    sealDropzone.style.display = 'none';
    gavelBtn.disabled = false;
    showToast(`Document "${file.name}" has been filed.`, 'success');
}

function clearFile() {
    selectedFile = null;
    fileInput.value = '';
    filedDoc.classList.add('hidden');
    sealDropzone.style.display = '';
    gavelBtn.disabled = true;
}

function formatBytes(bytes) {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// ==========================================
// 5. AUDIT
// ==========================================
gavelBtn.addEventListener('click', commenceAudit);

async function commenceAudit() {
    if (!selectedFile) return;

    chamberIdle.classList.add('hidden');
    chamberResults.classList.add('hidden');
    chamberDelib.classList.remove('hidden');
    gavelBtn.disabled = true;
    courtBreakdown.classList.add('hidden');
    document.getElementById('post-audit-actions').classList.add('hidden');

    ['ds-1','ds-2','ds-3','ds-4','ds-5'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });
    document.getElementById('ds-1').classList.add('active');
    animateDeliberation();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const resp = await fetch(`${API_BASE}/audit-document`, { method: 'POST', body: formData });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Court error: ${resp.status}`);
        }

        const data = await resp.json();
        auditData = data.results || [];
        lastAuditResponse = data;
        finishDeliberation();

        setTimeout(() => {
            renderJudgments(data);
            saveToHistory(data, selectedFile.name);
            const scCount = data.supreme_court_count || 0;
            const hcCount = data.high_court_count || 0;
            showToast(`Judgment delivered: ${data.total_citations_found || 0} citations reviewed (${scCount} SC, ${hcCount} HC).`, 'info');
        }, 700);

    } catch (error) {
        showToast(`Audit failed: ${error.message}`, 'error');
        chamberDelib.classList.add('hidden');
        chamberIdle.classList.remove('hidden');
    } finally {
        gavelBtn.disabled = false;
    }
}

function animateDeliberation() {
    const steps = ['ds-1', 'ds-2', 'ds-3', 'ds-4', 'ds-5'];
    const texts = [
        ['READING DOCUMENT', 'Extracting text from filed PDF...'],
        ['IDENTIFYING AUTHORITIES', 'AI is finding all cited cases...'],
        ['CLASSIFYING COURTS', 'Separating High Court and Supreme Court citations...'],
        ['SEARCHING COURT RECORDS', 'Cross-referencing SC cases against the archive...'],
        ['PRONOUNCING JUDGMENT', 'Running hallucination detection...']
    ];
    steps.forEach((id, i) => {
        setTimeout(() => {
            if (i > 0) {
                document.getElementById(steps[i - 1]).classList.remove('active');
                document.getElementById(steps[i - 1]).classList.add('done');
            }
            document.getElementById(id).classList.add('active');
            document.getElementById('delib-title').textContent = texts[i][0];
            document.getElementById('delib-sub').textContent = texts[i][1];
        }, i * 1800);
    });
}

function finishDeliberation() {
    ['ds-1','ds-2','ds-3','ds-4','ds-5'].forEach(id => {
        document.getElementById(id).classList.remove('active');
        document.getElementById(id).classList.add('done');
    });
    document.getElementById('delib-title').textContent = 'JUDGMENT READY';
    document.getElementById('delib-sub').textContent = 'The bench has concluded its review.';
}

// ==========================================
// 6. RENDER RESULTS
// ==========================================
function renderJudgments(data) {
    chamberDelib.classList.add('hidden');
    chamberResults.classList.remove('hidden');

    const results = data.results || [];
    const scCount = data.supreme_court_count || 0;
    const hcCount = data.high_court_count || 0;
    const total   = data.total_citations_found || results.length;

    let upheld = 0, fabricated = 0, skipped = 0, unverified = 0;
    judgmentRoll.innerHTML = '';

    results.forEach((item, i) => {
        const status = classifyStatus(item);
        if (status === 'verified')      upheld++;
        else if (status === 'hallucinated') fabricated++;
        else if (status === 'skipped')  skipped++;
        else                            unverified++;
        const card = buildJudgmentCard(item, i, status);
        judgmentRoll.appendChild(card);
    });

    animateNum(vcTotal, total);
    animateNum(vcUpheld, upheld);
    animateNum(vcOverruled, fabricated);
    animateNum(vcSkipped, skipped);
    animateNum(vcUnheard, unverified);

    jfAll.textContent        = total;
    jfUpheld.textContent     = upheld;
    jfFabricated.textContent = fabricated;
    jfSkipped.textContent    = skipped;
    jfUnverified.textContent = unverified;

    courtBreakdown.classList.remove('hidden');
    cbScCount.textContent = scCount;
    cbHcCount.textContent = hcCount;
    const maxCount = Math.max(scCount, hcCount, 1);
    cbScFill.style.width = `${(scCount / maxCount) * 100}%`;
    cbHcFill.style.width = `${(hcCount / maxCount) * 100}%`;

    document.getElementById('post-audit-actions').classList.remove('hidden');
}

function classifyStatus(item) {
    const raw = (item.verification?.status || '').toLowerCase();
    const courtType = (item.court_type || '').toLowerCase();
    if (raw.includes('skipped') || raw.includes('⚠️')) return 'skipped';
    if (courtType === 'high court' && !raw.includes('verified') && !raw.includes('hallucination')) return 'skipped';
    if (raw.includes('verified') || raw.includes('🟢')) return 'verified';
    if (raw.includes('hallucination') || raw.includes('🔴')) return 'hallucinated';
    return 'no-match';
}

function buildJudgmentCard(item, index, status) {
    const card = document.createElement('div');
    card.className = `j-card ${status}`;
    card.style.animationDelay = `${index * 0.08}s`;
    card.dataset.status = status;

    const v = item.verification || {};
    const verdictLabels = {
        'verified': 'UPHELD', 'hallucinated': 'FABRICATED',
        'skipped': 'HIGH COURT', 'no-match': 'UNVERIFIED'
    };
    const courtType   = item.court_type || 'Unknown';
    const matchedName = v.matched_name || '—';
    const reason      = v.reason || v.message || 'No observations by the Court.';
    const confidence  = typeof v.confidence === 'number' ? v.confidence : null;
    const courtIcon   = courtType.toLowerCase().includes('high')
        ? '<i class="fas fa-university"></i>'
        : '<i class="fas fa-landmark"></i>';

    const confidenceBar = confidence !== null ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">AI Confidence</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                     style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : '';

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${esc(item.target_citation)}</div>
            <span class="j-verdict-badge">${verdictLabels[status]}</span>
        </div>
        ${confidenceBar}
        <div class="j-card-details">
            <div class="j-detail">
                ${courtIcon}
                <span class="j-label">COURT</span>
                <span class="j-value">${esc(courtType)}</span>
            </div>
            ${status === 'verified' ? `
                <div class="j-detail">
                    <i class="fas fa-gavel"></i>
                    <span class="j-label">MATCH</span>
                    <span class="j-value">${esc(matchedName)}</span>
                </div>
            ` : ''}
            <div class="j-detail">
                <i class="fas fa-feather-alt"></i>
                <span class="j-label">NOTE</span>
                <span class="j-value">${esc(truncate(reason, 90))}</span>
            </div>
        </div>
        <div class="j-card-foot">
            <div class="j-read-order">
                <i class="fas fa-scroll"></i> READ FULL ORDER
            </div>
            <span class="j-serial">MATTER #${String(index + 1).padStart(3, '0')}</span>
        </div>
    `;
    card.addEventListener('click', () => openOrder(item, status));
    return card;
}

// ==========================================
// 7. FILTER TABS
// ==========================================
document.querySelectorAll('.jf-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.jf-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const filter = tab.dataset.filter;
        document.querySelectorAll('.j-card').forEach(card => {
            card.style.display = (filter === 'all' || card.dataset.status === filter) ? '' : 'none';
        });
    });
});

// ==========================================
// 8. ORDER MODAL
// ==========================================
function openOrder(item, status) {
    const v = item.verification || {};
    const courtType = item.court_type || 'Unknown';
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;

    const verdictText = {
        'verified': '✅ CITATION UPHELD — Exists in Supreme Court Records',
        'hallucinated': '❌ CITATION FABRICATED — Not Found in Any Record',
        'skipped': '⚠️ HIGH COURT CITATION — Bypassed Supreme Court Verification',
        'no-match': '⚠️ CITATION UNVERIFIED — No Matching Candidates Found'
    };

    let sectionIdx = 0;
    const nextSection = () => ['I','II','III','IV','V','VI'][sectionIdx++];
    let html = '';

    html += `<div class="order-section"><div class="order-verdict-banner ${status}">${verdictText[status]}</div></div>`;

    if (confidence !== null) {
        html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. AI CONFIDENCE SCORE</div>
            <div class="confidence-bar-wrap" style="padding:0.5rem 0;">
                <div class="confidence-track" style="height:12px;">
                    <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                         style="width:${confidence}%;height:100%;"></div>
                </div>
                <span class="confidence-pct" style="font-size:1.2rem;font-weight:700;color:var(--gold);">${confidence}%</span>
            </div>
        </div>`;
    }

    html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. CITATION AS SUBMITTED</div>
            <div class="order-field">
                <div class="of-label">CASE NAME / REFERENCE</div>
                <div class="of-value">${esc(item.target_citation)}</div>
            </div>
            <div class="order-field">
                <div class="of-label">COURT CLASSIFICATION</div>
                <div class="of-value">
                    ${courtType.toLowerCase().includes('high')
                        ? '<i class="fas fa-university" style="color:var(--amber);margin-right:6px;"></i>'
                        : '<i class="fas fa-landmark" style="color:var(--gold);margin-right:6px;"></i>'}
                    ${esc(courtType)}
                </div>
            </div>
        </div>`;

    if (status === 'verified') {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. MATCHING RECORD</div>
                <div class="order-field">
                    <div class="of-label">CASE ON RECORD</div>
                    <div class="of-value">${esc(v.matched_name || '—')}</div>
                </div>
                <div class="order-field">
                    <div class="of-label">SOURCE FILE</div>
                    <div class="of-value" style="font-family:var(--font-mono);font-size:0.8rem;color:var(--gold);">
                        ${esc(v.file_to_open || '—')}
                    </div>
                </div>
            </div>`;
    }

    if (status === 'skipped') {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. HIGH COURT DISPOSITION</div>
                <div class="order-field">
                    <div class="of-label">STATUS</div>
                    <div class="of-value" style="color:var(--amber);">
                        <i class="fas fa-exclamation-triangle" style="margin-right:6px;"></i>
                        This citation was identified as a High Court case by the AI classifier.
                    </div>
                </div>
                <div class="order-field">
                    <div class="of-label">RECOMMENDATION</div>
                    <div class="of-value" style="font-style:italic;">
                        Verify this citation against the relevant High Court database independently.
                    </div>
                </div>
            </div>`;
    }

    html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. OBSERVATIONS OF THE COURT</div>
            <div class="order-field">
                <div class="of-label">AI REASONING</div>
                <div class="of-value" style="font-style:italic;">
                    "${esc(v.reason || v.message || 'The Court has no further observations.')}"
                </div>
            </div>
        </div>
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. VERIFICATION DATA</div>
            <div class="order-field">
                <div class="of-label">RAW JSON RESPONSE</div>
                <pre style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0;">${esc(JSON.stringify({ court_type: courtType, verification: v }, null, 2))}</pre>
            </div>
        </div>`;

    orderBody.innerHTML = html;
    orderOverlay.classList.remove('hidden');
}

orderClose.addEventListener('click', () => orderOverlay.classList.add('hidden'));
orderOverlay.addEventListener('click', (e) => { if (e.target === orderOverlay) orderOverlay.classList.add('hidden'); });
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') { orderOverlay.classList.add('hidden'); document.getElementById('summary-overlay').classList.add('hidden'); } });

// ==========================================
// 9. MANUAL CITATION SEARCH
// ==========================================
async function runManualSearch() {
    const input = document.getElementById('manual-search-input');
    const btn = document.getElementById('search-btn');
    const area = document.getElementById('search-results-area');
    const query = input.value.trim();

    if (!query) { showToast('Please enter a citation to search.', 'warning'); return; }

    btn.disabled = true;
    btn.innerHTML = '<i class="fas fa-spinner fa-spin"></i> Verifying...';
    area.innerHTML = `<div class="search-loading"><i class="fas fa-gavel fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Cross-referencing with court archive...</p></div>`;

    try {
        const resp = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation: query })
        });
        if (!resp.ok) throw new Error(`Error: ${resp.status}`);
        const data = await resp.json();
        renderSearchResult(area, data);
    } catch (err) {
        area.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle"></i><p>Search failed: ${esc(err.message)}</p></div>`;
    } finally {
        btn.disabled = false;
        btn.innerHTML = '<i class="fas fa-gavel"></i> VERIFY';
    }
}

function fillSearch(text) {
    document.getElementById('manual-search-input').value = text;
    runManualSearch();
}

function renderSearchResult(area, data) {
    const v = data.verification || {};
    const status = classifyStatus(data);
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;

    const statusColors = {
        'verified': '#4caf8a', 'hallucinated': '#e87777',
        'skipped': '#e8b877', 'no-match': '#aaa'
    };
    const statusLabels = {
        'verified': '🟢 VERIFIED', 'hallucinated': '🔴 FABRICATED',
        'skipped': '⚠️ HIGH COURT', 'no-match': '❓ UNVERIFIED'
    };

    area.innerHTML = `
        <div class="search-result-card ${status}">
            <div class="src-header">
                <div class="src-citation">${esc(data.target_citation)}</div>
                <span class="src-verdict" style="color:${statusColors[status]}">${statusLabels[status]}</span>
            </div>
            ${confidence !== null ? `
            <div class="confidence-bar-wrap" style="margin:0.75rem 0;">
                <span class="confidence-label">AI Confidence</span>
                <div class="confidence-track"><div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" style="width:${confidence}%"></div></div>
                <span class="confidence-pct">${confidence}%</span>
            </div>` : ''}
            <div class="src-details">
                <div class="src-field"><span class="src-label">COURT</span><span>${esc(data.court_type)}</span></div>
                ${v.matched_name ? `<div class="src-field"><span class="src-label">MATCH</span><span>${esc(v.matched_name)}</span></div>` : ''}
                ${v.file_to_open ? `<div class="src-field"><span class="src-label">FILE</span><span style="font-family:var(--font-mono);font-size:0.8rem;color:var(--gold)">${esc(v.file_to_open)}</span></div>` : ''}
                <div class="src-field"><span class="src-label">REASON</span><span style="font-style:italic">${esc(v.reason || v.message || '—')}</span></div>
            </div>
        </div>`;
}

// ==========================================
// 10. BULK UPLOAD
// ==========================================
let bulkFiles = [];
const bulkFileInput = document.getElementById('bulk-file-input');
const bulkDropzone  = document.getElementById('bulk-dropzone');
const bulkAuditBtn  = document.getElementById('bulk-audit-btn');
const bulkFileList  = document.getElementById('bulk-file-list');

bulkDropzone.addEventListener('dragover', (e) => { e.preventDefault(); bulkDropzone.classList.add('drag-over'); });
bulkDropzone.addEventListener('dragleave', () => bulkDropzone.classList.remove('drag-over'));
bulkDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    bulkDropzone.classList.remove('drag-over');
    addBulkFiles([...e.dataTransfer.files]);
});
bulkFileInput.addEventListener('change', (e) => addBulkFiles([...e.target.files]));

function addBulkFiles(files) {
    const pdfs = files.filter(f => f.name.toLowerCase().endsWith('.pdf'));
    if (pdfs.length === 0) { showToast('Please select PDF files only.', 'error'); return; }
    bulkFiles = [...bulkFiles, ...pdfs].slice(0, 10); // Max 10 files
    renderBulkFileList();
    bulkAuditBtn.disabled = bulkFiles.length === 0;
}

function renderBulkFileList() {
    if (bulkFiles.length === 0) { bulkFileList.innerHTML = ''; return; }
    bulkFileList.innerHTML = bulkFiles.map((f, i) => `
        <div class="bulk-file-item">
            <i class="fas fa-file-pdf" style="color:var(--gold)"></i>
            <span class="bfi-name">${esc(f.name)}</span>
            <span class="bfi-size">${formatBytes(f.size)}</span>
            <button class="bfi-remove" onclick="removeBulkFile(${i})"><i class="fas fa-times"></i></button>
        </div>`).join('');
}

function removeBulkFile(idx) {
    bulkFiles.splice(idx, 1);
    renderBulkFileList();
    bulkAuditBtn.disabled = bulkFiles.length === 0;
}

async function runBulkAudit() {
    if (bulkFiles.length === 0) return;

    const progressEl = document.getElementById('bulk-progress');
    const progressFill = document.getElementById('bulk-progress-fill');
    const progressText = document.getElementById('bulk-progress-text');
    const resultsArea = document.getElementById('bulk-results-area');

    progressEl.classList.remove('hidden');
    bulkAuditBtn.disabled = true;
    resultsArea.innerHTML = '';

    const allResults = [];

    for (let i = 0; i < bulkFiles.length; i++) {
        const file = bulkFiles[i];
        const pct = Math.round(((i + 0.5) / bulkFiles.length) * 100);
        progressFill.style.width = `${pct}%`;
        progressText.textContent = `Processing ${i + 1}/${bulkFiles.length}: ${file.name}...`;

        try {
            const formData = new FormData();
            formData.append('file', file);
            const resp = await fetch(`${API_BASE}/audit-document`, { method: 'POST', body: formData });
            if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
            const data = await resp.json();
            allResults.push({ ...data, filename: file.name, success: true });
        } catch (err) {
            allResults.push({ filename: file.name, success: false, error: err.message });
        }
    }

    progressFill.style.width = '100%';
    progressText.textContent = 'All documents processed!';

    renderBulkResults(allResults, resultsArea);
    bulkAuditBtn.disabled = false;

    setTimeout(() => progressEl.classList.add('hidden'), 2000);
}

function renderBulkResults(results, container) {
    const totalCitations = results.reduce((s, r) => s + (r.total_citations_found || 0), 0);
    const totalVerified  = results.reduce((s, r) => s + (r.results || []).filter(x => classifyStatus(x) === 'verified').length, 0);
    const totalFabricatedC = results.reduce((s, r) => s + (r.results || []).filter(x => classifyStatus(x) === 'hallucinated').length, 0);

    container.innerHTML = `
        <div class="bulk-summary-header">
            <div class="bulk-stat"><div class="bs-num">${results.length}</div><div class="bs-label">Documents</div></div>
            <div class="bulk-stat"><div class="bs-num">${totalCitations}</div><div class="bs-label">Citations</div></div>
            <div class="bulk-stat upheld"><div class="bs-num">${totalVerified}</div><div class="bs-label">Verified</div></div>
            <div class="bulk-stat fabricated"><div class="bs-num">${totalFabricatedC}</div><div class="bs-label">Fabricated</div></div>
        </div>
        ${results.map(r => `
        <div class="bulk-doc-card ${r.success ? '' : 'error'}">
            <div class="bdc-header">
                <i class="fas fa-file-pdf" style="color:var(--gold)"></i>
                <span class="bdc-name">${esc(r.filename)}</span>
                ${r.success ? `<span class="bdc-badge">${r.total_citations_found || 0} citations</span>` : `<span class="bdc-badge error">ERROR</span>`}
            </div>
            ${r.success ? `
            <div class="bdc-stats">
                <span>✅ ${(r.results || []).filter(x => classifyStatus(x) === 'verified').length} verified</span>
                <span>❌ ${(r.results || []).filter(x => classifyStatus(x) === 'hallucinated').length} fabricated</span>
                <span>⚠️ ${(r.results || []).filter(x => classifyStatus(x) === 'skipped').length} HC</span>
            </div>` : `<p style="color:#e87777;font-size:0.8rem;">${esc(r.error)}</p>`}
        </div>`).join('')}`;
}

// ==========================================
// 11. EXPORT FUNCTIONS
// ==========================================
function exportCSV() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const rows = [['#', 'Citation', 'Court Type', 'Status', 'Confidence', 'Matched Name', 'Reason/Message']];
    auditData.forEach((item, i) => {
        const v = item.verification || {};
        rows.push([
            i + 1,
            `"${(item.target_citation || '').replace(/"/g, '""')}"`,
            `"${(item.court_type || '').replace(/"/g, '""')}"`,
            `"${(v.status || '').replace(/"/g, '""')}"`,
            v.confidence ?? '',
            `"${(v.matched_name || '').replace(/"/g, '""')}"`,
            `"${(v.reason || v.message || '').replace(/"/g, '""')}"`
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    downloadFile('audit-report.csv', csv, 'text/csv');
    showToast('CSV report downloaded.', 'success');
}

function exportPDF() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF({ orientation: 'portrait', unit: 'mm', format: 'a4' });
    const pageWidth = doc.internal.pageSize.getWidth();
    let y = 20;

    // Header
    doc.setFillColor(15, 12, 30);
    doc.rect(0, 0, pageWidth, 40, 'F');
    doc.setTextColor(212, 175, 55);
    doc.setFontSize(16);
    doc.setFont('helvetica', 'bold');
    doc.text('LEGAL CITATION AUDIT REPORT', pageWidth / 2, 18, { align: 'center' });
    doc.setFontSize(9);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(180, 180, 200);
    doc.text('Supreme Court of India • AI Citation Integrity Verification', pageWidth / 2, 26, { align: 'center' });
    doc.text(`Generated: ${new Date().toLocaleString('en-IN')} • Session: ${sessionId.textContent}`, pageWidth / 2, 33, { align: 'center' });
    y = 55;

    // Stats box
    const total = auditData.length;
    const verified = auditData.filter(r => classifyStatus(r) === 'verified').length;
    const fabricated = auditData.filter(r => classifyStatus(r) === 'hallucinated').length;
    const skipped = auditData.filter(r => classifyStatus(r) === 'skipped').length;
    const unverified = total - verified - fabricated - skipped;

    doc.setFillColor(30, 25, 60);
    doc.roundedRect(10, y, pageWidth - 20, 28, 3, 3, 'F');
    doc.setFontSize(10);
    doc.setFont('helvetica', 'bold');

    const statItems = [
        { label: 'TOTAL', val: total, color: [212, 175, 55] },
        { label: 'UPHELD', val: verified, color: [76, 175, 130] },
        { label: 'FABRICATED', val: fabricated, color: [232, 119, 119] },
        { label: 'HC', val: skipped, color: [232, 184, 119] },
        { label: 'UNVERIFIED', val: unverified, color: [170, 170, 170] }
    ];
    statItems.forEach((s, i) => {
        const x = 15 + i * 38;
        doc.setTextColor(...s.color);
        doc.setFontSize(14);
        doc.text(String(s.val), x, y + 14);
        doc.setFontSize(7);
        doc.setTextColor(150, 150, 170);
        doc.text(s.label, x, y + 22);
    });
    y += 38;

    // Citation rows
    auditData.forEach((item, i) => {
        if (y > 260) { doc.addPage(); y = 20; }
        const status = classifyStatus(item);
        const v = item.verification || {};
        const bgColors = {
            'verified': [20, 50, 35], 'hallucinated': [60, 20, 20],
            'skipped': [50, 40, 10], 'no-match': [30, 30, 50]
        };
        doc.setFillColor(...(bgColors[status] || [30, 30, 60]));
        doc.roundedRect(10, y, pageWidth - 20, 22, 2, 2, 'F');

        doc.setFontSize(8);
        doc.setFont('helvetica', 'bold');
        doc.setTextColor(212, 175, 55);
        doc.text(`#${String(i+1).padStart(3,'0')}`, 14, y + 8);

        const citation = (item.target_citation || '').substring(0, 65);
        doc.setTextColor(230, 230, 240);
        doc.text(citation + (item.target_citation.length > 65 ? '...' : ''), 28, y + 8);

        const statusColors2 = {
            'verified': [76, 200, 130], 'hallucinated': [232, 100, 100],
            'skipped': [232, 184, 119], 'no-match': [150, 150, 170]
        };
        const statusLabels2 = { 'verified': 'UPHELD', 'hallucinated': 'FABRICATED', 'skipped': 'HC', 'no-match': 'UNVERIFIED' };
        doc.setTextColor(...(statusColors2[status] || [150, 150, 170]));
        doc.setFont('helvetica', 'bold');
        doc.setFontSize(7);
        doc.text(statusLabels2[status] || 'UNKNOWN', pageWidth - 12, y + 8, { align: 'right' });

        if (v.matched_name) {
            doc.setFont('helvetica', 'italic');
            doc.setTextColor(170, 170, 200);
            doc.setFontSize(7);
            doc.text(`Match: ${v.matched_name.substring(0, 80)}`, 28, y + 15);
        }
        if (typeof v.confidence === 'number') {
            doc.setFont('helvetica', 'normal');
            doc.setTextColor(150, 150, 170);
            doc.text(`Confidence: ${v.confidence}%`, pageWidth - 12, y + 15, { align: 'right' });
        }
        y += 26;
    });

    // Footer
    doc.setFontSize(7);
    doc.setTextColor(100, 100, 130);
    doc.text('Powered by Groq × LLaMA 3.3 70B • This report is for informational purposes only and does not constitute legal advice.', pageWidth / 2, 290, { align: 'center' });

    doc.save(`legal-audit-${Date.now()}.pdf`);
    showToast('PDF report downloaded.', 'success');
}

function exportSummaryPDF() {
    const bodyEl = document.getElementById('summary-body');
    const text = bodyEl.innerText;
    if (!text || text.includes('Generating')) { showToast('Summary not ready yet.', 'warning'); return; }
    const { jsPDF } = window.jspdf;
    const doc = new jsPDF();
    doc.setFontSize(14);
    doc.setFont('helvetica', 'bold');
    doc.text('Legal Audit Summary Report', 20, 20);
    doc.setFontSize(10);
    doc.setFont('helvetica', 'normal');
    doc.setTextColor(80, 80, 80);
    const lines = doc.splitTextToSize(text, 170);
    doc.text(lines, 20, 35);
    doc.save(`audit-summary-${Date.now()}.pdf`);
}

function downloadFile(name, content, type) {
    const blob = new Blob([content], { type });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = name;
    document.body.appendChild(a); a.click();
    document.body.removeChild(a); URL.revokeObjectURL(url);
}

// ==========================================
// 12. AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!lastAuditResponse) { showToast('Run an audit first.', 'warning'); return; }
    const overlay = document.getElementById('summary-overlay');
    const body = document.getElementById('summary-body');
    overlay.classList.remove('hidden');
    body.innerHTML = `<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary...</p></div>`;

    try {
        const resp = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: lastAuditResponse.results || [],
                total: lastAuditResponse.total_citations_found || 0,
                sc_count: lastAuditResponse.supreme_court_count || 0,
                hc_count: lastAuditResponse.high_court_count || 0
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        const riskColors = { 'Low': '#4caf8a', 'Medium': '#e8b877', 'High': '#e87777' };
        const riskColor = riskColors[data.risk_level] || '#aaa';

        body.innerHTML = `
            <div class="summary-risk-badge" style="background:${riskColor}22;border:1px solid ${riskColor};color:${riskColor};">
                <i class="fas fa-shield-alt"></i> RISK LEVEL: ${esc(data.risk_level)}
            </div>
            <div class="summary-stats-row">
                <div class="sum-stat up"><span>${data.stats.verified}</span>Verified</div>
                <div class="sum-stat fab"><span>${data.stats.fabricated}</span>Fabricated</div>
                <div class="sum-stat sk"><span>${data.stats.skipped}</span>HC</div>
                <div class="sum-stat un"><span>${data.stats.unverified}</span>Unverified</div>
            </div>
            <div class="summary-text">${esc(data.summary).replace(/\n\n/g, '</p><p>').replace(/\n/g, '<br>')}</div>`;
    } catch (err) {
        body.innerHTML = `<p style="color:#e87777;">Failed to generate summary: ${esc(err.message)}</p>`;
    }
}

// ==========================================
// 13. AUDIT HISTORY
// ==========================================
function saveToHistory(data, filename) {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const entry = {
        id: Date.now(),
        filename,
        timestamp: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified:    (data.results || []).filter(r => classifyStatus(r) === 'verified').length,
        fabricated:  (data.results || []).filter(r => classifyStatus(r) === 'hallucinated').length,
        skipped:     (data.results || []).filter(r => classifyStatus(r) === 'skipped').length,
        results:     data.results || []
    };
    history.unshift(entry);
    if (history.length > 50) history.pop();
    localStorage.setItem('lca_history', JSON.stringify(history));
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const list = document.getElementById('history-list');
    const countEl = document.getElementById('history-count');
    countEl.textContent = `${history.length} record${history.length !== 1 ? 's' : ''}`;

    if (!history.length) {
        list.innerHTML = `<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>`;
        return;
    }

    list.innerHTML = history.map(entry => `
        <div class="history-item" onclick="restoreFromHistory('${entry.id}')">
            <div class="hi-left">
                <div class="hi-icon"><i class="fas fa-file-contract"></i></div>
                <div class="hi-info">
                    <div class="hi-filename">${esc(entry.filename)}</div>
                    <div class="hi-date">${new Date(entry.timestamp).toLocaleString('en-IN')}</div>
                </div>
            </div>
            <div class="hi-stats">
                <span class="hi-stat up" title="Verified">✅ ${entry.verified}</span>
                <span class="hi-stat fab" title="Fabricated">❌ ${entry.fabricated}</span>
                <span class="hi-stat hc" title="High Court">🏛️ ${entry.hc}</span>
                <span class="hi-stat total" title="Total">${entry.total} total</span>
            </div>
            <button class="hi-delete" onclick="event.stopPropagation();deleteHistoryItem(${entry.id})" title="Delete">
                <i class="fas fa-trash"></i>
            </button>
        </div>`).join('');
}

function restoreFromHistory(id) {
    const history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    const entry = history.find(h => h.id == id);
    if (!entry) return;
    auditData = entry.results;
    lastAuditResponse = {
        results: entry.results,
        total_citations_found: entry.total,
        supreme_court_count: entry.sc,
        high_court_count: entry.hc
    };
    switchTab('audit');
    renderJudgments(lastAuditResponse);
    showToast(`Loaded audit: ${entry.filename}`, 'info');
}

function deleteHistoryItem(id) {
    let history = JSON.parse(localStorage.getItem('lca_history') || '[]');
    history = history.filter(h => h.id != id);
    localStorage.setItem('lca_history', JSON.stringify(history));
    renderHistory();
}

function clearHistory() {
    if (!confirm('Clear all audit history?')) return;
    localStorage.removeItem('lca_history');
    renderHistory();
    showToast('History cleared.', 'info');
}

// ==========================================
// 14. CHATBOT
// ==========================================
let chatHistory = [];
let auditContextEnabled = false;

function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    if (!pane.classList.contains('hidden')) {
        document.getElementById('chat-notification').style.display = 'none';
        document.getElementById('chat-input').focus();
    }
}

function toggleAuditContext() {
    auditContextEnabled = !auditContextEnabled;
    const btn = document.getElementById('ctx-btn');
    const bar = document.getElementById('chat-context-bar');
    btn.style.color = auditContextEnabled ? 'var(--gold)' : '';
    bar.classList.toggle('hidden', !auditContextEnabled);
    if (auditContextEnabled && !lastAuditResponse) {
        showToast('Run an audit first to use audit context.', 'warning');
        auditContextEnabled = false;
        btn.style.color = '';
        bar.classList.add('hidden');
    }
}

function clearChat() {
    chatHistory = [];
    document.getElementById('chat-messages').innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><p>Chat cleared. How can I assist you?</p></div>
        </div>`;
}

function handleChatKey(e) {
    if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 120) + 'px';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
    document.getElementById('chat-suggestions').style.display = 'none';
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const sendBtn = document.getElementById('chat-send-btn');
    const messages = document.getElementById('chat-messages');
    const text = input.value.trim();
    if (!text) return;

    // Add user message
    chatHistory.push({ role: 'user', content: text });
    appendChatMessage('user', text, messages);
    input.value = '';
    input.style.height = 'auto';
    sendBtn.disabled = true;

    // Typing indicator
    const typingId = 'typing-' + Date.now();
    messages.insertAdjacentHTML('beforeend', `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble"><div class="typing-dots"><span></span><span></span><span></span></div></div>
        </div>`);
    messages.scrollTop = messages.scrollHeight;

    let auditContext = null;
    if (auditContextEnabled && lastAuditResponse) {
        const v = lastAuditResponse.results?.filter(r => classifyStatus(r) === 'verified').length || 0;
        const f = lastAuditResponse.results?.filter(r => classifyStatus(r) === 'hallucinated').length || 0;
        auditContext = `Last audit found ${lastAuditResponse.total_citations_found} citations: ${v} verified, ${f} fabricated.`;
    }

    try {
        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: text,
                history: chatHistory.slice(-10),
                audit_context: auditContext
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();

        document.getElementById(typingId)?.remove();
        const reply = data.reply || 'I apologize, I could not process that request.';
        chatHistory.push({ role: 'assistant', content: reply });
        appendChatMessage('assistant', reply, messages);
    } catch (err) {
        document.getElementById(typingId)?.remove();
        appendChatMessage('assistant', `⚠️ Error: ${err.message}`, messages);
    } finally {
        sendBtn.disabled = false;
        input.focus();
    }
}

function appendChatMessage(role, text, container) {
    const div = document.createElement('div');
    div.className = `chat-msg ${role}`;
    // Format text — convert **bold** and line breaks
    const formatted = esc(text)
        .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
        .replace(/\n/g, '<br>');
    div.innerHTML = role === 'assistant'
        ? `<div class="msg-avatar">⚖️</div><div class="msg-bubble">${formatted}</div>`
        : `<div class="msg-bubble">${formatted}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
}

// ==========================================
// 15. UTILITIES
// ==========================================
function esc(str) {
    if (!str) return '';
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
}

function truncate(str, max) {
    if (!str) return '';
    return str.length > max ? str.substring(0, max) + '…' : str;
}

function animateNum(el, target) {
    const duration = 900;
    const startTime = performance.now();
    function update(now) {
        const elapsed = now - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.round(target * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function showToast(message, type = 'info') {
    const icons = { success: 'check-circle', error: 'exclamation-triangle', info: 'info-circle', warning: 'exclamation-circle' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<i class="fas fa-${icons[type] || 'info-circle'}"></i><span>${esc(message)}</span>`;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3800);
}

// ==========================================
// 16. INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => { runOathSequence(); });

// API health check
fetch(`${API_BASE}/`)
    .then(r => r.json())
    .then(d => {
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.add('online');
        apiInd.querySelector('span').textContent = 'API';
        // Check DB stats
        return fetch(`${API_BASE}/db-stats`);
    })
    .then(r => r.json())
    .then(d => {
        const dbInd = document.getElementById('ind-db');
        if (d.loaded) {
            dbInd.classList.add('online');
            document.getElementById('registry-records').textContent = `${d.record_count.toLocaleString()} cases in archive`;
        } else {
            dbInd.classList.remove('online');
            document.getElementById('registry-records').textContent = 'No database loaded';
        }
    })
    .catch(() => {
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.remove('online');
        apiInd.querySelector('span').textContent = 'OFFLINE';
    });