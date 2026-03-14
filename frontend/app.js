// ==========================================
// ⚖️ LEGAL CITATION AUDITOR — ENGINE
// Updated to handle HC/SC court classification from backend
// ==========================================

const API_BASE = '';

// ===== DOM REFS =====
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

// ==========================================
// 1. OATH SCREEN (Boot)
// ==========================================
function runOathSequence() {
    setTimeout(() => {
        oathScreen.classList.add('dismissed');
        appEl.classList.remove('hidden');
        setTimeout(() => {
            oathScreen.style.display = 'none';
        }, 1000);
    }, 4000);
}

// ==========================================
// 2. CLOCK
// ==========================================
function updateClock() {
    const now = new Date();
    const opts = { hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false };
    const dateStr = now.toLocaleDateString('en-IN', {
        day: '2-digit', month: 'short', year: 'numeric'
    });
    benchClock.textContent = `${dateStr}  ${now.toLocaleTimeString('en-IN', opts)}`;
}
setInterval(updateClock, 1000);
updateClock();

// Session ID
sessionId.textContent = `SCI-${Date.now().toString(36).toUpperCase().slice(-6)}`;

// ==========================================
// 3. FILE HANDLING
// ==========================================
sealDropzone.addEventListener('click', () => fileInput.click());

sealDropzone.addEventListener('dragover', (e) => {
    e.preventDefault();
    sealDropzone.classList.add('drag-over');
});

sealDropzone.addEventListener('dragleave', () => {
    sealDropzone.classList.remove('drag-over');
});

sealDropzone.addEventListener('drop', (e) => {
    e.preventDefault();
    sealDropzone.classList.remove('drag-over');
    if (e.dataTransfer.files[0]) handleFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', (e) => {
    if (e.target.files[0]) handleFile(e.target.files[0]);
});

filedRemove.addEventListener('click', (e) => {
    e.stopPropagation();
    clearFile();
});

function handleFile(file) {
    if (!file.name.toLowerCase().endsWith('.pdf')) {
        showToast('The Court only accepts PDF documents.', 'error');
        return;
    }
    if (file.size > 50 * 1024 * 1024) {
        showToast('Document exceeds the 50 MB filing limit.', 'error');
        return;
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
// 4. AUDIT
// ==========================================
gavelBtn.addEventListener('click', commenceAudit);

async function commenceAudit() {
    if (!selectedFile) return;

    // Show deliberation
    chamberIdle.classList.add('hidden');
    chamberResults.classList.add('hidden');
    chamberDelib.classList.remove('hidden');
    gavelBtn.disabled = true;

    // Hide court breakdown until results
    courtBreakdown.classList.add('hidden');

    // Reset steps (now 5 steps)
    ['ds-1','ds-2','ds-3','ds-4','ds-5'].forEach(id => {
        const el = document.getElementById(id);
        el.classList.remove('active', 'done');
    });
    document.getElementById('ds-1').classList.add('active');

    animateDeliberation();

    try {
        const formData = new FormData();
        formData.append('file', selectedFile);

        const resp = await fetch(`${API_BASE}/audit-document`, {
            method: 'POST',
            body: formData
        });

        if (!resp.ok) {
            const err = await resp.json().catch(() => ({}));
            throw new Error(err.detail || `Court error: ${resp.status}`);
        }

        const data = await resp.json();
        auditData = data.results || [];

        // Complete all steps
        finishDeliberation();

        setTimeout(() => {
            renderJudgments(data);

            const scCount = data.supreme_court_count || 0;
            const hcCount = data.high_court_count || 0;

            showToast(
                `Judgment delivered: ${data.total_citations_found || 0} citations reviewed `
                + `(${scCount} SC, ${hcCount} HC).`,
                'info'
            );
        }, 700);

    } catch (error) {
        console.error(error);
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
        const el = document.getElementById(id);
        el.classList.remove('active');
        el.classList.add('done');
    });
    document.getElementById('delib-title').textContent = 'JUDGMENT READY';
    document.getElementById('delib-sub').textContent = 'The bench has concluded its review.';
}

// ==========================================
// 5. RENDER RESULTS
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

    // Animate counters
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

    // Court breakdown bar
    courtBreakdown.classList.remove('hidden');
    cbScCount.textContent = scCount;
    cbHcCount.textContent = hcCount;

    const maxCount = Math.max(scCount, hcCount, 1);
    cbScFill.style.width = `${(scCount / maxCount) * 100}%`;
    cbHcFill.style.width = `${(hcCount / maxCount) * 100}%`;
}

/**
 * Classify status from backend response.
 * The backend sends:
 *   - "🟢 VERIFIED BY AI"      → verified
 *   - "🔴 HALLUCINATION DETECTED" → hallucinated
 *   - "🔴 NO CANDIDATES"       → no-match
 *   - "⚠️ SKIPPED"             → skipped (High Court)
 *   - "ERROR"                   → no-match
 */
function classifyStatus(item) {
    const raw = (item.verification?.status || '').toLowerCase();
    const courtType = (item.court_type || '').toLowerCase();

    // High Court cases are skipped by backend
    if (raw.includes('skipped') || raw.includes('⚠️')) return 'skipped';
    if (courtType === 'high court') return 'skipped';

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
        'verified':     'UPHELD',
        'hallucinated': 'FABRICATED',
        'skipped':      'HIGH COURT',
        'no-match':     'UNVERIFIED'
    };

    const courtType   = item.court_type || 'Unknown';
    const matchedName = v.matched_name || '—';
    const reason      = v.reason || v.message || 'No observations by the Court.';
    const filePath    = v.file_to_open || '—';

    // Court type badge icon
    const courtIcon = courtType.toLowerCase().includes('high')
        ? '<i class="fas fa-university"></i>'
        : '<i class="fas fa-landmark"></i>';

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${esc(item.target_citation)}</div>
            <span class="j-verdict-badge">${verdictLabels[status]}</span>
        </div>
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
// 6. FILTER TABS
// ==========================================
document.querySelectorAll('.jf-tab').forEach(tab => {
    tab.addEventListener('click', () => {
        document.querySelectorAll('.jf-tab').forEach(t => t.classList.remove('active'));
        tab.classList.add('active');
        const filter = tab.dataset.filter;

        document.querySelectorAll('.j-card').forEach(card => {
            card.style.display =
                (filter === 'all' || card.dataset.status === filter) ? '' : 'none';
        });
    });
});

// ==========================================
// 7. ORDER MODAL
// ==========================================
function openOrder(item, status) {
    const v = item.verification || {};
    const courtType = item.court_type || 'Unknown';

    const verdictText = {
        'verified':
            '✅ CITATION UPHELD — Exists in Supreme Court Records',
        'hallucinated':
            '❌ CITATION FABRICATED — Not Found in Any Record',
        'skipped':
            '⚠️ HIGH COURT CITATION — Bypassed Supreme Court Verification',
        'no-match':
            '⚠️ CITATION UNVERIFIED — No Matching Candidates Found'
    };

    const sectionNum = { a: 'I', b: 'II', c: 'III', d: 'IV', e: 'V' };
    let sectionIdx = 0;
    const nextSection = () => ['I','II','III','IV','V','VI'][sectionIdx++];

    let html = '';

    // Verdict banner
    html += `
        <div class="order-section">
            <div class="order-verdict-banner ${status}">
                ${verdictText[status]}
            </div>
        </div>
    `;

    // Section: Citation as submitted
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
        </div>
    `;

    // Section: Matching record (only for verified)
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
            </div>
        `;
    }

    // Section: High Court explanation (only for skipped)
    if (status === 'skipped') {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. HIGH COURT DISPOSITION</div>
                <div class="order-field">
                    <div class="of-label">STATUS</div>
                    <div class="of-value" style="color:var(--amber);">
                        <i class="fas fa-exclamation-triangle" style="margin-right:6px;"></i>
                        This citation was identified as a High Court case by the AI classifier.
                        It has been intentionally excluded from Supreme Court registry verification.
                    </div>
                </div>
                <div class="order-field">
                    <div class="of-label">RECOMMENDATION</div>
                    <div class="of-value" style="font-style:italic;">
                        Verify this citation against the relevant High Court database independently.
                    </div>
                </div>
            </div>
        `;
    }

    // Section: Observations
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
    `;

    // Section: Raw JSON
    html += `
        <div class="order-section">
            <div class="order-section-title">${nextSection()}. VERIFICATION DATA</div>
            <div class="order-field">
                <div class="of-label">RAW JSON RESPONSE</div>
                <pre style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);
                     white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0;">${esc(JSON.stringify({
                         court_type: courtType,
                         verification: v
                     }, null, 2))}</pre>
            </div>
        </div>
    `;

    orderBody.innerHTML = html;
    orderOverlay.classList.remove('hidden');
}

orderClose.addEventListener('click', () => orderOverlay.classList.add('hidden'));
orderOverlay.addEventListener('click', (e) => {
    if (e.target === orderOverlay) orderOverlay.classList.add('hidden');
});
document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') orderOverlay.classList.add('hidden');
});

// ==========================================
// 8. UTILITIES
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
    const icons = {
        success: 'check-circle',
        error: 'exclamation-triangle',
        info: 'info-circle',
        warning: 'exclamation-circle'
    };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `
        <i class="fas fa-${icons[type] || 'info-circle'}"></i>
        <span>${esc(message)}</span>
    `;
    toastContainer.appendChild(toast);
    setTimeout(() => toast.remove(), 3800);
}

// ==========================================
// 9. INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    runOathSequence();
});

// API health check
fetch(`${API_BASE}/`)
    .then(r => r.json())
    .then(d => {
        console.log('⚖️ API:', d);
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.add('online');
        apiInd.querySelector('span').textContent = 'API';
    })
    .catch(err => {
        console.warn('API not reachable:', err);
        const apiInd = document.getElementById('ind-api');
        apiInd.classList.remove('online');
        apiInd.querySelector('span').textContent = 'OFFLINE';
    });