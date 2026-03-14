// ==========================================
// ⚖️ LEGAL CITATION AUDITOR v2.1 — ENGINE
// Features: Audit, Search, Bulk, History, Export, Chatbot, Summary, RAG Quote Verification
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
const sessionIdEl      = document.getElementById('session-id');
const toastContainer   = document.getElementById('toast-container');

// Stats
const vcTotal      = document.getElementById('vc-total');
const vcUpheld     = document.getElementById('vc-upheld');
const vcOverruled  = document.getElementById('vc-overruled');
const vcSkipped    = document.getElementById('vc-skipped');
const vcUnheard    = document.getElementById('vc-unheard');

// Quote verification stats
const vcQuoteVerified     = document.getElementById('vc-quote-verified');
const vcQuoteContradicted = document.getElementById('vc-quote-contradicted');
const vcQuoteUnsupported  = document.getElementById('vc-quote-unsupported');
const vcQuoteFabricated   = document.getElementById('vc-quote-fabricated');

// Filter counts
const jfAll        = document.getElementById('jf-all');
const jfUpheld     = document.getElementById('jf-upheld');
const jfFabricated = document.getElementById('jf-fabricated');
const jfSkipped    = document.getElementById('jf-skipped');
const jfUnverified = document.getElementById('jf-unverified');
const jfQuoteIssues = document.getElementById('jf-quote-issues');

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
    // Check backend health during boot
    checkBackendHealth();
    setTimeout(() => {
        oathScreen.classList.add('dismissed');
        appEl.classList.remove('hidden');
        setTimeout(() => { oathScreen.style.display = 'none'; }, 1000);
    }, 4500);
}

async function checkBackendHealth() {
    try {
        const resp = await fetch(`${API_BASE}/db-stats`);
        const data = await resp.json();
        if (data.loaded) {
            document.getElementById('registry-records').textContent = `${data.record_count.toLocaleString()} cases in registry`;
        } else {
            document.getElementById('ind-db').classList.remove('online');
            document.getElementById('ind-db').classList.add('offline');
        }
    } catch (e) {
        console.warn('Backend not reachable:', e);
        ['ind-api', 'ind-llm', 'ind-db', 'ind-rag'].forEach(id => {
            const el = document.getElementById(id);
            if (el) { el.classList.remove('online'); el.classList.add('offline'); }
        });
    }
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

sessionIdEl.textContent = `SCI-${Date.now().toString(36).toUpperCase().slice(-6)}`;

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
    document.getElementById('quote-summary').classList.add('hidden');

    ['ds-1','ds-2','ds-3','ds-4','ds-5','ds-6'].forEach(id => {
        const el = document.getElementById(id);
        if (el) { el.classList.remove('active', 'done'); }
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
    const steps = ['ds-1', 'ds-2', 'ds-3', 'ds-4', 'ds-5', 'ds-6'];
    const texts = [
        ['READING DOCUMENT', 'Extracting text from filed PDF...'],
        ['IDENTIFYING AUTHORITIES', 'AI is finding all cited cases and attributed claims...'],
        ['CLASSIFYING COURTS', 'Separating High Court and Supreme Court citations...'],
        ['SEARCHING COURT RECORDS', 'Cross-referencing SC cases against the archive...'],
        ['RAG QUOTE VERIFICATION', 'Embedding quotes & searching source judgments...'],
        ['PRONOUNCING JUDGMENT', 'Running hallucination detection...']
    ];
    steps.forEach((id, i) => {
        setTimeout(() => {
            const el = document.getElementById(id);
            if (!el) return;
            if (i > 0) {
                const prevEl = document.getElementById(steps[i - 1]);
                if (prevEl) {
                    prevEl.classList.remove('active');
                    prevEl.classList.add('done');
                }
            }
            el.classList.add('active');
            document.getElementById('delib-title').textContent = texts[i][0];
            document.getElementById('delib-sub').textContent = texts[i][1];
        }, i * 1500);
    });
}

function finishDeliberation() {
    ['ds-1','ds-2','ds-3','ds-4','ds-5','ds-6'].forEach(id => {
        const el = document.getElementById(id);
        if (el) {
            el.classList.remove('active');
            el.classList.add('done');
        }
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
    let quoteVerified = 0, quoteContradicted = 0, quoteUnsupported = 0, quoteFabricated = 0;
    let quoteIssueCount = 0;
    
    judgmentRoll.innerHTML = '';

    results.forEach((item, i) => {
        const status = classifyStatus(item);
        if (status === 'verified')      upheld++;
        else if (status === 'hallucinated') fabricated++;
        else if (status === 'skipped')  skipped++;
        else                            unverified++;
        
        // Count quote verification stats
        const qv = item.quote_verification || {};
        const qStatus = (qv.status || '').toLowerCase();
        if (qStatus.includes('verified')) quoteVerified++;
        else if (qStatus.includes('contradicted')) { quoteContradicted++; quoteIssueCount++; }
        else if (qStatus.includes('unsupported')) { quoteUnsupported++; quoteIssueCount++; }
        else if (qStatus.includes('fabricated')) { quoteFabricated++; quoteIssueCount++; }
        
        const hasQuoteIssue = qStatus.includes('contradicted') || qStatus.includes('unsupported') || qStatus.includes('fabricated');
        const card = buildJudgmentCard(item, i, status, hasQuoteIssue);
        judgmentRoll.appendChild(card);
    });

    animateNum(vcTotal, total);
    animateNum(vcUpheld, upheld);
    animateNum(vcOverruled, fabricated);
    animateNum(vcSkipped, skipped);
    animateNum(vcUnheard, unverified);

    // Quote verification summary
    const quoteSummaryEl = document.getElementById('quote-summary');
    if (quoteVerified + quoteContradicted + quoteUnsupported + quoteFabricated > 0) {
        quoteSummaryEl.classList.remove('hidden');
        animateNum(vcQuoteVerified, quoteVerified);
        animateNum(vcQuoteContradicted, quoteContradicted);
        animateNum(vcQuoteUnsupported, quoteUnsupported);
        animateNum(vcQuoteFabricated, quoteFabricated);
    }

    jfAll.textContent        = total;
    jfUpheld.textContent     = upheld;
    jfFabricated.textContent = fabricated;
    jfSkipped.textContent    = skipped;
    jfUnverified.textContent = unverified;
    jfQuoteIssues.textContent = quoteIssueCount;

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
    if (raw.includes('skipped') || (raw.includes('⚠️') && !raw.includes('hc-'))) return 'skipped';
    if (courtType === 'high court' && !raw.includes('verified') && !raw.includes('hallucination') && !raw.includes('hc-')) return 'skipped';
    if (raw.includes('verified') || raw.includes('🟢')) return 'verified';
    if (raw.includes('hallucination') || raw.includes('🔴')) return 'hallucinated';
    if (raw.includes('hc-')) {
        if (raw.includes('verified')) return 'verified';
        if (raw.includes('hallucination')) return 'hallucinated';
    }
    return 'no-match';
}

function classifyQuoteStatus(quoteVerification) {
    if (!quoteVerification) return null;
    const s = (quoteVerification.status || '').toLowerCase();
    if (s.includes('verified')) return 'quote-ok';
    if (s.includes('contradicted')) return 'quote-contradicted';
    if (s.includes('unsupported')) return 'quote-unsupported';
    if (s.includes('fabricated')) return 'quote-fabricated';
    if (s.includes('skipped')) return 'quote-skipped';
    if (s.includes('error')) return 'quote-error';
    return null;
}

function buildJudgmentCard(item, index, status, hasQuoteIssue) {
    const card = document.createElement('div');
    card.className = `j-card ${status}${hasQuoteIssue ? ' quote-issue' : ''}`;
    card.style.animationDelay = `${index * 0.08}s`;
    card.dataset.status = status;
    card.dataset.hasQuoteIssue = hasQuoteIssue ? 'true' : 'false';

    const v = item.verification || {};
    const qv = item.quote_verification || {};
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
    
    const attributedClaim = item.attributed_claim || '';

    const confidenceBar = confidence !== null ? `
        <div class="confidence-bar-wrap">
            <span class="confidence-label">AI Confidence</span>
            <div class="confidence-track">
                <div class="confidence-fill ${confidence >= 80 ? 'high' : confidence >= 50 ? 'mid' : 'low'}" 
                     style="width:${confidence}%"></div>
            </div>
            <span class="confidence-pct">${confidence}%</span>
        </div>` : '';

    // Quote verification badge
    const qvStatus = classifyQuoteStatus(qv);
    const qvBadgeMap = {
        'quote-ok': '<span class="qv-badge qv-ok"><i class="fas fa-check-double"></i> Quote Verified</span>',
        'quote-contradicted': '<span class="qv-badge qv-bad"><i class="fas fa-exclamation-triangle"></i> Quote Contradicted</span>',
        'quote-unsupported': '<span class="qv-badge qv-warn"><i class="fas fa-question-circle"></i> Quote Unsupported</span>',
        'quote-fabricated': '<span class="qv-badge qv-bad"><i class="fas fa-ghost"></i> Quote Fabricated</span>',
        'quote-skipped': '<span class="qv-badge qv-skip"><i class="fas fa-forward"></i> Quote Check Skipped</span>',
        'quote-error': '<span class="qv-badge qv-skip"><i class="fas fa-bug"></i> Quote Check Error</span>'
    };
    const qvBadge = qvBadgeMap[qvStatus] || '';

    // Attributed claim preview
    const claimPreview = attributedClaim 
        ? `<div class="j-detail j-claim-preview">
                <i class="fas fa-quote-left"></i>
                <span class="j-label">CLAIM</span>
                <span class="j-value">"${esc(truncate(attributedClaim, 80))}"</span>
           </div>` 
        : '';

    card.innerHTML = `
        <div class="j-card-top">
            <div class="j-case-name">${esc(item.target_citation)}</div>
            <div class="j-badges">
                <span class="j-verdict-badge">${verdictLabels[status]}</span>
                ${qvBadge}
            </div>
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
            ${claimPreview}
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
            if (filter === 'all') {
                card.style.display = '';
            } else if (filter === 'quote-issue') {
                card.style.display = card.dataset.hasQuoteIssue === 'true' ? '' : 'none';
            } else {
                card.style.display = card.dataset.status === filter ? '' : 'none';
            }
        });
    });
});

// ==========================================
// 8. ORDER MODAL
// ==========================================
function openOrder(item, status) {
    const v = item.verification || {};
    const qv = item.quote_verification || {};
    const courtType = item.court_type || 'Unknown';
    const confidence = typeof v.confidence === 'number' ? v.confidence : null;
    const attributedClaim = item.attributed_claim || '';

    const verdictText = {
        'verified': '✅ CITATION UPHELD — Exists in Supreme Court Records',
        'hallucinated': '❌ CITATION FABRICATED — Not Found in Any Record',
        'skipped': '⚠️ HIGH COURT CITATION — Bypassed Supreme Court Verification',
        'no-match': '⚠️ CITATION UNVERIFIED — No Matching Candidates Found'
    };

    let sectionIdx = 0;
    const nextSection = () => ['I','II','III','IV','V','VI','VII','VIII'][sectionIdx++];
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

    // Attributed claim section
    if (attributedClaim) {
        html += `
            <div class="order-section">
                <div class="order-section-title">${nextSection()}. ATTRIBUTED CLAIM / QUOTE</div>
                <div class="order-field">
                    <div class="of-label">WHAT THE LAWYER CLAIMED THIS CASE STATES</div>
                    <div class="of-value order-quote-block">
                        <i class="fas fa-quote-left" style="color:var(--gold);opacity:0.5;margin-right:6px;"></i>
                        "${esc(attributedClaim)}"
                    </div>
                </div>
            </div>`;
    }

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

    // QUOTE VERIFICATION SECTION (RAG)
    if (qv && qv.status) {
        const qvStatusClass = classifyQuoteStatus(qv);
        const qvStatusColors = {
            'quote-ok': '#4caf8a',
            'quote-contradicted': '#e87777',
            'quote-unsupported': '#e8b877',
            'quote-fabricated': '#e87777',
            'quote-skipped': '#aaa',
            'quote-error': '#aaa'
        };
        
        html += `
            <div class="order-section order-quote-section">
                <div class="order-section-title">${nextSection()}. QUOTE VERIFICATION (RAG ENGINE)</div>
                <div class="order-field">
                    <div class="of-label">VERIFICATION STATUS</div>
                    <div class="of-value" style="color:${qvStatusColors[qvStatusClass] || 'var(--text-primary)'}; font-weight:600;">
                        ${esc(qv.status)}
                    </div>
                </div>
                <div class="order-field">
                    <div class="of-label">REASONING</div>
                    <div class="of-value" style="font-style:italic;">
                        "${esc(qv.reason || qv.explanation || 'No reasoning provided.')}"
                    </div>
                </div>`;
        
        if (qv.found_paragraph) {
            html += `
                <div class="order-field">
                    <div class="of-label">MOST RELEVANT PARAGRAPH FROM SOURCE JUDGMENT</div>
                    <div class="of-value order-source-paragraph">
                        ${esc(qv.found_paragraph)}
                    </div>
                </div>`;
        }
        
        if (qv.closest_text_found) {
            html += `
                <div class="order-field">
                    <div class="of-label">CLOSEST TEXT FOUND (LOW SIMILARITY)</div>
                    <div class="of-value order-source-paragraph" style="border-color:rgba(232,119,119,0.3);">
                        ${esc(qv.closest_text_found)}
                    </div>
                </div>`;
        }
        
        html += `</div>`;
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
                <pre style="font-family:var(--font-mono);font-size:0.65rem;color:var(--text-secondary);white-space:pre-wrap;word-break:break-all;line-height:1.6;margin:0;">${esc(JSON.stringify({ 
                    court_type: courtType, 
                    attributed_claim: attributedClaim || undefined,
                    verification: v,
                    quote_verification: Object.keys(qv).length > 0 ? qv : undefined
                }, null, 2))}</pre>
            </div>
        </div>`;

    orderBody.innerHTML = html;
    orderOverlay.classList.remove('hidden');
}

orderClose.addEventListener('click', () => orderOverlay.classList.add('hidden'));
orderOverlay.addEventListener('click', (e) => { if (e.target === orderOverlay) orderOverlay.classList.add('hidden'); });
document.addEventListener('keydown', (e) => { 
    if (e.key === 'Escape') { 
        orderOverlay.classList.add('hidden'); 
        document.getElementById('summary-overlay').classList.add('hidden'); 
    } 
});

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
    bulkFiles = [...bulkFiles, ...pdfs].slice(0, 10);
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
    const totalQuoteIssues = results.reduce((s, r) => s + (r.results || []).filter(x => {
        const qs = (x.quote_verification?.status || '').toLowerCase();
        return qs.includes('contradicted') || qs.includes('unsupported') || qs.includes('fabricated');
    }).length, 0);

    container.innerHTML = `
        <div class="bulk-summary-header">
            <div class="bulk-stat"><div class="bs-num">${results.length}</div><div class="bs-label">Documents</div></div>
            <div class="bulk-stat"><div class="bs-num">${totalCitations}</div><div class="bs-label">Citations</div></div>
            <div class="bulk-stat upheld"><div class="bs-num">${totalVerified}</div><div class="bs-label">Verified</div></div>
            <div class="bulk-stat fabricated"><div class="bs-num">${totalFabricatedC}</div><div class="bs-label">Fabricated</div></div>
            <div class="bulk-stat quote-issues"><div class="bs-num">${totalQuoteIssues}</div><div class="bs-label">Quote Issues</div></div>
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
                <span>📝 ${(r.results || []).filter(x => {
                    const qs = (x.quote_verification?.status || '').toLowerCase();
                    return qs.includes('contradicted') || qs.includes('fabricated');
                }).length} quote issues</span>
            </div>` : `<p style="color:#e87777;font-size:0.8rem;">${esc(r.error)}</p>`}
        </div>`).join('')}`;
}

// ==========================================
// 11. EXPORT FUNCTIONS
// ==========================================
function exportCSV() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    const rows = [['#', 'Citation', 'Court Type', 'Status', 'Confidence', 'Matched Name', 'Reason/Message', 'Attributed Claim', 'Quote Status', 'Quote Reason']];
    auditData.forEach((item, i) => {
        const v = item.verification || {};
        const qv = item.quote_verification || {};
        rows.push([
            i + 1,
            `"${(item.target_citation || '').replace(/"/g, '""')}"`,
            item.court_type || '',
            v.status || '',
            v.confidence ?? '',
            v.matched_name || '',
            `"${(v.reason || v.message || '').replace(/"/g, '""')}"`,
            `"${(item.attributed_claim || '').replace(/"/g, '""')}"`,
            qv.status || '',
            `"${(qv.reason || qv.explanation || '').replace(/"/g, '""')}"`
        ]);
    });
    const csv = rows.map(r => r.join(',')).join('\n');
    downloadFile(csv, 'citation_audit_report.csv', 'text/csv');
    showToast('CSV exported successfully.', 'success');
}

function exportPDF() {
    if (!auditData.length) { showToast('No audit data to export.', 'warning'); return; }
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        
        doc.setFontSize(18);
        doc.text('Legal Citation Audit Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString('en-IN')}`, 20, 28);
        doc.text(`Session: ${sessionIdEl.textContent}`, 20, 34);
        
        let y = 45;
        
        // Summary
        const verified = auditData.filter(x => classifyStatus(x) === 'verified').length;
        const fabricated = auditData.filter(x => classifyStatus(x) === 'hallucinated').length;
        const quoteIssues = auditData.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).length;
        
        doc.setFontSize(12);
        doc.text('Summary', 20, y); y += 8;
        doc.setFontSize(9);
        doc.text(`Total Citations: ${auditData.length}`, 25, y); y += 5;
        doc.text(`Verified: ${verified}  |  Fabricated: ${fabricated}  |  Quote Issues: ${quoteIssues}`, 25, y); y += 10;
        
        doc.setFontSize(12);
        doc.text('Detailed Results', 20, y); y += 8;
        
        auditData.forEach((item, i) => {
            if (y > 270) { doc.addPage(); y = 20; }
            const v = item.verification || {};
            const qv = item.quote_verification || {};
            doc.setFontSize(9);
            doc.setFont(undefined, 'bold');
            doc.text(`${i + 1}. ${truncate(item.target_citation || '', 70)}`, 20, y); y += 5;
            doc.setFont(undefined, 'normal');
            doc.text(`   Status: ${v.status || 'Unknown'}  |  Court: ${item.court_type || 'Unknown'}`, 20, y); y += 5;
            if (v.matched_name) {
                doc.text(`   Match: ${v.matched_name}`, 20, y); y += 5;
            }
            if (item.attributed_claim) {
                const claimText = truncate(item.attributed_claim, 80);
                doc.text(`   Claim: "${claimText}"`, 20, y); y += 5;
            }
            if (qv.status) {
                doc.text(`   Quote: ${qv.status}`, 20, y); y += 5;
            }
            y += 3;
        });
        
        doc.save('citation_audit_report.pdf');
        showToast('PDF exported successfully.', 'success');
    } catch (e) {
        showToast('PDF export failed: ' + e.message, 'error');
    }
}

function downloadFile(content, filename, type) {
    const blob = new Blob([content], { type });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = filename;
    a.click();
    URL.revokeObjectURL(a.href);
}

// ==========================================
// 12. AI SUMMARY
// ==========================================
async function generateSummary() {
    if (!auditData.length) { showToast('No audit data available.', 'warning'); return; }
    
    const summaryOverlay = document.getElementById('summary-overlay');
    const summaryBody = document.getElementById('summary-body');
    summaryOverlay.classList.remove('hidden');
    summaryBody.innerHTML = `<div class="summary-loading"><i class="fas fa-feather-alt fa-spin" style="font-size:2rem;color:var(--gold);"></i><p>Generating professional summary with quote verification analysis...</p></div>`;

    const scCount = lastAuditResponse?.supreme_court_count || 0;
    const hcCount = lastAuditResponse?.high_court_count || 0;

    try {
        const resp = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: auditData,
                total: auditData.length,
                sc_count: scCount,
                hc_count: hcCount
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        
        const riskColors = { 'Low': '#4caf8a', 'Medium': '#e8b877', 'High': '#e87777' };
        
        summaryBody.innerHTML = `
            <div class="summary-content">
                <div class="summary-risk-badge" style="background:${riskColors[data.risk_level] || '#aaa'}20; border:1px solid ${riskColors[data.risk_level] || '#aaa'}; color:${riskColors[data.risk_level] || '#aaa'}; padding:0.5rem 1rem; border-radius:6px; text-align:center; font-weight:700; margin-bottom:1rem;">
                    RISK LEVEL: ${data.risk_level || 'Unknown'}
                </div>
                <div class="summary-stats" style="display:grid;grid-template-columns:repeat(4,1fr);gap:0.5rem;margin-bottom:1rem;">
                    <div style="text-align:center;padding:0.5rem;background:rgba(76,175,138,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#4caf8a;">${data.stats?.verified || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Verified</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(232,119,119,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#e87777;">${data.stats?.fabricated || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Fabricated</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(232,184,119,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#e8b877;">${data.stats?.skipped || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">HC Skipped</div>
                    </div>
                    <div style="text-align:center;padding:0.5rem;background:rgba(170,170,170,0.1);border-radius:6px;">
                        <div style="font-size:1.2rem;font-weight:700;color:#aaa;">${data.stats?.unverified || 0}</div>
                        <div style="font-size:0.7rem;color:var(--text-secondary);">Unverified</div>
                    </div>
                </div>
                <div class="summary-text" style="line-height:1.8;color:var(--text-primary);white-space:pre-wrap;">${esc(data.summary)}</div>
            </div>`;
        
        lastSummaryText = data.summary;
    } catch (err) {
        summaryBody.innerHTML = `<div class="search-error"><i class="fas fa-exclamation-triangle"></i><p>Summary generation failed: ${esc(err.message)}</p></div>`;
    }
}

let lastSummaryText = '';

function exportSummaryPDF() {
    if (!lastSummaryText) { showToast('No summary to export.', 'warning'); return; }
    try {
        const { jsPDF } = window.jspdf;
        const doc = new jsPDF();
        doc.setFontSize(16);
        doc.text('AI Audit Summary Report', 20, 20);
        doc.setFontSize(10);
        doc.text(`Generated: ${new Date().toLocaleString('en-IN')}`, 20, 28);
        doc.setFontSize(10);
        const lines = doc.splitTextToSize(lastSummaryText, 170);
        doc.text(lines, 20, 40);
        doc.save('ai_audit_summary.pdf');
        showToast('Summary PDF exported.', 'success');
    } catch (e) {
        showToast('Export failed: ' + e.message, 'error');
    }
}

// ==========================================
// 13. HISTORY
// ==========================================
function saveToHistory(data, filename) {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    
    const quoteIssues = (data.results || []).filter(x => {
        const qs = (x.quote_verification?.status || '').toLowerCase();
        return qs.includes('contradicted') || qs.includes('fabricated');
    }).length;
    
    history.unshift({
        id: Date.now(),
        filename,
        date: new Date().toISOString(),
        total: data.total_citations_found || 0,
        sc: data.supreme_court_count || 0,
        hc: data.high_court_count || 0,
        verified: (data.results || []).filter(x => classifyStatus(x) === 'verified').length,
        fabricated: (data.results || []).filter(x => classifyStatus(x) === 'hallucinated').length,
        quoteIssues: quoteIssues,
        results: data.results
    });
    
    // Keep only last 50
    if (history.length > 50) history.length = 50;
    localStorage.setItem('auditHistory', JSON.stringify(history));
}

function renderHistory() {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    const list = document.getElementById('history-list');
    const count = document.getElementById('history-count');
    count.textContent = `${history.length} records`;

    if (history.length === 0) {
        list.innerHTML = `<div class="history-empty"><i class="fas fa-history" style="font-size:3rem;color:var(--gold);opacity:0.3;"></i><p>No audit history yet.</p></div>`;
        return;
    }

    list.innerHTML = history.map(h => `
        <div class="history-item" onclick='loadHistoryItem(${h.id})'>
            <div class="hi-top">
                <div class="hi-file"><i class="fas fa-file-pdf" style="color:var(--gold);margin-right:6px;"></i>${esc(h.filename)}</div>
                <div class="hi-date">${new Date(h.date).toLocaleDateString('en-IN', { day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })}</div>
            </div>
            <div class="hi-stats">
                <span>📜 ${h.total} citations</span>
                <span>✅ ${h.verified} verified</span>
                <span>❌ ${h.fabricated} fabricated</span>
                <span>⚖️ ${h.sc} SC / ${h.hc} HC</span>
                ${h.quoteIssues ? `<span>📝 ${h.quoteIssues} quote issues</span>` : ''}
            </div>
        </div>`).join('');
}

function loadHistoryItem(id) {
    const history = JSON.parse(localStorage.getItem('auditHistory') || '[]');
    const item = history.find(h => h.id === id);
    if (!item) return;
    
    auditData = item.results || [];
    lastAuditResponse = {
        results: item.results,
        total_citations_found: item.total,
        supreme_court_count: item.sc,
        high_court_count: item.hc
    };
    
    switchTab('audit');
    renderJudgments(lastAuditResponse);
    showToast(`Loaded history: ${item.filename}`, 'info');
}

function clearHistory() {
    if (confirm('Clear all audit history?')) {
        localStorage.removeItem('auditHistory');
        renderHistory();
        showToast('History cleared.', 'info');
    }
}

// ==========================================
// 14. CHATBOT
// ==========================================
let chatHistory = [];
let useAuditContext = false;

function toggleChat() {
    const pane = document.getElementById('chat-pane');
    pane.classList.toggle('hidden');
    document.getElementById('chat-notification').style.display = 'none';
}

function toggleAuditContext() {
    useAuditContext = !useAuditContext;
    const bar = document.getElementById('chat-context-bar');
    const btn = document.getElementById('ctx-btn');
    if (useAuditContext) {
        bar.classList.remove('hidden');
        btn.classList.add('active');
        if (!lastAuditResponse) {
            showToast('No audit data yet. Run an audit first!', 'warning');
        }
    } else {
        bar.classList.add('hidden');
        btn.classList.remove('active');
    }
}

function clearChat() {
    chatHistory = [];
    const messages = document.getElementById('chat-messages');
    messages.innerHTML = `
        <div class="chat-msg assistant">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble">
                <p>Chat cleared. How can I help you?</p>
            </div>
        </div>`;
    document.getElementById('chat-suggestions').style.display = '';
}

function sendSuggestion(text) {
    document.getElementById('chat-input').value = text;
    sendChatMessage();
}

function handleChatKey(event) {
    if (event.key === 'Enter' && !event.shiftKey) {
        event.preventDefault();
        sendChatMessage();
    }
}

function autoResizeChat(el) {
    el.style.height = 'auto';
    el.style.height = Math.min(el.scrollHeight, 100) + 'px';
}

async function sendChatMessage() {
    const input = document.getElementById('chat-input');
    const messages = document.getElementById('chat-messages');
    const msg = input.value.trim();
    if (!msg) return;

    // Hide suggestions
    document.getElementById('chat-suggestions').style.display = 'none';

    // Add user message
    messages.innerHTML += `
        <div class="chat-msg user">
            <div class="msg-bubble">${esc(msg)}</div>
        </div>`;
    chatHistory.push({ role: 'user', content: msg });
    input.value = '';
    input.style.height = 'auto';

    // Scroll down
    messages.scrollTop = messages.scrollHeight;

    // Show typing indicator
    const typingId = 'typing-' + Date.now();
    messages.innerHTML += `
        <div class="chat-msg assistant" id="${typingId}">
            <div class="msg-avatar">⚖️</div>
            <div class="msg-bubble typing-indicator">
                <span></span><span></span><span></span>
            </div>
        </div>`;
    messages.scrollTop = messages.scrollHeight;

    // Build audit context string
    let auditCtx = null;
    if (useAuditContext && lastAuditResponse) {
        const results = lastAuditResponse.results || [];
        const verified = results.filter(x => classifyStatus(x) === 'verified').length;
        const fabricated = results.filter(x => classifyStatus(x) === 'hallucinated').length;
        const quoteIssues = results.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).length;
        
        const fabricatedNames = results.filter(x => classifyStatus(x) === 'hallucinated').map(x => x.target_citation).slice(0, 5);
        const quoteIssueNames = results.filter(x => {
            const qs = (x.quote_verification?.status || '').toLowerCase();
            return qs.includes('contradicted') || qs.includes('fabricated');
        }).map(x => `${x.target_citation}: ${x.quote_verification?.status}`).slice(0, 5);
        
        auditCtx = `Last audit: ${results.length} citations total. ${verified} verified, ${fabricated} fabricated, ${quoteIssues} quote issues. ` +
            (fabricatedNames.length ? `Fabricated: ${fabricatedNames.join(', ')}. ` : '') +
            (quoteIssueNames.length ? `Quote issues: ${quoteIssueNames.join('; ')}. ` : '');
    }

    try {
        const resp = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
                history: chatHistory.slice(-10),
                audit_context: auditCtx
            })
        });
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        
        // Remove typing indicator
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();

        // Add assistant response
        const reply = data.reply || 'I apologize, I could not generate a response.';
        messages.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble">${formatChatResponse(reply)}</div>
            </div>`;
        chatHistory.push({ role: 'assistant', content: reply });
        messages.scrollTop = messages.scrollHeight;
    } catch (err) {
        const typingEl = document.getElementById(typingId);
        if (typingEl) typingEl.remove();
        messages.innerHTML += `
            <div class="chat-msg assistant">
                <div class="msg-avatar">⚖️</div>
                <div class="msg-bubble" style="color:#e87777;">Error: ${esc(err.message)}</div>
            </div>`;
        messages.scrollTop = messages.scrollHeight;
    }
}

function formatChatResponse(text) {
    // Convert markdown-like formatting
    let html = esc(text);
    // Bold
    html = html.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    // Italic
    html = html.replace(/\*(.*?)\*/g, '<em>$1</em>');
    // Bullet points
    html = html.replace(/^- (.*)/gm, '<li>$1</li>');
    html = html.replace(/(<li>.*<\/li>)/s, '<ul>$1</ul>');
    // Line breaks
    html = html.replace(/\n/g, '<br>');
    return html;
}

// ==========================================
// 15. UTILITY FUNCTIONS
// ==========================================
function esc(str) {
    if (!str) return '';
    const div = document.createElement('div');
    div.textContent = String(str);
    return div.innerHTML;
}

function truncate(str, maxLen) {
    if (!str) return '';
    return str.length > maxLen ? str.substring(0, maxLen) + '...' : str;
}

function animateNum(el, target) {
    if (!el) return;
    const duration = 600;
    const start = parseInt(el.textContent) || 0;
    const startTime = performance.now();
    
    function update(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3); // easeOutCubic
        el.textContent = Math.round(start + (target - start) * eased);
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function showToast(message, type = 'info') {
    const toast = document.createElement('div');
    toast.className = `toast toast-${type}`;
    
    const icons = { success: 'fa-check-circle', error: 'fa-times-circle', warning: 'fa-exclamation-triangle', info: 'fa-info-circle' };
    toast.innerHTML = `<i class="fas ${icons[type] || icons.info}"></i><span>${esc(message)}</span>`;
    
    toastContainer.appendChild(toast);
    
    // Trigger animation
    requestAnimationFrame(() => toast.classList.add('show'));
    
    setTimeout(() => {
        toast.classList.remove('show');
        toast.classList.add('hide');
        setTimeout(() => toast.remove(), 300);
    }, 4000);
}

// ==========================================
// 16. INIT
// ==========================================
document.addEventListener('DOMContentLoaded', () => {
    runOathSequence();
});