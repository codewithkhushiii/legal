// ==========================================
// CITATION-AUDITOR.JS
// ==========================================

let selectedFiles = [];
let auditResults = null;
let chatHistory = [];

document.addEventListener('DOMContentLoaded', () => {
    initUploadZone();
    initButtons();
    initChat();
    initFilters();
});

// ==========================================
// UPLOAD ZONE
// ==========================================
function initUploadZone() {
    const zone = document.getElementById('uploadZone');
    const input = document.getElementById('fileInput');
    
    zone.addEventListener('click', () => input.click());
    
    zone.addEventListener('dragover', (e) => {
        e.preventDefault();
        zone.classList.add('drag-over');
    });
    
    zone.addEventListener('dragleave', () => {
        zone.classList.remove('drag-over');
    });
    
    zone.addEventListener('drop', (e) => {
        e.preventDefault();
        zone.classList.remove('drag-over');
        handleFiles(e.dataTransfer.files);
    });
    
    input.addEventListener('change', (e) => {
        handleFiles(e.target.files);
    });
}

function handleFiles(fileList) {
    for (const file of fileList) {
        if (file.type !== 'application/pdf') {
            showToast(`"${file.name}" is not a PDF file`, 'error');
            continue;
        }
        if (selectedFiles.some(f => f.name === file.name)) {
            showToast(`"${file.name}" already added`, 'info');
            continue;
        }
        selectedFiles.push(file);
    }
    renderFileList();
    updateButtons();
}

function renderFileList() {
    const list = document.getElementById('fileList');
    list.innerHTML = '';
    
    selectedFiles.forEach((file, index) => {
        const item = document.createElement('div');
        item.className = 'file-item';
        item.innerHTML = `
            <div class="file-item-info">
                <i class="fas fa-file-pdf"></i>
                <span class="file-item-name">${file.name}</span>
            </div>
            <span class="file-item-size">${formatFileSize(file.size)}</span>
            <button class="file-item-remove" onclick="removeFile(${index})">
                <i class="fas fa-times"></i>
            </button>
        `;
        list.appendChild(item);
    });
}

function removeFile(index) {
    selectedFiles.splice(index, 1);
    renderFileList();
    updateButtons();
}

function updateButtons() {
    document.getElementById('btnAudit').disabled = selectedFiles.length === 0;
    document.getElementById('btnClear').disabled = selectedFiles.length === 0;
}

// ==========================================
// BUTTONS
// ==========================================
function initButtons() {
    document.getElementById('btnAudit').addEventListener('click', startAudit);
    document.getElementById('btnClear').addEventListener('click', clearAll);
    document.getElementById('btnQuickVerify').addEventListener('click', quickVerify);
    
    // Quick verify on Enter key
    document.getElementById('quickCitation').addEventListener('keypress', (e) => {
        if (e.key === 'Enter') quickVerify();
    });
}

function clearAll() {
    selectedFiles = [];
    auditResults = null;
    renderFileList();
    updateButtons();
    
    document.getElementById('summaryBanner').style.display = 'none';
    document.getElementById('aiSummary').style.display = 'none';
    document.getElementById('filterTabs').style.display = 'none';
    document.getElementById('progressSection').style.display = 'none';
    
    document.getElementById('resultsContainer').innerHTML = `
        <div class="empty-state">
            <div class="empty-icon"><i class="fas fa-balance-scale"></i></div>
            <h3>No Audit Results Yet</h3>
            <p>Upload a legal document to begin citation verification</p>
        </div>
    `;
    
    showToast('Cleared all data', 'info');
}

// ==========================================
// QUICK VERIFY
// ==========================================
async function quickVerify() {
    const input = document.getElementById('quickCitation');
    const resultDiv = document.getElementById('quickResult');
    const citation = input.value.trim();
    
    if (!citation) {
        showToast('Please enter a citation to verify', 'error');
        return;
    }
    
    resultDiv.innerHTML = `
        <div class="file-item" style="border-color: var(--gold-glow); margin-top: 8px;">
            <div class="file-item-info">
                <i class="fas fa-spinner fa-spin" style="color: var(--gold);"></i>
                <span>Verifying citation...</span>
            </div>
        </div>
    `;
    
    try {
        const response = await fetch(`${API_BASE}/verify-citation`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ citation })
        });
        
        const data = await response.json();
        const v = data.verification || {};
        const status = v.status || 'Unknown';
        
        let statusClass = 'skipped';
        let icon = 'question-circle';
        
        if (status.includes('VERIFIED')) {
            statusClass = 'verified';
            icon = 'check-circle';
        } else if (status.includes('HALLUCINATION')) {
            statusClass = 'fabricated';
            icon = 'exclamation-triangle';
        }
        
        resultDiv.innerHTML = `
            <div class="result-card" style="margin-top: 8px;">
                <div class="result-card-header">
                    <div class="result-card-left">
                        <div class="result-status-icon ${statusClass}">
                            <i class="fas fa-${icon}"></i>
                        </div>
                        <div>
                            <div class="result-citation-name">${citation}</div>
                            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:4px;">
                                ${v.matched_name || v.message || 'No match found'}
                            </div>
                        </div>
                    </div>
                </div>
                ${v.reason ? `
                <div style="padding: 12px 18px; border-top: 1px solid var(--border-color);">
                    <div class="detail-label">Reason</div>
                    <div class="detail-value">${v.reason}</div>
                </div>` : ''}
                ${v.confidence !== undefined ? `
                <div style="padding: 0 18px 12px;">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${v.confidence}%</div>
                </div>` : ''}
            </div>
        `;
    } catch (e) {
        resultDiv.innerHTML = `
            <div class="file-item" style="border-color: var(--red-border); margin-top: 8px;">
                <div class="file-item-info">
                    <i class="fas fa-exclamation-circle" style="color: var(--red);"></i>
                    <span style="color: var(--red);">Error: ${e.message}</span>
                </div>
            </div>
        `;
    }
}

// ==========================================
// MAIN AUDIT
// ==========================================
async function startAudit() {
    if (selectedFiles.length === 0) return;
    
    const progressSection = document.getElementById('progressSection');
    const btnAudit = document.getElementById('btnAudit');
    const btnClear = document.getElementById('btnClear');
    
    progressSection.style.display = 'block';
    btnAudit.disabled = true;
    btnClear.disabled = true;
    
    // Reset progress steps
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        step.classList.remove('active', 'done');
    }
    
    updateProgress(10, 'Uploading documents...', 1);
    
    try {
        const formData = new FormData();
        const isMultiple = selectedFiles.length > 1;
        
        if (isMultiple) {
            selectedFiles.forEach(f => formData.append('files', f));
        } else {
            formData.append('file', selectedFiles[0]);
        }
        
        updateProgress(20, 'Extracting text and identifying citations...', 2);
        
        const endpoint = isMultiple ? '/audit-multiple' : '/audit-document';
        const response = await fetch(`${API_BASE}${endpoint}`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`Server error: ${response.status}`);
        }
        
        updateProgress(60, 'Cross-referencing database...', 3);
        
        const data = await response.json();
        
        updateProgress(80, 'Verifying quotations...', 4);
        
        // Process results
        let results;
        if (isMultiple) {
            results = [];
            (data.documents || []).forEach(doc => {
                (doc.results || []).forEach(r => {
                    r._filename = doc.filename;
                    results.push(r);
                });
            });
            auditResults = {
                results,
                total: data.total_sc_citations + data.total_hc_citations,
                sc_count: data.total_sc_citations,
                hc_count: data.total_hc_citations
            };
        } else {
            results = data.results || [];
            auditResults = {
                results,
                total: data.total_citations_found,
                sc_count: data.supreme_court_count,
                hc_count: data.high_court_count
            };
        }
        
        updateProgress(90, 'Generating report...', 5);
        
        renderResults(auditResults);
        
        // Get AI summary
        await generateSummary(auditResults);
        
        updateProgress(100, 'Audit complete!', 5);
        
        // Mark all steps done
        for (let i = 1; i <= 5; i++) {
            document.getElementById(`step${i}`).classList.remove('active');
            document.getElementById(`step${i}`).classList.add('done');
        }
        
        showToast('Audit completed successfully!', 'success');
        
        setTimeout(() => {
            progressSection.style.display = 'none';
        }, 2000);
        
    } catch (e) {
        showToast(`Audit failed: ${e.message}`, 'error');
        progressSection.style.display = 'none';
    }
    
    btnAudit.disabled = false;
    btnClear.disabled = false;
}

function updateProgress(percent, text, activeStep) {
    document.getElementById('progressFill').style.width = `${percent}%`;
    document.getElementById('progressText').textContent = text;
    
    for (let i = 1; i <= 5; i++) {
        const step = document.getElementById(`step${i}`);
        if (i < activeStep) {
            step.classList.remove('active');
            step.classList.add('done');
        } else if (i === activeStep) {
            step.classList.add('active');
            step.classList.remove('done');
        }
    }
}

// ==========================================
// RENDER RESULTS
// ==========================================
function renderResults(data) {
    const results = data.results || [];
    
    // Count categories
    let verified = 0, fabricated = 0, skipped = 0;
    results.forEach(r => {
        const status = (r.verification || {}).status || '';
        if (status.includes('VERIFIED') && !status.includes('HC-')) verified++;
        else if (status.includes('HALLUCINATION')) fabricated++;
        else skipped++;
    });
    
    // Update summary banner
    const banner = document.getElementById('summaryBanner');
    banner.style.display = 'flex';
    document.getElementById('totalCount').textContent = results.length;
    document.getElementById('verifiedCount').textContent = verified;
    document.getElementById('fabricatedCount').textContent = fabricated;
    document.getElementById('skippedCount').textContent = skipped;
    
    // Risk badge
    const riskBadge = document.getElementById('riskBadge');
    const riskLevel = document.getElementById('riskLevel');
    if (fabricated > 2) {
        riskBadge.className = 'risk-badge risk-high';
        riskLevel.textContent = 'High Risk';
    } else if (fabricated > 0) {
        riskBadge.className = 'risk-badge risk-medium';
        riskLevel.textContent = 'Medium Risk';
    } else {
        riskBadge.className = 'risk-badge risk-low';
        riskLevel.textContent = 'Low Risk';
    }
    
    // Show filter tabs
    document.getElementById('filterTabs').style.display = 'flex';
    
    // Render cards
    renderResultCards(results, 'all');
}

function renderResultCards(results, filter) {
    const container = document.getElementById('resultsContainer');
    container.innerHTML = '';
    
    const filtered = results.filter(r => {
        if (filter === 'all') return true;
        const status = (r.verification || {}).status || '';
        if (filter === 'verified') return status.includes('VERIFIED') && !status.includes('HC-');
        if (filter === 'fabricated') return status.includes('HALLUCINATION');
        if (filter === 'skipped') return !status.includes('VERIFIED') && !status.includes('HALLUCINATION');
        return true;
    });
    
    if (filtered.length === 0) {
        container.innerHTML = `
            <div class="empty-state" style="padding: 3rem;">
                <p style="color: var(--text-muted);">No results match this filter.</p>
            </div>
        `;
        return;
    }
    
    filtered.forEach((r, idx) => {
        const v = r.verification || {};
        const q = r.quote_verification || {};
        const status = v.status || 'Unknown';
        
        let statusClass = 'skipped';
        let icon = 'question-circle';
        
        if (status.includes('VERIFIED') && !status.includes('HC-')) {
            statusClass = 'verified';
            icon = 'check-circle';
        } else if (status.includes('HALLUCINATION')) {
            statusClass = 'fabricated';
            icon = 'exclamation-triangle';
        }
        
        const courtType = (r.court_type || '').includes('High Court') ? 'hc' : 'sc';
        const courtLabel = courtType === 'hc' ? 'HC' : 'SC';
        
        // Quote verification status
        let qStatusClass = 'skipped';
        let qLabel = q.status || 'N/A';
        if (qLabel.includes('VERIFIED')) qStatusClass = 'verified';
        else if (qLabel.includes('CONTRADICTED')) qStatusClass = 'contradicted';
        else if (qLabel.includes('PARTIAL')) qStatusClass = 'partial';
        
        const card = document.createElement('div');
        card.className = `result-card`;
        card.dataset.status = statusClass;
        card.innerHTML = `
            <div class="result-card-header" onclick="toggleCard(this)">
                <div class="result-card-left">
                    <div class="result-status-icon ${statusClass}">
                        <i class="fas fa-${icon}"></i>
                    </div>
                    <span class="result-citation-name">${r.target_citation || 'Unknown'}</span>
                </div>
                <div style="display:flex; align-items:center; gap:10px;">
                    <span class="result-court-badge ${courtType}">${courtLabel}</span>
                    <button class="result-expand-btn">
                        <i class="fas fa-chevron-down"></i>
                    </button>
                </div>
            </div>
            <div class="result-card-details" id="details-${idx}">
                <div class="detail-section">
                    <div class="detail-label">Verification Status</div>
                    <div class="detail-value">${status}</div>
                </div>
                ${v.matched_name ? `
                <div class="detail-section">
                    <div class="detail-label">Matched Case</div>
                    <div class="detail-value">${v.matched_name}</div>
                </div>` : ''}
                ${v.matched_citation ? `
                <div class="detail-section">
                    <div class="detail-label">Database Citation</div>
                    <div class="detail-value mono">${v.matched_citation}</div>
                </div>` : ''}
                ${v.reason ? `
                <div class="detail-section">
                    <div class="detail-label">Matching Reason</div>
                    <div class="detail-value">${v.reason}</div>
                </div>` : ''}
                ${v.message ? `
                <div class="detail-section">
                    <div class="detail-label">Note</div>
                    <div class="detail-value">${v.message}</div>
                </div>` : ''}
                ${v.confidence !== undefined ? `
                <div class="detail-section">
                    <div class="detail-label">Confidence</div>
                    <div class="detail-value">${v.confidence}%</div>
                </div>` : ''}
                <div class="detail-section">
                    <div class="detail-label">Quote Verification</div>
                    <div>
                        <span class="quote-status ${qStatusClass}">${qLabel}</span>
                    </div>
                    ${q.reason ? `<div class="detail-value" style="margin-top:8px;">${q.reason}</div>` : ''}
                    ${q.verdict ? `<div class="detail-value" style="margin-top:4px; font-weight:600;">Verdict: ${q.verdict}</div>` : ''}
                </div>
                ${r._filename ? `
                <div class="detail-section">
                    <div class="detail-label">Source File</div>
                    <div class="detail-value mono">${r._filename}</div>
                </div>` : ''}
            </div>
        `;
        container.appendChild(card);
    });
}

function toggleCard(header) {
    const card = header.closest('.result-card');
    const details = card.querySelector('.result-card-details');
    const icon = header.querySelector('.result-expand-btn i');
    
    details.classList.toggle('open');
    icon.classList.toggle('fa-chevron-down');
    icon.classList.toggle('fa-chevron-up');
}

// ==========================================
// FILTERS
// ==========================================
function initFilters() {
    document.addEventListener('click', (e) => {
        if (e.target.classList.contains('filter-tab') || e.target.closest('.filter-tab')) {
            const tab = e.target.classList.contains('filter-tab') ? e.target : e.target.closest('.filter-tab');
            
            document.querySelectorAll('.filter-tab').forEach(t => t.classList.remove('active'));
            tab.classList.add('active');
            
            const filter = tab.dataset.filter;
            if (auditResults) {
                renderResultCards(auditResults.results, filter);
            }
        }
    });
}

// ==========================================
// AI SUMMARY
// ==========================================
async function generateSummary(data) {
    try {
        const response = await fetch(`${API_BASE}/summarize`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                results: data.results,
                total: data.total,
                sc_count: data.sc_count,
                hc_count: data.hc_count
            })
        });
        
        const summaryData = await response.json();
        
        const aiSummary = document.getElementById('aiSummary');
        aiSummary.style.display = 'block';
        document.getElementById('aiSummaryText').textContent = summaryData.summary;
        
    } catch (e) {
        console.error('Summary generation failed:', e);
    }
}

// ==========================================
// CHAT WIDGET
// ==========================================
function initChat() {
    const toggle = document.getElementById('chatToggle');
    const panel = document.getElementById('chatPanel');
    const close = document.getElementById('chatClose');
    const input = document.getElementById('chatInput');
    const send = document.getElementById('chatSend');
    
    toggle.addEventListener('click', () => {
        panel.classList.toggle('open');
    });
    
    close.addEventListener('click', () => {
        panel.classList.remove('open');
    });
    
    send.addEventListener('click', sendChatMessage);
    input.addEventListener('keypress', (e) => {
        if (e.key === 'Enter') sendChatMessage();
    });
}

async function sendChatMessage() {
    const input = document.getElementById('chatInput');
    const messages = document.getElementById('chatMessages');
    const message = input.value.trim();
    
    if (!message) return;
    
    // Add user message
    const userMsg = document.createElement('div');
    userMsg.className = 'chat-message user-message';
    userMsg.innerHTML = `
        <div class="message-avatar"><i class="fas fa-user"></i></div>
        <div class="message-content"><p>${escapeHtml(message)}</p></div>
    `;
    messages.appendChild(userMsg);
    
    input.value = '';
    messages.scrollTop = messages.scrollHeight;
    
    // Show typing indicator
    const typing = document.createElement('div');
    typing.className = 'chat-message bot-message';
    typing.id = 'typing-indicator';
    typing.innerHTML = `
        <div class="message-avatar"><i class="fas fa-robot"></i></div>
        <div class="message-content"><p><i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite;"></i> 
        <i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite 0.2s;"></i> 
        <i class="fas fa-circle" style="font-size:0.4rem;animation:pulse-dot 1s infinite 0.4s;"></i></p></div>
    `;
    messages.appendChild(typing);
    messages.scrollTop = messages.scrollHeight;
    
    // Build audit context
    let auditContext = '';
    if (auditResults) {
        const verified = auditResults.results.filter(r => (r.verification?.status || '').includes('VERIFIED')).length;
        const fabricated = auditResults.results.filter(r => (r.verification?.status || '').includes('HALLUCINATION')).length;
        auditContext = `Current audit: ${auditResults.total} citations found, ${verified} verified, ${fabricated} fabricated.`;
    }
    
    try {
        const response = await fetch(`${API_BASE}/chat`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message,
                history: chatHistory,
                audit_context: auditContext || null
            })
        });
        
        const data = await response.json();
        
        // Remove typing indicator
        document.getElementById('typing-indicator')?.remove();
        
        // Add bot response
        const botMsg = document.createElement('div');
        botMsg.className = 'chat-message bot-message';
        botMsg.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content"><p>${formatChatResponse(data.reply)}</p></div>
        `;
        messages.appendChild(botMsg);
        messages.scrollTop = messages.scrollHeight;
        
        // Update history
        chatHistory.push({ role: 'user', content: message });
        chatHistory.push({ role: 'assistant', content: data.reply });
        
        // Keep history manageable
        if (chatHistory.length > 20) {
            chatHistory = chatHistory.slice(-20);
        }
        
    } catch (e) {
        document.getElementById('typing-indicator')?.remove();
        
        const errorMsg = document.createElement('div');
        errorMsg.className = 'chat-message bot-message';
        errorMsg.innerHTML = `
            <div class="message-avatar"><i class="fas fa-robot"></i></div>
            <div class="message-content"><p style="color:var(--red);">Sorry, I encountered an error. Please try again.</p></div>
        `;
        messages.appendChild(errorMsg);
    }
}

function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

function formatChatResponse(text) {
    // Basic formatting
    let formatted = escapeHtml(text);
    
    // Bold text between **
    formatted = formatted.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Bullet points
    formatted = formatted.replace(/^[-•]\s+(.+)$/gm, '<br>• $1');
    
    // Newlines to br
    formatted = formatted.replace(/\n/g, '<br>');
    
    return formatted;
}