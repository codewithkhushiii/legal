const API_BASE = window.location.origin;

async function checkServerStatus() {
    const dot = document.getElementById('serverStatus');
    const text = document.getElementById('serverStatusText');
    if (!dot || !text) return;
    try {
        const res = await fetch(`${API_BASE}/db-stats`);
        const data = await res.json();
        dot.classList.remove('offline');
        dot.classList.add('online');
        text.textContent = `${data.record_count?.toLocaleString() || 0} records`;
        const dbRecords = document.getElementById('dbRecords');
        if (dbRecords) animateNumber(dbRecords, data.record_count || 0);
    } catch (e) {
        dot.classList.remove('online');
        dot.classList.add('offline');
        text.textContent = 'Offline';
    }
}

function animateNumber(el, target, duration = 1500) {
    const startTime = performance.now();
    function update(now) {
        const progress = Math.min((now - startTime) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        el.textContent = Math.floor(target * eased).toLocaleString();
        if (progress < 1) requestAnimationFrame(update);
    }
    requestAnimationFrame(update);
}

function formatFileSize(bytes) {
    if (bytes === 0) return '0 B';
    const k = 1024;
    const sizes = ['B', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
}

function showToast(message, type = 'info') {
    const existing = document.querySelector('.home-toast');
    if (existing) existing.remove();
    const toast = document.createElement('div');
    toast.className = `home-toast home-toast-${type}`;
    toast.style.cssText = 'position:fixed;top:85px;right:24px;z-index:2000;display:flex;align-items:center;gap:10px;padding:14px 20px;border-radius:10px;font-size:0.88rem;font-weight:500;animation:fadeInUp 0.4s ease;box-shadow:0 8px 30px rgba(0,0,0,0.3);';
    if (type === 'success') toast.style.cssText += 'background:rgba(76,175,138,0.15);border:1px solid rgba(76,175,138,0.3);color:#4caf8a;';
    else if (type === 'error') toast.style.cssText += 'background:rgba(232,119,119,0.15);border:1px solid rgba(232,119,119,0.3);color:#e87777;';
    else toast.style.cssText += 'background:rgba(119,184,232,0.15);border:1px solid rgba(119,184,232,0.3);color:#77b8e8;';
    toast.textContent = message;
    document.body.appendChild(toast);
    setTimeout(() => toast.remove(), 3500);
}

document.addEventListener('DOMContentLoaded', () => {
    checkServerStatus();
    setInterval(checkServerStatus, 30000);
});