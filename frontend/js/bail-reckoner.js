const API_BASE = window.location.origin;

document.addEventListener('DOMContentLoaded', () => {
    document.getElementById('bailForm').addEventListener('submit', handleBailSubmit);
});

async function handleBailSubmit(e) {
    e.preventDefault();
    const statute = document.getElementById('statute').value;
    const offenseCategory = document.getElementById('offenseCategory').value;
    const imprisonmentDuration = parseInt(document.getElementById('imprisonmentDuration').value) || 0;
    const riskEscape = document.getElementById('riskEscape').checked;
    const riskInfluence = document.getElementById('riskInfluence').checked;
    const servedHalfTerm = document.getElementById('servedHalfTerm').checked;
    if (!statute || !offenseCategory) return;

    document.getElementById('bailEmpty').style.display = 'none';
    document.getElementById('bailResults').style.display = 'none';
    document.getElementById('bailNoData').style.display = 'none';
    document.getElementById('bailLoading').style.display = 'block';

    try {
        const res = await fetch(`${API_BASE}/reckoner/bail`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ statute, offense_category: offenseCategory, imprisonment_duration_served: imprisonmentDuration, risk_of_escape: riskEscape, risk_of_influence: riskInfluence, served_half_term: servedHalfTerm })
        });
        const data = await res.json();
        document.getElementById('bailLoading').style.display = 'none';

        if (data.status === 'NO_DATA') {
            document.getElementById('bailNoData').style.display = 'block';
            document.getElementById('noDataMessage').textContent = data.message || 'No matching cases found.';
            return;
        }
        if (data.status === 'SUCCESS') renderBailResults(data);
        else document.getElementById('bailEmpty').style.display = 'block';
    } catch (e) {
        document.getElementById('bailLoading').style.display = 'none';
        document.getElementById('bailEmpty').style.display = 'block';
    }
}

function renderBailResults(data) {
    document.getElementById('bailResults').style.display = 'block';
    const insights = data.historical_insights || {};
    const conditions = data.likely_conditions || {};
    const probability = parseFloat(insights.historical_bail_probability || '0');

    animateGauge(probability);
    document.getElementById('casesAnalyzed').textContent = insights.similar_cases_analyzed || '—';
    document.getElementById('riskScore').textContent = insights.average_risk_score !== undefined ? insights.average_risk_score.toFixed(2) : '—';

    const surety = document.getElementById('suretyBond');
    const personal = document.getElementById('personalBond');
    surety.className = conditions.surety_bond_likely ? 'bond-card likely' : 'bond-card unlikely';
    surety.querySelector('.bond-status').textContent = conditions.surety_bond_likely ? 'Likely Required' : 'Unlikely';
    personal.className = conditions.personal_bond_likely ? 'bond-card likely' : 'bond-card unlikely';
    personal.querySelector('.bond-status').textContent = conditions.personal_bond_likely ? 'Likely Required' : 'Unlikely';

    document.getElementById('strategyText').textContent = data.legal_strategy_note || 'Standard bail arguments apply.';
    const warning = document.getElementById('statutoryWarning');
    if (data.legal_strategy_note && data.legal_strategy_note.includes('⚠️')) {
        warning.style.display = 'flex';
        document.getElementById('statutoryWarningText').textContent = data.legal_strategy_note;
    } else {
        warning.style.display = 'none';
    }
}

function animateGauge(percentage) {
    const arc = document.getElementById('gaugeArc');
    const valueEl = document.getElementById('gaugeValue');
    const totalLength = 251.2;
    const startTime = performance.now();

    function animate(now) {
        const progress = Math.min((now - startTime) / 1500, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const current = percentage * eased;
        arc.setAttribute('stroke-dashoffset', totalLength - (totalLength * (current / 100)));
        valueEl.textContent = `${Math.round(current)}%`;
        valueEl.style.color = current > 70 ? '#4caf8a' : current > 40 ? '#e8b877' : '#e87777';
        if (progress < 1) requestAnimationFrame(animate);
    }
    requestAnimationFrame(animate);
}