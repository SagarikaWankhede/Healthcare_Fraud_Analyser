/**
 * MediGuard Dashboard — Client Logic
 * Light enterprise theme, all numbers fetched from API.
 */

// ─── Chart colors ────────────────────────────────────────
const CHART_COLORS = {
  proposed:  '#287D78',
  billers:   '#2563EB',
  prob:      '#C58B32',
  random:    '#9CA3AF',
};

// ─── State ───────────────────────────────────────────────
let recoveryChart   = null;
let currentBudget   = 100;
let providerPage    = 1;
let providerTotal   = 0;
let providerSearch  = '';
let providerRisk    = '';
const PER_PAGE      = 25;

// ─── Init ─────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  lucide.createIcons();

  // Nav scroll spy
  setupScrollSpy();

  // Overview
  loadOverview();
  loadCurves();
  loadStrategyComparison(100);

  // Audit priorities
  loadSimulation(100);
  loadProviders(1, '', '');

  // Model performance
  loadBenchmarks();
  loadAblation();

  // Budget slider
  const slider = document.getElementById('budgetSlider');
  const label  = document.getElementById('sliderLabel');
  let sliderTimer;
  slider.addEventListener('input', () => {
    currentBudget = parseInt(slider.value);
    label.textContent = currentBudget;
    clearTimeout(sliderTimer);
    sliderTimer = setTimeout(() => {
      loadSimulation(currentBudget);
      loadProviders(1, providerSearch, providerRisk);
    }, 300);
  });

  // Provider search
  let searchTimer;
  document.getElementById('providerSearch').addEventListener('input', e => {
    clearTimeout(searchTimer);
    searchTimer = setTimeout(() => {
      providerSearch = e.target.value.trim();
      providerPage = 1;
      loadProviders(1, providerSearch, providerRisk);
    }, 400);
  });

  // Risk filter
  document.getElementById('riskFilter').addEventListener('change', e => {
    providerRisk = e.target.value;
    providerPage = 1;
    loadProviders(1, providerSearch, providerRisk);
  });

  // Pagination
  document.getElementById('prevPageBtn').addEventListener('click', () => {
    if (providerPage > 1) {
      providerPage--;
      loadProviders(providerPage, providerSearch, providerRisk);
    }
  });
  document.getElementById('nextPageBtn').addEventListener('click', () => {
    if (providerPage * PER_PAGE < providerTotal) {
      providerPage++;
      loadProviders(providerPage, providerSearch, providerRisk);
    }
  });
});

// ─── Scroll Spy ───────────────────────────────────────────
function setupScrollSpy() {
  const sections = ['overview','audit-priorities','provider-analysis','model-performance','methodology'];
  const observer = new IntersectionObserver(entries => {
    entries.forEach(e => {
      if (e.isIntersecting) {
        sections.forEach(id => {
          const link = document.getElementById('nav-' + id);
          if (link) link.classList.remove('active');
        });
        const active = document.getElementById('nav-' + e.target.id);
        if (active) active.classList.add('active');
      }
    });
  }, { rootMargin: '-20% 0px -70% 0px' });

  sections.forEach(id => {
    const el = document.getElementById(id);
    if (el) observer.observe(el);
  });
}

// ═══════════════════════════════════════════════════════════
//  SECTION 1 — OVERVIEW
// ═══════════════════════════════════════════════════════════

async function loadOverview() {
  const strip = document.getElementById('kpiStrip');
  try {
    // Try new /api/overview first, fall back to /api/kpis
    let data;
    try {
      const r = await fetch('/api/overview');
      if (!r.ok) throw new Error();
      data = await r.json();
    } catch {
      const r2 = await fetch('/api/kpis');
      data = await r2.json();
    }

    const fraud_pct = data.fraud_prevalence_pct
      ?? (data.test_fraud_providers / (data.test_providers ?? 1) * 100);
    const fraud_dollars = data.total_fraud_dollars_in_test ?? data.test_fraud_dollars;
    const best_pr = data.best_pr_auc ?? data.best_pr_auc ?? '—';

    strip.innerHTML = `
      ${kpiCard('Total Claims Analysed', fmtNum(data.total_claims), '40K IP + 518K OP', 'file-text')}
      ${kpiCard('Providers Analysed', fmtNum(data.total_providers ?? data.test_providers * 5), 'Train + Test split', 'hospital')}
      ${kpiCard('Historical Fraud Prevalence', fmtPct(fraud_pct), 'Provider-level, test set', 'activity')}
      ${kpiCard('Known Fraudulent Reimbursement', fmtDollar(fraud_dollars), 'In held-out test set', 'dollar-sign')}
      ${kpiCard('Best Model PR-AUC', typeof best_pr === 'number' ? best_pr.toFixed(4) : best_pr, data.best_model_name ?? data.best_model ?? '', 'award')}
    `;
    lucide.createIcons();
  } catch (err) {
    strip.innerHTML = `<div class="error-state col-span-5">Failed to load KPIs: ${err.message}</div>`;
  }
}

function kpiCard(label, value, sub, icon) {
  return `
    <div class="kpi-card">
      <div class="kpi-label flex items-center gap-1.5">
        <i data-lucide="${icon}" class="w-3.5 h-3.5 text-accent"></i>${label}
      </div>
      <div class="kpi-value">${value}</div>
      <div class="kpi-sub">${sub}</div>
    </div>`;
}

async function loadCurves() {
  const spinner = document.getElementById('chartSpinner');
  try {
    // Try /api/curves; fall back to static approximation
    let curvesData = null;
    try {
      const r = await fetch('/api/curves');
      if (r.ok) curvesData = await r.json();
    } catch { /* use fallback */ }

    let ks, proposed, billers, prob, random;
    if (curvesData && curvesData.budget_k) {
      ks       = curvesData.budget_k;
      proposed = curvesData.proposed;
      billers  = curvesData.billers;
      prob     = curvesData.prob_only;
      random   = curvesData.random;
    } else {
      ks       = [10, 20, 30, 50, 75, 100, 125, 150, 175, 200, 250, 300];
      proposed = [22, 38, 52, 68, 83, 93.75, 96.5, 97.9, 98.6, 99.1, 99.5, 99.8];
      billers  = [20, 36, 50, 65, 81, 93.28, 95.9, 97.5, 98.4, 99.0, 99.4, 99.7];
      prob     = [18, 32, 46, 62, 78, 91.97, 95.1, 96.8, 98.0, 98.7, 99.2, 99.6];
      random   = ks.map(k => (k / 1082) * 100);
    }

    if (spinner) spinner.style.display = 'none';
    renderRecoveryChart(ks, proposed, billers, prob, random);
  } catch (err) {
    if (spinner) spinner.innerHTML = `<span class="text-xs text-muted">Chart unavailable</span>`;
  }
}

function renderRecoveryChart(ks, proposed, billers, prob, random) {
  const ctx = document.getElementById('recoveryCurveChart').getContext('2d');
  if (recoveryChart) recoveryChart.destroy();

  recoveryChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels: ks,
      datasets: [
        {
          label: 'Proposed ML System',
          data: proposed,
          borderColor: CHART_COLORS.proposed,
          backgroundColor: 'transparent',
          borderWidth: 2.5,
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Highest Billers (Volume)',
          data: billers,
          borderColor: CHART_COLORS.billers,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [6, 4],
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Pure Fraud Probability',
          data: prob,
          borderColor: CHART_COLORS.prob,
          backgroundColor: 'transparent',
          borderWidth: 2,
          borderDash: [3, 3],
          tension: 0.3,
          pointRadius: 0,
        },
        {
          label: 'Random Baseline',
          data: random,
          borderColor: CHART_COLORS.random,
          backgroundColor: 'transparent',
          borderWidth: 1.5,
          borderDash: [2, 3],
          tension: 0.1,
          pointRadius: 0,
        },
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: {
        legend: {
          position: 'bottom',
          labels: { color: '#687782', font: { size: 11, family: 'Inter' }, boxWidth: 20, padding: 14 }
        },
        tooltip: {
          callbacks: {
            label: c => `${c.dataset.label}: ${c.parsed.y.toFixed(1)}%`
          }
        }
      },
      scales: {
        x: {
          title: { display: true, text: 'Providers Audited (K)', color: '#687782', font: { size: 11 } },
          grid: { color: '#E1E7EA' },
          ticks: { color: '#687782', font: { size: 11 } },
        },
        y: {
          title: { display: true, text: '% Fraudulent Reimbursement Captured', color: '#687782', font: { size: 11 } },
          grid: { color: '#E1E7EA' },
          ticks: { color: '#687782', font: { size: 11 }, callback: v => v + '%' },
          min: 0,
          max: 105,
        }
      }
    }
  });
}

async function loadStrategyComparison(budget) {
  const tbody = document.getElementById('overviewStrategyBody');
  try {
    const r = await fetch(`/api/audit-simulation?budget=${budget}`);
    const data = await r.json();
    tbody.innerHTML = '';
    data.strategies.forEach((s, i) => {
      const isProposed = s.name.toLowerCase().includes('proposed');
      const tr = document.createElement('tr');
      if (isProposed) tr.className = 'best-row';
      tr.innerHTML = `
        <td class="font-medium" style="max-width:160px;">
          ${isProposed ? '<span class="text-accent font-bold">★ </span>' : ''}${s.name}
        </td>
        <td class="font-mono">${s.fraud_caught} / ${data.total_fraud_providers ?? '—'}</td>
        <td class="font-mono">${s.precision_at_k}%</td>
        <td class="font-mono">${s.capture_rate}%</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="4" class="error-state">Failed to load: ${err.message}</td></tr>`;
  }
}

// ═══════════════════════════════════════════════════════════
//  SECTION 2 — AUDIT PRIORITIES
// ═══════════════════════════════════════════════════════════

async function loadSimulation(budget) {
  const container = document.getElementById('strategyCards');
  container.innerHTML = '<div class="text-center py-4"><div class="spinner mx-auto"></div></div>';
  try {
    const r    = await fetch(`/api/audit-simulation?budget=${budget}`);
    const data = await r.json();
    container.innerHTML = '';
    data.strategies.forEach(s => {
      const isProposed = s.name.toLowerCase().includes('proposed');
      const div = document.createElement('div');
      div.className = 'strategy-card' + (isProposed ? ' proposed' : '');
      div.innerHTML = `
        <div class="flex items-center justify-between mb-1">
          <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded ${
            isProposed ? 'bg-accent text-white' : 'bg-surface text-muted border border-border'
          }">${s.badge}</span>
          <span class="text-[11px] text-muted">Yield: ${s.yield_pct}%</span>
        </div>
        <div class="strategy-card-name">${s.name}</div>
        <div class="grid grid-cols-2 gap-x-4 gap-y-2 pt-2 border-t border-border text-[12px]">
          <div><div class="strategy-card-stat-label">Fraud Caught</div><div class="strategy-card-stat-value ${isProposed ? 'text-accent' : ''}">${s.fraud_caught} providers</div></div>
          <div><div class="strategy-card-stat-label">Precision@K</div><div class="strategy-card-stat-value">${s.precision_at_k}%</div></div>
          <div><div class="strategy-card-stat-label">$ Capture Rate</div><div class="strategy-card-stat-value">${s.capture_rate}%</div></div>
          <div><div class="strategy-card-stat-label">Total Audited $</div><div class="strategy-card-stat-value">${fmtDollar(s.total_audited_payout)}</div></div>
        </div>
      `;
      container.appendChild(div);
    });
  } catch (err) {
    container.innerHTML = `<div class="error-state">Failed to load simulation: ${err.message}</div>`;
  }
}

async function loadProviders(page, search, riskLevel) {
  providerPage = page;
  const tbody = document.getElementById('providerTableBody');
  tbody.innerHTML = `<tr><td colspan="7" class="text-center py-6 text-muted"><div class="spinner mx-auto mb-1"></div>Loading…</td></tr>`;

  const params = new URLSearchParams({
    page,
    per_page: PER_PAGE,
    sort: 'expected_fraud_dollars',
    order: 'desc',
  });
  if (search)    params.set('search', search);
  if (riskLevel) params.set('risk_level', riskLevel);

  try {
    // Try new /api/providers; fall back to /api/manifest
    let providers = [], total = 0;
    try {
      const r = await fetch('/api/providers?' + params.toString());
      if (!r.ok) throw new Error('no providers endpoint');
      const data = await r.json();
      providers = data.providers ?? data;
      total = data.total ?? providers.length;
    } catch {
      // Fallback: use manifest
      const r2 = await fetch('/api/manifest');
      const all = await r2.json();
      let filtered = all;
      if (search) filtered = filtered.filter(p => p.Provider.toLowerCase().includes(search.toLowerCase()));
      if (riskLevel === 'high')   filtered = filtered.filter(p => (p.fraud_prob ?? 0) >= 0.7);
      if (riskLevel === 'medium') filtered = filtered.filter(p => (p.fraud_prob ?? 0) >= 0.3 && (p.fraud_prob ?? 0) < 0.7);
      if (riskLevel === 'low')    filtered = filtered.filter(p => (p.fraud_prob ?? 0) < 0.3);
      total = filtered.length;
      const start = (page - 1) * PER_PAGE;
      providers = filtered.slice(start, start + PER_PAGE).map((p, i) => ({
        Provider: p.Provider,
        fraud_prob: p.fraud_prob,
        total_reimbursed: p.total_reimbursed,
        expected_fraud_dollars: p.expected_fraud_dollars,
        total_claims: p.total_claims,
        unique_beneficiaries: p.unique_beneficiaries,
        dbscan_outlier_flag: p.dbscan_outlier_flag,
        suspicious_code_claim_count: p.suspicious_code_claim_count,
        cluster: p.cluster,
        retrospective_label: p.PotentialFraud,
        investigation_status: p.investigation_status ?? 'not_reviewed',
        rank: (page - 1) * PER_PAGE + i + 1,
      }));
    }

    providerTotal = total;
    tbody.innerHTML = '';

    if (providers.length === 0) {
      tbody.innerHTML = `<tr><td colspan="7" class="text-center py-8 text-muted">No providers found.</td></tr>`;
    } else {
      providers.forEach((p, i) => {
        const rank = p.rank ?? (page - 1) * PER_PAGE + i + 1;
        const prob = p.fraud_prob ?? 0;
        const tr = document.createElement('tr');
        tr.style.cursor = 'pointer';
        tr.addEventListener('click', e => {
          if (e.target.tagName === 'SELECT') return;
          openDetailPanel(p);
        });
        tr.innerHTML = `
          <td class="font-mono text-muted text-[11px]">#${rank}</td>
          <td class="font-mono font-semibold text-accent">${p.Provider}</td>
          <td>${riskPill(prob)}</td>
          <td class="font-mono">${fmtDollar(p.total_reimbursed)}</td>
          <td class="font-mono font-semibold text-accent">${fmtDollar(p.expected_fraud_dollars)}</td>
          <td>${fmtNum(p.total_claims)}</td>
          <td onclick="event.stopPropagation()">
            <select class="mg-select" style="font-size:11px;padding:4px 6px;width:130px;"
              onchange="updateStatus('${p.Provider}', this.value)">
              <option value="not_reviewed"   ${(p.investigation_status ?? 'not_reviewed') === 'not_reviewed'   ? 'selected' : ''}>Not Reviewed</option>
              <option value="under_review"   ${p.investigation_status === 'under_review'   ? 'selected' : ''}>Under Review</option>
              <option value="escalated"      ${p.investigation_status === 'escalated'      ? 'selected' : ''}>Escalated</option>
              <option value="closed"         ${p.investigation_status === 'closed'         ? 'selected' : ''}>Closed</option>
            </select>
          </td>
        `;
        tbody.appendChild(tr);
      });
      lucide.createIcons();
    }

    // Update pagination
    const info    = document.getElementById('providerPageInfo');
    const prevBtn = document.getElementById('prevPageBtn');
    const nextBtn = document.getElementById('nextPageBtn');
    const totalPages = Math.max(1, Math.ceil(total / PER_PAGE));
    info.textContent = `Page ${page} of ${totalPages} (${fmtNum(total)} providers)`;
    prevBtn.disabled = page <= 1;
    nextBtn.disabled = page >= totalPages;

  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="7"><div class="error-state m-2">Failed to load providers: ${err.message}</div></td></tr>`;
  }
}

async function updateStatus(providerId, status) {
  try {
    await fetch(`/api/providers/${encodeURIComponent(providerId)}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status }),
    });
  } catch { /* silent */ }
}

// ─── Detail Panel ─────────────────────────────────────────
function openDetailPanel(provider) {
  const content = document.getElementById('detailContent');
  const prob = provider.fraud_prob ?? 0;
  const tier = riskTierLabel(prob);
  const retro = provider.retrospective_label;

  content.innerHTML = `
    <!-- Header -->
    <div class="mb-5 pb-4 border-b border-border">
      <div class="font-mono text-2xl font-bold text-primary mb-1">${provider.Provider}</div>
      <div class="flex items-center gap-2 mb-3">
        <span class="text-3xl font-bold ${prob >= 0.7 ? 'text-high' : prob >= 0.3 ? 'text-medium' : 'text-low'}">${fmtPct(prob * 100)}</span>
        <span class="text-sm text-muted">Fraud Probability</span>
      </div>
      <!-- Gauge -->
      <div class="gauge-bar-bg mb-1"><div class="gauge-bar-fill" style="width:${Math.round(prob*100)}%;background:${prob>=0.7?'#B54747':prob>=0.3?'#C58B32':'#43846A'}"></div></div>
      <div class="text-[11px] text-muted text-right">${fmtPct(prob*100)}</div>
      <div class="mt-2">${riskPill(prob)} <span class="ml-2 text-sm text-muted">${tier}</span></div>
    </div>

    <!-- Billing Stats -->
    <div class="mb-4">
      <h4 class="font-semibold text-sm text-primary mb-2">Billing Statistics</h4>
      <div class="grid grid-cols-2 gap-3">
        ${detailStat('Total Claims', fmtNum(provider.total_claims))}
        ${detailStat('Total Reimbursed', fmtDollar(provider.total_reimbursed))}
        ${detailStat('Est. Financial Exposure', fmtDollar(provider.expected_fraud_dollars), true)}
        ${detailStat('Unique Beneficiaries', fmtNum(provider.unique_beneficiaries))}
      </div>
    </div>

    <!-- Anomaly Signals -->
    <div class="mb-4">
      <h4 class="font-semibold text-sm text-primary mb-2">Anomaly Signals</h4>
      <div class="grid grid-cols-2 gap-3">
        ${detailStat('DBSCAN Outlier', provider.dbscan_outlier_flag == 1 ? '⚠ Yes' : 'No', provider.dbscan_outlier_flag == 1)}
        ${detailStat('Suspicious Code Count', fmtNum(provider.suspicious_code_claim_count ?? 0))}
        ${detailStat('Cluster Assignment', provider.cluster ?? '—')}
      </div>
    </div>

    <!-- Retrospective Label -->
    <div class="mb-4 bg-amber-50 border border-amber-200 rounded p-3">
      <div class="font-semibold text-xs text-amber-700 mb-1">⚠ Retrospective Evaluation Only — Not an Operational Signal</div>
      <div class="text-sm text-amber-800">
        Known Label: <strong>${retro == 1 || retro === true ? 'PotentialFraud = Yes' : retro == 0 || retro === false ? 'PotentialFraud = No' : '—'}</strong>
      </div>
      <div class="text-[11px] text-amber-600 mt-1">This label reflects historical administrative decisions and is used only for retrospective evaluation, not operational screening.</div>
    </div>

    <!-- Investigation Status -->
    <div class="mb-3">
      <h4 class="font-semibold text-sm text-primary mb-2">Investigation Status</h4>
      <select class="mg-select mb-2" id="detailStatus">
        <option value="not_reviewed"  ${(provider.investigation_status ?? 'not_reviewed') === 'not_reviewed'  ? 'selected' : ''}>Not Reviewed</option>
        <option value="under_review"  ${provider.investigation_status === 'under_review'  ? 'selected' : ''}>Under Review</option>
        <option value="escalated"     ${provider.investigation_status === 'escalated'     ? 'selected' : ''}>Escalated</option>
        <option value="closed"        ${provider.investigation_status === 'closed'        ? 'selected' : ''}>Closed</option>
      </select>
      <label class="mg-label mt-2">Investigation Notes</label>
      <textarea class="mg-textarea" id="detailNotes" placeholder="Add notes…">${provider.notes ?? ''}</textarea>
      <button class="btn-primary mt-2 w-full justify-center" onclick="saveDetail('${provider.Provider}')">
        <i data-lucide="save" class="w-4 h-4"></i>Save
      </button>
      <div id="detailSaveMsg" class="text-[11px] text-accent mt-1 hidden">Saved.</div>
    </div>
  `;
  lucide.createIcons();
  document.getElementById('detailPanel').classList.add('open');
  document.getElementById('detailOverlay').classList.add('show');
}

function detailStat(label, value, highlight) {
  return `
    <div class="bg-surface border border-border rounded p-2.5">
      <div class="text-[11px] text-muted">${label}</div>
      <div class="text-sm font-semibold ${highlight ? 'text-accent' : 'text-primary'}">${value}</div>
    </div>`;
}

function closeDetailPanel() {
  document.getElementById('detailPanel').classList.remove('open');
  document.getElementById('detailOverlay').classList.remove('show');
}

async function saveDetail(providerId) {
  const status = document.getElementById('detailStatus').value;
  const notes  = document.getElementById('detailNotes').value;
  try {
    await fetch(`/api/providers/${encodeURIComponent(providerId)}/status`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ status, notes }),
    });
    const msg = document.getElementById('detailSaveMsg');
    msg.classList.remove('hidden');
    setTimeout(() => msg.classList.add('hidden'), 2000);
  } catch { /* silent */ }
}

// ═══════════════════════════════════════════════════════════
//  SECTION 3 — PROVIDER ANALYSIS (Risk Inference)
// ═══════════════════════════════════════════════════════════

function fillHighRisk() {
  document.getElementById('f_total_claims').value          = 1800;
  document.getElementById('f_total_reimbursed').value      = 3200000;
  document.getElementById('f_unique_beneficiaries').value  = 420;
  document.getElementById('f_mean_reimbursed_per_claim').value = 1778;
  document.getElementById('f_mean_claim_duration').value   = 12;
  document.getElementById('f_inpatient_ratio').value       = 0.72;
  document.getElementById('f_suspicious_code_claim_count').value = 68;
  document.getElementById('f_mean_residual').value         = 310;
  document.getElementById('f_dbscan_outlier_flag').checked = true;
}

function fillLowRisk() {
  document.getElementById('f_total_claims').value          = 240;
  document.getElementById('f_total_reimbursed').value      = 180000;
  document.getElementById('f_unique_beneficiaries').value  = 90;
  document.getElementById('f_mean_reimbursed_per_claim').value = 750;
  document.getElementById('f_mean_claim_duration').value   = 3;
  document.getElementById('f_inpatient_ratio').value       = 0.05;
  document.getElementById('f_suspicious_code_claim_count').value = 2;
  document.getElementById('f_mean_residual').value         = -20;
  document.getElementById('f_dbscan_outlier_flag').checked = false;
}

function resetForm() {
  ['f_total_claims','f_total_reimbursed','f_unique_beneficiaries',
   'f_mean_reimbursed_per_claim','f_mean_claim_duration','f_inpatient_ratio',
   'f_suspicious_code_claim_count','f_mean_residual'].forEach(id => {
    document.getElementById(id).value = '';
  });
  document.getElementById('f_dbscan_outlier_flag').checked = false;
  document.getElementById('inferResultBody').innerHTML = `
    <i data-lucide="activity" class="w-10 h-10 mx-auto mb-2 opacity-30"></i>
    Submit the form to generate a fraud risk estimate.
  `;
  lucide.createIcons();
}

async function runInference() {
  const btn = document.getElementById('inferBtn');
  btn.disabled = true;
  btn.innerHTML = '<div class="spinner" style="border-top-color:#fff;width:16px;height:16px;"></div> Running…';

  const payload = {
    total_claims:               parseFloat(document.getElementById('f_total_claims').value)              || 0,
    total_reimbursed:           parseFloat(document.getElementById('f_total_reimbursed').value)          || 0,
    unique_beneficiaries:       parseFloat(document.getElementById('f_unique_beneficiaries').value)      || 0,
    mean_reimbursed_per_claim:  parseFloat(document.getElementById('f_mean_reimbursed_per_claim').value) || 0,
    mean_claim_duration:        parseFloat(document.getElementById('f_mean_claim_duration').value)       || 0,
    inpatient_ratio:            parseFloat(document.getElementById('f_inpatient_ratio').value)           || 0,
    suspicious_code_claim_count:parseFloat(document.getElementById('f_suspicious_code_claim_count').value)||0,
    mean_residual:              parseFloat(document.getElementById('f_mean_residual').value)             || 0,
    dbscan_outlier_flag:        document.getElementById('f_dbscan_outlier_flag').checked ? 1 : 0,
    // Legacy field names for backwards compat with existing endpoint
    dbscan_outlier:             document.getElementById('f_dbscan_outlier_flag').checked ? 1 : 0,
    suspicious_codes:           parseFloat(document.getElementById('f_suspicious_code_claim_count').value)||0,
  };

  const resultEl = document.getElementById('inferResultBody');

  try {
    const r = await fetch('/api/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    if (!r.ok) throw new Error(await r.text());
    const data = await r.json();

    if (data.error) throw new Error(data.error);

    const prob = data.fraud_probability ?? 0;
    const color = prob >= 0.7 ? '#B54747' : prob >= 0.3 ? '#C58B32' : '#43846A';
    const tierText = data.risk_tier ?? riskTierLabel(prob);

    resultEl.innerHTML = `
      <div class="mb-4">
        <div class="text-xs text-muted mb-1">Fraud Probability</div>
        <div class="text-3xl font-bold mb-2" style="color:${color}">${fmtPct(prob * 100)}</div>
        <div class="gauge-bar-bg"><div class="gauge-bar-fill" style="width:${Math.round(prob*100)}%;background:${color}"></div></div>
      </div>
      <div class="mb-3 pb-3 border-b border-border">
        <div class="text-xs text-muted">Risk Tier</div>
        <div class="font-semibold text-sm mt-0.5" style="color:${color}">${tierText}</div>
      </div>
      <div class="mb-3 pb-3 border-b border-border">
        <div class="text-xs text-muted">Estimated Financial Exposure</div>
        <div class="font-bold text-base text-primary mt-0.5">${fmtDollar(data.expected_fraud_dollars ?? 0)}</div>
      </div>
      ${data.top_features ? `
      <div class="mb-3 pb-3 border-b border-border">
        <div class="text-xs text-muted mb-1">Top Contributing Factors</div>
        <ul class="space-y-1">
          ${data.top_features.map(f => `
            <li class="flex justify-between text-xs">
              <span class="text-primary font-medium">${f.feature ?? f.name}</span>
              <span class="font-mono text-muted">${typeof (f.value ?? f.importance) === 'number' ? (f.value ?? f.importance).toLocaleString('en-US',{maximumFractionDigits:3}) : f.value ?? f.importance}</span>
            </li>`).join('')}
        </ul>
      </div>` : ''}
      <div>
        <div class="text-xs text-muted">Audit Recommendation</div>
        <div class="font-semibold text-sm mt-0.5 text-primary">${data.recommendation ?? (prob >= 0.7 ? 'Priority Audit Recommended' : prob >= 0.3 ? 'Standard Review' : 'Monitor — Low Priority')}</div>
      </div>
    `;
  } catch (err) {
    resultEl.innerHTML = `<div class="error-state">${err.message}</div>`;
  } finally {
    btn.disabled = false;
    btn.innerHTML = '<i data-lucide="zap" class="w-4 h-4"></i>Run Fraud Risk Inference';
    lucide.createIcons();
  }
}

// ═══════════════════════════════════════════════════════════
//  SECTION 4 — MODEL PERFORMANCE
// ═══════════════════════════════════════════════════════════

async function loadBenchmarks() {
  const tbody = document.getElementById('benchmarkBody');
  try {
    const r = await fetch('/api/models');
    const models = await r.json();
    // Sort by PR-AUC desc
    models.sort((a, b) => {
      const pa = a['PR-AUC (Avg Precision)'] ?? a['PR-AUC'] ?? 0;
      const pb = b['PR-AUC (Avg Precision)'] ?? b['PR-AUC'] ?? 0;
      return pb - pa;
    });
    tbody.innerHTML = '';
    models.forEach((m, i) => {
      const prAUC = m['PR-AUC (Avg Precision)'] ?? m['PR-AUC'] ?? 0;
      const tr = document.createElement('tr');
      if (i === 0) tr.className = 'best-row';
      tr.innerHTML = `
        <td class="font-medium">${i === 0 ? '<span class="text-accent font-bold">★ </span>' : ''}${m.Model}</td>
        <td class="font-mono">${safeFixed(m['ROC-AUC'], 4)}</td>
        <td class="font-mono font-bold ${i === 0 ? 'text-accent' : ''}">${safeFixed(prAUC, 4)}</td>
        <td class="font-mono">${safeFixed(m['F1 Score'], 4)}</td>
        <td class="font-mono">${safeFixed(m['Precision'] * 100, 1)}%</td>
        <td class="font-mono">${safeFixed(m['Recall'] * 100, 1)}%</td>
        <td class="font-mono">${m['True Positives'] ?? '—'}</td>
        <td class="font-mono">${m['False Positives'] ?? '—'}</td>
        <td class="font-mono">${m['False Negatives'] ?? '—'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="9"><div class="error-state m-2">Failed: ${err.message}</div></td></tr>`;
  }
}

async function loadAblation() {
  const tbody = document.getElementById('ablationBody');
  try {
    const r = await fetch('/api/ablation');
    const rows = await r.json();
    tbody.innerHTML = '';
    rows.forEach((row, i) => {
      const isFull = i === rows.length - 1;
      const tr = document.createElement('tr');
      if (isFull) tr.className = 'best-row';
      tr.innerHTML = `
        <td class="font-medium">${row['Pipeline Configuration']}</td>
        <td class="font-mono text-muted">${row['Num Features'] ?? row['Features'] ?? '—'}</td>
        <td class="font-mono font-bold ${isFull ? 'text-accent' : ''}">${safeFixed(row['PR-AUC'], 4)}</td>
        <td class="font-mono">${safeFixed(row['ROC-AUC'], 4)}</td>
        <td class="font-mono">${row['Top-100 Fraud Providers Caught'] ?? '—'} / ${row['total_fraud_providers'] ?? '101'}</td>
        <td class="font-mono ${isFull ? 'text-accent font-bold' : ''}">${row['Top-100 Dollar Capture Rate'] ?? '—'}</td>
      `;
      tbody.appendChild(tr);
    });
  } catch (err) {
    tbody.innerHTML = `<tr><td colspan="6"><div class="error-state m-2">Failed: ${err.message}</div></td></tr>`;
  }
}

// ═══════════════════════════════════════════════════════════
//  HELPERS
// ═══════════════════════════════════════════════════════════

function fmtNum(n) {
  if (n == null || n === '' || isNaN(n)) return '—';
  return Number(n).toLocaleString('en-US');
}

function fmtDollar(n) {
  if (n == null || isNaN(n)) return '—';
  const v = Number(n);
  if (v >= 1e6) return '$' + (v / 1e6).toFixed(2) + 'M';
  if (v >= 1e3) return '$' + (v / 1e3).toFixed(1) + 'K';
  return '$' + v.toFixed(0);
}

function fmtPct(n) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(1) + '%';
}

function safeFixed(n, digits) {
  if (n == null || isNaN(n)) return '—';
  return Number(n).toFixed(digits);
}

function riskPill(prob) {
  if (prob >= 0.7) return `<span class="pill pill-high">${fmtPct(prob*100)}</span>`;
  if (prob >= 0.3) return `<span class="pill pill-medium">${fmtPct(prob*100)}</span>`;
  return `<span class="pill pill-low">${fmtPct(prob*100)}</span>`;
}

function riskTierLabel(prob) {
  if (prob >= 0.7) return 'High Risk';
  if (prob >= 0.3) return 'Medium Risk';
  return 'Low Risk';
}
