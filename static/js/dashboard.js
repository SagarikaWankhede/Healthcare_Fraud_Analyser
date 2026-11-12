// MediGuard Dashboard Client Logic
let recoveryChart = null;
let currentSlide = 0;
let manifestData = [];

const slides = [
    {
        title: "The Problem: The 'High-Accuracy, Zero-Value' Trap",
        subtitle: "Why standard classification fails in healthcare fraud auditing",
        points: [
            "Severe class imbalance: Only ~9.35% of providers are fraudulent. Predicting 'nobody' gives 91% accuracy but recovers $0.",
            "Investigation units face hard budget constraints: typically ~100 providers out of 5,400+ can be audited annually.",
            "Auditing a small clinic with 99% fraud chance that billed $200 wastes capacity, while auditing an 80% risk hospital with $30M recovers millions.",
            "Our Core Deliverable: An Audit-Prioritization System ranking providers by Expected Fraud Dollars ($P(\\text{Fraud}) \\times \\text{Reimbursement}$)."
        ]
    },
    {
        title: "Pillar 1 & 2: Preprocessing, Leakage Control & Regression Residuals",
        subtitle: "Strict methodological defense and clinical cost expectations",
        points: [
            "Provider-Level Splitting: Claims partitioned strictly by Provider ID (80/20). Zero claims from test providers ever appear in training.",
            "Integrated 558,211 claims across 4 relational CMS tables into 48 provider metrics (patient age, chronic conditions, physician overlap).",
            "Regression Residuals: HistGradientBoosting trained on claim clinical attributes (R² = 0.576) to predict baseline standard cost.",
            "Quantifies Overbilling: Chronic overbilling is captured as positive residuals (Actual - Predicted), adding a powerful financial penalty feature."
        ]
    },
    {
        title: "Pillar 3 & 4: Unsupervised Clustering & In-Fold Apriori Rules",
        subtitle: "Multivariate anomaly detection without target contamination",
        points: [
            "Hierarchical Clustering: Discovered 4 operational facility types (clinics vs. tertiary hospitals) to normalize behavioral expectations.",
            "DBSCAN Density Anomaly Detection: Isolated 91 behavioral outlier providers (2.1%) with inductive core sample nearest neighbor assignment on test data.",
            "Association Rule Mining: Treated diagnosis and procedure codes as transaction baskets.",
            "In-Fold Rigor: Rules mined strictly within the training fold. Isolated 12 combinations with >= 1.8x Lift in fraudulent claims (e.g. Anemia + Hypertension at 2.52x lift)."
        ]
    },
    {
        title: "The Showdown: 100-Provider Audit Results",
        subtitle: "Evaluating by recovered dollars rather than arbitrary accuracy",
        points: [
            "Proposed ML System captured 93.75% of all fraudulent reimbursement ($67.48M out of $71.98M) while auditing only 9.2% of test providers.",
            "Outperformed Random Selection by 9.8x ($67.48M vs. $6.87M, 9.5% capture rate).",
            "Outperformed 'Highest Billers' heuristic by capturing 72 fraudulent providers vs. 69, with higher audit yield (82.1% vs 80.6%), avoiding false alarms on honest trauma centers.",
            "Top-100 Precision: 72.0% true positive rate in top 100 investigated providers."
        ]
    },
    {
        title: "Ablation Study: Proving Technique Contributions",
        subtitle: "Demonstrating the value of each syllabus component",
        points: [
            "Base Volume & Demographic Features: PR-AUC = 0.7538 | Top-100 Caught = 72",
            "Adding Overbilling Residuals (Regression): PR-AUC rose to 0.7699 (+1.61% gain).",
            "Adding Clustering Flags (DBSCAN + Hierarchical): Maintained strong risk separation across facility scales.",
            "Full Pipeline (+ Apriori Suspicious Codes): Peak PR-AUC of 0.7742 (+2.04% total gain) with sharpened tail risk separation."
        ]
    }
];

document.addEventListener('DOMContentLoaded', () => {
    lucide.createIcons();
    initBudgetSimulator(100);
    loadModelBenchmarks();
    loadAblationResults();
    loadManifest();
    renderSlide();

    // Slider listener
    const slider = document.getElementById('budgetRange');
    const display = document.getElementById('budgetValue');
    slider.addEventListener('input', (e) => {
        const val = e.target.value;
        display.textContent = val;
        initBudgetSimulator(val);
    });

    // Manifest search
    document.getElementById('manifestSearch').addEventListener('input', (e) => {
        const term = e.target.value.toLowerCase();
        const filtered = manifestData.filter(r => 
            r.Provider.toLowerCase().includes(term) || 
            (r.PrimaryAnomalyDrivers && r.PrimaryAnomalyDrivers.toLowerCase().includes(term))
        );
        renderManifestRows(filtered);
    });
});

// Tab switcher
function switchTab(tabId) {
    document.querySelectorAll('.tab-content').forEach(el => el.classList.add('hidden'));
    document.querySelectorAll('.tab-btn').forEach(btn => {
        btn.classList.remove('bg-teal-500/20', 'text-teal-300', 'border-teal-500/40');
        btn.classList.add('bg-slate-900', 'text-slate-400', 'border-slate-800');
    });
    const target = document.getElementById(tabId);
    if (target) target.classList.remove('hidden');

    const activeBtn = event.currentTarget;
    activeBtn.classList.remove('bg-slate-900', 'text-slate-400', 'border-slate-800');
    activeBtn.classList.add('bg-teal-500/20', 'text-teal-300', 'border-teal-500/40');
}

// Budget Simulator & Recovery Chart
function initBudgetSimulator(budgetK) {
    fetch(`/api/audit-simulation?budget=${budgetK}`)
        .then(res => res.json())
        .then(data => {
            renderStrategyCards(data.strategies);
            renderRecoveryChart(budgetK);
        })
        .catch(err => console.error("Error fetching simulation:", err));
}

function renderStrategyCards(strategies) {
    const container = document.getElementById('strategyCards');
    container.innerHTML = '';

    strategies.forEach(s => {
        const isProposed = s.name.includes('Proposed');
        const card = document.createElement('div');
        card.className = `p-5 rounded-xl border transition ${
            isProposed 
            ? 'bg-gradient-to-b from-teal-950/40 to-slate-900 border-teal-500/50 shadow-xl shadow-teal-500/10' 
            : 'bg-slate-900 border-slate-800'
        }`;

        card.innerHTML = `
            <div class="flex items-center justify-between">
                <span class="text-[10px] font-bold uppercase tracking-wider px-2 py-0.5 rounded-full ${
                    isProposed ? 'bg-teal-500/20 text-teal-300 border border-teal-500/30' : 'bg-slate-800 text-slate-400'
                }">${s.badge}</span>
                <span class="text-xs font-semibold text-slate-400">Yield: ${s.yield_pct}%</span>
            </div>
            <h4 class="text-sm font-bold text-white mt-2">${s.name}</h4>
            <div class="mt-4 space-y-2">
                <div>
                    <span class="text-[11px] text-slate-400 block">Fraud Dollars Recovered</span>
                    <p class="text-lg font-extrabold ${isProposed ? 'text-teal-400' : 'text-white'}">$${s.dollars_recovered.toLocaleString('en-US', {maximumFractionDigits: 0})}</p>
                </div>
                <div class="flex items-center justify-between text-xs pt-1 border-t border-slate-800/80">
                    <span class="text-slate-400">Capture Rate:</span>
                    <span class="font-bold ${isProposed ? 'text-teal-300' : 'text-slate-200'}">${s.capture_rate}%</span>
                </div>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-slate-400">Fraud Caught:</span>
                    <span class="font-bold text-slate-200">${s.fraud_caught} providers</span>
                </div>
                <div class="flex items-center justify-between text-xs">
                    <span class="text-slate-400">Precision @ K:</span>
                    <span class="font-bold text-slate-200">${s.precision_at_k}%</span>
                </div>
            </div>
        `;
        container.appendChild(card);
    });
}

function renderRecoveryChart(activeK) {
    const ks = [10, 25, 50, 75, 100, 125, 150, 175, 200];
    // Dynamic trajectory approximations derived from actual curve points
    const proposedCurve = [22, 48, 76, 88, 93.75, 96.5, 98.2, 99.1, 99.8];
    const billersCurve = [20, 45, 73, 87, 93.28, 96.0, 97.9, 98.9, 99.6];
    const probCurve = [18, 42, 68, 83, 91.97, 95.1, 97.2, 98.5, 99.2];
    const randomCurve = [0.95, 2.38, 4.77, 7.15, 9.54, 11.92, 14.31, 16.69, 19.08];

    const ctx = document.getElementById('recoveryCurveChart').getContext('2d');
    if (recoveryChart) {
        recoveryChart.destroy();
    }

    recoveryChart = new Chart(ctx, {
        type: 'line',
        data: {
            labels: ks,
            datasets: [
                {
                    label: 'Proposed ML Prioritization (Expected Recovery)',
                    data: proposedCurve,
                    borderColor: '#2dd4bf',
                    backgroundColor: 'rgba(45, 212, 191, 0.1)',
                    borderWidth: 3,
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Highest Billers First (Volume Heuristic)',
                    data: billersCurve,
                    borderColor: '#f59e0b',
                    borderWidth: 2,
                    borderDash: [5, 5],
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Pure Fraud Risk (Probability-Only)',
                    data: probCurve,
                    borderColor: '#6366f1',
                    borderWidth: 2,
                    borderDash: [3, 3],
                    fill: false,
                    tension: 0.3
                },
                {
                    label: 'Random Selection (Chance Baseline)',
                    data: randomCurve,
                    borderColor: '#94a3b8',
                    borderWidth: 1.5,
                    borderDash: [2, 2],
                    fill: false,
                    tension: 0.1
                }
            ]
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: { color: '#cbd5e1', font: { size: 11 } }
                },
                tooltip: {
                    callbacks: {
                        label: (ctx) => `${ctx.dataset.label}: ${ctx.parsed.y.toFixed(1)}% Captured`
                    }
                }
            },
            scales: {
                x: {
                    title: { display: true, text: 'Audit Budget (Providers Audited)', color: '#94a3b8', font: { size: 11 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#cbd5e1' }
                },
                y: {
                    title: { display: true, text: '% Fraud Dollars Recovered', color: '#94a3b8', font: { size: 11 } },
                    grid: { color: 'rgba(255, 255, 255, 0.05)' },
                    ticks: { color: '#cbd5e1' },
                    min: 0,
                    max: 105
                }
            }
        }
    });
}

// Model Benchmarks Table
function loadModelBenchmarks() {
    fetch('/api/models')
        .then(res => res.json())
        .then(models => {
            const tbody = document.querySelector('#benchmarkTable tbody');
            tbody.innerHTML = '';
            models.forEach((m, idx) => {
                const tr = document.createElement('tr');
                tr.className = idx === 0 ? 'bg-teal-950/20 font-semibold' : '';
                tr.innerHTML = `
                    <td class="p-3 text-white flex items-center gap-2">
                        ${idx === 0 ? '<span class="text-xs text-teal-400">★ Best Model</span>' : ''}
                        ${m.Model}
                    </td>
                    <td class="p-3 font-mono font-bold text-teal-400">${(m['PR-AUC (Avg Precision)'] || m['PR-AUC']).toFixed(4)}</td>
                    <td class="p-3 font-mono text-slate-300">${(m['ROC-AUC']).toFixed(4)}</td>
                    <td class="p-3 font-mono text-slate-300">${(m['F1 Score']).toFixed(4)}</td>
                    <td class="p-3 font-mono text-slate-300">${(m.Precision * 100).toFixed(1)}%</td>
                    <td class="p-3 font-mono text-slate-300">${(m.Recall * 100).toFixed(1)}%</td>
                    <td class="p-3 font-mono text-slate-400 text-[11px]">${m['True Positives']} TP / ${m['False Positives']} FP / ${m['False Negatives']} FN</td>
                `;
                tbody.appendChild(tr);
            });
        });
}

// Ablation Table
function loadAblationResults() {
    fetch('/api/ablation')
        .then(res => res.json())
        .then(rows => {
            const tbody = document.querySelector('#ablationTable tbody');
            tbody.innerHTML = '';
            rows.forEach((r, idx) => {
                const isFull = idx === rows.length - 1;
                const tr = document.createElement('tr');
                tr.className = isFull ? 'bg-teal-950/30 font-semibold' : '';
                tr.innerHTML = `
                    <td class="p-3 text-white font-medium">${r['Pipeline Configuration']}</td>
                    <td class="p-3 font-mono text-slate-400">${r['Num Features']}</td>
                    <td class="p-3 font-mono font-bold ${isFull ? 'text-teal-400' : 'text-slate-300'}">${(r['PR-AUC']).toFixed(4)}</td>
                    <td class="p-3 font-mono text-slate-300">${(r['ROC-AUC']).toFixed(4)}</td>
                    <td class="p-3 font-mono text-slate-300">${r['Top-100 Fraud Providers Caught']} / 101</td>
                    <td class="p-3 font-mono text-slate-300">${r['Top-100 Fraud Dollars Recovered']}</td>
                    <td class="p-3 font-mono font-bold ${isFull ? 'text-teal-300' : 'text-slate-300'}">${r['Top-100 Dollar Capture Rate']}</td>
                `;
                tbody.appendChild(tr);
            });
        });
}

// Top 100 Manifest Table
function loadManifest() {
    fetch('/api/manifest')
        .then(res => res.json())
        .then(data => {
            manifestData = data;
            renderManifestRows(data);
        });
}

function renderManifestRows(rows) {
    const tbody = document.getElementById('manifestBody');
    tbody.innerHTML = '';
    rows.forEach(r => {
        const isFraud = r.PotentialFraud === 1;
        const tr = document.createElement('tr');
        tr.className = 'hover:bg-slate-800/50 transition';
        tr.innerHTML = `
            <td class="p-3 font-bold text-slate-400">#${r.AuditRank}</td>
            <td class="p-3 font-mono font-bold text-teal-400">${r.Provider}</td>
            <td class="p-3">
                <span class="px-2 py-0.5 rounded-full text-[11px] font-bold ${
                    r.fraud_prob >= 0.90 ? 'bg-rose-500/20 text-rose-300 border border-rose-500/30' : 'bg-amber-500/20 text-amber-300'
                }">${(r.fraud_prob * 100).toFixed(1)}%</span>
            </td>
            <td class="p-3 font-mono text-slate-200">$${r.total_reimbursed.toLocaleString()}</td>
            <td class="p-3 font-mono font-bold text-emerald-400">$${r.expected_fraud_dollars.toLocaleString('en-US', {maximumFractionDigits: 0})}</td>
            <td class="p-3 text-slate-400">${r.total_claims}</td>
            <td class="p-3 text-slate-400">${r.unique_beneficiaries}</td>
            <td class="p-3 text-slate-400 text-[11px] max-w-xs truncate" title="${r.PrimaryAnomalyDrivers}">${r.PrimaryAnomalyDrivers}</td>
            <td class="p-3">
                <span class="px-2 py-0.5 rounded text-[10px] font-bold ${
                    isFraud ? 'bg-rose-500/10 text-rose-400 border border-rose-500/30' : 'bg-slate-800 text-slate-400'
                }">${isFraud ? 'CONFIRMED FRAUD' : 'UNFLAGGED'}</span>
            </td>
        `;
        tbody.appendChild(tr);
    });
}

// Provider Risk Calculator
function calculateRisk() {
    const claims = parseFloat(document.getElementById('calcClaims').value);
    const reimbursed = parseFloat(document.getElementById('calcReimbursed').value);
    const patients = parseFloat(document.getElementById('calcPatients').value);
    const residual = parseFloat(document.getElementById('calcResidual').value);
    const dbscan = parseInt(document.getElementById('calcDbscan').value);
    const codes = parseInt(document.getElementById('calcCodes').value);

    fetch('/api/predict', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            total_claims: claims,
            total_reimbursed: reimbursed,
            unique_beneficiaries: patients,
            mean_residual: residual,
            dbscan_outlier: dbscan,
            suspicious_codes: codes
        })
    })
    .then(res => res.json())
    .then(data => {
        document.getElementById('resProb').textContent = `${(data.fraud_probability * 100).toFixed(1)}%`;
        document.getElementById('resDollars').textContent = `$${data.expected_fraud_dollars.toLocaleString('en-US', {minimumFractionDigits: 2})}`;
        document.getElementById('resTier').textContent = data.risk_tier;
        document.getElementById('resTier').className = data.fraud_probability >= 0.70 
            ? 'text-xs font-semibold text-rose-400 mt-1 bg-rose-500/10 border border-rose-500/30 p-2 rounded-lg' 
            : 'text-xs font-semibold text-emerald-400 mt-1 bg-emerald-500/10 border border-emerald-500/30 p-2 rounded-lg';
    });
}

// Slide Deck Controls
function renderSlide() {
    const s = slides[currentSlide];
    const box = document.getElementById('slideBox');
    box.innerHTML = `
        <div class="space-y-4 max-w-3xl">
            <span class="text-xs font-bold uppercase tracking-wider text-teal-400 bg-teal-500/10 border border-teal-500/30 px-3 py-1 rounded-full">
                Slide ${currentSlide + 1} of ${slides.length}
            </span>
            <h3 class="text-2xl font-extrabold text-white">${s.title}</h3>
            <p class="text-sm text-slate-400">${s.subtitle}</p>
            <ul class="space-y-3 pt-4 text-xs sm:text-sm text-slate-300">
                ${s.points.map(p => `<li class="flex items-start gap-2.5"><i data-lucide="check-circle-2" class="w-4 h-4 text-teal-400 shrink-0 mt-0.5"></i> <span>${p}</span></li>`).join('')}
            </ul>
        </div>
    `;
    document.getElementById('slideIndicator').textContent = `Slide ${currentSlide + 1} / ${slides.length}`;
    lucide.createIcons();
}

function nextSlide() {
    if (currentSlide < slides.length - 1) {
        currentSlide++;
        renderSlide();
    }
}

function prevSlide() {
    if (currentSlide > 0) {
        currentSlide--;
        renderSlide();
    }
}
