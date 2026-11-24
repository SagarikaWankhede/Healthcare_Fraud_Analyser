import { api } from "../api.js";
import { STRATEGY_STYLE, axis, lineDataset, makeChart } from "../charts.js";
import { fmtCurrency, fmtNumber, fmtPct } from "../format.js";
import { card, clear, errorBox, h, kpi, loading, storageGet, table } from "../ui.js";

const ORDER = ["exposure", "probability", "reimbursement", "random"];

export function legend(keys) {
  return h("div", { class: "legend" }, keys.map((k) => h("span", {},
    h("span", { class: `sw${STRATEGY_STYLE[k].dash ? " dash" : ""}`, style: { background: STRATEGY_STYLE[k].color } }),
    STRATEGY_STYLE[k].label)));
}

export function strategyTable(rows, { compact = false } = {}) {
  const byKey = Object.fromEntries(rows.map((r) => [r.strategy, r]));
  const ordered = ORDER.map((k) => byKey[k]).filter(Boolean);
  const cols = [
    { label: "Strategy", render: (r) => h("span", { style: { whiteSpace: "nowrap" } }, h("span", { class: `sw${STRATEGY_STYLE[r.strategy].dash ? " dash" : ""}`, style: { display: "inline-block", width: "10px", height: "3px", marginRight: "8px", verticalAlign: "middle", background: STRATEGY_STYLE[r.strategy].color } }), r.label) },
    { label: "Fraud-labelled found", num: true, render: (r) => r.strategy === "random"
      ? `${fmtNumber(r.fraud_labelled_identified, 1)} (${fmtNumber(r.fraud_labelled_identified_p05, 0)}–${fmtNumber(r.fraud_labelled_identified_p95, 0)})`
      : fmtNumber(r.fraud_labelled_identified) },
    { label: "Precision at K", num: true, render: (r) => fmtPct(r.precision_at_k) },
    { label: "Fraud-labelled reimb. captured", num: true, title: "Total reimbursement of selected providers that carry a historical fraud label", render: (r) => fmtCurrency(r.known_fraud_reimbursement_captured) },
    { label: "Capture rate", num: true, render: (r) => r.strategy === "random"
      ? `${fmtPct(r.capture_rate)} (${fmtPct(r.capture_rate_p05)}–${fmtPct(r.capture_rate_p95)})` : fmtPct(r.capture_rate) },
  ];
  if (!compact) {
    cols.push({ label: "Reimbursement reviewed", num: true, render: (r) => fmtCurrency(r.reimbursement_reviewed) });
    cols.push({ label: "Same as proposed", num: true, title: "Providers also selected by the proposed ranking at this capacity",
      render: (r) => (r.overlap_with_proposed == null ? "—" : `${r.overlap_with_proposed} / ${r.providers_selected}`) });
  }
  return table(cols, ordered, { caption: "Audit strategy comparison" });
}

export function captureChart(canvas, curves, { capacity, maxK, metric = "capture" }) {
  const n = Math.min(maxK, curves.k.length);
  const src = metric === "capture" ? curves : curves.identified;
  const datasets = ORDER.map((k) => lineDataset(STRATEGY_STYLE[k].label, src[k].slice(0, n), STRATEGY_STYLE[k].color,
    { dash: STRATEGY_STYLE[k].dash, width: k === "exposure" ? 2.5 : 2 }));
  // 5th–95th percentile band for the random baseline
  datasets.push({ ...lineDataset("Random 95th pct.", src.random_p95.slice(0, n), "rgba(138,150,158,0)"), fill: "+1", backgroundColor: "rgba(138,150,158,0.14)" });
  datasets.push(lineDataset("Random 5th pct.", src.random_p05.slice(0, n), "rgba(138,150,158,0)"));
  const fmtY = metric === "capture" ? (v) => fmtPct(v, 0) : (v) => fmtNumber(v);
  return makeChart(canvas, {
    type: "line",
    data: { labels: curves.k.slice(0, n), datasets },
    options: {
      interaction: { mode: "index", intersect: false },
      scales: {
        x: { ...axis("Audit capacity (providers reviewed)"), type: "linear", min: 1, max: n },
        y: axis(metric === "capture" ? "Known fraud-labelled reimbursement captured" : "Fraud-labelled providers identified",
          { format: fmtY, min: 0, max: metric === "capture" ? 1 : undefined }),
      },
      plugins: {
        marker: { x: capacity, label: `Capacity ${capacity}` },
        tooltip: {
          filter: (item) => !item.dataset.label.startsWith("Random 5") && !item.dataset.label.startsWith("Random 9"),
          callbacks: {
            title: (items) => `Top ${items[0].parsed.x} providers`,
            label: (item) => ` ${item.dataset.label}: ${fmtY(item.parsed.y)}`,
          },
        },
      },
      parsing: false,
      normalized: true,
    },
  });
}

export function toXY(curves) {
  // Chart.js with parsing:false wants {x, y}
  const mk = (arr) => arr.map((y, i) => ({ x: i + 1, y }));
  const out = { k: curves.k, identified: {} };
  for (const key of [...ORDER, "random_p05", "random_p95"]) {
    out[key] = mk(curves[key]);
    out.identified[key] = mk(curves.identified[key]);
  }
  return out;
}

export async function render(root, ctx) {
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Overview"),
    h("p", {}, "Provider-level fraud risk and audit prioritisation on 2009 Medicare-style claims. Audit outcomes are evaluated retrospectively on providers held out from all model development."))));
  const body = h("div", {}, loading());
  root.append(body);
  let d;
  try {
    d = await api.get("/overview");
  } catch (err) {
    if (ctx.isCurrent()) clear(body).append(errorBox(err, () => render(clear(root), ctx)));
    return;
  }
  if (!ctx.isCurrent()) return;
  clear(body);
  const capacity = Math.max(1, Math.min(storageGet("mg.capacity", d.default_capacity), d.cohort.providers));
  const atCap = await api.get("/audit-simulation", { capacity }).catch(() => ({ strategies: d.strategies_at_default, capacity: d.default_capacity }));
  if (!ctx.isCurrent()) return;
  const byKey = Object.fromEntries(atCap.strategies.map((r) => [r.strategy, r]));
  const prop = byKey.exposure;
  const rand = byKey.random;

  body.append(h("div", { class: "grid grid-6" },
    kpi("Total claims analysed", fmtNumber(d.data.claims), `${fmtNumber(d.data.inpatient_claims)} inpatient · ${fmtNumber(d.data.outpatient_claims)} outpatient`, { "data-kpi": "claims" }),
    kpi("Providers analysed", fmtNumber(d.data.providers), `${fmtNumber(d.data.beneficiaries)} beneficiaries`, { "data-kpi": "providers" }),
    kpi("Historical fraud prevalence", fmtPct(d.data.fraud_prevalence), `${fmtNumber(d.data.fraud_labelled_providers)} fraud-labelled providers`, { "data-kpi": "prevalence" }),
    kpi("Reimbursement of fraud-labelled providers", fmtCurrency(d.data.fraud_labelled_reimbursement),
      d.data.total_reimbursement ? `${fmtPct(d.data.fraud_labelled_reimbursement / d.data.total_reimbursement)} of ${fmtCurrency(d.data.total_reimbursement)} total` : "", { "data-kpi": "fraud-reimbursement" }),
    kpi("Current audit capacity", `${fmtNumber(atCap.capacity)} providers`, h("a", { href: "#/audit" }, "Adjust in Audit Priorities"), { "data-kpi": "capacity" }),
    kpi("Captured at this capacity", fmtPct(prop.capture_rate),
      `vs ${fmtPct(rand.capture_rate)} random · held-out cohort`, { "data-kpi": "capture" }),
  ));

  const xy = toXY(d.curves);
  const capCanvas = h("canvas", { "aria-label": "Cumulative fraud-labelled reimbursement capture by audit strategy", role: "img" });
  const idCanvas = h("canvas", { "aria-label": "Fraud-labelled providers identified by audit strategy", role: "img" });
  let range = 300;
  const rangeSeg = h("div", { class: "seg", role: "group", "aria-label": "Capacity range" });
  const charts = [];
  function drawCharts() {
    charts.splice(0).forEach((c) => c && c.destroy());
    charts.push(captureChart(capCanvas, xy, { capacity: atCap.capacity, maxK: range, metric: "capture" }));
    charts.push(captureChart(idCanvas, xy, { capacity: atCap.capacity, maxK: range, metric: "identified" }));
  }
  for (const [label, v] of [["Top 300", 300], [`All ${fmtNumber(d.cohort.providers)}`, d.cohort.providers]]) {
    const b = h("button", { class: v === range ? "active" : null, onclick: () => {
      range = v; rangeSeg.querySelectorAll("button").forEach((x) => x.classList.toggle("active", x === b)); drawCharts();
    } }, label);
    rangeSeg.append(b);
  }

  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    h("section", { class: "card" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Cumulative fraud capture curve"),
        h("p", {}, `Share of the ${fmtCurrency(d.cohort.known_fraud_reimbursement)} reimbursement billed by the ${d.cohort.fraud_labelled_providers} fraud-labelled providers in the ${fmtNumber(d.cohort.providers)}-provider held-out cohort, captured as audit capacity grows.`)),
        rangeSeg),
      h("div", { class: "chart-box tall" }, capCanvas), legend(ORDER),
      h("p", { class: "small muted", style: { marginTop: "6px" } }, `Random baseline: ${atCap.random_method || "seeded random orderings"} (shaded band).`)),
    h("section", { class: "card" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Fraud-labelled providers identified"),
        h("p", {}, "Count of historically fraud-labelled providers within the top K of each ranking (same held-out cohort)."))),
      h("div", { class: "chart-box tall" }, idCanvas), legend(ORDER)),
  ));
  drawCharts();

  body.append(h("div", { class: "grid grid-3", style: { marginTop: "16px" } },
    h("section", { class: "card span-2" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, `Strategy comparison at capacity ${atCap.capacity}`),
        h("p", {}, "Retrospective evaluation: identical held-out providers for every strategy."))),
      strategyTable(atCap.strategies, { compact: true })),
    card("Model", "Selected by cross-validated PR-AUC; evaluated once on the held-out cohort.",
      h("dl", { class: "kv" },
        h("dt", {}, "Classifier"), h("dd", {}, d.model.name),
        h("dt", {}, "Version"), h("dd", { class: "mono" }, d.model.version),
        h("dt", {}, "Held-out PR-AUC"), h("dd", {}, d.model.test_pr_auc.toFixed(3)),
        h("dt", {}, "Held-out ROC-AUC"), h("dd", {}, d.model.test_roc_auc.toFixed(3)),
        h("dt", {}, "5-fold CV PR-AUC"), h("dd", {}, `${d.model.cv_pr_auc.toFixed(3)} ± ${d.model.cv_pr_auc_std.toFixed(3)}`),
        h("dt", {}, "No-skill PR-AUC (prevalence)"), h("dd", {}, (d.cohort.fraud_labelled_providers / d.cohort.providers).toFixed(3))),
      h("p", { class: "small", style: { marginTop: "10px" } }, h("a", { href: "#/performance" }, "Full model evaluation"))),
  ));

  body.append(h("div", { class: "notice", style: { marginTop: "16px" } },
    h("strong", {}, "Reading these numbers. "),
    "Before an audit the system provides a calibrated fraud probability, an estimated financial exposure (probability × reimbursement) and an audit priority. ",
    "“Known fraud-labelled reimbursement captured” is a retrospective measure: all reimbursement of selected providers that carry a historical fraud label. ",
    "It is not verified fraudulent billing and not money recovered."));
}
