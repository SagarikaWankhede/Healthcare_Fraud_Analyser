import { api } from "../api.js";
import { COLORS, axis, makeChart } from "../charts.js";
import { STATUS_LABELS, fmtByKind, fmtCurrency, fmtNumber, fmtPct, fmtSigned } from "../format.js";
import { card, clear, errorBox, h, kpi, kvList, loading, priorityBadge, riskBadge, statusBadge, table } from "../ui.js";

const GROUP_ORDER = ["billing", "patients", "physicians", "coding", "regression", "association"];
const GROUP_TITLES = {
  billing: "Billing behaviour", patients: "Patient mix", physicians: "Physician patterns", coding: "Clinical coding",
  regression: "Reimbursement deviation (from claim-level regression)", association: "Code combinations (association rules)",
};
const KEY_INPUTS = ["total_claims", "total_reimbursed", "unique_beneficiaries", "inpatient_claims", "max_claim_duration",
  "mean_residual", "total_positive_deviation", "positive_residual_ratio", "suspicious_itemset_claims"];

export async function render(root, ctx) {
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Provider Analysis"),
    h("p", {}, "Investigate an individual provider's billing, anomaly signals and model explanation, or score a provider profile with the trained model."))));
  const tabs = h("div", { class: "tabs", role: "tablist" });
  const panel = h("div", {});
  root.append(tabs, panel);
  const mode = ctx.param === "score" ? "score" : "profile";
  for (const [key, label] of [["profile", "Provider profile"], ["score", "Score a provider"]]) {
    tabs.append(h("button", { role: "tab", class: key === mode ? "active" : null, "aria-selected": String(key === mode),
      onclick: () => { window.location.hash = key === "score" ? "#/providers/score" : "#/providers"; } }, label));
  }
  if (mode === "score") return renderScorer(panel, ctx);
  return renderProfileTab(panel, ctx);
}

// ---------------------------------------------------------------------------------------------
async function renderProfileTab(panel, ctx) {
  const input = h("input", { type: "search", placeholder: "Provider ID, e.g. PRV51001", value: ctx.param || "", "aria-label": "Provider ID", "data-testid": "provider-search" });
  const go = () => { const v = input.value.trim().toUpperCase(); if (v) window.location.hash = `#/providers/${encodeURIComponent(v)}`; };
  input.addEventListener("keydown", (e) => { if (e.key === "Enter") go(); });
  panel.append(h("div", { class: "row", style: { marginBottom: "16px" } }, input, h("button", { class: "primary", onclick: go }, "Open provider")));
  const body = h("div", {});
  panel.append(body);
  if (!ctx.param) {
    body.append(loading("Loading highest-priority providers…"));
    try {
      const res = await api.get("/providers", { per_page: 10 });
      if (!ctx.isCurrent()) return;
      clear(body).append(card("Highest estimated exposure", "Top of the audit queue. Open any provider to see its full profile.",
        table([
          { label: "Rank", num: true, render: (r) => String(r.rank) },
          { label: "Provider ID", render: (r) => h("a", { class: "mono", href: `#/providers/${r.provider_id}` }, r.provider_id) },
          { label: "Fraud probability", num: true, render: (r) => fmtPct(r.fraud_probability) },
          { label: "Estimated exposure", num: true, render: (r) => fmtCurrency(r.estimated_exposure) },
          { label: "Priority", render: (r) => priorityBadge(r.audit_priority) },
          { label: "Status", render: (r) => statusBadge(r.investigation_status) },
        ], res.providers, { onRowClick: (r) => { window.location.hash = `#/providers/${r.provider_id}`; } })));
    } catch (err) {
      if (ctx.isCurrent()) clear(body).append(errorBox(err));
    }
    return;
  }
  await renderProfile(body, ctx, ctx.param, false);
}

async function renderProfile(body, ctx, id, retrospective) {
  clear(body).append(loading(`Loading ${id}…`));
  let p;
  try {
    p = await api.get(`/providers/${encodeURIComponent(id)}`, { evaluation_mode: retrospective ? "retrospective" : "" });
  } catch (err) {
    if (ctx.isCurrent()) clear(body).append(errorBox(err, err.status === 404 ? null : () => renderProfile(body, ctx, id, retrospective)));
    return;
  }
  if (!ctx.isCurrent()) return;
  clear(body);
  const s = p.scores;

  // 1. Overview header
  body.append(h("section", { class: "card" },
    h("div", { class: "row" },
      h("h2", { class: "mono", "data-testid": "profile-id" }, p.provider_id), riskBadge(s.risk_level), priorityBadge(s.audit_priority),
      h("span", { class: "muted small" }, `Rank ${s.rank_exposure} of ${s.cohort_size} by estimated exposure · rank ${s.rank_probability} by probability · rank ${s.rank_reimbursement} by reimbursement`),
      h("span", { class: "spacer" }),
      h("label", { class: "row small", style: { gap: "6px" } },
        h("input", { type: "checkbox", checked: retrospective, "data-testid": "retro-toggle",
          onchange: (e) => renderProfile(body, ctx, id, e.target.checked) }), "Retrospective evaluation mode"))));

  // 2. Scores
  body.append(h("div", { class: "grid grid-6", style: { marginTop: "16px" } },
    kpi("Fraud probability (calibrated)", fmtPct(s.fraud_probability), s.above_decision_threshold ? `Above decision threshold (${fmtPct(s.decision_threshold)})` : `Below decision threshold (${fmtPct(s.decision_threshold)})`, { "data-kpi": "probability" }),
    kpi("Estimated exposure", fmtCurrency(s.estimated_exposure), "Probability × reimbursement (proxy)"),
    kpi("Total reimbursement", fmtCurrency(p.billing.total_reimbursed), `${fmtNumber(p.billing.total_claims)} claims`),
    kpi("Beneficiaries", fmtNumber(p.billing.unique_beneficiaries), `${fmtNumber(p.billing.claims_per_beneficiary, 2)} claims each`),
    kpi("Inpatient claims", fmtNumber(p.billing.inpatient_claims), `${fmtPct(p.billing.inpatient_claims / p.billing.total_claims)} of claims`),
    kpi("Peer group", `Group ${p.anomaly.clustering.peer_group}`, p.anomaly.clustering.dbscan_outlier ? "DBSCAN outlier" : "Within a dense region"),
  ));

  // 3. Recommendation + investigation
  const statusSel = h("select", { "data-testid": "status-select", "aria-label": "Investigation status" },
    Object.entries(STATUS_LABELS).map(([v, t]) => h("option", { value: v, selected: v === p.investigation.status }, t)));
  const note = h("textarea", { rows: "2", maxlength: "2000", placeholder: "Investigator note (optional)", style: { width: "100%" } }, p.investigation.note || "");
  const saveMsg = h("span", { class: "small muted", "data-testid": "status-msg" });
  const history = h("div", {});
  const drawHistory = (inv) => {
    clear(history).append(inv.history.length
      ? table([{ label: "When (UTC)", render: (e) => e.created_at.replace("T", " ").replace("+00:00", "") },
          { label: "Status", render: (e) => statusBadge(e.status) }, { label: "Note", render: (e) => e.note || "—" },
          { label: "Model", render: (e) => h("span", { class: "mono small" }, e.model_version || "—") }], inv.history)
      : h("p", { class: "small muted" }, "No investigation activity recorded."));
  };
  drawHistory(p.investigation);
  const save = h("button", { class: "primary", "data-testid": "status-save", onclick: async () => {
    save.disabled = true;
    saveMsg.textContent = "Saving…";
    try {
      const inv = await api.patch(`/providers/${encodeURIComponent(p.provider_id)}/status`, { status: statusSel.value, note: note.value.trim() || null });
      saveMsg.textContent = `Saved: ${STATUS_LABELS[inv.status]}`;
      drawHistory(inv);
    } catch (err) {
      saveMsg.textContent = `Could not save: ${err.message}`;
    } finally {
      save.disabled = false;
    }
  } }, "Save status");
  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    h("section", { class: "card" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Audit recommendation"),
      h("p", {}, "Application-generated, from model output."))),
      h("p", {}, h("strong", { "data-testid": "recommendation" }, p.recommendation.action), " — ", p.recommendation.rationale),
      h("p", { class: "small muted" }, p.recommendation.disclaimer)),
    h("section", { class: "card" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Investigation record"),
      h("p", {}, "Decisions made by investigators. Stored separately from model output."))),
      h("div", { class: "row" }, statusSel, save, saveMsg), h("div", { style: { margin: "8px 0" } }, note), history)));

  // 4. Trends + claim mix
  const trendCanvas = h("canvas", { role: "img", "aria-label": "Monthly reimbursement" });
  const mixCanvas = h("canvas", { role: "img", "aria-label": "Claim amount distribution" });
  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    card("Reimbursement trend", `Monthly reimbursement by claim start date (${p.claim_period.start} to ${p.claim_period.end}). Hover for claim counts.`, h("div", { class: "chart-box short" }, trendCanvas)),
    card("Claim distribution", "Claims by reimbursement amount and care setting.", h("div", { class: "chart-box short" }, mixCanvas),
      h("div", { class: "legend" }, h("span", {}, h("span", { class: "sw", style: { background: "#5468E0" } }), "Outpatient"),
        h("span", {}, h("span", { class: "sw", style: { background: "#D86A2A" } }), "Inpatient")))));
  makeChart(trendCanvas, {
    type: "bar",
    data: { labels: p.trend.map((t) => t.month), datasets: [{ data: p.trend.map((t) => t.reimbursed), backgroundColor: COLORS.accent, borderRadius: 3, maxBarThickness: 28 }] },
    options: { scales: { x: { grid: { display: false } }, y: axis(null, { min: 0, format: (v) => fmtCurrency(v) }) },
      plugins: { tooltip: { callbacks: { label: (it) => { const t = p.trend[it.dataIndex]; return ` ${fmtCurrency(t.reimbursed, { compact: false })} · ${t.claims} claims (${t.inpatient_claims} inpatient)`; } } } } },
  });
  const buckets = ["$0", "$1-99", "$100-499", "$500-1,999", "$2,000-9,999", "$10,000+"];
  const mixOf = (setting) => buckets.map((b) => (p.claim_mix.find((m) => m.setting === setting && m.bucket === b) || { claims: 0 }).claims);
  makeChart(mixCanvas, {
    type: "bar",
    data: { labels: buckets, datasets: [
      { label: "Outpatient", data: mixOf("outpatient"), backgroundColor: "#5468E0", borderRadius: 3, maxBarThickness: 22 },
      { label: "Inpatient", data: mixOf("inpatient"), backgroundColor: "#D86A2A", borderRadius: 3, maxBarThickness: 22 }] },
    options: { scales: { x: { grid: { display: false } }, y: axis("Claims", { min: 0 }) },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: ${fmtNumber(it.parsed.y)} claims` } } } },
  });

  // 5. Billing statistics + comparisons
  const b = p.billing;
  body.append(h("div", { class: "grid grid-3", style: { marginTop: "16px" } },
    card("Billing statistics", null, kvList([
      ["Total claims", fmtNumber(b.total_claims)], ["Inpatient claims", fmtNumber(b.inpatient_claims)],
      ["Total reimbursement", fmtCurrency(b.total_reimbursed, { compact: false })],
      ["Mean per claim", fmtCurrency(b.mean_reimbursed, { compact: false })], ["Median per claim", fmtCurrency(b.median_reimbursed, { compact: false })],
      ["Largest claim", fmtCurrency(b.max_reimbursed, { compact: false })], ["Billing variability (CV)", fmtNumber(b.cv_reimbursed, 2)],
      ["Mean inpatient claim", fmtCurrency(b.inpatient_mean_reimbursed, { compact: false })],
      ["Mean outpatient claim", fmtCurrency(b.outpatient_mean_reimbursed, { compact: false })],
      ["Deductible paid", fmtCurrency(b.total_deductible_paid, { compact: false })], ["Total claim days", fmtNumber(b.total_claim_days)],
      ["Mean claim duration (days)", fmtNumber(b.mean_claim_duration, 1)], ["Reimbursement per beneficiary", fmtCurrency(b.reimbursement_per_beneficiary, { compact: false })],
      ["Active months", fmtNumber(b.active_months)], ["Attending physicians", fmtNumber(b.unique_attending_physicians)]])),
    h("section", { class: "card span-2" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Compared with the cohort"),
      h("p", {}, `Percentile among the ${s.cohort_size} held-out providers (50th = typical).`))),
      table([
        { label: "Measure", render: (c) => c.label },
        { label: "Provider", num: true, render: (c) => fmtByKind(c.value, c.kind) },
        { label: "Cohort median", num: true, render: (c) => fmtByKind(c.cohort_median, c.kind) },
        { label: "Percentile", num: true, render: (c) => (c.percentile == null ? "—" : fmtNumber(c.percentile * 100, 0)) },
        { label: "", render: (c) => h("div", { class: "bar-inline", title: c.percentile == null ? "" : `${Math.round(c.percentile * 100)}th percentile` }, h("span", { style: { width: `${Math.round((c.percentile || 0) * 100)}%` } })) },
      ], p.comparisons))));

  // 6. Anomaly indicators
  const reg = p.anomaly.regression;
  const clu = p.anomaly.clustering;
  const asc = p.anomaly.association;
  body.append(h("h2", { class: "section-title" }, "Anomaly indicators"));
  body.append(h("div", { class: "grid grid-3" },
    card("Reimbursement deviation", "Actual minus model-predicted reimbursement per claim. High values mean billing above comparable claims — not proof of fraud.",
      kvList([["Mean residual per claim", fmtCurrency(reg.mean_residual, { compact: false })], ["Median residual", fmtCurrency(reg.median_residual, { compact: false })],
        ["Claims above prediction", fmtPct(reg.positive_residual_ratio)], ["Total above prediction", fmtCurrency(reg.total_positive_deviation)],
        ["Above-prediction share of reimbursement", fmtPct(reg.excess_reimbursement_ratio)]])),
    card("Peer-group behaviour", "Unsupervised clustering of provider billing profiles. Outliers bill differently; they are not confirmed fraud.",
      kvList([["Hierarchical peer group", `Group ${clu.peer_group}`], ["DBSCAN status", clu.dbscan_outlier ? "Outlier (low-density region)" : "Inside a dense region"],
        ["Distance to nearest dense region", `${fmtNumber(clu.dbscan_core_distance, 2)} × eps`], ["Distance to peer-group centre", fmtNumber(clu.cluster_centroid_distance, 2)]])),
    card("Flagged code combinations", "Diagnosis/procedure itemsets over-represented among fraud-labelled providers in training data (lift > 1).",
      kvList([["Claims with a flagged combination", `${fmtNumber(asc.suspicious_itemset_claims)} (${fmtPct(asc.suspicious_itemset_claim_ratio)})`],
        ["Distinct combinations billed", fmtNumber(asc.suspicious_itemset_variety)]]),
      asc.itemsets.length ? table([{ label: "Itemset", render: (i) => h("span", { class: "mono small" }, i.itemset) },
        { label: "Claims", num: true, render: (i) => fmtNumber(i.claims) }, { label: "Lift", num: true, render: (i) => fmtNumber(i.lift, 2) }], asc.itemsets.slice(0, 6)) : null)));

  // 7. Model explanation
  const ex = p.explanation;
  const exCanvas = h("canvas", { role: "img", "aria-label": "Feature contributions to fraud probability" });
  body.append(h("h2", { class: "section-title" }, "Model explanation"));
  body.append(h("div", { class: "grid grid-2" },
    card("Main contributing factors", p.explanation_method, h("div", { class: "chart-box short" }, exCanvas),
      h("div", { class: "legend" }, h("span", {}, h("span", { class: "sw", style: { background: COLORS.riskHigh } }), "Raises the estimate"),
        h("span", {}, h("span", { class: "sw", style: { background: COLORS.accent } }), "Lowers the estimate"))),
    card("Factor detail", "Provider value compared with the training-population median used as reference.",
      table([{ label: "Feature", render: (e) => e.label }, { label: "Provider", num: true, render: (e) => fmtByKind(e.value, e.kind) },
        { label: "Reference", num: true, render: (e) => fmtByKind(e.reference, e.kind) },
        { label: "Effect on probability", num: true, render: (e) => `${fmtSigned(e.contribution * 100, 1)} pts` }], ex))));
  if (!ex.length) exCanvas.parentElement.replaceWith(h("p", { class: "empty" }, "No single feature changes this estimate measurably (the probability is saturated)."));
  else makeChart(exCanvas, {
    type: "bar",
    data: { labels: ex.map((e) => e.label), datasets: [{ data: ex.map((e) => e.contribution * 100), borderRadius: 3, barThickness: 14,
      backgroundColor: ex.map((e) => (e.contribution > 0 ? COLORS.riskHigh : COLORS.accent)) }] },
    options: { indexAxis: "y", scales: { x: axis("Change in probability (percentage points)", { format: (v) => `${v > 0 ? "+" : ""}${v}` }), y: { grid: { display: false }, ticks: { autoSkip: false } } },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${fmtSigned(it.parsed.x, 2)} pts` } } } },
  });

  // 8. Retrospective (separated)
  if (p.retrospective) {
    body.append(h("section", { class: "card retro", style: { marginTop: "16px" }, "data-testid": "retro-panel" },
      h("h2", {}, "Retrospective evaluation"),
      h("p", { style: { marginTop: "6px" } }, "Historical fraud label: ", h("strong", {}, p.retrospective.fraud_label ? "Fraud-labelled (PotentialFraud = Yes)" : "Not fraud-labelled (PotentialFraud = No)")),
      h("p", { class: "small muted" }, p.retrospective.note)));
  }
}

// ---------------------------------------------------------------------------------------------
async function renderScorer(panel, ctx) {
  panel.append(loading("Loading model schema…"));
  let schema;
  try {
    schema = await api.get("/predict/schema");
  } catch (err) {
    if (ctx.isCurrent()) clear(panel).append(errorBox(err, () => renderScorer(clear(panel), ctx)));
    return;
  }
  if (!ctx.isCurrent()) return;
  clear(panel);
  const feats = Object.fromEntries(schema.features.map((f) => [f.feature, f]));
  const inputs = {};
  const errors = {};
  const field = (f) => {
    const inp = h("input", { type: "number", step: "any", name: f.feature, "data-feature": f.feature,
      placeholder: f.train_median == null ? "" : `median ${fmtNumber(f.train_median, f.kind === "count" ? 0 : 2)}`,
      min: ["count", "amount", "days", "ratio", "years", "score", "binary"].includes(f.kind) ? "0" : null,
      max: f.kind === "ratio" || f.kind === "binary" ? "1" : null });
    const err = h("span", { class: "field-error" });
    inputs[f.feature] = inp;
    errors[f.feature] = err;
    const unit = { amount: "$", signed_amount: "$", ratio: "0–1", days: "days", years: "years" }[f.kind];
    return h("label", { class: "field", "data-field": f.feature }, `${f.label}${unit ? ` (${unit})` : ""}${f.required ? " *" : ""}`, inp, err);
  };

  const examples = h("div", { class: "row" }, h("span", { class: "small muted" }, "Load example:"),
    schema.examples.map((ex) => h("button", { "data-example": ex.provider_id, title: ex.note, onclick: () => fill(ex) }, `${ex.title} (${ex.provider_id})`)),
    h("button", { onclick: reset, "data-testid": "reset" }, "Reset"));
  const exampleNote = h("p", { class: "small muted" });
  const key = KEY_INPUTS.filter((f) => feats[f]).map((f) => field(feats[f]));
  const rest = h("details", {}, h("summary", {}, `All model inputs (${schema.features.length})`),
    GROUP_ORDER.map((g) => {
      const fs = schema.features.filter((f) => f.group === g && !KEY_INPUTS.includes(f.feature));
      return fs.length ? [h("div", { class: "group-title" }, GROUP_TITLES[g]), h("div", { class: "form-grid" }, fs.map(field))] : null;
    }));
  const formError = h("div", {});
  const result = h("div", {});
  const submit = h("button", { class: "primary", type: "submit", "data-testid": "predict-submit" }, "Estimate risk");
  const form = h("form", { novalidate: true, onsubmit: (e) => { e.preventDefault(); predict(); } },
    h("div", { class: "form-grid" }, key), rest, h("div", { class: "row", style: { marginTop: "14px" } }, submit), formError);

  panel.append(h("div", { class: "grid grid-2" },
    h("section", { class: "card" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Provider profile"),
      h("p", {}, `Enter provider-level aggregates. Fields left blank are filled with the training median by the model's own fitted imputer. Ratios, peer group and DBSCAN status are computed server-side. Model ${schema.model_name} · ${schema.model_version}.`))),
      examples, exampleNote, form),
    h("div", {}, result)));

  function setValues(values) {
    for (const [f, inp] of Object.entries(inputs)) inp.value = values[f] == null ? "" : String(values[f]);
  }
  function fill(ex) {
    setValues(ex.features);
    exampleNote.textContent = ex.note;
    clearErrors();
  }
  function reset() {
    setValues({});
    exampleNote.textContent = "";
    clearErrors();
    clear(result);
  }
  function clearErrors() {
    for (const [f, e] of Object.entries(errors)) { e.textContent = ""; e.parentElement.classList.remove("invalid"); }
    clear(formError);
  }
  function showErrors(list) {
    for (const d of list) {
      const e = errors[d.field] || errors[d.field.replace(/^features\./, "")];
      if (e) { e.textContent = d.message; e.parentElement.classList.add("invalid"); }
    }
    const unplaced = list.filter((d) => !errors[d.field]);
    clear(formError).append(h("div", { class: "error-box", style: { marginTop: "10px" }, role: "alert", "data-testid": "predict-error" },
      "Please correct the highlighted fields.", unplaced.length ? h("ul", {}, unplaced.map((d) => h("li", {}, `${d.field}: ${d.message}`))) : null));
  }

  async function predict() {
    clearErrors();
    const payload = {};
    const local = [];
    for (const [f, inp] of Object.entries(inputs)) {
      if (inp.value.trim() === "") continue;
      const v = Number(inp.value);
      if (!Number.isFinite(v)) local.push({ field: f, message: "must be a number" });
      else payload[f] = v;
    }
    for (const f of schema.features.filter((x) => x.required)) if (!(f.feature in payload) && !local.some((l) => l.field === f.feature)) local.push({ field: f.feature, message: "is required" });
    if (local.length) return showErrors(local);
    submit.disabled = true;
    clear(result).append(loading("Scoring with the trained model…"));
    try {
      const r = await api.post("/predict", { features: payload });
      if (!ctx.isCurrent()) return;
      drawResult(r);
    } catch (err) {
      clear(result);
      if (err.status === 422 && Array.isArray(err.details)) showErrors(err.details);
      else clear(formError).append(errorBox(err));
    } finally {
      submit.disabled = false;
    }
  }

  function drawResult(r) {
    const cc = r.computed_features;
    const exCanvas = h("canvas", { role: "img", "aria-label": "Feature contributions" });
    clear(result).append(
      h("section", { class: "card", "data-testid": "predict-result" },
        h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Model estimate"), h("p", {}, `${r.model_name} · ${r.model_version}`))),
        h("div", { class: "grid grid-2" },
          kpi("Fraud probability (calibrated)", h("span", { "data-testid": "predicted-probability" }, fmtPct(r.fraud_probability)), r.above_decision_threshold ? `Above decision threshold (${fmtPct(r.decision_threshold)})` : `Below decision threshold (${fmtPct(r.decision_threshold)})`),
          kpi("Estimated exposure", fmtCurrency(r.estimated_exposure), "Probability × reimbursement (proxy)"),
          kpi("Audit priority", h("span", {}, priorityBadge(r.audit_priority), " ", riskBadge(r.risk_level)), `Would rank ${r.cohort_rank} of ${r.cohort_size} in the held-out cohort`),
          kpi("Peer group / DBSCAN", `Group ${cc.peer_group}`, cc.dbscan_outlier ? "DBSCAN outlier (computed)" : "Inside a dense region (computed)")),
        h("p", { style: { marginTop: "12px" } }, h("strong", {}, r.recommendation.action), " — ", r.recommendation.rationale),
        r.warnings.length ? h("div", { class: "notice warn" }, r.warnings.map((w) => h("div", {}, w))) : null,
        h("p", { class: "small muted", style: { marginTop: "8px" } },
          `${r.imputed_features.length} of the model's features were not supplied and were imputed with training medians.`),
        r.imputed_features.length ? h("details", {}, h("summary", { class: "small" }, "Imputed features"), h("p", { class: "small mono" }, r.imputed_features.join(", "))) : null,
        h("h3", { style: { margin: "14px 0 6px" } }, "Main contributing factors"),
        h("div", { class: "chart-box short" }, exCanvas),
        h("p", { class: "small muted" }, "Change in probability when each feature is set to the training median, one at a time."),
        h("p", { class: "small muted" }, r.recommendation.disclaimer)));
    const ex = r.contributions;
    if (!ex.length) {
      exCanvas.parentElement.replaceWith(h("p", { class: "empty" }, "No single feature changes this estimate measurably (the probability is saturated)."));
      return;
    }
    makeChart(exCanvas, {
      type: "bar",
      data: { labels: ex.map((e) => e.label), datasets: [{ data: ex.map((e) => e.contribution * 100), borderRadius: 3, barThickness: 12,
        backgroundColor: ex.map((e) => (e.contribution > 0 ? COLORS.riskHigh : COLORS.accent)) }] },
      options: { indexAxis: "y", scales: { x: axis("Percentage points"), y: { grid: { display: false }, ticks: { autoSkip: false } } },
        plugins: { tooltip: { callbacks: { label: (it) => ` ${fmtSigned(it.parsed.x, 2)} pts` } } } },
    });
  }
}
