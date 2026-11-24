import { api } from "../api.js";
import { COLORS, MODEL_COLORS, axis, lineDataset, makeChart } from "../charts.js";
import { fmtCurrency, fmtNumber, fmtPct, fmtSigned } from "../format.js";
import { card, clear, errorBox, h, kvList, loading, table } from "../ui.js";

const f3 = (v) => (v == null ? "—" : v.toFixed(3));
const pm = (m, s) => `${f3(m)} ± ${f3(s)}`;
const MODEL_ORDER = ["xgboost", "random_forest", "adaboost", "logistic_regression"];

function modelLegend(models) {
  return h("div", { class: "legend" }, models.map((m) => h("span", {},
    h("span", { class: "sw", style: { background: MODEL_COLORS[m.model] } }), m.model_name)));
}

function curveChart(canvas, models, kind, prevalence) {
  const datasets = models.map((m) => ({
    ...lineDataset(m.model_name, m.curves[kind].map(([x, y]) => ({ x, y })), MODEL_COLORS[m.model], { width: m.selected ? 2.5 : 1.75 }),
  }));
  const ref = kind === "roc" ? [{ x: 0, y: 0 }, { x: 1, y: 1 }] : [{ x: 0, y: prevalence }, { x: 1, y: prevalence }];
  datasets.push(lineDataset(kind === "roc" ? "Chance" : `No-skill (prevalence ${fmtPct(prevalence)})`, ref, COLORS.baseline, { dash: [4, 4], width: 1 }));
  return makeChart(canvas, {
    type: "line", data: { datasets },
    options: {
      parsing: false, interaction: { mode: "nearest", intersect: false },
      scales: { x: { ...axis(kind === "roc" ? "False positive rate" : "Recall"), type: "linear", min: 0, max: 1 },
        y: axis(kind === "roc" ? "True positive rate" : "Precision", { min: 0, max: 1 }) },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: (${it.parsed.x.toFixed(2)}, ${it.parsed.y.toFixed(2)})` } } },
    },
  });
}

function confusion(cm) {
  const cell = (label, v, strong) => h("td", { class: "num", style: { fontWeight: strong ? "600" : "400", padding: "14px" } }, h("div", { class: "small muted" }, label), fmtNumber(v));
  return h("table", { class: "confusion", style: { maxWidth: "420px" } },
    h("thead", {}, h("tr", {}, h("th", {}, ""), h("th", { class: "num" }, "Predicted not fraud"), h("th", { class: "num" }, "Predicted fraud"))),
    h("tbody", {},
      h("tr", {}, h("th", {}, "Not fraud-labelled"), cell("True negative", cm.tn), cell("False positive", cm.fp)),
      h("tr", {}, h("th", {}, "Fraud-labelled"), cell("False negative", cm.fn), cell("True positive", cm.tp, true))));
}

export async function render(root, ctx) {
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Model Performance"),
    h("p", {}, "Cross-validated model selection on development providers and a single final evaluation on held-out providers. PR-AUC is the primary metric because only ~9% of providers are fraud-labelled."))));
  const body = h("div", {}, loading());
  root.append(body);
  let perf, analysis;
  try {
    [perf, analysis] = await Promise.all([api.get("/model-performance"), api.get("/analysis")]);
  } catch (err) {
    if (ctx.isCurrent()) clear(body).append(errorBox(err, () => render(clear(root), ctx)));
    return;
  }
  if (!ctx.isCurrent()) return;
  clear(body);
  const models = MODEL_ORDER.map((k) => perf.test.find((t) => t.model === k)).filter(Boolean);
  const sel = models.find((m) => m.selected);
  const prevalence = perf.test_prevalence;

  // Comparison table
  body.append(card("Model comparison",
    `${perf.selection_criterion}. Held-out metrics use each model's calibrated probabilities and its own out-of-fold F1-optimal threshold.`,
    table([
      { label: "Model", render: (m) => h("span", { style: { whiteSpace: "nowrap" } }, h("span", { class: "sw", style: { display: "inline-block", width: "10px", height: "3px", marginRight: "8px", verticalAlign: "middle", background: MODEL_COLORS[m.model] } }), m.model_name, m.selected ? h("strong", { style: { marginLeft: "6px" } }, "(selected)") : null) },
      { label: "Best configuration", render: (m) => h("span", { class: "small mono" }, m.config) },
      { label: "CV PR-AUC", num: true, render: (m) => pm(m.cv_pr_auc_mean, m.cv_pr_auc_std) },
      { label: "CV ROC-AUC", num: true, render: (m) => pm(m.cv_roc_auc_mean, m.cv_roc_auc_std) },
      { label: "Held-out PR-AUC", num: true, render: (m) => f3(m.pr_auc) },
      { label: "Held-out ROC-AUC", num: true, render: (m) => f3(m.roc_auc) },
      { label: "Precision", num: true, render: (m) => f3(m.precision) },
      { label: "Recall", num: true, render: (m) => f3(m.recall) },
      { label: "F1", num: true, render: (m) => f3(m.f1) },
      { label: "Brier", num: true, render: (m) => f3(m.brier) },
      { label: "Threshold", num: true, render: (m) => fmtPct(m.threshold) },
    ], models, { rowClass: (m) => (m.selected ? "selected-row" : "") }),
    h("p", { class: "small muted", style: { marginTop: "8px" } },
      `Held-out cohort: ${perf.split.test_providers} providers (${perf.split.test_fraud} fraud-labelled). With about 100 positives, held-out PR-AUC has a sampling uncertainty of several hundredths; the CV mean ± s.d. is the more stable comparison.`)));

  // Curves
  const roc = h("canvas", { role: "img", "aria-label": "ROC curves" });
  const pr = h("canvas", { role: "img", "aria-label": "Precision-recall curves" });
  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    card("Precision–recall curves", "Held-out cohort.", h("div", { class: "chart-box" }, pr), modelLegend(models)),
    card("ROC curves", "Held-out cohort.", h("div", { class: "chart-box" }, roc), modelLegend(models))));
  curveChart(pr, models, "pr", prevalence);
  curveChart(roc, models, "roc", prevalence);

  // Confusion + calibration
  const cmBox = h("div", {});
  const modelSel = h("select", { "aria-label": "Model for confusion matrix", onchange: () => drawCm() },
    models.map((m) => h("option", { value: m.model, selected: m.selected }, m.model_name)));
  const drawCm = () => {
    const m = models.find((x) => x.model === modelSel.value);
    clear(cmBox).append(confusion(m.confusion_matrix), h("p", { class: "small muted", style: { marginTop: "8px" } },
      `Threshold ${fmtPct(m.threshold)} (chosen on out-of-fold development predictions). Precision ${f3(m.precision)}, recall ${f3(m.recall)}.`));
  };
  drawCm();
  const cal = h("canvas", { role: "img", "aria-label": "Calibration curve" });
  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    h("section", { class: "card" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Confusion matrix"), h("p", {}, "Held-out cohort at the decision threshold.")), modelSel), cmBox),
    card("Probability calibration", `${perf.calibration_method}. Equal-count bins of the held-out cohort (${sel.model_name}).`,
      h("div", { class: "chart-box short" }, cal),
      h("div", { class: "legend" }, h("span", {}, h("span", { class: "sw", style: { background: MODEL_COLORS[sel.model] } }), `Calibrated (Brier ${f3(sel.calibration.calibrated_brier)}, ECE ${f3(sel.calibration.calibrated_ece)})`),
        h("span", {}, h("span", { class: "sw", style: { background: COLORS.baseline } }), `Raw class-weighted score (Brier ${f3(sel.calibration.raw_brier)}, ECE ${f3(sel.calibration.raw_ece)})`)))));
  const relPts = (arr) => arr.map((b) => ({ x: b.mean_predicted, y: b.observed_rate }));
  makeChart(cal, {
    type: "line",
    data: { datasets: [
      { ...lineDataset("Calibrated", relPts(sel.calibration.calibrated_reliability), MODEL_COLORS[sel.model]), pointRadius: 4 },
      { ...lineDataset("Raw score", relPts(sel.calibration.raw_reliability), COLORS.baseline), pointRadius: 4 },
      lineDataset("Perfect calibration", [{ x: 0, y: 0 }, { x: 1, y: 1 }], COLORS.border, { dash: [4, 4], width: 1 })] },
    options: { parsing: false, scales: { x: { ...axis("Mean predicted probability"), type: "linear", min: 0, max: 1 }, y: axis("Observed fraud-labelled rate", { min: 0, max: 1 }) },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}: predicted ${fmtPct(it.parsed.x)}, observed ${fmtPct(it.parsed.y)}` } } } },
  });

  // Feature importance
  const imp = perf.feature_importance.slice(0, 15);
  const impCanvas = h("canvas", { role: "img", "aria-label": "Permutation feature importance" });
  body.append(h("div", { style: { marginTop: "16px" } }, card("Feature importance", `${perf.importance_method}. Error bars omitted; fold s.d. shown in the table.`,
    h("div", { class: "grid grid-2" }, h("div", { class: "chart-box tall" }, impCanvas),
      table([{ label: "Feature", render: (f) => f.label }, { label: "Group", render: (f) => h("span", { class: "small muted" }, f.group) },
        { label: "PR-AUC drop", num: true, render: (f) => `${f.mean.toFixed(4)} ± ${f.std.toFixed(4)}` }], imp)))));
  makeChart(impCanvas, {
    type: "bar",
    data: { labels: imp.map((f) => f.label), datasets: [{ data: imp.map((f) => f.mean), backgroundColor: MODEL_COLORS[sel.model], borderRadius: 3, barThickness: 12 }] },
    options: { indexAxis: "y", scales: { x: axis("Mean drop in validation PR-AUC", { min: 0 }), y: { grid: { display: false }, ticks: { autoSkip: false } } },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.parsed.x.toFixed(4)}` } } } },
  });

  // Ablation
  const ab = perf.ablation.variants;
  const abCanvas = h("canvas", { role: "img", "aria-label": "Ablation: CV PR-AUC by feature set" });
  body.append(h("div", { style: { marginTop: "16px" } }, card("Technique ablation",
    `${perf.ablation.selected_model} (${perf.ablation.config}) refitted on each feature set. ${perf.ablation.note}`,
    h("div", { class: "chart-box short" }, abCanvas),
    h("div", { style: { marginTop: "12px" } },
      table([
        { label: "Feature set", render: (v) => v.label }, { label: "Features", num: true, render: (v) => String(v.n_features) },
        { label: "CV PR-AUC", num: true, render: (v) => pm(v.cv_pr_auc_mean, v.cv_pr_auc_std) },
        { label: "Δ vs base (paired s.d.)", num: true, render: (v) => (v.variant === "base" ? "—" : `${fmtSigned(v.delta_cv_pr_auc_vs_base, 3)} (${f3(v.delta_fold_std)})`) },
        { label: "Held-out PR-AUC", num: true, render: (v) => f3(v.test_pr_auc) },
        { label: "Held-out capture @100", num: true, render: (v) => fmtPct(v.test_capture_rate_at_100) },
      ], ab)),
    h("p", { class: "small muted", style: { marginTop: "8px" } }, ablationReading(ab)))));
  makeChart(abCanvas, {
    type: "bar",
    data: { labels: ab.map((v) => v.label.replace("Base + ", "+ ").replace(" (all techniques)", "")), datasets: [{ data: ab.map((v) => v.cv_pr_auc_mean),
      backgroundColor: ab.map((v) => (v.variant === "full" ? MODEL_COLORS[sel.model] : "#9FB4BA")), borderRadius: 3, barThickness: 18 }] },
    options: { indexAxis: "y", scales: { x: axis("Mean CV PR-AUC", { format: (v) => v.toFixed(2), min: Math.max(0, Math.floor((Math.min(...ab.map((v) => v.cv_pr_auc_mean)) - 0.08) * 20) / 20), max: Math.min(1, Math.ceil((Math.max(...ab.map((v) => v.cv_pr_auc_mean)) + 0.03) * 20) / 20) }), y: { grid: { display: false } } },
      plugins: { tooltip: { callbacks: { label: (it) => { const v = ab[it.dataIndex]; return ` ${pm(v.cv_pr_auc_mean, v.cv_pr_auc_std)}`; } } } } },
  });

  // Technique outputs
  body.append(h("h2", { class: "section-title" }, "Analytical technique outputs"));
  const reg = analysis.regression;
  const clu = analysis.clustering;
  const rules = analysis.association_rules;
  const scat = h("canvas", { role: "img", "aria-label": "Predicted versus actual claim reimbursement" });
  const pca = h("canvas", { role: "img", "aria-label": "PCA projection of providers" });
  body.append(h("div", { class: "grid grid-2" },
    card("Regression: expected claim reimbursement", `${reg.model} trained on development-provider claims; metrics on claims of providers it never saw.`,
      kvList([
        ["Held-out claims", fmtNumber(reg.test.n_claims)], ["MAE", fmtCurrency(reg.test.mae, { compact: false })],
        ["RMSE", fmtCurrency(reg.test.rmse, { compact: false })], ["R²", reg.test.r2.toFixed(3)],
        ["Development out-of-fold R²", reg.development_out_of_fold.r2.toFixed(3)],
        ["CV validation-fold R² range", `${Math.min(...reg.cv_validation_folds.map((f) => f.r2)).toFixed(3)} – ${Math.max(...reg.cv_validation_folds.map((f) => f.r2)).toFixed(3)}`]]),
      h("div", { class: "chart-box short", style: { marginTop: "12px" } }, scat),
      h("p", { class: "small muted" }, "Random sample of 600 held-out claims. A positive residual means the claim was reimbursed above the model's estimate for comparable claims.")),
    card("Clustering: provider peer groups", `Ward clustering (k = ${clu.n_clusters}, chosen by silhouette) and DBSCAN (eps = ${clu.dbscan_eps.toFixed(2)}, min_samples = ${clu.dbscan_min_samples}); PCA projection of held-out providers (${fmtPct(clu.pca_explained_variance[0] + clu.pca_explained_variance[1], 0)} of variance).`,
      h("div", { class: "chart-box short" }, pca),
      h("div", { class: "legend" }, clu.test_cohort.map((c) => h("span", {}, h("span", { class: "sw", style: { background: PEER_COLORS[c.peer_group % PEER_COLORS.length] } }), `Peer group ${c.peer_group}`)),
        h("span", {}, h("span", { class: "sw", style: { background: COLORS.riskHigh } }), "DBSCAN outlier (×)")),
      table([
        { label: "Group", render: (c) => String(c.peer_group) }, { label: "Providers", num: true, render: (c) => fmtNumber(c.providers) },
        { label: "Outliers", num: true, render: (c) => fmtNumber(c.dbscan_outliers) },
        { label: "Median claims", num: true, render: (c) => fmtNumber(c.median_profile.total_claims) },
        { label: "Median reimbursement", num: true, render: (c) => fmtCurrency(c.median_profile.total_reimbursed) },
        { label: "Inpatient share", num: true, render: (c) => fmtPct(c.median_profile.inpatient_ratio) },
        { label: "Mean predicted prob.", num: true, render: (c) => fmtPct(c.mean_fraud_probability) },
        { label: "Fraud-labelled rate*", num: true, render: (c) => fmtPct(c.retrospective_fraud_rate) },
      ], clu.test_cohort),
      h("p", { class: "small muted", style: { marginTop: "6px" } },
        `*Retrospective. DBSCAN flagged ${clu.test_outliers} held-out providers; their fraud-labelled rate is ${fmtPct(clu.retrospective_fraud_rate_outliers)} vs ${fmtPct(clu.retrospective_fraud_rate_non_outliers)} for the rest — outliers are unusual, not necessarily fraudulent.`))));

  makeChart(scat, {
    type: "scatter",
    data: { datasets: [{ data: reg.scatter_sample.map(([a, p]) => ({ x: p, y: a })), backgroundColor: "rgba(0,137,123,0.45)", pointRadius: 2.5 },
      lineDataset("y = x", [{ x: 0, y: 0 }, { x: 60000, y: 60000 }], COLORS.baseline, { dash: [4, 4], width: 1 })] },
    options: { parsing: false, scales: { x: { ...axis("Predicted reimbursement", { format: (v) => fmtCurrency(v) }), type: "logarithmic", min: 10, max: 60000 },
      y: { ...axis("Actual reimbursement", { format: (v) => fmtCurrency(v) }), type: "logarithmic", min: 10, max: 60000 } },
      plugins: { tooltip: { callbacks: { label: (it) => ` predicted ${fmtCurrency(it.parsed.x)}, actual ${fmtCurrency(it.parsed.y)}` } } } },
  });
  const groups = [...new Set(clu.pca_points.map((p) => p[2]))].sort();
  makeChart(pca, {
    type: "scatter",
    data: { datasets: [
      ...groups.map((g) => ({ label: `Peer group ${g}`, data: clu.pca_points.filter((p) => p[2] === g && !p[3]).map((p) => ({ x: p[0], y: p[1] })),
        backgroundColor: PEER_COLORS[g % PEER_COLORS.length] + "99", pointRadius: 2.5 })),
      { label: "DBSCAN outlier", data: clu.pca_points.filter((p) => p[3]).map((p) => ({ x: p[0], y: p[1] })), backgroundColor: COLORS.riskHigh, borderColor: COLORS.riskHigh, pointStyle: "crossRot", pointRadius: 5, borderWidth: 1.5 }] },
    options: { parsing: false, scales: { x: { ...axis("PC 1"), type: "linear" }, y: axis("PC 2") },
      plugins: { tooltip: { callbacks: { label: (it) => ` ${it.dataset.label}` } } } },
  });

  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    card("Association rules: code combinations linked to fraud-labelled providers",
      `Apriori on ${fmtNumber(rules.n_transactions)} development claims (min support ${fmtPct(rules.params.min_support, 2)}, ${rules.n_items} items, ${rules.n_frequent_itemsets} frequent itemsets). ${rules.n_selected_rules} itemsets pass the provider-diversity and lift filters.`,
      table([
        { label: "Itemset", render: (r) => h("span", { class: "mono small" }, r.itemset.join(" + ")) },
        { label: "Claims", num: true, render: (r) => fmtNumber(r.claims) },
        { label: "Providers", num: true, render: (r) => `${fmtNumber(r.providers)} (${fmtNumber(r.fraud_labelled_providers)})` },
        { label: "Support", num: true, render: (r) => fmtPct(r.support, 2) },
        { label: "Confidence", num: true, render: (r) => fmtPct(r.confidence) },
        { label: "Lift", num: true, render: (r) => r.lift.toFixed(2) },
      ], rules.fraud_rules.slice(0, 12)),
      h("p", { class: "small muted", style: { marginTop: "6px" } },
        "Rule form {itemset} → {claim billed by a fraud-labelled provider}. Support = share of all claims containing the itemset and billed by a fraud-labelled provider; confidence = share of the itemset's claims billed by fraud-labelled providers; lift = confidence ÷ overall fraud-labelled share of claims. Providers column: distinct providers (of which fraud-labelled). Lift > 1 is an association with historically labelled providers, not evidence that a claim is fraudulent.")),
    card("Association rules: frequent code-to-code patterns", "Top rules among codes themselves (unsupervised). These describe normal clinical coding co-occurrence.",
      table([
        { label: "Antecedent → consequent", render: (r) => h("span", { class: "mono small" }, `${r.antecedent.join(" + ")} → ${r.consequent.join(" + ")}`) },
        { label: "Support", num: true, render: (r) => fmtPct(r.support, 2) },
        { label: "Confidence", num: true, render: (r) => fmtPct(r.confidence) },
        { label: "Lift", num: true, render: (r) => r.lift.toFixed(1) },
      ], rules.code_rules.slice(0, 12)))));
}

const PEER_COLORS = ["#00897B", "#5468E0", "#D86A2A", "#A64D9E", "#6B8E23", "#8C6D46", "#3B8EA5", "#9C6644"];

function ablationReading(ab) {
  const base = ab.find((v) => v.variant === "base");
  const parts = ab.filter((v) => v.variant !== "base").map((v) => {
    const sig = Math.abs(v.delta_cv_pr_auc_vs_base) > 2 * (v.delta_fold_std / Math.sqrt(v.fold_pr_auc.length));
    return `${v.label.replace("Base + ", "")}: ${fmtSigned(v.delta_cv_pr_auc_vs_base, 3)}${sig ? "" : " (within fold noise)"}`;
  });
  return `Change in mean CV PR-AUC relative to base features (${f3(base.cv_pr_auc_mean)}): ${parts.join("; ")}. "Within fold noise" means the paired difference is smaller than about two standard errors across the five folds, so the technique cannot be shown to help or hurt on this data.`;
}
