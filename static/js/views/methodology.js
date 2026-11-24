import { api } from "../api.js";
import { fmtNumber, fmtPct } from "../format.js";
import { card, clear, errorBox, h, kvList, loading, table } from "../ui.js";

const STAGES = [
  ["1", "Ingest & validate", "Four raw CSVs, schema checks, typed codes"],
  ["2", "Preprocess claims", "Joins, dates, missing deductibles kept as unknown"],
  ["3", "Provider split", "80% development / 20% held-out, stratified"],
  ["4", "Feature pipeline", "Aggregates + regression + clustering + Apriori, fitted per fold"],
  ["5", "Classify & calibrate", "4 model families, CV selection, Platt calibration"],
  ["6", "Prioritise audits", "Exposure ranking, capacity simulation"],
];

export async function render(root, ctx) {
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Methodology"),
    h("p", {}, "How the data is processed, how each technique contributes, and which safeguards keep evaluation honest. Figures on this page are read from the training run's artifacts."))));
  const body = h("div", {}, loading());
  root.append(body);
  let m;
  try {
    m = await api.get("/methodology");
  } catch (err) {
    if (ctx.isCurrent()) clear(body).append(errorBox(err, () => render(clear(root), ctx)));
    return;
  }
  if (!ctx.isCurrent()) return;
  clear(body);
  const d = m.data_report;
  const s = m.split;
  const cfg = m.manifest.config;

  body.append(card("Pipeline", null, h("div", { class: "pipeline" }, STAGES.map(([n, t, p]) =>
    h("div", { class: "stage" }, h("div", { class: "step" }, `STEP ${n}`), h("h4", {}, t), h("p", {}, p))))));

  const prose = h("div", { class: "prose" },
    h("h3", {}, "Data sources"),
    h("p", {}, `Kaggle “Healthcare Provider Fraud Detection Analysis” training files: ${fmtNumber(d.inpatient_claims)} inpatient and ${fmtNumber(d.outpatient_claims)} outpatient claims (${fmtNumber(d.claims)} total, ${d.claim_start_min} to ${d.claim_start_max}), ${fmtNumber(d.beneficiaries)} beneficiaries and provider-level labels for ${fmtNumber(d.providers)} providers, of which ${fmtNumber(d.fraud_labelled_providers)} (${fmtPct(d.fraud_prevalence)}) are labelled PotentialFraud = Yes. The label belongs to the provider, not to individual claims.`),
    h("h3", {}, "Preprocessing"),
    h("ul", {},
      h("li", {}, `Claims are stacked with an inpatient flag and joined to beneficiaries on BeneID (many-to-one, validated). Duplicate claim IDs dropped: ${d.duplicate_claim_ids_dropped}.`),
      h("li", {}, `Reimbursement is never imputed (missing: ${d.missing_reimbursement}). ${fmtNumber(d.zero_reimbursement_claims)} $0 claims are kept and flagged.`),
      h("li", {}, `${fmtNumber(d.missing_deductible_claims)} inpatient claims lack a deductible while every other inpatient claim carries the fixed $1,068 Part A deductible, so the value is treated as unknown (NaN + indicator), not zero.`),
      h("li", {}, "Beneficiary annual reimbursement totals are dropped because they include the claim's own reimbursement (target leakage for the regression). Race and gender are excluded from every model."),
      h("li", {}, "Diagnosis and procedure codes are handled as strings; heavy-tailed counts and dollar amounts are log-transformed where distances or linear models are involved.")),
    h("h3", {}, "Provider-level feature engineering"),
    h("p", {}, "One row per provider: billing volume and variability, inpatient/outpatient mix, claim durations, beneficiaries and claims per beneficiary, patient chronic-condition mix, physician patterns (including the share of a provider's physicians who also bill elsewhere) and coding intensity. These aggregations never use the fraud label."),
    h("h3", {}, "Regression — reimbursement deviation"),
    h("p", {}, "A gradient-boosted regressor predicts each claim's reimbursement from patient and treatment information (inpatient flag, DRG, primary and admitting diagnosis category, stay length, coding counts, chronic conditions, age, coverage). Residual = actual − predicted, aggregated per provider (mean, median, share of claims above prediction, total above prediction). Training providers receive out-of-fold residuals so their features are out-of-sample, like those of unseen providers. A positive residual signals billing above comparable claims; it is not proof of fraud."),
    h("h3", {}, "Clustering — unusual provider behaviour"),
    h("p", {}, "Ward hierarchical clustering groups providers into peer groups (k chosen by silhouette). DBSCAN marks providers outside every dense region of behaviour space, with eps from the k-distance distribution. Both are unsupervised and fitted on training providers only; new providers are assigned to the nearest centroid and flagged as outliers when no DBSCAN core point lies within eps."),
    h("h3", {}, "Association rules — code combinations"),
    h("p", {}, `Apriori (min support ${fmtPct(cfg.apriori_min_support, 2)}, itemsets up to ${cfg.apriori_max_len} codes) finds frequent diagnosis/procedure combinations in training claims. Class rules {itemset} → {fraud-labelled provider} are then scored by support, confidence and lift, keeping itemsets seen in ≥ ${cfg.rule_min_claims} claims from ≥ ${cfg.rule_min_providers} providers with lift ≥ ${cfg.rule_min_lift}. Training providers get out-of-fold rule features (rules re-mined without their own fold) so their label never shapes their own feature.`),
    h("h3", {}, "Classification"),
    h("p", {}, "Logistic Regression, Random Forest, AdaBoost and XGBoost are compared, each with a small hyper-parameter grid and class weighting for the ~9% minority class. Hyper-parameters and the model family are chosen by mean 5-fold cross-validated PR-AUC. Class weighting inflates raw scores, so probabilities are recalibrated with Platt scaling fitted on out-of-fold predictions; the decision threshold maximises out-of-fold F1."),
    h("h3", {}, "Audit prioritisation"),
    h("p", {}, h("span", { class: "formula" }, "estimated exposure = calibrated P(fraud-labelled) × total reimbursement")),
    h("p", {}, "This is an exposure proxy for ranking, not an estimate of fraudulent dollars or recoverable money. The simulator compares it with ranking by reimbursement, by probability and at random (seeded repetitions) on the same held-out providers. Retrospective outcomes count reimbursement of selected providers that carry a historical fraud label."),
    h("h3", {}, "Limitations"),
    h("ul", {},
      h("li", {}, "Labels are historical, provider-level and of unknown provenance; unlabelled providers are not verified as legitimate, and a fraud-labelled provider's reimbursement is not all fraudulent."),
      h("li", {}, "Fraud-labelled providers are concentrated among high billers, so “highest reimbursement first” is a strong baseline and the margins between ranked strategies are small."),
      h("li", {}, "The held-out cohort contains about 100 fraud-labelled providers, so held-out metrics carry sampling uncertainty of several percentage points."),
      h("li", {}, "The data is synthetic CMS-style data from a single year; performance on real, current claims is unknown.")));
  body.append(h("div", { style: { marginTop: "16px" } }, card("Technical documentation", null, prose)));

  const checks = m.leakage_checks;
  body.append(h("div", { class: "grid grid-2", style: { marginTop: "16px" } },
    card("Leakage prevention — verified checks", `Assertions executed during training run ${m.manifest.model_version}; training aborts if any fails.`,
      table([{ label: "Check", render: (c) => c.check }, { label: "Result", render: (c) => h("span", { class: c.passed ? "pass" : "fail" }, c.passed ? "Passed" : "Failed") },
        { label: "Detail", render: (c) => h("span", { class: "small muted" }, c.detail) }], checks),
      h("p", { class: "small muted", style: { marginTop: "8px" } }, "Also covered by the automated test-suite: provider disjointness of every split, rules mined only from training claims, out-of-fold features, and feature-order consistency between training and inference.")),
    h("div", {},
      card("Evaluation design", null, kvList([
        ["Development providers", `${fmtNumber(s.development_providers)} (${fmtNumber(s.development_fraud)} fraud-labelled)`],
        ["Held-out providers", `${fmtNumber(s.test_providers)} (${fmtNumber(s.test_fraud)} fraud-labelled)`],
        ["Development claims", fmtNumber(s.development_claims)], ["Held-out claims", fmtNumber(s.test_claims)],
        ["Cross-validation", `${s.cv_folds}-fold stratified, provider-level`], ["Inner out-of-fold folds", String(s.inner_folds)],
        ["Random seed", String(s.random_state)], ["Model version", m.manifest.model_version], ["Trained", m.manifest.trained_at],
        ["Git commit", m.manifest.git_commit || "—"], ["Pipeline runtime", `${fmtNumber(m.manifest.runtime_seconds)} s`]])),
      h("div", { style: { marginTop: "16px" } }, card("Data quality report", null, kvList([
        ["Claims with unlabelled provider", fmtNumber(d.claims_with_unlabelled_provider)], ["Claims end before start", fmtNumber(d.claims_end_before_start)],
        ["Unparseable claim dates", fmtNumber(d.unparseable_claim_dates)], ["Claims without beneficiary record", fmtNumber(d.claims_without_beneficiary_record)],
        ["Claims after beneficiary death", fmtNumber(d.claims_after_beneficiary_death)],
        ["Beneficiary rows with negative annual amounts (dropped columns)", fmtNumber(d.negative_annual_reimbursement_rows)]]))))));

  const fc = m.feature_catalog;
  body.append(h("div", { style: { marginTop: "16px" } }, card(`Model feature schema (${fc.features.length} features, in model order)`,
    "Derived features are recomputed from primary inputs by the same function in training and inference.",
    table([{ label: "#", num: true, render: (f) => String(fc.features.indexOf(f) + 1) }, { label: "Feature", render: (f) => h("span", { class: "mono small" }, f.feature) },
      { label: "Description", render: (f) => f.label }, { label: "Group", render: (f) => fc.groups[f.group] || f.group },
      { label: "Derived", render: (f) => (f.derived ? "Yes" : "") }], fc.features))));
}
