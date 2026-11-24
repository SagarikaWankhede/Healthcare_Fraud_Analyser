import { api } from "../api.js";
import { STRATEGY_STYLE, axis, makeChart } from "../charts.js";
import { STATUS_LABELS, clamp, fmtCurrency, fmtNumber, fmtPct } from "../format.js";
import { clear, debounce, errorBox, h, loading, priorityBadge, riskBadge, statusBadge, storageGet, storageSet, table } from "../ui.js";
import { strategyTable } from "./overview.js";

const PRESETS = [10, 25, 50, 100, 150, 200];
const ORDER = ["exposure", "probability", "reimbursement", "random"];

export async function render(root, ctx) {
  root.append(h("div", { class: "page-head" }, h("div", {}, h("h1", {}, "Audit Priorities"),
    h("p", {}, "Choose how many providers the investigation team can review. The queue ranks held-out providers by estimated financial exposure; the comparison shows how each selection strategy would have performed against historical labels."))));

  let schema;
  try {
    schema = await api.get("/audit-simulation", { capacity: 1 });
  } catch (err) {
    if (ctx.isCurrent()) root.append(errorBox(err, () => render(clear(root), ctx)));
    return;
  }
  if (!ctx.isCurrent()) return;
  const N = schema.cohort.providers;

  const state = {
    capacity: clamp(storageGet("mg.capacity", 100), 1, N),
    q: { page: 1, per_page: 25, sort: "rank_exposure", order: "asc", search: "", risk_level: "", priority: "", status: "",
      min_probability: "", min_exposure: "", onlyCapacity: true },
  };

  // ---- capacity control ----
  const capValue = h("strong", { class: "num", "data-testid": "capacity-value" });
  const slider = h("input", { type: "range", min: "1", max: String(N), value: String(state.capacity), "aria-label": "Audit capacity", "data-testid": "capacity-slider" });
  const presetSeg = h("div", { class: "seg", role: "group", "aria-label": "Capacity presets" },
    PRESETS.filter((p) => p <= N).map((p) => h("button", { "data-capacity": String(p), onclick: () => setCapacity(p) }, String(p))));
  const capSummary = h("div", { class: "row small muted" });
  const compareBox = h("div", {}, loading());
  const barCanvas = h("canvas", { role: "img", "aria-label": "Capture rate by strategy" });
  let barChart = null;

  root.append(h("div", { class: "grid grid-3" },
    h("section", { class: "card span-2" },
      h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Audit capacity"),
        h("p", {}, `Number of providers the team can investigate (cohort of ${fmtNumber(N)} held-out providers).`))),
      h("div", { class: "row" }, h("span", {}, "Providers to review: ", capValue), presetSeg),
      h("div", { style: { margin: "14px 0 8px" } }, slider),
      capSummary),
    h("section", { class: "card" }, h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Capture rate at this capacity"),
      h("p", {}, "Known fraud-labelled reimbursement captured ÷ total in cohort."))), h("div", { class: "chart-box short" }, barCanvas))));

  root.append(h("section", { class: "card", style: { marginTop: "16px" } }, h("div", { class: "card-head" }, h("div", {}, h("h2", { "data-testid": "compare-title" }, "Strategy comparison"),
      h("p", {}, "Retrospective evaluation on identical held-out providers. Random selection: mean (5th–95th percentile)."))), compareBox));

  async function refreshComparison() {
    const k = state.capacity;
    try {
      const sim = await api.get("/audit-simulation", { capacity: k });
      if (!ctx.isCurrent() || k !== state.capacity) return;
      clear(compareBox).append(strategyTable(sim.strategies));
      const prop = sim.strategies.find((s) => s.strategy === "exposure");
      const by = Object.fromEntries(sim.strategies.map((s) => [s.strategy, s]));
      const same = ["probability", "reimbursement"].filter((s) => by[s].overlap_with_proposed === k);
      if (same.length) {
        compareBox.append(h("p", { class: "small muted", style: { marginTop: "8px" } },
          `At capacity ${k}, ${same.map((s) => by[s].label.toLowerCase()).join(" and ")} selects exactly the same providers as the proposed ranking, so the outcomes are identical.`));
      }
      const data = ORDER.map((s) => by[s].capture_rate);
      if (barChart) {
        barChart.data.datasets[0].data = data;
        barChart.update();
      } else {
        barChart = makeChart(barCanvas, {
          type: "bar",
          data: { labels: ORDER.map((s) => STRATEGY_STYLE[s].label.replace(" (proposed)", "").replace(" (mean)", "")),
            datasets: [{ data, backgroundColor: ORDER.map((s) => STRATEGY_STYLE[s].color), borderRadius: 4, barThickness: 18 }] },
          options: { indexAxis: "y", scales: { x: axis(null, { min: 0, max: 1, format: (v) => fmtPct(v, 0) }), y: { grid: { display: false } } },
            plugins: { tooltip: { callbacks: { label: (it) => ` ${fmtPct(it.parsed.x)}` } } } },
        });
      }
      capSummary.textContent = "";
      capSummary.append(`Selected providers carry ${fmtCurrency(prop.estimated_exposure_selected)} estimated exposure and ${fmtCurrency(prop.reimbursement_reviewed)} total reimbursement.`);
    } catch (err) {
      if (ctx.isCurrent()) clear(compareBox).append(errorBox(err));
    }
  }

  // ---- queue ----
  const queueBox = h("div", { "data-testid": "queue" }, loading());
  const pager = h("div", { class: "pagination" });
  const search = h("input", { type: "search", placeholder: "e.g. PRV51001", "aria-label": "Search provider ID", "data-testid": "search" });
  const sel = (name, label, options) => h("label", { class: "field" }, label,
    h("select", { "data-filter": name, onchange: (e) => { state.q[name] = e.target.value; state.q.page = 1; loadQueue(); } },
      options.map(([v, t]) => h("option", { value: v }, t))));
  const minProb = h("input", { type: "number", min: "0", max: "100", step: "1", placeholder: "0", "aria-label": "Minimum fraud probability (%)", "data-testid": "min-prob" });
  const minExp = h("input", { type: "number", min: "0", step: "1000", placeholder: "0", "aria-label": "Minimum estimated exposure ($)", "data-testid": "min-exposure" });
  const onlyCap = h("input", { type: "checkbox", checked: true, "data-testid": "only-capacity" });
  const filterError = h("div", { class: "field-error" });
  search.addEventListener("input", debounce(() => { state.q.search = search.value.trim(); state.q.page = 1; loadQueue(); }, 250));
  const numFilter = (input, key, scale) => input.addEventListener("input", debounce(() => {
    const v = input.value === "" ? "" : Number(input.value);
    if (v !== "" && (!Number.isFinite(v) || v < 0 || (key === "min_probability" && v > 100))) {
      filterError.textContent = key === "min_probability" ? "Probability must be between 0 and 100." : "Exposure must be 0 or more.";
      return;
    }
    filterError.textContent = "";
    state.q[key] = v === "" ? "" : v * scale;
    state.q.page = 1;
    loadQueue();
  }, 300));
  numFilter(minProb, "min_probability", 0.01);
  numFilter(minExp, "min_exposure", 1);
  onlyCap.addEventListener("change", () => { state.q.onlyCapacity = onlyCap.checked; state.q.page = 1; loadQueue(); });

  root.append(h("section", { class: "card", style: { marginTop: "16px" } },
    h("div", { class: "card-head" }, h("div", {}, h("h2", {}, "Provider Audit Queue"),
      h("p", {}, "Ranked by estimated exposure = calibrated fraud probability × total reimbursement. Select a provider to open its profile."))),
    h("div", { class: "filters" },
      h("label", { class: "field" }, "Provider ID", search),
      sel("risk_level", "Risk level", [["", "All"], ["high", "High (≥ 50%)"], ["medium", "Medium (15–50%)"], ["low", "Low (< 15%)"]]),
      sel("priority", "Audit priority", [["", "All"], ["P1", "P1"], ["P2", "P2"], ["P3", "P3"], ["Routine", "Routine"]]),
      sel("status", "Investigation status", [["", "All"], ...Object.entries(STATUS_LABELS)]),
      h("label", { class: "field" }, "Min. probability (%)", minProb),
      h("label", { class: "field" }, "Min. exposure ($)", minExp),
      h("label", { class: "field" }, "Rows per page", h("select", { "data-testid": "per-page", onchange: (e) => { state.q.per_page = Number(e.target.value); state.q.page = 1; loadQueue(); } },
        [10, 25, 50, 100].map((n) => h("option", { value: String(n), selected: n === 25 }, String(n))))),
      h("label", { class: "row small", style: { gap: "6px", alignSelf: "center" } }, onlyCap, "Only within capacity")),
    filterError, queueBox, pager,
    h("p", { class: "small muted", style: { marginTop: "8px" } },
      "Priority: P1 = top 5% of cohort exposure, P2 = next 10%, P3 = other medium/high-risk providers, Routine = remainder. ",
      "Main indicators are the features that most increase this provider's estimate relative to a typical provider.")));

  const setSort = (key) => {
    if (state.q.sort === key) state.q.order = state.q.order === "asc" ? "desc" : "asc";
    else { state.q.sort = key; state.q.order = key === "rank_exposure" || key === "Provider" ? "asc" : "desc"; }
    state.q.page = 1;
    loadQueue();
  };

  let queueToken = 0;
  async function loadQueue() {
    const token = ++queueToken;
    queueBox.style.opacity = "0.6";
    const q = state.q;
    const params = { page: q.page, per_page: q.per_page, sort: q.sort, order: q.order, search: q.search, risk_level: q.risk_level,
      priority: q.priority, status: q.status, min_probability: q.min_probability, min_exposure: q.min_exposure,
      within_capacity: q.onlyCapacity ? state.capacity : "" };
    try {
      const res = await api.get("/providers", params);
      if (!ctx.isCurrent() || token !== queueToken) return;
      queueBox.style.opacity = "1";
      clear(queueBox).append(table([
        { label: "Rank", num: true, sortKey: "rank_exposure", render: (r) => String(r.rank) },
        { label: "Provider ID", sortKey: "Provider", render: (r) => h("a", { href: `#/providers/${encodeURIComponent(r.provider_id)}`, class: "mono" }, r.provider_id) },
        { label: "Fraud probability", num: true, sortKey: "fraud_probability", render: (r) => fmtPct(r.fraud_probability) },
        { label: "Total reimbursement", num: true, sortKey: "total_reimbursed", render: (r) => fmtCurrency(r.total_reimbursed) },
        { label: "Estimated exposure", num: true, sortKey: "estimated_exposure", render: (r) => fmtCurrency(r.estimated_exposure) },
        { label: "Claims", num: true, sortKey: "total_claims", render: (r) => fmtNumber(r.total_claims) },
        { label: "Beneficiaries", num: true, sortKey: "unique_beneficiaries", render: (r) => fmtNumber(r.unique_beneficiaries) },
        { label: "Main risk indicators", render: (r) => h("span", { class: "small" }, r.main_indicators.join("; ") || "—") },
        { label: "Risk", render: (r) => riskBadge(r.risk_level) },
        { label: "Priority", render: (r) => priorityBadge(r.audit_priority) },
        { label: "Status", render: (r) => statusBadge(r.investigation_status) },
      ], res.providers, {
        sort: { key: q.sort, order: q.order }, onSort: setSort,
        onRowClick: (r) => { window.location.hash = `#/providers/${encodeURIComponent(r.provider_id)}`; },
        rowClass: (r) => (!q.onlyCapacity && r.rank === state.capacity + 1 && q.sort === "rank_exposure" && q.order === "asc" ? "capacity-break" : ""),
        caption: "Provider audit queue",
      }));
      clear(pager).append(
        h("span", { class: "muted", "data-testid": "queue-total" }, `${fmtNumber(res.total)} providers`),
        h("span", { class: "spacer" }),
        h("button", { disabled: res.page <= 1, onclick: () => { state.q.page = res.page - 1; loadQueue(); }, "data-testid": "prev-page" }, "Previous"),
        h("span", { "data-testid": "page-indicator" }, `Page ${res.page} of ${res.pages}`),
        h("button", { disabled: res.page >= res.pages, onclick: () => { state.q.page = res.page + 1; loadQueue(); }, "data-testid": "next-page" }, "Next"));
    } catch (err) {
      if (ctx.isCurrent() && token === queueToken) {
        queueBox.style.opacity = "1";
        clear(queueBox).append(errorBox(err, loadQueue));
      }
    }
  }

  const refreshAll = debounce(() => { refreshComparison(); if (state.q.onlyCapacity) { state.q.page = 1; loadQueue(); } }, 120);
  function setCapacity(k) {
    state.capacity = clamp(Math.round(k), 1, N);
    storageSet("mg.capacity", state.capacity);
    capValue.textContent = fmtNumber(state.capacity);
    slider.value = String(state.capacity);
    presetSeg.querySelectorAll("button").forEach((b) => b.classList.toggle("active", Number(b.dataset.capacity) === state.capacity));
    refreshAll();
  }
  slider.addEventListener("input", () => setCapacity(Number(slider.value)));

  setCapacity(state.capacity);
}
