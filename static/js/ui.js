// Minimal DOM helpers. All text is set via textContent, never innerHTML, so API data cannot inject markup.
import { PRIORITY_LABELS, RISK_LABELS, STATUS_LABELS } from "./format.js";

export function h(tag, attrs = {}, ...children) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs || {})) {
    if (v === undefined || v === null || v === false) continue;
    if (k === "class") el.className = v;
    else if (k === "dataset") Object.assign(el.dataset, v);
    else if (k.startsWith("on") && typeof v === "function") el.addEventListener(k.slice(2), v);
    else if (k === "style" && typeof v === "object") Object.assign(el.style, v);
    else if (v === true) el.setAttribute(k, "");
    else el.setAttribute(k, v);
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children.flat(Infinity)) {
    if (c === null || c === undefined || c === false) continue;
    el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) {
  while (el.firstChild) el.removeChild(el.firstChild);
  return el;
}

export function loading(text = "Loading…") {
  return h("div", { class: "loading", role: "status" }, text);
}

export function errorBox(err, onRetry) {
  const details = Array.isArray(err?.details) ? err.details : [];
  return h("div", { class: "error-box", role: "alert" },
    h("strong", {}, err?.status === 503 ? "Service unavailable. " : "Something went wrong. "),
    err?.message || String(err),
    details.length ? h("ul", {}, details.map((d) => h("li", {}, `${d.field}: ${d.message}`))) : null,
    onRetry ? h("div", { style: { marginTop: "8px" } }, h("button", { onclick: onRetry }, "Retry")) : null);
}

export const riskBadge = (r) => h("span", { class: `badge risk-${r}` }, RISK_LABELS[r] || r);
export const priorityBadge = (p) => h("span", { class: `badge prio prio-${p}` }, PRIORITY_LABELS[p] || p);
export const statusBadge = (s) => h("span", { class: `badge status status-${s}` }, STATUS_LABELS[s] || s);

export function card(title, subtitle, ...body) {
  return h("section", { class: "card" },
    title || subtitle ? h("div", { class: "card-head" }, h("div", {}, title ? h("h2", {}, title) : null,
      subtitle ? h("p", {}, subtitle) : null)) : null,
    ...body);
}

export function kpi(label, value, sub, attrs = {}) {
  return h("div", { class: "card kpi", ...attrs }, h("div", { class: "label" }, label),
    h("div", { class: "value" }, value), sub ? h("div", { class: "sub" }, sub) : null);
}

// columns: [{key, label, num, render(row) -> node|string, sortKey}]
export function table(columns, rows, { onRowClick, rowClass, sort, onSort, caption } = {}) {
  const thead = h("thead", {}, h("tr", {}, columns.map((c) => {
    const sortable = Boolean(c.sortKey && onSort);
    const active = sortable && sort && sort.key === c.sortKey;
    return h("th", {
      class: [c.num ? "num" : "", sortable ? "sortable" : ""].join(" ").trim() || null,
      scope: "col",
      "aria-sort": active ? (sort.order === "asc" ? "ascending" : "descending") : null,
      tabindex: sortable ? "0" : null,
      onclick: sortable ? () => onSort(c.sortKey) : null,
      onkeydown: sortable ? (e) => { if (e.key === "Enter") onSort(c.sortKey); } : null,
      title: c.title || null,
    }, c.label, active ? h("span", { class: "sort-ind" }, sort.order === "asc" ? "▲" : "▼") : null);
  })));
  const tbody = h("tbody", {}, rows.length ? rows.map((r) => h("tr", {
    class: [onRowClick ? "clickable" : "", rowClass ? rowClass(r) : ""].join(" ").trim() || null,
    onclick: onRowClick ? () => onRowClick(r) : null,
  }, columns.map((c) => h("td", { class: c.num ? "num" : null }, c.render ? c.render(r) : r[c.key]))))
    : h("tr", {}, h("td", { colspan: String(columns.length), class: "empty" }, "No records match the current filters.")));
  return h("div", { class: "table-wrap" }, h("table", {}, caption ? h("caption", { class: "visually-hidden" }, caption) : null, thead, tbody));
}

export function kvList(pairs) {
  return h("dl", { class: "kv" }, pairs.flatMap(([k, v]) => [h("dt", {}, k), h("dd", {}, v)]));
}

export function storageGet(key, fallback) {
  try {
    const v = window.localStorage.getItem(key);
    return v === null ? fallback : JSON.parse(v);
  } catch {
    return fallback;
  }
}

export function storageSet(key, value) {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage unavailable: non-essential */
  }
}

export function debounce(fn, ms) {
  let t;
  return (...args) => {
    clearTimeout(t);
    t = setTimeout(() => fn(...args), ms);
  };
}
