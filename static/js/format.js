// Pure formatting helpers (no DOM) — unit tested with `node --test`.

const isNum = (v) => typeof v === "number" && Number.isFinite(v);

export function fmtNumber(v, digits = 0) {
  if (!isNum(v)) return "—";
  return v.toLocaleString("en-US", { minimumFractionDigits: digits, maximumFractionDigits: digits });
}

export function fmtCurrency(v, { compact = true, digits } = {}) {
  if (!isNum(v)) return "—";
  const abs = Math.abs(v);
  const sign = v < 0 ? "-" : "";
  if (compact && abs >= 1e9) return `${sign}$${(abs / 1e9).toFixed(digits ?? 2)}B`;
  if (compact && abs >= 1e6) return `${sign}$${(abs / 1e6).toFixed(digits ?? 2)}M`;
  if (compact && abs >= 1e4) return `${sign}$${(abs / 1e3).toFixed(digits ?? 1)}K`;
  return `${sign}$${abs.toLocaleString("en-US", { maximumFractionDigits: digits ?? 0 })}`;
}

// v is a fraction in [0, 1]
export function fmtPct(v, digits = 1) {
  if (!isNum(v)) return "—";
  return `${(v * 100).toFixed(digits)}%`;
}

export function fmtSigned(v, digits = 3) {
  if (!isNum(v)) return "—";
  return `${v > 0 ? "+" : ""}${v.toFixed(digits)}`;
}

export function fmtByKind(v, kind) {
  switch (kind) {
    case "amount":
    case "signed_amount":
      return fmtCurrency(v, { compact: Math.abs(v ?? 0) >= 1e4 });
    case "ratio":
      return fmtPct(v, 1);
    case "count":
      return fmtNumber(v, 0);
    case "binary":
      return v === 1 ? "Yes" : v === 0 ? "No" : "—";
    case "days":
    case "years":
      return isNum(v) ? fmtNumber(v, 1) : "—";
    default:
      return isNum(v) ? fmtNumber(v, Math.abs(v) < 10 ? 2 : 1) : "—";
  }
}

export const STATUS_LABELS = {
  not_reviewed: "Not reviewed",
  under_review: "Under review",
  escalated: "Escalated",
  closed: "Closed",
};

export const RISK_LABELS = { high: "High", medium: "Medium", low: "Low" };

export const PRIORITY_LABELS = { P1: "P1", P2: "P2", P3: "P3", Routine: "Routine" };

// Build a query string, dropping empty values.
export function buildQuery(params = {}) {
  const q = new URLSearchParams();
  for (const [k, v] of Object.entries(params)) {
    if (v === undefined || v === null || v === "" || (typeof v === "number" && Number.isNaN(v))) continue;
    q.set(k, String(v));
  }
  const s = q.toString();
  return s ? `?${s}` : "";
}

// Parse the hash route: "#/providers/PRV1?x=1" -> { name: "providers", param: "PRV1" }
export function parseRoute(hash) {
  const clean = (hash || "").replace(/^#\/?/, "").split("?")[0];
  const [name, ...rest] = clean.split("/").filter(Boolean);
  return { name: name || "overview", param: rest.length ? decodeURIComponent(rest.join("/")) : null };
}

export function clamp(v, lo, hi) {
  return Math.min(hi, Math.max(lo, v));
}
