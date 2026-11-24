/* global Chart */
// Chart.js helpers: consistent theme, fixed entity colours, and a registry so charts are destroyed on navigation.

export const COLORS = {
  text: "#243746",
  text2: "#687782",
  grid: "#EEF1F3",
  border: "#E1E7EA",
  baseline: "#8A969E",
  riskHigh: "#B54747",
  accent: "#287D78",
};

// Colour follows the entity (never its rank). Validated categorical palette (see README).
export const STRATEGY_STYLE = {
  exposure: { color: "#00897B", label: "Risk-weighted exposure (proposed)" },
  probability: { color: "#5468E0", label: "Fraud probability only" },
  reimbursement: { color: "#D86A2A", label: "Highest reimbursement first" },
  random: { color: "#8A969E", label: "Random selection (mean)", dash: [5, 4] },
};
export const MODEL_COLORS = {
  xgboost: "#00897B",
  random_forest: "#5468E0",
  adaboost: "#D86A2A",
  logistic_regression: "#A64D9E",
};

const registry = new Set();

export function destroyCharts() {
  for (const c of registry) c.destroy();
  registry.clear();
}

let configured = false;
function configure() {
  if (configured || typeof Chart === "undefined") return;
  Chart.defaults.font.family = '"IBM Plex Sans", system-ui, sans-serif';
  Chart.defaults.font.size = 12;
  Chart.defaults.color = COLORS.text2;
  Chart.defaults.borderColor = COLORS.grid;
  Chart.defaults.animation = false;
  Chart.defaults.maintainAspectRatio = false;
  Chart.defaults.plugins.legend.display = false; // HTML legends are rendered next to charts
  Chart.defaults.plugins.tooltip.backgroundColor = "#FFFFFF";
  Chart.defaults.plugins.tooltip.titleColor = COLORS.text;
  Chart.defaults.plugins.tooltip.bodyColor = COLORS.text;
  Chart.defaults.plugins.tooltip.borderColor = COLORS.border;
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.padding = 10;
  Chart.defaults.plugins.tooltip.boxPadding = 4;
  configured = true;
}

// Draws a vertical reference line at x = options.plugins.marker.x (category index or linear value).
const markerPlugin = {
  id: "marker",
  afterDatasetsDraw(chart, _args, opts) {
    if (opts == null || opts.x == null) return;
    const xScale = chart.scales.x;
    const x = xScale.getPixelForValue(opts.x);
    if (!Number.isFinite(x) || x < chart.chartArea.left || x > chart.chartArea.right) return;
    const { ctx, chartArea } = chart;
    ctx.save();
    ctx.strokeStyle = COLORS.text;
    ctx.setLineDash([3, 3]);
    ctx.lineWidth = 1;
    ctx.beginPath();
    ctx.moveTo(x, chartArea.top);
    ctx.lineTo(x, chartArea.bottom);
    ctx.stroke();
    if (opts.label) {
      ctx.setLineDash([]);
      ctx.fillStyle = COLORS.text;
      ctx.font = '500 11px "IBM Plex Sans", sans-serif';
      ctx.textAlign = x > chartArea.right - 120 ? "right" : "left";
      ctx.fillText(opts.label, x + (ctx.textAlign === "left" ? 6 : -6), chartArea.top + 12);
    }
    ctx.restore();
  },
};

export function makeChart(canvas, config) {
  configure();
  if (typeof Chart === "undefined") {
    canvas.replaceWith(Object.assign(document.createElement("div"), { className: "empty", textContent: "Charting library failed to load." }));
    return null;
  }
  config.plugins = [...(config.plugins || []), markerPlugin];
  const chart = new Chart(canvas, config);
  registry.add(chart);
  return chart;
}

export function lineDataset(label, data, color, { dash, fill, width = 2 } = {}) {
  return {
    label, data, borderColor: color, backgroundColor: color, borderWidth: width, pointRadius: 0,
    pointHoverRadius: 4, borderDash: dash || [], fill: fill ?? false, tension: 0,
  };
}

export function axis(title, { format, min, max, type = "linear", grid = true } = {}) {
  return {
    type, min, max,
    title: title ? { display: true, text: title, color: COLORS.text2 } : undefined,
    grid: { display: grid, color: COLORS.grid, drawTicks: false },
    border: { color: COLORS.border },
    ticks: { padding: 6, callback: format ? (v) => format(v) : undefined, maxTicksLimit: 8 },
  };
}
